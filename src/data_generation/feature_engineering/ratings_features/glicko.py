"""
Glicko-2 Rating System with Hyperparameter Tuning using Optuna

This module mirrors the Elo-based infrastructure but replaces the Elo methods
with Glicko-2 functions from the 'glicko2' library.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import optuna
from glicko2 import Glicko2, Rating
from sklearn.metrics import log_loss
from tqdm import tqdm

from utils.io_utils import get_sorting_keys, json_loader
from utils.logger import LOG_TOPIC, instantiate_logger, logger
from utils.paths import (
    CONSIDERED_LEAGUES,
    DEFAULT_MODELS_PARAMETERS,
    ENTITY_GLICKO_HYPERPARAMETERS,
    LEAGUE_ELO,
)
from utils.pd import pd

# ----------------------------------------------------------------------
# Global Config / Constants
# ----------------------------------------------------------------------
config = json_loader(DEFAULT_MODELS_PARAMETERS)
considered_leagues_config = json_loader(CONSIDERED_LEAGUES)
data_pipeline_logger = instantiate_logger(LOG_TOPIC.DATA_PIPELINE)

MAJOR_LEAGUES = considered_leagues_config["major_leagues"]
CROSS_LEAGUE_COMPETITIONS = considered_leagues_config["cross_league_competitions"]
TRIALS_NUM = 25

# Default Glicko-2 parameters (parallel to how Elo had baseline ratings)
DEFAULT_MU = config.get("glicko2", {}).get("mu", 1500.0)
DEFAULT_PHI = config.get("glicko2", {}).get("phi", 350.0)
DEFAULT_SIGMA = config.get("glicko2", {}).get("sigma", 0.06)

WIN = 1  # Helper constant for clarity in `rate_match_using_mean`


# ----------------------------------------------------------------------
# Helper Utilities
# ----------------------------------------------------------------------
def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a numeric `value` between `min_val` and `max_val`."""
    return max(min_val, min(value, max_val))


def is_major_league(league: str) -> bool:
    """Check if a league is considered major."""
    return league in MAJOR_LEAGUES


# ------------------------------------------------------------------------------
# 1. Preprocessing (mirrors Elo's preprocessing)
# ------------------------------------------------------------------------------
def calculate_mean_rating(ratings: list[Rating]) -> Rating:
    """Mean Glicko-2 rating for a group of Ratings (mu/phi/sigma averaged)."""
    if not ratings:
        return Rating(mu=DEFAULT_MU, phi=DEFAULT_PHI, sigma=DEFAULT_SIGMA)
    n = float(len(ratings))
    return Rating(
        mu=sum(r.mu for r in ratings) / n,
        phi=sum(r.phi for r in ratings) / n,
        sigma=sum(r.sigma for r in ratings) / n,
    )


def rate_match_using_mean(
    model: Glicko2, team_ratings: list[Rating], opp_mean_rating: Rating, result: int
) -> list[Rating]:
    """
    Rate each player in a team using the team's mean rating vs the opponent's mean rating.
    """
    updated_ratings: list[Rating] = []
    for rating in team_ratings:
        if result == WIN:
            new_rating, _ = model.rate_1vs1(rating, opp_mean_rating)
        else:
            _, new_rating = model.rate_1vs1(opp_mean_rating, rating)
        updated_ratings.append(new_rating)
    return updated_ratings


def preprocess_glicko2_dataframe(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Validate, normalize, and stably sort the input DataFrame for Glicko-2 computation.
    """
    if entity.lower() not in ["team", "player"]:
        msg = "Entity must be 'team' or 'player'"
        raise ValueError(msg)

    entity_key = "teamid" if entity.lower() == "team" else "playerid"

    required_columns = [
        "season",
        "date",
        "gameid",
        entity_key,
        "league",
        "side",
        "result",
    ]
    if entity.lower() == "player":
        required_columns.append("position")

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        msg = f"Input DataFrame is missing required columns: {missing_columns}"
        raise ValueError(msg)

    df = df.copy()

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        null_count = df["date"].isna().sum()
        if null_count > 0:
            logger.warning(
                f"{null_count} 'date' entries could not be converted; dropping them."
            )
            data_pipeline_logger.warning(
                f"{null_count} 'date' entries could not be converted; dropping them."
            )
            df = df.dropna(subset=["date"]).copy()

    # Drop rows missing 'league' or 'result'
    if df["league"].isna().any() or df["result"].isna().any():
        n_missing_leagues = df["league"].isna().sum()
        n_missing_results = df["result"].isna().sum()
        logger.warning(
            f"{n_missing_leagues} 'league' and {n_missing_results} 'result' missing; dropping them."
        )
        data_pipeline_logger.warning(
            f"{n_missing_leagues} 'league' and {n_missing_results} 'result' missing; dropping them."
        )
        df = df.dropna(subset=["league", "result"]).reset_index(drop=True)

    # Normalize 'result' to numeric 0/1 if needed (mirror Elo)
    if not pd.api.types.is_numeric_dtype(df["result"]):
        valmap = {
            "W": 1,
            "Win": 1,
            "win": 1,
            True: 1,
            "L": 0,
            "Loss": 0,
            "loss": 0,
            False: 0,
        }
        df["result"] = df["result"].map(valmap).astype("float64")

    # Stable sort by canonical keys (ensures player row alignment)
    return df.sort_values(by=get_sorting_keys(entity), kind="mergesort").reset_index(
        drop=True
    )


# ----------------------------------------------------------------------
# 2. Core Glicko-2 Functions
# ----------------------------------------------------------------------
def update_glicko2_rating(
    own_rating: Rating, opp_rating: Rating, actual_result: float, model: Glicko2
) -> Rating:
    """Update a single entity's Glicko-2 rating for a single match using 1vs1."""
    if actual_result >= 1.0:
        new_win_rating, _ = model.rate_1vs1(own_rating, opp_rating)
        return new_win_rating
    _, new_loser_rating = model.rate_1vs1(opp_rating, own_rating)
    return new_loser_rating


# ----------------------------------------------------------------------
# 3. Season / Entity Lifecycle Utilities
# ----------------------------------------------------------------------
def linear_decay_reset(
    glicko2_ratings: dict[int | str, dict[str, Any]],
    current_season: int,
    baseline_mu: float,
    decay_factor: float,
) -> dict[int | str, dict[str, Any]]:
    """Partially decay mu toward baseline if old season < current_season."""
    for entity_id, data in glicko2_ratings.items():
        if data["season"] < current_season:
            old_rating: Rating = data["rating"]
            delta_mu = old_rating.mu - baseline_mu
            reset_mu = baseline_mu + delta_mu * decay_factor
            glicko2_ratings[entity_id]["rating"] = Rating(
                mu=reset_mu, phi=old_rating.phi, sigma=old_rating.sigma
            )
            glicko2_ratings[entity_id]["season"] = current_season
    return glicko2_ratings


def handle_position_switch(
    entity_id: int | str,
    new_position: str | None,
    glicko2_ratings: dict[int | str, dict[str, Any]],
    baseline_mu: float,
    position_reset_factor: float,
) -> None:
    """Partial mu reset toward baseline when a player's position changes."""
    if not new_position:
        return
    last_position = glicko2_ratings[entity_id].get("last_position")
    if last_position and last_position != new_position:
        old_rating: Rating = glicko2_ratings[entity_id]["rating"]
        delta_mu = old_rating.mu - baseline_mu
        reset_mu = baseline_mu + delta_mu * (1.0 - position_reset_factor)
        glicko2_ratings[entity_id]["rating"] = Rating(
            mu=reset_mu, phi=old_rating.phi, sigma=old_rating.sigma
        )
    glicko2_ratings[entity_id]["last_position"] = new_position


def handle_new_entity(
    ent_id: int | str,
    glicko2_ratings: dict[int | str, dict[str, Any]],
    league_elo_dict: dict[str, float],
    new_league: str,
    current_season: int,
    baseline_mu: float,
    init_adjust_factor: float,
) -> None:
    """Initialize Glicko-2 rating for a new entity, referencing league offsets."""
    max_diff = 0.2 * baseline_mu
    avg_league_elo = (
        sum(league_elo_dict.values()) / len(league_elo_dict)
        if league_elo_dict
        else baseline_mu
    )
    league_elo = league_elo_dict.get(new_league, baseline_mu)
    init_adjustment = (league_elo - avg_league_elo) * init_adjust_factor
    initial_mu = baseline_mu + clamp(init_adjustment, -max_diff, max_diff)

    glicko2_ratings[ent_id] = {
        "rating": Rating(mu=initial_mu, phi=DEFAULT_PHI, sigma=DEFAULT_SIGMA),
        "season": current_season,
        "league": new_league,
    }


def handle_league_swap(
    ent_id: int | str,
    new_league: str,
    glicko2_ratings: dict[int | str, dict[str, Any]],
    league_elo_dict: dict[str, float],
    baseline_mu: float,
    transfer_factor: float,
    transfer_factor_minor_to_major: float = 0.4,
) -> None:
    """Handle league changes with partial mu adjustments."""
    curr_league = glicko2_ratings[ent_id].get("league")
    if not curr_league or curr_league == new_league:
        glicko2_ratings[ent_id]["league"] = new_league
        return

    curr_is_major = is_major_league(curr_league)
    new_is_major = is_major_league(new_league)

    curr_elo_val = league_elo_dict.get(curr_league, baseline_mu)
    new_elo_val = league_elo_dict.get(new_league, baseline_mu)
    diff = new_elo_val - curr_elo_val

    old_rating: Rating = glicko2_ratings[ent_id]["rating"]
    old_mu = old_rating.mu

    if "player" in str(ent_id).lower() and not curr_is_major and new_is_major:
        adjusted_diff = transfer_factor_minor_to_major * diff
        new_mu = old_mu - adjusted_diff
    else:
        adjusted_diff = transfer_factor * diff
        new_mu = old_mu + adjusted_diff

    glicko2_ratings[ent_id]["rating"] = Rating(
        mu=new_mu, phi=old_rating.phi, sigma=old_rating.sigma
    )
    glicko2_ratings[ent_id]["league"] = new_league


# ----------------------------------------------------------------------
# 4. Game Processing
# ----------------------------------------------------------------------
def process_game(
    df: pd.DataFrame,
    game_group: pd.DataFrame,
    glicko2_ratings: dict[int | str, dict[str, Any]],
    glicko2_model: Glicko2,
    entity: str,
    entity_key: str,
    baseline_mu: float,
    decay_factor: float,
    league_elo_dict: dict[str, float],
    transfer_factor: float,
    initial_elo_adjustment_factor: float,
    position_reset_factor: float = 0.2,
) -> dict[int | str, dict[str, Any]]:
    """Process a single grouped game, updating Glicko-2 ratings for both sides."""
    current_season = game_group.iloc[0]["season"]

    # Seasonal decay
    linear_decay_reset(
        glicko2_ratings=glicko2_ratings,
        current_season=current_season,
        baseline_mu=baseline_mu,
        decay_factor=decay_factor,
    )

    # Check/init each entity
    entity_ids = game_group[entity_key].unique()
    for ent_id in entity_ids:
        new_league = game_group.loc[game_group[entity_key] == ent_id, "league"].iloc[0]
        if ent_id not in glicko2_ratings:
            handle_new_entity(
                ent_id=ent_id,
                glicko2_ratings=glicko2_ratings,
                league_elo_dict=league_elo_dict,
                new_league=new_league,
                current_season=current_season,
                baseline_mu=baseline_mu,
                init_adjust_factor=initial_elo_adjustment_factor,
            )
        elif new_league not in CROSS_LEAGUE_COMPETITIONS:
            handle_league_swap(
                ent_id=ent_id,
                new_league=new_league,
                glicko2_ratings=glicko2_ratings,
                league_elo_dict=league_elo_dict,
                baseline_mu=baseline_mu,
                transfer_factor=transfer_factor,
            )

    # Position switching (players)
    if entity.lower() == "player":
        for _, row in game_group.iterrows():
            handle_position_switch(
                entity_id=row[entity_key],
                new_position=row.get("position"),
                glicko2_ratings=glicko2_ratings,
                baseline_mu=baseline_mu,
                position_reset_factor=position_reset_factor,
            )

    # Split sides, gather old ratings
    blue_rows = game_group[game_group["side"] == "Blue"]
    red_rows = game_group[game_group["side"] == "Red"]

    blue_ids = blue_rows[entity_key].to_numpy()
    red_ids = red_rows[entity_key].to_numpy()

    blue_old_ratings = [glicko2_ratings[b]["rating"] for b in blue_ids]
    red_old_ratings = [glicko2_ratings[r]["rating"] for r in red_ids]

    # Team means, expected outcome
    blue_mean_rating = calculate_mean_rating(blue_old_ratings)
    red_mean_rating = calculate_mean_rating(red_old_ratings)
    opp_impact = glicko2_model.reduce_impact(red_mean_rating)
    blue_expected = glicko2_model.expect_score(
        blue_mean_rating, red_mean_rating, opp_impact
    )
    result = float(blue_rows.iloc[0]["result"])
    red_result = 1.0 - result

    # Rate both sides with mean-vs-mean
    updated_blue_ratings = rate_match_using_mean(
        glicko2_model,
        team_ratings=blue_old_ratings,
        opp_mean_rating=red_mean_rating,
        result=int(result),
    )
    updated_red_ratings = rate_match_using_mean(
        glicko2_model,
        team_ratings=red_old_ratings,
        opp_mean_rating=blue_mean_rating,
        result=int(red_result),
    )

    # Save updates in dictionary
    for i, bid in enumerate(blue_ids):
        glicko2_ratings[bid]["rating"] = updated_blue_ratings[i]
    for i, rid in enumerate(red_ids):
        glicko2_ratings[rid]["rating"] = updated_red_ratings[i]

    # Write columns back to df (per row) — opp columns are per-opponent entity, mirroring Elo behavior
    df.loc[blue_rows.index, "glicko2_mu_before"] = [r.mu for r in blue_old_ratings]
    df.loc[blue_rows.index, "glicko2_phi_before"] = [r.phi for r in blue_old_ratings]
    df.loc[blue_rows.index, "opp_glicko2_mu_before"] = [r.mu for r in red_old_ratings]
    df.loc[blue_rows.index, "opp_glicko2_phi_before"] = [r.phi for r in red_old_ratings]
    df.loc[blue_rows.index, "glicko2_win_likelihood"] = blue_expected
    df.loc[blue_rows.index, "glicko2_mu_after"] = [r.mu for r in updated_blue_ratings]
    df.loc[blue_rows.index, "glicko2_phi_after"] = [r.phi for r in updated_blue_ratings]

    df.loc[red_rows.index, "glicko2_mu_before"] = [r.mu for r in red_old_ratings]
    df.loc[red_rows.index, "glicko2_phi_before"] = [r.phi for r in red_old_ratings]
    df.loc[red_rows.index, "opp_glicko2_mu_before"] = [r.mu for r in blue_old_ratings]
    df.loc[red_rows.index, "opp_glicko2_phi_before"] = [r.phi for r in blue_old_ratings]
    df.loc[red_rows.index, "glicko2_win_likelihood"] = 1.0 - blue_expected
    df.loc[red_rows.index, "glicko2_mu_after"] = [r.mu for r in updated_red_ratings]
    df.loc[red_rows.index, "glicko2_phi_after"] = [r.phi for r in updated_red_ratings]

    return glicko2_ratings


# ----------------------------------------------------------------------
# 5. Hyperparameter Tuning
# ----------------------------------------------------------------------
def tune_glicko2_hyperparameters(
    df: pd.DataFrame,
    entity: str,
    hyperparameters_path: Path,
    league_elo_dict: dict[str, float],
) -> dict[str, float]:
    """Load Glicko-2 hyperparameters or compute them via Optuna."""
    best_params = load_hyperparameters(hyperparameters_path)
    if best_params:
        return best_params

    logger.info(
        f"No hyperparameters found at {hyperparameters_path}. Starting tuning process..."
    )
    data_pipeline_logger.info(
        f"No hyperparameters found at {hyperparameters_path}. Starting tuning process..."
    )

    def objective(trial: optuna.trial.Trial) -> float:
        hyperparams = suggest_glicko2_hyperparameters(trial)

        df_train, df_valid = split_and_validate_data(df, entity)
        if df_train.empty or df_valid.empty:
            logger.warning("Training or validation DataFrame is empty after splitting.")
            data_pipeline_logger.warning(
                "Training or validation DataFrame is empty after splitting."
            )
            return float("inf")

        try:
            df_train_res = run_glicko2_computation(
                df=df_train.copy(),
                entity=entity,
                mu=hyperparams["mu"],
                phi=hyperparams["phi"],
                sigma=hyperparams["sigma"],
                decay_factor=hyperparams["decay_factor"],
                transfer_factor=hyperparams["transfer_factor"],
                initial_elo_adjustment_factor=hyperparams[
                    "initial_elo_adjustment_factor"
                ],
                position_reset_factor=hyperparams["position_reset_factor"],
                league_elo_dict=league_elo_dict,
                show_progress=False,
            )
        except Exception as err:
            logger.error(f"Error during Glicko-2 computation in training phase: {err}")
            data_pipeline_logger.exception(
                f"Error during Glicko-2 computation in training phase: {err}"
            )
            return float("inf")

        val_ratings = initialize_validation_ratings(
            df_train_res,
            entity,
            hyperparams["mu"],
            hyperparams["phi"],
            hyperparams["sigma"],
        )

        try:
            loss = evaluate_validation(df_valid, val_ratings, entity, hyperparams)
        except Exception as err:
            logger.error(f"Error during validation phase: {err}")
            data_pipeline_logger.exception(f"Error during validation phase: {err}")
            return float("inf")

        return loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=TRIALS_NUM, show_progress_bar=True)

    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")
    data_pipeline_logger.info(f"Best hyperparameters: {best_params}")

    save_hyperparameters(best_params, hyperparameters_path)
    return best_params


def load_hyperparameters(path: Path) -> dict[str, float]:
    """Load hyperparameters from a JSON file if it exists."""
    if path.exists():
        logger.info(f"Loading hyperparameters from {path}")
        data_pipeline_logger.info(f"Loading hyperparameters from {path}")
        try:
            with path.open() as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load hyperparameters from {path}: {e}")
            data_pipeline_logger.exception(
                f"Failed to load hyperparameters from {path}"
            )
    return {}


def save_hyperparameters(params: dict[str, float], path: Path) -> None:
    """Save hyperparameters to a JSON file."""
    try:
        logger.info(f"Storing hyperparameters to {path}")
        data_pipeline_logger.info(f"Storing hyperparameters to {path}")
        with path.open("w") as f:
            json.dump(params, f)
    except Exception as e:
        logger.error(f"Failed to save hyperparameters to {path}: {e}")
        data_pipeline_logger.exception(f"Failed to save hyperparameters to {path}")


def suggest_glicko2_hyperparameters(trial: optuna.trial.Trial) -> dict[str, float]:
    """Suggest Glicko-2 hyperparameters using Optuna's trial."""
    return {
        "mu": trial.suggest_float("mu", 1200, 1800, step=100),
        "phi": trial.suggest_float("phi", 100, 500, step=50),
        "sigma": trial.suggest_float("sigma", 0.01, 0.3, step=0.01),
        "decay_factor": trial.suggest_float("decay_factor", 0.5, 1.0, step=0.05),
        "transfer_factor": trial.suggest_float("transfer_factor", 0.1, 1.0, step=0.1),
        "initial_elo_adjustment_factor": trial.suggest_float(
            "initial_elo_adjustment_factor", 0.0, 1.0, step=0.1
        ),
        "position_reset_factor": trial.suggest_float(
            "position_reset_factor", 0.0, 1.0, step=0.1
        ),
    }


def split_and_validate_data(
    df: pd.DataFrame, entity: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split on year boundary; fallback to 80/20 if needed.
    Validates group sizes (2 for teams, 10 for players).
    """
    df_sorted = df.sort_values(by=["date", "gameid", "side"]).reset_index(drop=True)
    if df_sorted.empty:
        logger.warning("DataFrame is empty after sorting.")
        data_pipeline_logger.warning("DataFrame is empty after sorting.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        split_year = df_sorted["date"].dt.year.max()
        split_date = pd.to_datetime(f"{split_year}-01-01")
    except AttributeError as e:
        logger.error(f"Error accessing 'date' column with .dt accessor: {e}")
        data_pipeline_logger.exception(
            f"Error accessing 'date' column with .dt accessor: {e}"
        )
        return pd.DataFrame(), pd.DataFrame()

    df_train = df_sorted[df_sorted["date"] < split_date].reset_index(drop=True)
    df_valid = df_sorted[df_sorted["date"] >= split_date].reset_index(drop=True)

    # Fallback to 80/20 if either side is empty (mirror Elo behavior)
    if df_train.empty or df_valid.empty:
        q80 = df_sorted["date"].quantile(0.8)
        df_train = df_sorted[df_sorted["date"] < q80].reset_index(drop=True)
        df_valid = df_sorted[df_sorted["date"] >= q80].reset_index(drop=True)

    expected_count = 10 if entity.lower() == "player" else 2
    for split_df, name in [(df_train, "Training"), (df_valid, "Validation")]:
        if not split_df.empty:
            group_sizes = split_df.groupby("gameid").size()
            if not (group_sizes == expected_count).all():
                logger.warning(
                    f"{name} data has gameids with incorrect number of entities."
                )
                data_pipeline_logger.warning(
                    f"{name} data has gameids with incorrect number of entities."
                )
                return pd.DataFrame(), pd.DataFrame()

    return df_train, df_valid


def initialize_validation_ratings(
    df_train_res: pd.DataFrame, entity: str, mu: float, phi: float, sigma: float
) -> defaultdict:
    """Initialize validation ratings from final training mu (phi/sigma baseline)."""
    entity_key = "teamid" if entity.lower() == "team" else "playerid"
    last_mu_map = df_train_res.groupby(entity_key)["glicko2_mu_after"].last().to_dict()

    val_ratings = defaultdict(lambda: {"rating": Rating(mu=mu, phi=phi, sigma=sigma)})
    for e_id, final_mu in last_mu_map.items():
        if pd.notna(final_mu):
            val_ratings[e_id]["rating"] = Rating(
                mu=float(final_mu), phi=phi, sigma=sigma
            )
    return val_ratings


def evaluate_validation(
    df_valid: pd.DataFrame,
    val_ratings: defaultdict,
    entity: str,
    hyperparams: dict[str, float],
) -> float:
    """Evaluate validation set by mean-vs-mean team ratings; compute log loss."""
    expected_probs: list[float] = []
    df_valid_sorted = df_valid.sort_values(by=["date", "gameid"]).reset_index(drop=True)
    model = Glicko2(
        mu=hyperparams["mu"], phi=hyperparams["phi"], sigma=hyperparams["sigma"]
    )
    entity_key = "teamid" if entity.lower() == "team" else "playerid"

    for _, grp in df_valid_sorted.groupby(["date", "gameid"], sort=False):
        blue_side = grp[grp["side"] == "Blue"]
        red_side = grp[grp["side"] == "Red"]

        blue_mu_sum = sum(
            val_ratings[bid]["rating"].mu for bid in blue_side[entity_key]
        )
        red_mu_sum = sum(val_ratings[rid]["rating"].mu for rid in red_side[entity_key])
        blue_count = max(1, len(blue_side))
        red_count = max(1, len(red_side))

        blue_team_rating = Rating(
            mu=blue_mu_sum / blue_count,
            phi=hyperparams["phi"],
            sigma=hyperparams["sigma"],
        )
        red_team_rating = Rating(
            mu=red_mu_sum / red_count,
            phi=hyperparams["phi"],
            sigma=hyperparams["sigma"],
        )

        opp_impact = model.reduce_impact(red_team_rating)
        exp = model.expect_score(blue_team_rating, red_team_rating, opp_impact)
        result = float(blue_side.iloc[0]["result"])

        expected_probs.extend([exp] * len(blue_side))

        # Update validation trajectory
        for bid in blue_side[entity_key]:
            old = val_ratings[bid]["rating"]
            val_ratings[bid]["rating"] = update_glicko2_rating(
                old, red_team_rating, result, model
            )
        for rid in red_side[entity_key]:
            old = val_ratings[rid]["rating"]
            val_ratings[rid]["rating"] = update_glicko2_rating(
                old, blue_team_rating, 1.0 - result, model
            )

    y_true = df_valid_sorted.loc[df_valid_sorted["side"] == "Blue", "result"]
    y_pred = pd.Series(expected_probs).clip(0.0001, 0.9999)

    if len(y_true) != len(y_pred):
        logger.error("Mismatch in lengths of y_true and y_pred.")
        data_pipeline_logger.error("Mismatch in lengths of y_true and y_pred.")
        return float("inf")

    return log_loss(y_true, y_pred)


# ----------------------------------------------------------------------
# 6. Main Glicko-2 Computation
# ----------------------------------------------------------------------
def run_glicko2_computation(
    df: pd.DataFrame,
    entity: str,
    mu: float,
    phi: float,
    sigma: float,
    decay_factor: float,
    transfer_factor: float,
    initial_elo_adjustment_factor: float,
    position_reset_factor: float,
    league_elo_dict: dict[str, float],
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Main procedure to update Glicko-2 ratings across the entire DataFrame.
    """
    df = df.copy()
    entity_key = "teamid" if entity.lower() == "team" else "playerid"
    glicko2_model = Glicko2(mu=mu, phi=phi, sigma=sigma)

    # Initial rating dict
    glicko2_ratings = defaultdict(
        lambda: {
            "rating": Rating(mu=mu, phi=phi, sigma=sigma),
            "season": int(df["season"].min()),
            "league": None,
        }
    )

    # Pre-allocate numeric columns (float64)
    for col in [
        "glicko2_mu_before",
        "glicko2_phi_before",
        "opp_glicko2_mu_before",
        "opp_glicko2_phi_before",
        "glicko2_win_likelihood",
        "glicko2_mu_after",
        "glicko2_phi_after",
    ]:
        df[col] = pd.Series(index=df.index, dtype="float64")

    grouped = df.groupby(["date", "gameid"], sort=False)
    if show_progress:
        grouped = tqdm(grouped, desc="Processing games", total=grouped.ngroups)

    for _, game_grp in grouped:
        glicko2_ratings = process_game(
            df=df,
            game_group=game_grp,
            glicko2_ratings=glicko2_ratings,
            glicko2_model=glicko2_model,
            entity=entity,
            entity_key=entity_key,
            baseline_mu=mu,
            decay_factor=decay_factor,
            league_elo_dict=league_elo_dict,
            transfer_factor=transfer_factor,
            initial_elo_adjustment_factor=initial_elo_adjustment_factor,
            position_reset_factor=position_reset_factor,
        )

    return df


def calculate_glicko2(
    df: pd.DataFrame,
    entity: str,
    league_elo_dict: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Main entry point for Glicko-2 rating computation.
    Mirrors the structure of the Elo code but uses Glicko-2 updates
    and columns named glicko2_*.
    """
    df_pre = preprocess_glicko2_dataframe(df, entity)

    if league_elo_dict is None:
        league_elo_dict = {}
        if LEAGUE_ELO.exists():
            league_elo_df = pd.read_parquet(LEAGUE_ELO)
            league_elo_dict = league_elo_df.set_index("league")["elo"].to_dict()

    hyperparameters_path = Path(
        str(ENTITY_GLICKO_HYPERPARAMETERS).replace("entity", entity)
    )
    best_params = tune_glicko2_hyperparameters(
        df_pre, entity, hyperparameters_path, league_elo_dict
    )

    return run_glicko2_computation(
        df=df_pre,
        entity=entity,
        mu=best_params["mu"],
        phi=best_params["phi"],
        sigma=best_params["sigma"],
        decay_factor=best_params["decay_factor"],
        transfer_factor=best_params["transfer_factor"],
        initial_elo_adjustment_factor=best_params["initial_elo_adjustment_factor"],
        position_reset_factor=best_params.get("position_reset_factor", 0.2),
        league_elo_dict=league_elo_dict,
        show_progress=True,
    )
