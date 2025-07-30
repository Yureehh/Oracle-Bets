"""
TrueSkill Rating System with Hyperparameter Tuning using Optuna

Mirrors the structure of the Plackett-Luce (and Elo/Glicko) modules, but uses
the TrueSkill library from the `trueskill` package.
"""

import itertools
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import trueskill
from sklearn.metrics import log_loss
from tqdm import tqdm
from trueskill import Rating, TrueSkill

from utils.io_utils import get_sorting_keys, json_loader
from utils.logger import instantiate_conf_logger, logger
from utils.paths import (
    CONSIDERED_LEAGUES,
    DEFAULT_MODELS_PARAMETERS,
    ENTITY_TRUESKILL_HYPERPARAMETERS,
    LEAGUE_ELO,
)

# ------------------------------------------------------------------------------
# 1. Global Config / Constants
# ------------------------------------------------------------------------------
config = json_loader(DEFAULT_MODELS_PARAMETERS)
ts_config = config.get("trueskill", {})
DEFAULT_MU = ts_config.get("mu", 25.0)
DEFAULT_SIGMA = ts_config.get("sigma", 8.333)
DEFAULT_BETA = ts_config.get(
    "beta", DEFAULT_SIGMA / 2
)  # Standard practice: beta = sigma/2
TRIALS_NUM = 50

considered_leagues_config = json_loader(CONSIDERED_LEAGUES)
MAJOR_LEAGUES = considered_leagues_config["major_leagues"]
CROSS_LEAGUE_COMPETITIONS = considered_leagues_config["cross_league_competitions"]

data_pipeline_logger = instantiate_conf_logger("data_pipeline")


# ------------------------------------------------------------------------------
# 2. Helper Utilities
# ------------------------------------------------------------------------------
def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp numeric `value` between `min_val` and `max_val`."""
    return max(min_val, min(value, max_val))


def is_major_league(league: str) -> bool:
    """Check if league is considered major."""
    return league in MAJOR_LEAGUES


# ------------------------------------------------------------------------------
# 3. DataFrame Preprocessing
# ------------------------------------------------------------------------------
def preprocess_trueskill_dataframe(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Validate the DataFrame's required columns, convert date to datetime, drop missing entries,
    and sort by the appropriate keys. Mirrors the approach in Elo/Glicko/PL.
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

    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        msg = f"Input DataFrame is missing required columns: {missing_cols}"
        raise ValueError(msg)

    # Convert 'date' to datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        null_count = df["date"].isnull().sum()
        if null_count > 0:
            logger.warning(
                f"{null_count} 'date' entries could not be converted; dropping them."
            )
            data_pipeline_logger.warning(
                f"{null_count} 'date' entries could not be converted; dropping them."
            )
            df = df.dropna(subset=["date"]).copy()

    # Drop rows missing league or result
    if df["league"].isna().any() or df["result"].isna().any():
        n_missing_league = df["league"].isnull().sum()
        n_missing_result = df["result"].isnull().sum()
        logger.warning(
            f"{n_missing_league} 'league' and {n_missing_result} 'result' missing; dropping them."
        )
        data_pipeline_logger.warning(
            f"{n_missing_league} 'league' and {n_missing_result} 'result' missing; dropping them."
        )
        df = df.dropna(subset=["league", "result"]).reset_index(drop=True)

    # Sort by date and game
    return df.sort_values(by=get_sorting_keys(entity)).reset_index(drop=True)


# ------------------------------------------------------------------------------
# 4. Core TrueSkill Functions
# ------------------------------------------------------------------------------
def create_ts_rating(mu: float, sigma: float) -> Rating:
    """Create a TrueSkill Rating with given mu and sigma."""
    return Rating(mu=mu, sigma=sigma)


def expected_win_probability(
    team1: list[Rating], team2: list[Rating], beta: float
) -> float:
    """Return the probability that team1 beats team2 using the TrueSkill formula."""
    delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
    sum_sigma_sq = sum(r.sigma**2 for r in itertools.chain(team1, team2))
    denom = math.sqrt(len(team1 + team2) * (beta**2) + sum_sigma_sq)
    return trueskill.global_env().cdf(delta_mu / denom)


def update_ts_ratings(
    model: TrueSkill,
    teams_ratings: tuple[list[Rating], list[Rating]],
    ranks: list[int],
) -> list[list[Rating]]:
    """
    Update TrueSkill ratings for two teams given their ranks (0 => winner, 1 => loser).
    Returns updated ratings for [team1, team2].
    """
    return model.rate(deepcopy(teams_ratings), ranks=ranks)


# ------------------------------------------------------------------------------
# 5. Seasonal, Position, and League Adjustments
# ------------------------------------------------------------------------------
def linear_decay_reset(
    ts_ratings: dict[int | str, dict[str, Any]],
    current_season: int,
    baseline_mu: float,
    baseline_sigma: float,
    decay_factor: float,
) -> dict[int | str, dict[str, Any]]:
    """
    In-place: partially reset mu/sigma toward baseline if stored season < current_season.
    new_mu = baseline_mu + (old_mu - baseline_mu)*decay_factor
    new_sigma = baseline_sigma + (old_sigma - baseline_sigma)*decay_factor
    """
    for data in ts_ratings.values():
        if data["season"] < current_season:
            old_rating = data["rating"]
            new_mu = baseline_mu + (old_rating.mu - baseline_mu) * decay_factor
            new_sigma = (
                baseline_sigma + (old_rating.sigma - baseline_sigma) * decay_factor
            )
            data["rating"] = create_ts_rating(new_mu, new_sigma)
            data["season"] = current_season

    return ts_ratings  # return same dict reference


def handle_position_switch(
    entity_id: int | str,
    new_position: str | None,
    ts_ratings: dict[int | str, dict[str, Any]],
    baseline_mu: float,
    baseline_sigma: float,
    position_reset_factor: float,
) -> None:
    """For players switching position, partially reset rating toward baseline by `position_reset_factor`."""
    if new_position is None:
        return
    last_position = ts_ratings[entity_id].get("last_position")
    if last_position and last_position != new_position:
        old_rating = ts_ratings[entity_id]["rating"]
        new_mu = baseline_mu + (old_rating.mu - baseline_mu) * (
            1.0 - position_reset_factor
        )
        new_sigma = baseline_sigma + (old_rating.sigma - baseline_sigma) * (
            1.0 - position_reset_factor
        )
        ts_ratings[entity_id]["rating"] = create_ts_rating(new_mu, new_sigma)
    ts_ratings[entity_id]["last_position"] = new_position


def handle_new_entity(
    ent_id: int | str,
    ts_ratings: dict[int | str, dict[str, Any]],
    league_elo_dict: dict[str, float],
    new_league: str,
    current_season: int,
    baseline_mu: float,
    baseline_sigma: float,
    init_adjust_factor: float,
) -> None:
    """Initialize a new entity rating, offset by partial league difference from average."""
    max_diff = 0.2 * baseline_mu
    if league_elo_dict:
        avg_league_elo = sum(league_elo_dict.values()) / len(league_elo_dict)
    else:
        avg_league_elo = baseline_mu
    league_elo = league_elo_dict.get(new_league, baseline_mu)

    init_adjust = (league_elo - avg_league_elo) * init_adjust_factor
    initial_mu = baseline_mu + init_adjust

    offset = clamp(initial_mu - baseline_mu, -max_diff, max_diff)
    initial_mu = baseline_mu + offset

    ts_ratings[ent_id] = {
        "rating": create_ts_rating(initial_mu, baseline_sigma),
        "season": current_season,
        "league": new_league,
    }


def handle_league_swap(
    ent_id: int | str,
    new_league: str,
    ts_ratings: dict[int | str, dict[str, Any]],
    league_elo_dict: dict[str, float],
    baseline_mu: float,
    baseline_sigma: float,
    transfer_factor: float,
    transfer_factor_minor_to_major: float = 0.4,
) -> None:
    """
    Partially shift rating if an entity changes leagues.
    If minor->major for a "player", apply bigger penalty.
    Otherwise, apply a fraction of the difference in league Elo.
    """
    curr_league = ts_ratings[ent_id].get("league")
    if not curr_league or curr_league == new_league:
        ts_ratings[ent_id]["league"] = new_league
        return

    curr_is_major = is_major_league(curr_league)
    new_is_major = is_major_league(new_league)
    curr_elo_val = league_elo_dict.get(curr_league, baseline_mu)
    new_elo_val = league_elo_dict.get(new_league, baseline_mu)
    diff = new_elo_val - curr_elo_val

    old_rating = ts_ratings[ent_id]["rating"]
    old_mu = old_rating.mu

    # Define the maximum adjustment to prevent extreme changes
    max_diff = 0.2 * baseline_mu

    # Calculate adjusted difference based on the league swap type
    if (
        isinstance(ent_id, str)
        and "player" in ent_id.lower()
        and not curr_is_major
        and new_is_major
    ):
        # Apply a bigger penalty for minor->major player transitions
        adjusted_diff = transfer_factor_minor_to_major * diff
    else:
        # Standard transfer adjustment
        adjusted_diff = transfer_factor * diff

    # Clamp the adjusted difference to prevent extreme changes
    adjusted_diff = clamp(adjusted_diff, -max_diff, max_diff)

    # Apply the adjusted difference to old_mu and clamp the result
    new_mu = clamp(old_mu + adjusted_diff, -baseline_mu, 3 * baseline_mu)

    # Update the rating and league
    ts_ratings[ent_id]["rating"] = create_ts_rating(new_mu, baseline_sigma)
    ts_ratings[ent_id]["league"] = new_league


# ------------------------------------------------------------------------------
# 6. Game Processing
# ------------------------------------------------------------------------------
def process_game(
    df: pd.DataFrame,
    game_group: pd.DataFrame,
    ts_ratings: dict[int | str, dict[str, Any]],
    ts_model: TrueSkill,
    entity: str,
    entity_key: str,
    baseline_mu: float,
    baseline_sigma: float,
    decay_factor: float,
    transfer_factor: float,
    initial_elo_adjustment_factor: float,
    position_reset_factor: float,
    league_elo_dict: dict[str, float],
) -> dict[int | str, dict[str, Any]]:
    """
    Process a single match, updating TrueSkill ratings for each entity in the match.
    Return the updated dictionary so we don't lose changes.
    """
    current_season = game_group.iloc[0]["season"]

    # In-place seasonal decay
    linear_decay_reset(
        ts_ratings=ts_ratings,
        current_season=current_season,
        baseline_mu=baseline_mu,
        baseline_sigma=baseline_sigma,
        decay_factor=decay_factor,
    )

    entity_ids = game_group[entity_key].unique()
    for ent_id in entity_ids:
        if ent_id not in ts_ratings:
            handle_new_entity(
                ent_id=ent_id,
                ts_ratings=ts_ratings,
                league_elo_dict=league_elo_dict,
                new_league=game_group.loc[
                    game_group[entity_key] == ent_id, "league"
                ].iloc[0],
                current_season=current_season,
                baseline_mu=baseline_mu,
                baseline_sigma=baseline_sigma,
                init_adjust_factor=initial_elo_adjustment_factor,
            )
        else:
            new_league = game_group.loc[
                game_group[entity_key] == ent_id, "league"
            ].iloc[0]
            if new_league not in CROSS_LEAGUE_COMPETITIONS:
                handle_league_swap(
                    ent_id=ent_id,
                    new_league=new_league,
                    ts_ratings=ts_ratings,
                    league_elo_dict=league_elo_dict,
                    baseline_mu=baseline_mu,
                    baseline_sigma=baseline_sigma,
                    transfer_factor=transfer_factor,
                )

    # If entity=player, handle position switching
    if entity.lower() == "player":
        for _, row in game_group.iterrows():
            ent_id = row[entity_key]
            new_position = row.get("position")
            handle_position_switch(
                entity_id=ent_id,
                new_position=new_position,
                ts_ratings=ts_ratings,
                baseline_mu=baseline_mu,
                baseline_sigma=baseline_sigma,
                position_reset_factor=position_reset_factor,
            )

    # Split sides
    blue_side = game_group[game_group["side"] == "Blue"]
    red_side = game_group[game_group["side"] == "Red"]

    if entity.lower() == "player":
        blue_side = blue_side.sort_values(by="position")
        red_side = red_side.sort_values(by="position")

    blue_ids = blue_side[entity_key].values
    red_ids = red_side[entity_key].values

    # Current ratings
    blue_ratings_before = [ts_ratings[b]["rating"] for b in blue_ids]
    red_ratings_before = [ts_ratings[r]["rating"] for r in red_ids]

    # Determine match result: [0,1] => Blue wins, otherwise [1,0]
    blue_result = blue_side.iloc[0]["result"]
    ranks = [0, 1] if abs(blue_result - 1.0) < 1e-9 else [1, 0]

    # Probability that Blue wins
    prob_blue_wins = expected_win_probability(
        blue_ratings_before, red_ratings_before, ts_model.beta
    )

    # Update ratings
    updated = update_ts_ratings(
        ts_model, (blue_ratings_before, red_ratings_before), ranks=ranks
    )
    updated_blue_ratings = updated[0]
    updated_red_ratings = updated[1]

    # Write updated ratings back to dictionary + DataFrame
    for i, idx in enumerate(blue_side.index):
        ent_id = blue_side.loc[idx, entity_key]
        old_rating = blue_ratings_before[i]
        new_rating = updated_blue_ratings[i]
        ts_ratings[ent_id]["rating"] = new_rating

        df.loc[idx, "trueskill_mu_before"] = old_rating.mu
        df.loc[idx, "trueskill_sigma_before"] = old_rating.sigma
        df.loc[idx, "opp_trueskill_mu_before"] = red_ratings_before[i].mu
        df.loc[idx, "opp_trueskill_sigma_before"] = red_ratings_before[i].sigma
        df.loc[idx, "trueskill_win_likelihood"] = prob_blue_wins
        df.loc[idx, "trueskill_mu_after"] = new_rating.mu
        df.loc[idx, "trueskill_sigma_after"] = new_rating.sigma

    for i, idx in enumerate(red_side.index):
        ent_id = red_side.loc[idx, entity_key]
        old_rating = red_ratings_before[i]
        new_rating = updated_red_ratings[i]
        ts_ratings[ent_id]["rating"] = new_rating

        df.loc[idx, "trueskill_mu_before"] = old_rating.mu
        df.loc[idx, "trueskill_sigma_before"] = old_rating.sigma
        df.loc[idx, "opp_trueskill_mu_before"] = blue_ratings_before[i].mu
        df.loc[idx, "opp_trueskill_sigma_before"] = blue_ratings_before[i].sigma
        df.loc[idx, "trueskill_win_likelihood"] = 1.0 - prob_blue_wins
        df.loc[idx, "trueskill_mu_after"] = new_rating.mu
        df.loc[idx, "trueskill_sigma_after"] = new_rating.sigma

    # Return the updated dictionary to ensure we don't lose any changes
    return ts_ratings


# ------------------------------------------------------------------------------
# 7. Hyperparameter Tuning
# ------------------------------------------------------------------------------
def tune_trueskill_hyperparameters(
    df: pd.DataFrame,
    entity: str,
    hyperparameters_path: Path,
    league_elo_dict: dict[str, float],
) -> dict[str, float]:
    """
    If existing hyperparams are found, load them. Otherwise, run Optuna to find best
    TrueSkill params that minimize validation log loss.
    """
    best_params = load_hyperparameters(hyperparameters_path)
    if best_params:
        return best_params

    logger.info(
        f"No TrueSkill hyperparameters found at {hyperparameters_path}. Starting tuning..."
    )
    data_pipeline_logger.info(
        f"No TrueSkill hyperparameters found at {hyperparameters_path}. Starting tuning..."
    )

    def objective(trial: optuna.trial.Trial) -> float:
        hyperparams = suggest_trueskill_hyperparameters(trial)
        df_train, df_valid = split_and_validate_data(df, entity)
        if df_train.empty or df_valid.empty:
            logger.warning("Training or validation data is empty after splitting.")
            data_pipeline_logger.warning(
                "Training or validation data is empty after splitting."
            )
            return float("inf")

        try:
            df_train_res = run_trueskill_computation(
                df=df_train.copy(),
                entity=entity,
                mu=hyperparams["mu"],
                sigma=hyperparams["sigma"],
                beta=hyperparams["beta"],
                decay_factor=hyperparams["decay_factor"],
                transfer_factor=hyperparams["transfer_factor"],
                initial_elo_adjustment_factor=hyperparams[
                    "initial_elo_adjustment_factor"
                ],
                position_reset_factor=hyperparams["position_reset_factor"],
                league_elo_dict=league_elo_dict,
                show_progress=False,
            )
        except Exception as e:
            logger.error(f"Error in training TrueSkill computation: {e}")
            data_pipeline_logger.exception(
                f"Error in training TrueSkill computation: {e}"
            )
            return float("inf")

        val_ratings = initialize_validation_ratings(
            df_train_res, entity, hyperparams["mu"], hyperparams["sigma"]
        )

        try:
            loss = evaluate_validation(df_valid, val_ratings, entity, hyperparams)
        except Exception as e:
            logger.error(f"Error in validation step: {e}")
            data_pipeline_logger.exception(f"Error in validation step: {e}")
            return float("inf")

        return loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=TRIALS_NUM, show_progress_bar=True)
    best_params = study.best_params

    logger.info(f"Best TrueSkill hyperparameters: {best_params}")
    data_pipeline_logger.info(f"Best TrueSkill hyperparameters: {best_params}")

    save_hyperparameters(best_params, hyperparameters_path)
    return best_params


def load_hyperparameters(path: Path) -> dict[str, float]:
    """Load TrueSkill hyperparameters from JSON if available."""
    if path.exists():
        logger.info(f"Loading TrueSkill hyperparameters from {path}")
        data_pipeline_logger.info(f"Loading TrueSkill hyperparameters from {path}")
        try:
            with path.open("r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load TrueSkill hyperparameters: {e}")
            data_pipeline_logger.exception(
                f"Failed to load TrueSkill hyperparameters from {path}"
            )
    return {}


def save_hyperparameters(params: dict[str, float], path: Path) -> None:
    """Save hyperparameters to a JSON file."""
    try:
        logger.info(f"Storing TrueSkill hyperparameters to {path}")
        data_pipeline_logger.info(f"Storing TrueSkill hyperparameters to {path}")
        with path.open("w") as f:
            json.dump(params, f)
    except Exception as e:
        logger.error(f"Failed to save TrueSkill hyperparameters: {e}")
        data_pipeline_logger.exception(
            f"Failed to save TrueSkill hyperparameters to {path}"
        )


def suggest_trueskill_hyperparameters(trial: optuna.trial.Trial) -> dict[str, float]:
    """Suggest TrueSkill hyperparameters (mu, sigma, beta, decay_factor, etc.) via Optuna."""
    return {
        "mu": trial.suggest_float("mu", 15.0, 40.0, step=5.0),
        "sigma": trial.suggest_float("sigma", 2.0, 15.0, step=1.0),
        "beta": trial.suggest_float("beta", 0.5, 10.0, step=0.5),
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
    Sort, then split the DataFrame into training and validation sets by year boundary,
    verifying the group size (2 for teams, 10 for players).
    """
    df_sorted = df.sort_values(by=["date", "gameid", "side"]).reset_index(drop=True)
    if df_sorted.empty:
        logger.warning("DataFrame is empty after sorting in split_and_validate_data.")
        data_pipeline_logger.warning(
            "DataFrame is empty after sorting in split_and_validate_data."
        )
        return pd.DataFrame(), pd.DataFrame()

    try:
        split_year = df_sorted["date"].dt.year.max()
        split_date = pd.to_datetime(f"{split_year}-01-01")
    except AttributeError as e:
        logger.error(f"Error accessing date column with .dt: {e}")
        data_pipeline_logger.exception(f"Error accessing date column with .dt: {e}")
        return pd.DataFrame(), pd.DataFrame()

    df_train = df_sorted[df_sorted["date"] < split_date].reset_index(drop=True)
    df_valid = df_sorted[df_sorted["date"] >= split_date].reset_index(drop=True)

    # Verify group sizes
    expected_count = 10 if entity.lower() == "player" else 2
    for subset, name in [(df_train, "Training"), (df_valid, "Validation")]:
        if not subset.empty:
            group_sizes = subset.groupby("gameid").size()
            if not (group_sizes == expected_count).all():
                logger.warning(
                    f"{name} data has gameids with an incorrect number of entities."
                )
                data_pipeline_logger.warning(
                    f"{name} data has gameids with an incorrect number of entities."
                )
                return pd.DataFrame(), pd.DataFrame()

    return df_train, df_valid


def initialize_validation_ratings(
    df_train_res: pd.DataFrame, entity: str, mu: float, sigma: float
) -> defaultdict:
    """
    After training, read the final 'trueskill_mu_after'/'trueskill_sigma_after' columns
    from the training results to initialize validation ratings. If an entity never appeared,
    start from the baseline (mu, sigma).
    """
    from collections import defaultdict

    entity_key = "teamid" if entity.lower() == "team" else "playerid"

    final_mu_map = (
        df_train_res.groupby(entity_key)["trueskill_mu_after"].last().to_dict()
    )
    final_sigma_map = (
        df_train_res.groupby(entity_key)["trueskill_sigma_after"].last().to_dict()
    )

    val_ratings = defaultdict(lambda: {"rating": create_ts_rating(mu, sigma)})
    for e_id in final_mu_map:
        if pd.notna(final_mu_map[e_id]) and pd.notna(final_sigma_map[e_id]):
            val_ratings[e_id]["rating"] = create_ts_rating(
                final_mu_map[e_id], final_sigma_map[e_id]
            )

    return val_ratings


def evaluate_validation(
    df_valid: pd.DataFrame,
    val_ratings: defaultdict,
    entity: str,
    hyperparams: dict[str, float],
) -> float:
    """
    Compute log loss on validation. For each match:
      - Retrieve Blue & Red ratings
      - Predict probability that Blue wins
      - Compare vs. actual result
      - Update ratings in a simplified loop
    """
    expected_probs = []
    df_valid_sorted = df_valid.sort_values(by=["date", "gameid"]).reset_index(drop=True)
    ts_model = TrueSkill(
        mu=hyperparams["mu"],
        sigma=hyperparams["sigma"],
        beta=hyperparams["beta"],
        draw_probability=0.0,
    )
    entity_key = "teamid" if entity.lower() == "team" else "playerid"

    for _, grp in df_valid_sorted.groupby(["date", "gameid"]):
        blue_side = grp[grp["side"] == "Blue"]
        red_side = grp[grp["side"] == "Red"]

        if entity.lower() == "player":
            blue_side = blue_side.sort_values(by="position")
            red_side = red_side.sort_values(by="position")

        blue_ids = blue_side[entity_key].values
        red_ids = red_side[entity_key].values

        blue_ratings = [val_ratings[b]["rating"] for b in blue_ids]
        red_ratings = [val_ratings[r]["rating"] for r in red_ids]

        prob_blue = expected_win_probability(blue_ratings, red_ratings, ts_model.beta)
        result = blue_side.iloc[0]["result"]  # 1 => Blue wins, 0 => Red wins

        expected_probs.extend([prob_blue] * len(blue_side))

        # Update ratings for validation
        if abs(result - 1.0) < 1e-9:
            # Blue wins => ranks=[0,1]
            updated = update_ts_ratings(ts_model, (blue_ratings, red_ratings), [0, 1])
        else:
            # Red wins => ranks=[1,0]
            updated = update_ts_ratings(ts_model, (blue_ratings, red_ratings), [1, 0])

        # Write updated ratings back
        for i, b_id in enumerate(blue_ids):
            val_ratings[b_id]["rating"] = updated[0][i]
        for i, r_id in enumerate(red_ids):
            val_ratings[r_id]["rating"] = updated[1][i]

    # Evaluate log loss
    y_true = df_valid_sorted.loc[df_valid_sorted["side"] == "Blue", "result"]
    y_pred = pd.Series(expected_probs).clip(0.0001, 0.9999)

    if len(y_true) != len(y_pred):
        logger.error("Mismatch in lengths of y_true and y_pred in evaluate_validation.")
        data_pipeline_logger.error(
            "Mismatch in lengths of y_true and y_pred in evaluate_validation."
        )
        return float("inf")

    return log_loss(y_true, y_pred)


# ------------------------------------------------------------------------------
# 8. Main TrueSkill Computation
# ------------------------------------------------------------------------------
def run_trueskill_computation(
    df: pd.DataFrame,
    entity: str,
    mu: float,
    sigma: float,
    beta: float,
    decay_factor: float,
    transfer_factor: float,
    initial_elo_adjustment_factor: float,
    position_reset_factor: float,
    league_elo_dict: dict[str, float],
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Main procedure to update TrueSkill ratings across the entire DataFrame:
      1. Group matches by (date, gameid)
      2. For each match, call `process_game` (capture returned dictionary!)
      3. Return a DataFrame with new columns:
         trueskill_mu_before, trueskill_sigma_before, trueskill_mu_after,
         trueskill_sigma_after, trueskill_win_likelihood, etc.
    """
    df = df.copy()
    entity_key = "teamid" if entity.lower() == "team" else "playerid"

    # Initialize a TrueSkill model
    ts_model = TrueSkill(
        mu=mu,
        sigma=sigma,
        beta=beta,
        draw_probability=0.0,
    )

    # Store entity ratings in a dictionary
    ts_ratings = defaultdict(
        lambda: {
            "rating": create_ts_rating(mu, sigma),
            "season": df["season"].min(),
            "league": None,
        }
    )

    # Pre-allocate output columns
    for col in [
        "trueskill_mu_before",
        "trueskill_sigma_before",
        "opp_trueskill_mu_before",
        "opp_trueskill_sigma_before",
        "trueskill_win_likelihood",
        "trueskill_mu_after",
        "trueskill_sigma_after",
    ]:
        df[col] = None

    grouped = df.groupby(["date", "gameid"], sort=False)
    if show_progress:
        grouped = tqdm(grouped, desc="Processing games", total=grouped.ngroups)

    # Process each match, retaining updated dictionary
    for _, game_grp in grouped:
        ts_ratings = process_game(
            df=df,
            game_group=game_grp,
            ts_ratings=ts_ratings,
            ts_model=ts_model,
            entity=entity,
            entity_key=entity_key,
            baseline_mu=mu,
            baseline_sigma=sigma,
            decay_factor=decay_factor,
            transfer_factor=transfer_factor,
            initial_elo_adjustment_factor=initial_elo_adjustment_factor,
            position_reset_factor=position_reset_factor,
            league_elo_dict=league_elo_dict,
        )

    return df


def calculate_trueskill(
    df: pd.DataFrame,
    entity: str,
    league_elo_dict: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Main entry point for TrueSkill rating computation:
      1. Preprocess DataFrame
      2. Load or tune hyperparameters
      3. Run final TrueSkill computations
    """
    df_pre = preprocess_trueskill_dataframe(df, entity)

    # If league_elo_dict is not provided, attempt to load from LEAGUE_ELO
    if league_elo_dict is None:
        league_elo_dict = {}
        if LEAGUE_ELO.exists():
            league_elo_df = pd.read_parquet(LEAGUE_ELO)
            league_elo_dict = league_elo_df.set_index("league")["elo"].to_dict()

    # Attempt to load or tune hyperparameters
    hyperparameters_path = Path(
        str(ENTITY_TRUESKILL_HYPERPARAMETERS).replace("entity", entity)
    )
    best_params = tune_trueskill_hyperparameters(
        df_pre, entity, hyperparameters_path, league_elo_dict
    )

    # Final run with best hyperparams
    return run_trueskill_computation(
        df=df_pre,
        entity=entity,
        mu=best_params.get("mu", DEFAULT_MU),
        sigma=best_params.get("sigma", DEFAULT_SIGMA),
        beta=best_params.get("beta", DEFAULT_BETA),
        decay_factor=best_params.get("decay_factor", 0.9),
        transfer_factor=best_params.get("transfer_factor", 0.5),
        initial_elo_adjustment_factor=best_params.get(
            "initial_elo_adjustment_factor", 0.5
        ),
        position_reset_factor=best_params.get("position_reset_factor", 0.2),
        league_elo_dict=league_elo_dict,
        show_progress=True,
    )
