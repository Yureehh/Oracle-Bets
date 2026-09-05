"""
Glicko-2 Rating System with Hyperparameter Tuning using Optuna

This module mirrors the Elo-based infrastructure but replaces the Elo methods
with Glicko-2 functions from the 'glicko2' library.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import optuna
from glicko2 import Glicko2, Rating
from oracle_bets_core.io_utils import json_loader
from oracle_bets_core.logger import LOG_TOPIC, instantiate_logger, logger
from oracle_bets_core.paths import (
    DEFAULT_MODELS_PARAMETERS,
    ENTITY_GLICKO_HYPERPARAMETERS,
    RATING_LEAGUE_ELO,
)
from oracle_bets_core.pd import pd
from sklearn.metrics import log_loss
from tqdm import tqdm

from lol_bets.data_generation.feature_engineering.ratings_features import (
    freeze_same_date_rating_inputs,
)
from lol_bets.data_generation.feature_engineering.ratings_features.common import (
    clamp,
    is_cross_league_competition,
    is_major_league,
)
from lol_bets.data_generation.feature_engineering.ratings_features.common import (
    load_hyperparameters as _load_hyperparameters,
)
from lol_bets.data_generation.feature_engineering.ratings_features.common import (
    preprocess_rating_dataframe as preprocess_glicko2_dataframe,
)
from lol_bets.data_generation.feature_engineering.ratings_features.common import (
    save_hyperparameters as _save_hyperparameters,
)
from lol_bets.data_generation.feature_engineering.ratings_features.common import (
    split_and_validate_data as _split_and_validate_data,
)

# ----------------------------------------------------------------------
# Global Config / Constants
# ----------------------------------------------------------------------
config = json_loader(DEFAULT_MODELS_PARAMETERS)
data_pipeline_logger = instantiate_logger(LOG_TOPIC.DATA_PIPELINE)

TRIALS_NUM = int(config.get("optuna", {}).get("trials", 100))
OPTUNA_SEED = int(config.get("optuna", {}).get("seed", 42))

# Default Glicko-2 parameters (parallel to how Elo had baseline ratings)
DEFAULT_MU = config.get("glicko2", {}).get("mu", 1500.0)
DEFAULT_PHI = config.get("glicko2", {}).get("phi", 350.0)
DEFAULT_SIGMA = config.get("glicko2", {}).get("sigma", 0.06)
INACTIVITY_GRACE_DAYS = config.get("shared", {}).get("inactivity_grace_days", 45)
INACTIVITY_HALF_LIFE_DAYS = config.get("shared", {}).get(
    "inactivity_half_life_days", 180
)

WIN = 1  # Helper constant for clarity in `rate_match_using_mean`


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


def apply_inactivity_decay(
    rating_data: dict[str, Any],
    current_date: pd.Timestamp,
    baseline_mu: float,
    baseline_phi: float,
) -> None:
    """Regress stale strength and grow Glicko uncertainty toward its prior."""
    last_active = rating_data.get("last_active")
    if last_active is None:
        return
    inactive_days = max(0, (current_date - last_active).days - INACTIVITY_GRACE_DAYS)
    if inactive_days <= 0:
        return
    retention = 0.5 ** (inactive_days / INACTIVITY_HALF_LIFE_DAYS)
    old: Rating = rating_data["rating"]
    rating_data["rating"] = Rating(
        mu=baseline_mu + (old.mu - baseline_mu) * retention,
        phi=baseline_phi - (baseline_phi - old.phi) * retention,
        sigma=old.sigma,
    )


def handle_new_entity(
    ent_id: int | str,
    glicko2_ratings: dict[int | str, dict[str, Any]],
    league_elo_dict: dict[str, float],
    new_league: str,
    current_season: int,
    baseline_mu: float,
    baseline_phi: float,
    baseline_sigma: float,
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
        "rating": Rating(mu=initial_mu, phi=baseline_phi, sigma=baseline_sigma),
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
    baseline_phi: float,
    baseline_sigma: float,
    decay_factor: float,
    league_elo_dict: dict[str, float],
    transfer_factor: float,
    initial_elo_adjustment_factor: float,
    position_reset_factor: float = 0.2,
) -> dict[int | str, dict[str, Any]]:
    """Process a single grouped game, updating Glicko-2 ratings for both sides."""
    current_season = game_group.iloc[0]["season"]
    current_date = pd.Timestamp(game_group.iloc[0]["date"])
    if pd.isna(current_date):
        raise ValueError("rating game date cannot be missing")
    current_date = cast("pd.Timestamp", current_date)

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
                baseline_phi=baseline_phi,
                baseline_sigma=baseline_sigma,
                init_adjust_factor=initial_elo_adjustment_factor,
            )
        elif not is_cross_league_competition(new_league):
            handle_league_swap(
                ent_id=ent_id,
                new_league=new_league,
                glicko2_ratings=glicko2_ratings,
                league_elo_dict=league_elo_dict,
                baseline_mu=baseline_mu,
                transfer_factor=transfer_factor,
            )
        apply_inactivity_decay(
            glicko2_ratings[ent_id], current_date, baseline_mu, baseline_phi
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
    for ent_id in entity_ids:
        glicko2_ratings[ent_id]["last_active"] = current_date

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
    *,
    force_retune: bool = False,
) -> dict[str, float]:
    """Load Glicko-2 hyperparameters or compute them via Optuna."""
    best_params = _load_hyperparameters(hyperparameters_path, data_pipeline_logger)
    if best_params and not force_retune:
        return best_params
    if not force_retune:
        raise FileNotFoundError(
            f"Missing reviewed Glicko hyperparameters at {hyperparameters_path}; "
            "run the explicit rating retune workflow."
        )

    logger.info(
        f"No hyperparameters found at {hyperparameters_path}. Starting tuning process..."
    )
    data_pipeline_logger.info(
        f"No hyperparameters found at {hyperparameters_path}. Starting tuning process..."
    )

    def objective(trial: optuna.trial.Trial) -> float:
        hyperparams = suggest_glicko2_hyperparameters(trial)

        df_train, df_valid = _split_and_validate_data(df, entity, data_pipeline_logger)
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

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED),
    )
    study.optimize(objective, n_trials=TRIALS_NUM, show_progress_bar=False)

    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")
    data_pipeline_logger.info(f"Best hyperparameters: {best_params}")

    _save_hyperparameters(best_params, hyperparameters_path, data_pipeline_logger)
    return best_params


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
        grouped = tqdm(
            grouped,
            desc="Processing games",
            total=grouped.ngroups,
            disable=not sys.stderr.isatty(),
        )

    for _, game_grp in grouped:
        glicko2_ratings = process_game(
            df=df,
            game_group=game_grp,
            glicko2_ratings=glicko2_ratings,
            glicko2_model=glicko2_model,
            entity=entity,
            entity_key=entity_key,
            baseline_mu=mu,
            baseline_phi=phi,
            baseline_sigma=sigma,
            decay_factor=decay_factor,
            league_elo_dict=league_elo_dict,
            transfer_factor=transfer_factor,
            initial_elo_adjustment_factor=initial_elo_adjustment_factor,
            position_reset_factor=position_reset_factor,
        )

    df = freeze_same_date_rating_inputs(
        df,
        entity=entity.lower(),
        rating_columns=("glicko2_mu_before", "glicko2_phi_before"),
    )
    for _, game_grp in df.groupby(["date", "gameid"], sort=False):
        blue_rows = game_grp[game_grp["side"] == "Blue"]
        red_rows = game_grp[game_grp["side"] == "Red"]
        blue = [
            Rating(mu=row.glicko2_mu_before, phi=row.glicko2_phi_before, sigma=sigma)
            for row in blue_rows.itertuples()
        ]
        red = [
            Rating(mu=row.glicko2_mu_before, phi=row.glicko2_phi_before, sigma=sigma)
            for row in red_rows.itertuples()
        ]
        blue_mean = calculate_mean_rating(blue)
        red_mean = calculate_mean_rating(red)
        blue_expected = glicko2_model.expect_score(
            blue_mean, red_mean, glicko2_model.reduce_impact(red_mean)
        )
        df.loc[blue_rows.index, "glicko2_win_likelihood"] = blue_expected
        df.loc[red_rows.index, "glicko2_win_likelihood"] = 1.0 - blue_expected

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
        if RATING_LEAGUE_ELO.exists():
            league_elo_df = pd.read_parquet(RATING_LEAGUE_ELO)
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
