"""
Elo Rating System with Hyperparameter Tuning using Optuna

This module contains functions to calculate Elo ratings for teams or players based on match results,
with performance-minded implementations and leak-free temporal evolution.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import optuna
from sklearn.metrics import log_loss
from tqdm import tqdm

from utils.io_utils import get_sorting_keys, json_loader
from utils.logger import LOG_TOPIC, instantiate_logger, logger
from utils.paths import (
    CONSIDERED_LEAGUES,
    DEFAULT_MODELS_PARAMETERS,
    ENTITY_ELO_HYPERPARAMETERS,
    LEAGUE_ELO,
)
from utils.pd import pd

try:
    from numba import njit
except Exception:
    njit = None

# ------------------------------------------------------------------------------
# Global Config / Constants
# ------------------------------------------------------------------------------
config = json_loader(DEFAULT_MODELS_PARAMETERS)
considered_leagues_config = json_loader(CONSIDERED_LEAGUES)
data_pipeline_logger = instantiate_logger(LOG_TOPIC.DATA_PIPELINE)
CROSS_COMPETITION_LEAGUES = considered_leagues_config["cross_league_competitions"]
MAJOR_LEAGUES = considered_leagues_config["major_leagues"]
TRIALS_NUM = 25
MAX_EXPONENT = 8.0  # To prevent overflow in expected outcome calc


# ------------------------------------------------------------------------------
# Helper Utilities
# ------------------------------------------------------------------------------
def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp `value` between `min_val` and `max_val`."""
    return max(min_val, min(value, max_val))


def is_major_league(league: str) -> bool:
    """Check if league is considered major."""
    return league in MAJOR_LEAGUES


# ------------------------------------------------------------------------------
# 1. Preprocessing
# ------------------------------------------------------------------------------
def preprocess_elo_dataframe(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Ensure required columns, normalize dtypes, and sort rows by stable keys.
    """
    if entity.lower() not in ["team", "player"]:
        msg = "Entity must be 'team' or 'player'"
        raise ValueError(msg)

    entity_key = "teamid" if entity.lower() == "team" else "playerid"

    # Required columns
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

    # Check for missing columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        msg = f"Input DataFrame is missing required columns: {missing_columns}"
        raise ValueError(msg)

    df = df.copy()

    # Convert 'date' to datetime
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

    # Normalize 'result' to numeric 0/1 if needed
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

    # Stable sort by canonical keys
    return df.sort_values(by=get_sorting_keys(entity), kind="mergesort").reset_index(
        drop=True
    )


# ------------------------------------------------------------------------------
# 2. Core Elo Functions (Numba-accelerated when available)
# ------------------------------------------------------------------------------
def _expected_outcome_py(elo_a: float, elo_b: float, elo_divisor: float) -> float:
    """Expected match outcome between two Elo totals (logistic on Elo gap)."""
    exponent = (elo_b - elo_a) / elo_divisor
    if exponent > MAX_EXPONENT:
        exponent = MAX_EXPONENT
    elif exponent < -MAX_EXPONENT:
        exponent = -MAX_EXPONENT
    return 1.0 / (1.0 + 10.0**exponent)


def _update_elo_py(
    old_elo: float, expected: float, actual_result: float, k_factor: float
) -> float:
    """Update Elo rating based on the match result."""
    return old_elo + k_factor * (actual_result - expected)


if njit is not None:
    expected_outcome = njit(cache=True, fastmath=True)(_expected_outcome_py)  # type: ignore[assignment]
    update_elo_rating = njit(cache=True, fastmath=True)(_update_elo_py)  # type: ignore[assignment]
else:  # fallback when numba unavailable
    expected_outcome = _expected_outcome_py
    update_elo_rating = _update_elo_py


def aggregate_team_elo(
    rows: pd.DataFrame, elo_ratings: dict[int | str, dict[str, Any]], entity_key: str
) -> float:
    """
    Aggregate Elo ratings for a group of entities in 'rows' (e.g., all players on a team).
    Optimized to avoid building a full ratings DataFrame each call.
    """
    if rows.empty:
        return 0.0
    return (
        rows[entity_key].map(lambda eid: elo_ratings.get(eid, {}).get("elo", 0.0)).sum()
    )


def handle_position_switch(
    entity_id: int | str,
    new_position: str | None,
    elo_ratings: dict[int | str, dict[str, Any]],
    baseline_elo: float,
    position_reset_factor: float,
) -> None:
    """
    Partially reset a player's Elo if they switch position.
    Moves Elo toward baseline_elo by position_reset_factor.
    """
    if not new_position:
        return

    last_position = elo_ratings[entity_id].get("last_position")
    if last_position and last_position != new_position:
        old_elo = elo_ratings[entity_id]["elo"]
        # partial reset toward baseline
        elo_ratings[entity_id]["elo"] = baseline_elo + (old_elo - baseline_elo) * (
            1.0 - position_reset_factor
        )

    elo_ratings[entity_id]["last_position"] = new_position


def linear_decay_reset(
    elo_ratings: dict[int | str, dict[str, Any]],
    current_season: int,
    baseline_elo: float,
    decay_factor: float,
) -> dict[int | str, dict[str, Any]]:
    """
    In-place: Apply a seasonal decay if the stored season < current_season.
    Elo is partially reset toward baseline by decay_factor.
    """
    for data in elo_ratings.values():
        if data.get("season", current_season) < current_season:
            data["elo"] = baseline_elo + (data["elo"] - baseline_elo) * decay_factor
            data["season"] = current_season
    return elo_ratings  # Return the *same* dict reference


def handle_new_entity(
    ent_id: int | str,
    elo_ratings: dict[int | str, dict[str, Any]],
    league_elo_dict: dict[str, float],
    new_league: str,
    current_season: int,
    baseline_elo: float,
    init_adjust_factor: float,
) -> None:
    """
    Initialize the Elo rating for a brand-new entity.
    Partial adjustment based on the league's Elo relative to average league Elo,
    clamped to avoid extreme starts.
    """
    max_diff = 0.2 * baseline_elo

    avg_league_elo = (
        sum(league_elo_dict.values()) / len(league_elo_dict)
        if league_elo_dict
        else baseline_elo
    )
    league_elo = league_elo_dict.get(new_league, baseline_elo)

    init_adjustment = (league_elo - avg_league_elo) * init_adjust_factor
    initial_rating = baseline_elo + clamp(init_adjustment, -max_diff, max_diff)

    elo_ratings[ent_id] = {
        "elo": initial_rating,
        "season": current_season,
        "league": new_league,
    }


def handle_league_swap(
    ent_id: int | str,
    new_league: str,
    elo_ratings: dict[int | str, dict[str, Any]],
    league_elo_dict: dict[str, float],
    baseline_elo: float,
    transfer_factor: float,
    transfer_factor_minor_to_major: float = 0.4,
) -> None:
    """
    Handle league transitions with partial adjustments for Elo ratings.

    If the league changes, adjust Elo based on the difference in league Elo ratings and whether
    the entity is moving between minor and major leagues. Moves into a major league get a stronger penalty.
    """
    curr_league = elo_ratings[ent_id].get("league")
    if not curr_league or curr_league == new_league:
        elo_ratings[ent_id]["league"] = new_league
        return

    curr_is_major = is_major_league(curr_league)
    new_is_major = is_major_league(new_league)

    curr_elo_val = league_elo_dict.get(curr_league, baseline_elo)
    new_elo_val = league_elo_dict.get(new_league, baseline_elo)
    diff = new_elo_val - curr_elo_val

    old_elo = elo_ratings[ent_id]["elo"]
    if not curr_is_major and new_is_major:
        # Minor -> Major: penalize (lower confidence in prior rating)
        adjusted_diff = transfer_factor_minor_to_major * diff
        new_elo = old_elo - adjusted_diff
    else:
        adjusted_diff = transfer_factor * diff
        new_elo = old_elo + adjusted_diff

    elo_ratings[ent_id]["elo"] = new_elo
    elo_ratings[ent_id]["league"] = new_league


# ------------------------------------------------------------------------------
# 3. Game Processing
# ------------------------------------------------------------------------------
def process_game(
    df: pd.DataFrame,
    game_group: pd.DataFrame,
    elo_ratings: dict[int | str, dict[str, Any]],
    k_factor: float,
    entity: str,
    entity_key: str,
    baseline_elo: float,
    decay_factor: float,
    elo_divisor: float,
    league_elo_dict: dict[str, float],
    transfer_factor: float,
    initial_elo_adjustment_factor: float,
    position_reset_factor: float = 0.2,
) -> dict[int | str, dict[str, Any]]:
    """
    Process a single game group, updating Elo for both sides and writing results to df.
    Return the updated dictionary so it persists across matches.
    """
    current_season = game_group.iloc[0]["season"]

    # Seasonal decay reset in-place
    linear_decay_reset(
        elo_ratings=elo_ratings,
        current_season=current_season,
        baseline_elo=baseline_elo,
        decay_factor=decay_factor,
    )

    # Identify all entities in the group
    entity_ids = game_group[entity_key].unique()

    # Check/initialize each entity’s Elo rating
    for ent_id in entity_ids:
        new_league = game_group.loc[game_group[entity_key] == ent_id, "league"].iloc[0]
        if ent_id not in elo_ratings:
            handle_new_entity(
                ent_id=ent_id,
                elo_ratings=elo_ratings,
                league_elo_dict=league_elo_dict,
                new_league=new_league,
                current_season=current_season,
                baseline_elo=baseline_elo,
                init_adjust_factor=initial_elo_adjustment_factor,
            )
        elif new_league not in CROSS_COMPETITION_LEAGUES:
            handle_league_swap(
                ent_id=ent_id,
                new_league=new_league,
                elo_ratings=elo_ratings,
                league_elo_dict=league_elo_dict,
                baseline_elo=baseline_elo,
                transfer_factor=transfer_factor,
            )

    # If dealing with players, handle position switching
    if entity.lower() == "player":
        for _, row in game_group.iterrows():
            ent_id = row[entity_key]
            new_position = row.get("position")
            handle_position_switch(
                entity_id=ent_id,
                new_position=new_position,
                elo_ratings=elo_ratings,
                baseline_elo=baseline_elo,
                position_reset_factor=position_reset_factor,
            )

    # Split sides
    blue_rows = game_group[game_group["side"] == "Blue"]
    red_rows = game_group[game_group["side"] == "Red"]

    # Compute total Elo for each side
    blue_elo_sum = aggregate_team_elo(blue_rows, elo_ratings, entity_key)
    red_elo_sum = aggregate_team_elo(red_rows, elo_ratings, entity_key)

    # Calculate expected outcomes (blue perspective)
    blue_expected = expected_outcome(blue_elo_sum, red_elo_sum, elo_divisor)
    blue_result = float(blue_rows.iloc[0]["result"])
    red_result = 1.0 - blue_result

    # Retrieve old Elos per entity (for per-row writeback)
    blue_ids = blue_rows[entity_key].to_numpy()
    red_ids = red_rows[entity_key].to_numpy()
    blue_old_elos = [elo_ratings[b]["elo"] for b in blue_ids]
    red_old_elos = [elo_ratings[r]["elo"] for r in red_ids]

    # Update Elos per entity
    blue_new_elos = [
        update_elo_rating(old, blue_expected, blue_result, k_factor)
        for old in blue_old_elos
    ]
    red_new_elos = [
        update_elo_rating(old, 1.0 - blue_expected, red_result, k_factor)
        for old in red_old_elos
    ]

    # Commit new Elos
    for i, bid in enumerate(blue_ids):
        elo_ratings[bid]["elo"] = blue_new_elos[i]
    for i, rid in enumerate(red_ids):
        elo_ratings[rid]["elo"] = red_new_elos[i]

    # Write columns back to df (per row):
    # - 'elo_before' = own entity Elo before
    # - 'opp_elo_before' = single opponent Elo before
    # - 'elo_win_likelihood' = blue_expected or 1 - blue_expected
    # - 'elo_after' = own entity Elo after
    df.loc[blue_rows.index, "elo_before"] = blue_old_elos
    df.loc[blue_rows.index, "opp_elo_before"] = red_old_elos
    df.loc[blue_rows.index, "elo_win_likelihood"] = blue_expected
    df.loc[blue_rows.index, "elo_after"] = blue_new_elos

    df.loc[red_rows.index, "elo_before"] = red_old_elos
    df.loc[red_rows.index, "opp_elo_before"] = blue_old_elos
    df.loc[red_rows.index, "elo_win_likelihood"] = 1.0 - blue_expected
    df.loc[red_rows.index, "elo_after"] = red_new_elos

    return elo_ratings


# ------------------------------------------------------------------------------
# 4. Hyperparameter Tuning
# ------------------------------------------------------------------------------
def tune_elo_hyperparameters(
    df: pd.DataFrame,
    entity: str,
    hyperparameters_path: Path,
    league_elo_dict: dict[str, float],
) -> dict[str, float]:
    """
    Either load hyperparameters if they exist, or compute them via Optuna.
    Returns the best parameters for subsequent Elo calculations.
    """
    # Attempt to load existing hyperparameters
    best_params = load_hyperparameters(hyperparameters_path)
    if best_params:
        return best_params

    logger.info(
        f"No hyperparameters found at {hyperparameters_path}. Starting tuning process..."
    )
    data_pipeline_logger.info(
        f"No hyperparameters found at {hyperparameters_path}. Starting tuning process..."
    )

    # Define the objective function for Optuna
    def objective(trial: optuna.trial.Trial) -> float:
        # Suggest hyperparameters
        hyperparams = suggest_hyperparameters(trial)

        # Split data into training and validation sets
        df_train, df_valid = split_and_validate_data(df, entity)
        if df_train.empty or df_valid.empty:
            logger.warning("Training or validation DataFrame is empty after splitting.")
            data_pipeline_logger.warning(
                "Training or validation DataFrame is empty after splitting."
            )
            return float("inf")

        # Run Elo computation on training data
        try:
            df_train_res = run_elo_computation(
                df=df_train.copy(),
                entity=entity,
                initial_elo=hyperparams["initial_elo"],
                k_factor=hyperparams["k_factor"],
                decay_factor=hyperparams["decay_factor"],
                elo_divisor=hyperparams["elo_divisor"],
                transfer_factor=hyperparams["transfer_factor"],
                initial_elo_adjustment_factor=hyperparams[
                    "initial_elo_adjustment_factor"
                ],
                position_reset_factor=hyperparams["position_reset_factor"],
                league_elo_dict=league_elo_dict,
                show_progress=False,
            )
        except Exception as err:
            logger.error(f"Error during Elo computation in training phase: {err}")
            data_pipeline_logger.exception(
                f"Error during Elo computation in training phase: {err}"
            )
            return float("inf")

        # Initialize validation ratings based on training results
        val_ratings = initialize_validation_ratings(
            df_train_res, entity, hyperparams["initial_elo"]
        )

        # Evaluate on validation set and compute log loss
        try:
            loss = evaluate_validation(df_valid, val_ratings, entity, hyperparams)
        except Exception as err:
            logger.error(f"Error during validation phase: {err}")
            data_pipeline_logger.exception(f"Error during validation phase: {err}")
            return float("inf")

        return loss

    # Create and optimize the Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=TRIALS_NUM, show_progress_bar=True)

    # Retrieve and log the best parameters
    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")
    data_pipeline_logger.info(f"Best hyperparameters: {best_params}")

    # Save the best hyperparameters to the specified path
    save_hyperparameters(best_params, hyperparameters_path)

    return best_params


def load_hyperparameters(path: Path) -> dict[str, float]:
    """
    Load hyperparameters from a JSON file if it exists.
    Returns the loaded parameters or {} if the file doesn't exist or loading fails.
    """
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


def suggest_hyperparameters(trial: optuna.trial.Trial) -> dict[str, float]:
    """Suggest hyperparameters using Optuna's trial object."""
    return {
        "k_factor": trial.suggest_float("k_factor", 16, 64, step=8),
        "initial_elo": trial.suggest_float("initial_elo", 1200, 1800, step=100),
        "elo_divisor": trial.suggest_float("elo_divisor", 100, 500, step=100),
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
    Sort, split, and validate the DataFrame into training and validation sets.
    Returns the training and validation DataFrames.
    """
    # Sort the DataFrame
    df_sorted = df.sort_values(by=["date", "gameid", "side"]).reset_index(drop=True)
    if df_sorted.empty:
        logger.warning("DataFrame is empty after sorting.")
        data_pipeline_logger.warning("DataFrame is empty after sorting.")
        return pd.DataFrame(), pd.DataFrame()

    # Primary split: start of the last year present
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

    # Fallback: if train or valid is empty, do an 80/20 time split
    if df_train.empty or df_valid.empty:
        q80 = df_sorted["date"].quantile(0.8)
        df_train = df_sorted[df_sorted["date"] < q80].reset_index(drop=True)
        df_valid = df_sorted[df_sorted["date"] >= q80].reset_index(drop=True)

    # Validate group sizes
    expected_count = 10 if entity.lower() == "player" else 2
    for split_df, split_name in [(df_train, "Training"), (df_valid, "Validation")]:
        if not split_df.empty:
            group_sizes = split_df.groupby("gameid").size()
            if not (group_sizes == expected_count).all():
                logger.warning(
                    f"{split_name} data has gameids with incorrect number of entities."
                )
                data_pipeline_logger.warning(
                    f"{split_name} data has gameids with incorrect number of entities."
                )
                return pd.DataFrame(), pd.DataFrame()

    return df_train, df_valid


def initialize_validation_ratings(
    df_train_res: pd.DataFrame, entity: str, initial_elo: float
) -> defaultdict:
    """Initialize validation ratings based on training results."""
    entity_key = "teamid" if entity.lower() == "team" else "playerid"
    last_elo_map = df_train_res.groupby(entity_key)["elo_after"].last().to_dict()

    val_ratings = defaultdict(lambda: {"elo": initial_elo})
    for e_id, final_elo in last_elo_map.items():
        # final_elo should be scalar because we write scalar per row
        val_ratings[e_id]["elo"] = float(final_elo)
    return val_ratings


def evaluate_validation(
    df_valid: pd.DataFrame,
    val_ratings: defaultdict,
    entity: str,
    hyperparams: dict[str, float],
) -> float:
    """Evaluate the validation set and compute the log loss."""
    expected_probs: list[float] = []
    df_valid_sorted = df_valid.sort_values(by=["date", "gameid"]).reset_index(drop=True)
    elo_divisor = hyperparams["elo_divisor"]
    k_factor = hyperparams["k_factor"]
    entity_key = "teamid" if entity.lower() == "team" else "playerid"

    for _, grp in df_valid_sorted.groupby(["date", "gameid"], sort=False):
        blue_side = grp[grp["side"] == "Blue"]
        red_side = grp[grp["side"] == "Red"]

        blue_elo_sum = sum(val_ratings[bid]["elo"] for bid in blue_side[entity_key])
        red_elo_sum = sum(val_ratings[rid]["elo"] for rid in red_side[entity_key])

        exp = expected_outcome(blue_elo_sum, red_elo_sum, elo_divisor)
        result = float(blue_side.iloc[0]["result"])

        # One probability per blue-row (team: 1, player: 5)
        expected_probs.extend([exp] * len(blue_side))

        # Update Elo ratings for validation trajectory
        for bid in blue_side[entity_key]:
            val_ratings[bid]["elo"] = update_elo_rating(
                val_ratings[bid]["elo"], exp, result, k_factor
            )
        for rid in red_side[entity_key]:
            val_ratings[rid]["elo"] = update_elo_rating(
                val_ratings[rid]["elo"], 1.0 - exp, 1.0 - result, k_factor
            )

    # Prepare true labels and predictions
    y_true = df_valid_sorted.loc[df_valid_sorted["side"] == "Blue", "result"]
    y_pred = pd.Series(expected_probs).clip(0.0001, 0.9999)

    # Ensure matching lengths
    if len(y_true) != len(y_pred):
        logger.error("Mismatch in lengths of y_true and y_pred.")
        data_pipeline_logger.error("Mismatch in lengths of y_true and y_pred.")
        return float("inf")

    return log_loss(y_true, y_pred)


# ------------------------------------------------------------------------------
# 5. Main Elo Computation
# ------------------------------------------------------------------------------
def run_elo_computation(
    df: pd.DataFrame,
    entity: str,
    initial_elo: float,
    k_factor: float,
    decay_factor: float,
    elo_divisor: float,
    transfer_factor: float,
    initial_elo_adjustment_factor: float,
    position_reset_factor: float,
    league_elo_dict: dict[str, float],
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Main Elo update procedure:
      - Group matches by (date, gameid)
      - For each group, call `process_game` to update Elo ratings
      - Return df with new columns: elo_before, opp_elo_before, elo_win_likelihood, elo_after
    """
    df = df.copy()
    entity_key = "teamid" if entity.lower() == "team" else "playerid"

    # Initialize Elo ratings once
    elo_ratings = defaultdict(
        lambda: {
            "elo": initial_elo,
            "season": int(df["season"].min()),
            "league": None,
        }
    )

    # Pre-allocate output columns
    for col in ["elo_before", "opp_elo_before", "elo_win_likelihood", "elo_after"]:
        df[col] = pd.Series(index=df.index, dtype="float64")

    grouped = df.groupby(["date", "gameid"], sort=False)
    if show_progress:
        grouped = tqdm(grouped, desc="Processing games", total=grouped.ngroups)

    for _, game_grp in grouped:
        elo_ratings = process_game(
            df=df,
            game_group=game_grp,
            elo_ratings=elo_ratings,
            k_factor=k_factor,
            entity=entity,
            entity_key=entity_key,
            baseline_elo=initial_elo,
            decay_factor=decay_factor,
            elo_divisor=elo_divisor,
            league_elo_dict=league_elo_dict,
            transfer_factor=transfer_factor,
            initial_elo_adjustment_factor=initial_elo_adjustment_factor,
            position_reset_factor=position_reset_factor,
        )

    return df


def calculate_elo(
    df: pd.DataFrame,
    entity: str,
    league_elo_dict: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Main entry point for Elo computation."""
    df_pre = preprocess_elo_dataframe(df, entity)

    # Attempt to load league Elo if not provided
    if league_elo_dict is None:
        league_elo_dict = {}
        if LEAGUE_ELO.exists():
            league_elo_df = pd.read_parquet(LEAGUE_ELO)
            league_elo_dict = league_elo_df.set_index("league")["elo"].to_dict()

    # Load or compute hyperparams
    hyperparameters_path = Path(
        str(ENTITY_ELO_HYPERPARAMETERS).replace("entity", entity)
    )
    best_params = tune_elo_hyperparameters(
        df_pre, entity, hyperparameters_path, league_elo_dict
    )

    # Run final Elo computation
    return run_elo_computation(
        df=df_pre,
        entity=entity,
        initial_elo=best_params["initial_elo"],
        k_factor=best_params["k_factor"],
        decay_factor=best_params["decay_factor"],
        elo_divisor=best_params["elo_divisor"],
        transfer_factor=best_params["transfer_factor"],
        initial_elo_adjustment_factor=best_params["initial_elo_adjustment_factor"],
        position_reset_factor=best_params.get("position_reset_factor", 0.2),
        league_elo_dict=league_elo_dict,
        show_progress=True,
    )
