"""
Elo Rating System with Hyperparameter Tuning using Optuna

This module contains functions to calculate Elo ratings for teams or players based on match results,
with FireDucks-based performance optimizations.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import optuna
import pandas as pd
import polars as pl
from numba import njit
from sklearn.metrics import log_loss
from tqdm import tqdm

from src.utils.logger import logger
from src.utils.paths import CONSIDERED_LEAGUES, DEFAULT_MODELS_PARAMETERS, ENTITY_ELO_HYPERPARAMETERS, LEAGUE_ELO
from src.utils.utils import get_sorting_keys, json_loader

# ------------------------------------------------------------------------------
# Global Config / Constants
# ------------------------------------------------------------------------------
config = json_loader(DEFAULT_MODELS_PARAMETERS)
considered_leagues_config = json_loader(CONSIDERED_LEAGUES)
MAJOR_LEAGUES = considered_leagues_config["major_leagues"]
CROSS_LEAGUE_COMPETITIONS = considered_leagues_config["cross_league_competitions"]
TRIALS_NUM = 50


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
def preprocess_elo_dataframe(df: pd.DataFrame, entity: str) -> pl.DataFrame:
    """
    Ensures the DataFrame has all required columns, that 'date' is a datetime,
    and that rows are sorted by the appropriate keys.
    """
    entity_key = "teamid" if entity.lower() == "team" else "playerid"

    # Required columns
    required_columns = ["season", "date", "gameid", entity_key, "league", "side", "result"]
    if entity.lower() == "player":
        required_columns.append("position")

    # Check for missing columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_columns}")

    # Convert 'date' to datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        null_count = df["date"].isnull().sum()
        if null_count > 0:
            logger.warning(f"{null_count} 'date' entries could not be converted; dropping them.")
            df = df.dropna(subset=["date"]).copy()

    df = pl.DataFrame(df, orient="row")

    # Drop rows missing 'league' or 'result'
    null_league = df.filter(df["league"].is_null())
    null_result = df.filter(df["result"].is_null())
    if null_league.height > 0 or null_result.height > 0:
        logger.info(
            f"Dropping {null_league.height} rows with null 'league' and {null_result.height} rows with null 'result'"
        )
        df = df.filter(df["league"].is_not_null() & df["result"].is_not_null())

    # Sort keys
    return df.sort(get_sorting_keys(entity))


# ------------------------------------------------------------------------------
# 2. Core Elo Functions
# ------------------------------------------------------------------------------
@njit
def expected_outcome(elo_a: float, elo_b: float, elo_divisor: float) -> float:
    """Calculate the expected match outcome between two Elo ratings."""
    exponent = (elo_b - elo_a) / elo_divisor
    return 1.0 / (1.0 + 10.0**exponent)


@njit
def update_elo_rating(old_elo: float, expected: float, actual_result: float, k_factor: float) -> float:
    """Update Elo rating based on the match result."""
    adjustment = k_factor * (actual_result - expected)
    return old_elo + adjustment


def aggregate_team_elo(
    rows: pl.DataFrame, elo_ratings: Dict[Union[int, str], Dict[str, Any]], entity_key: str
) -> float:
    """
    Aggregate Elo ratings for a group of entities in 'rows' (e.g., all players on a team).
    Uses a merge-based approach for clarity.
    """
    if rows.height == 0:
        return 0.0

    # Build a small DataFrame of current Elo ratings
    rating_data = [(eid, info["elo"]) for eid, info in elo_ratings.items()]
    rating_df = pl.DataFrame(rating_data, schema=[entity_key, "elo"], orient="row")
    merged = rows.join(rating_df, on=entity_key, how="left")
    return merged["elo"].fill_null(0).sum()


def handle_position_switch(
    entity_id: Union[int, str],
    new_position: Optional[str],
    elo_ratings: Dict[Union[int, str], Dict[str, Any]],
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
        elo_ratings[entity_id]["elo"] = baseline_elo + (old_elo - baseline_elo) * (1.0 - position_reset_factor)

    elo_ratings[entity_id]["last_position"] = new_position


def linear_decay_reset(
    elo_ratings: Dict[Union[int, str], Dict[str, Any]],
    current_season: int,
    baseline_elo: float,
    decay_factor: float,
) -> None:
    """
    Apply a seasonal decay to each entity if stored season < current_season.
    Elo is partially reset toward baseline by decay_factor.
    """
    for data in elo_ratings.values():
        if data["season"] < current_season:
            old_elo = data["elo"]
            data["elo"] = baseline_elo + (old_elo - baseline_elo) * decay_factor
            data["season"] = current_season


def handle_new_entity(
    ent_id: Union[int, str],
    elo_ratings: Dict[Union[int, str], Dict[str, Any]],
    league_elo_dict: Dict[str, float],
    new_league: str,
    current_season: int,
    baseline_elo: float,
    init_adjust_factor: float,
) -> None:
    """
    Initialize the Elo rating for a brand-new entity.
    Use a partial adjustment based on the difference between the league's Elo and the average Elo.
    """
    # Cap max difference to avoid extreme starts
    max_diff = 0.2 * baseline_elo

    avg_league_elo = sum(league_elo_dict.values()) / len(league_elo_dict) if league_elo_dict else baseline_elo
    league_elo = league_elo_dict.get(new_league, baseline_elo)

    init_adjustment = (league_elo - avg_league_elo) * init_adjust_factor
    initial_rating = baseline_elo + init_adjustment

    offset = clamp(initial_rating - baseline_elo, -max_diff, max_diff)
    initial_rating = baseline_elo + offset

    elo_ratings[ent_id] = {
        "elo": initial_rating,
        "season": current_season,
        "league": new_league,
    }


def handle_league_swap(
    ent_id: Union[int, str],
    new_league: str,
    elo_ratings: Dict[Union[int, str], Dict[str, Any]],
    league_elo_dict: Dict[str, float],
    baseline_elo: float,
    transfer_factor: float,
    transfer_factor_minor_to_major: float = 0.4,
) -> None:
    """
    Handle league transitions with partial adjustments for Elo ratings.

    If the league changes, adjust Elo based on the difference in league Elo ratings and whether
    the entity is moving between minor and major leagues. Players moving from minor to major
    leagues are penalized more heavily.
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
    if "player" in ent_id.lower() and not curr_is_major and new_is_major:
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
    df: pl.DataFrame,
    game_group: pl.DataFrame,
    elo_ratings: Dict[Union[int, str], Dict[str, Any]],
    k_factor: float,
    entity: str,
    entity_key: str,
    baseline_elo: float,
    decay_factor: float,
    elo_divisor: float,
    league_elo_dict: Dict[str, float],
    transfer_factor: float,
    initial_elo_adjustment_factor: float,
    position_reset_factor: float = 0.2,
) -> None:
    """Process a single game group, updating Elo for both sides and writing results to df."""
    current_season = game_group["season"].to_list()[0]

    linear_decay_reset(
        elo_ratings=elo_ratings,
        current_season=current_season,
        baseline_elo=baseline_elo,
        decay_factor=decay_factor,
    )

    entity_ids = game_group[entity_key].unique().to_list()

    for ent_id in entity_ids:
        if ent_id not in elo_ratings:
            new_league = game_group.filter(game_group[entity_key] == ent_id)["league"].to_list()[0]
            handle_new_entity(
                ent_id=ent_id,
                elo_ratings=elo_ratings,
                league_elo_dict=league_elo_dict,
                new_league=new_league,
                current_season=current_season,
                baseline_elo=baseline_elo,
                init_adjust_factor=initial_elo_adjustment_factor,
            )
        else:
            new_league = game_group.filter(game_group[entity_key] == ent_id)["league"].to_list()[0]
            handle_league_swap(
                ent_id=ent_id,
                new_league=new_league,
                elo_ratings=elo_ratings,
                league_elo_dict=league_elo_dict,
                baseline_elo=baseline_elo,
                transfer_factor=transfer_factor,
            )

    if entity.lower() == "player":
        for row in game_group.iter_rows(named=True):
            ent_id = row[entity_key]
            new_position = row.get("position")
            handle_position_switch(
                entity_id=ent_id,
                new_position=new_position,
                elo_ratings=elo_ratings,
                baseline_elo=baseline_elo,
                position_reset_factor=position_reset_factor,
            )

    # Filter Blue and Red rows
    blue_rows = game_group.filter(pl.col("side") == "Blue")
    red_rows = game_group.filter(pl.col("side") == "Red")

    # Compute total Elo for each side
    blue_elo_sum = aggregate_team_elo(blue_rows, elo_ratings, entity_key)
    red_elo_sum = aggregate_team_elo(red_rows, elo_ratings, entity_key)

    # Calculate expected outcomes
    blue_expected = expected_outcome(blue_elo_sum, red_elo_sum, elo_divisor)
    blue_result = blue_rows["result"].to_list()[0]
    red_result = 1.0 - blue_result

    # Retrieve and calculate old and new Elos
    blue_ids = blue_rows[entity_key].to_list()
    red_ids = red_rows[entity_key].to_list()

    blue_old_elos = [elo_ratings[b]["elo"] for b in blue_ids]
    red_old_elos = [elo_ratings[r]["elo"] for r in red_ids]

    blue_new_elos = [update_elo_rating(old, blue_expected, blue_result, k_factor) for old in blue_old_elos]
    red_new_elos = [update_elo_rating(old, 1.0 - blue_expected, red_result, k_factor) for old in red_old_elos]

    # Commit new Elos
    for i, bid in enumerate(blue_ids):
        elo_ratings[bid]["elo"] = blue_new_elos[i]
    for i, rid in enumerate(red_ids):
        elo_ratings[rid]["elo"] = red_new_elos[i]

    # Add a temporary row index column to preserve alignment
    df = df.with_row_count("row_index")

    if not blue_rows.is_empty():
        blue_rows = blue_rows.with_columns(
            [
                pl.Series("elo_before", blue_old_elos),
                pl.Series("opp_elo_before", [sum(red_old_elos) / len(red_old_elos)] * len(blue_rows)),
                pl.Series("elo_win_likelihood", [blue_expected] * len(blue_rows)),
                pl.Series("elo_after", blue_new_elos),
            ]
        ).with_row_count("row_index")

    if not red_rows.is_empty():
        red_rows = red_rows.with_columns(
            [
                pl.Series("elo_before", red_old_elos),
                pl.Series("opp_elo_before", [sum(blue_old_elos) / len(blue_old_elos)] * len(red_rows)),
                pl.Series("elo_win_likelihood", [1.0 - blue_expected] * len(red_rows)),
                pl.Series("elo_after", red_new_elos),
            ]
        ).with_row_count("row_index")

    # Concatenate updated Blue and Red rows
    updated_game_group = pl.concat([blue_rows, red_rows])

    # Update the original DataFrame
    df = df.join(updated_game_group, on="row_index", how="left").drop("row_index")


# ------------------------------------------------------------------------------
# 4. Hyperparameter Tuning
# ------------------------------------------------------------------------------
def tune_elo_hyperparameters(
    df: pl.DataFrame,
    entity: str,
    hyperparameters_path: Path,
    league_elo_dict: Dict[str, float],
) -> Dict[str, float]:
    """
    Either load hyperparameters if they exist, or compute them via Optuna.
    Returns the best parameters for subsequent Elo calculations.
    """
    if os.path.exists(hyperparameters_path):
        logger.info(f"Loading hyperparameters from {hyperparameters_path}")
        with open(hyperparameters_path) as f:
            loaded_params = json.load(f)
        return loaded_params

    logger.info(f"No hyperparameters found at {hyperparameters_path}. Starting tuning process...")

    def objective(trial: optuna.trial.Trial) -> float:
        k_factor = trial.suggest_float("k_factor", 16, 64, step=8)
        initial_elo = trial.suggest_float("initial_elo", 1200, 1800, step=100)
        elo_divisor = trial.suggest_float("elo_divisor", 100, 500, step=100)
        decay_factor = trial.suggest_float("decay_factor", 0.5, 1.0, step=0.05)
        transfer_factor = trial.suggest_float("transfer_factor", 0.1, 1.0, step=0.1)
        init_adjust = trial.suggest_float("initial_elo_adjustment_factor", 0.0, 1.0, step=0.1)
        position_reset_factor = trial.suggest_float("position_reset_factor", 0.0, 1.0, step=0.1)

        df_sorted = df.sort(["date", "gameid", "side"])
        if df_sorted.height == 0:
            return float("inf")

        try:
            split_year = df_sorted["date"].dt.year().max()
        except Exception as err:
            logger.error(f"Error in year extraction: {err}")
            return float("inf")

        split_date = pl.date(f"{split_year}-01-01")
        df_train = df_sorted.filter(df_sorted["date"] < split_date)
        df_valid = df_sorted.filter(df_sorted["date"] >= split_date)

        if df_train.height == 0 or df_valid.height == 0:
            return float("inf")

        # Quick group-size check
        expected_count = 10 if entity.lower() == "player" else 2
        if not all(df_train.group_by(["gameid"]).count()["teamid"] == expected_count):
            logger.warning("Skipping trial due to group size mismatch")
            return float("inf")
        if not all(df_valid.group_by(["gameid"]).count()["teamid"] == expected_count):
            logger.warning("Skipping trial due to group size mismatch")
            return float("inf")

        try:
            df_train_res = run_elo_computation(
                df=df_train.clone(),
                entity=entity,
                initial_elo=initial_elo,
                k_factor=k_factor,
                decay_factor=decay_factor,
                elo_divisor=elo_divisor,
                transfer_factor=transfer_factor,
                initial_elo_adjustment_factor=init_adjust,
                position_reset_factor=position_reset_factor,
                league_elo_dict=league_elo_dict,
                show_progress=True,
            )
        except Exception as err:
            logger.error(f"Error in training phase of trial: {err}")
            return float("inf")

        entity_key = "teamid" if entity.lower() == "team" else "playerid"
        last_elo_map = df_train_res.group_by(entity_key).agg(pl.col("elo_after").last()).to_dict(as_series=False)

        val_ratings = defaultdict(lambda: {"elo": initial_elo})
        for e_id, final_elo in last_elo_map.items():
            val_ratings[e_id]["elo"] = final_elo

        expected_probs = []
        df_valid = df_valid.sort(["date", "gameid"])
        for _, grp in df_valid.group_by(["date", "gameid"]):
            blue_side = grp.filter(grp["side"] == "Blue")
            red_side = grp.filter(grp["side"] == "Red")

            blue_elo_sum = sum(val_ratings[bid]["elo"] for bid in blue_side[entity_key])
            red_elo_sum = sum(val_ratings[rid]["elo"] for rid in red_side[entity_key])

            exp = expected_outcome(blue_elo_sum, red_elo_sum, elo_divisor)
            result = blue_side["result"].to_list()[0]

            expected_probs.append(exp)

            for bid in blue_side[entity_key].to_list():
                val_ratings[bid]["elo"] = update_elo_rating(val_ratings[bid]["elo"], exp, result, k_factor)
            for rid in red_side[entity_key].to_list():
                val_ratings[rid]["elo"] = update_elo_rating(val_ratings[rid]["elo"], 1.0 - exp, 1.0 - result, k_factor)

        y_true = df_valid.filter(df_valid["side"] == "Blue")["result"]
        y_pred = pl.Series(expected_probs).clip(0.0001, 0.9999)

        assert len(y_true) == len(y_pred), "Mismatch in lengths of y_true and y_pred"
        return log_loss(y_true.to_list(), y_pred.to_list())

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=TRIALS_NUM, show_progress_bar=True)
    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")

    logger.info(f"Storing hyperparameters to {hyperparameters_path}")
    with open(hyperparameters_path, "w") as f:
        json.dump(best_params, f)

    return best_params


# ------------------------------------------------------------------------------
# 5. Main Elo Computation
# ------------------------------------------------------------------------------
def run_elo_computation(
    df: pl.DataFrame,
    entity: str,
    initial_elo: float,
    k_factor: float,
    decay_factor: float,
    elo_divisor: float,
    transfer_factor: float,
    initial_elo_adjustment_factor: float,
    position_reset_factor: float,
    league_elo_dict: Dict[str, float],
    show_progress: bool = True,
) -> pl.DataFrame:
    """
    Main Elo update procedure:
      - Group matches by (date, gameid)
      - For each group, call `process_game` to update Elo ratings
      - Return df with new columns: elo_before, opp_elo_before, elo_win_likelihood, elo_after
    """
    df = df.clone()
    entity_key = "teamid" if entity.lower() == "team" else "playerid"

    # Initialize Elo ratings
    elo_ratings = defaultdict(lambda: {"elo": initial_elo, "season": df["season"].min(), "league": None})

    # Add necessary columns for outputs
    df = df.with_columns(
        [
            pl.lit(None).alias("elo_before"),
            pl.lit(None).alias("opp_elo_before"),
            pl.lit(None).alias("elo_win_likelihood"),
            pl.lit(None).alias("elo_after"),
        ]
    )

    # Group by date and gameid
    unique_groups = df.select(["date", "gameid"]).unique()

    # Wrap the iterator with tqdm for progress visualization
    group_iterator = (
        tqdm(unique_groups.iter_rows(named=True), desc="Processing games")
        if show_progress
        else unique_groups.iter_rows(named=True)
    )

    # Iterate through unique groups
    for group in group_iterator:
        group_date, group_gameid = group["date"], group["gameid"]

        # Extract the current group
        game_group = df.filter((pl.col("date") == group_date) & (pl.col("gameid") == group_gameid))

        # Process the game group
        process_game(
            df=df,
            game_group=game_group,
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
    league_elo_dict: Optional[Dict[str, float]] = None,
) -> pl.DataFrame:
    """Main entry point for Elo computation"""
    df_pre = preprocess_elo_dataframe(df, entity)

    # If no league Elo dict is given, default to empty
    if league_elo_dict is None:
        league_elo_dict = {}

    # Attempt to load league Elo if not provided
    if not league_elo_dict and LEAGUE_ELO.exists():
        logger.info(f"Loading league Elo ratings from {LEAGUE_ELO}")
        league_elo_df = pl.read_parquet(LEAGUE_ELO)
        league_elo_dict = dict(zip(league_elo_df["league"], league_elo_df["elo"]))

    # Load or compute hyperparameters
    hyperparameters_path = Path(str(ENTITY_ELO_HYPERPARAMETERS).replace("entity", entity))
    best_params = tune_elo_hyperparameters(df_pre, entity, hyperparameters_path, league_elo_dict)

    # Run final Elo computation
    df_final = run_elo_computation(
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
    return df_final
