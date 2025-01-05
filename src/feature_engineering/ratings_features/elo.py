"""
Elo Rating System with Hyperparameter Tuning using Optuna

This module contains functions to calculate Elo ratings for teams or players based on match results,
with FireDucks-based performance optimizations.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Union

import optuna
import pandas as pd
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
# 1. Preprocessing
# ------------------------------------------------------------------------------
def preprocess_elo_dataframe(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Ensures the DataFrame has all required columns, that 'date' is a datetime,
    and that rows are sorted by the appropriate keys.
    """
    entity_key = "teamid" if entity.lower() == "team" else "playerid"

    # Required columns
    required_columns = ["season", "date", "gameid", entity_key, "league", "side", "result"]
    if entity.lower() == "player":
        required_columns.append("position")

    # Check missing columns
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

    # Drop rows missing 'league' or 'result'
    if df["league"].isna().any() or df["result"].isna().any():
        n_missing_leagues = df["league"].isnull().sum()
        n_missing_results = df["result"].isnull().sum()
        logger.warning(f"{n_missing_leagues} 'league' and {n_missing_results} 'result' missing; dropping them.")
        df = df.dropna(subset=["league", "result"]).reset_index(drop=True)

    # Sort by date/gameid/side/position
    df = df.sort_values(by=get_sorting_keys(entity)).reset_index(drop=True)
    return df


# ------------------------------------------------------------------------------
# 2. Elo Functions
# ------------------------------------------------------------------------------
@njit
def expected_outcome(elo_a: float, elo_b: float, elo_divisor: float) -> float:
    """Calculate the expected match outcome between two Elo ratings."""
    exponent = (elo_b - elo_a) / elo_divisor
    return 1 / (1 + 10**exponent)


@njit
def update_elo_rating(old_elo: float, expected: float, actual_result: float, k_factor: float) -> float:
    """Update Elo rating based on match result."""
    adjustment = k_factor * (actual_result - expected)
    return old_elo + adjustment


def aggregate_team_elo(
    rows: pd.DataFrame, elo_ratings: Dict[Union[int, str], Dict[str, Any]], entity_key: str
) -> float:
    """
    Aggregate Elo ratings for a team or group of players in 'rows'.
    Uses a merge-based approach to avoid row-by-row .map calls.
    """
    if rows.empty:
        return 0.0

    rating_data = [(eid, info["elo"]) for eid, info in elo_ratings.items()]
    rating_df = pd.DataFrame(rating_data, columns=[entity_key, "elo"])
    merged = rows[[entity_key]].merge(rating_df, on=entity_key, how="left")
    return merged["elo"].fillna(0).sum()


def linear_decay_reset(
    elo_ratings: Dict[Union[int, str], Dict[str, Any]],
    baseline: float,
    current_season: int,
    decay_factor: float,
) -> None:
    """Apply linear decay reset to Elo ratings at the beginning of a new season."""
    for data in elo_ratings.values():
        if data["season"] < current_season:
            data["elo"] = baseline + (data["elo"] - baseline) * decay_factor
            data["season"] = current_season


def handle_entity_swap(
    entity_id: Union[int, str],
    new_league: str,
    elo_ratings: Dict[Union[int, str], Dict[str, Any]],
    league_elo_dict: Dict[str, float],
    baseline_elo: float,
    transfer_factor: float = 0.5,
) -> None:
    """Adjust an entity's Elo when swapping leagues, if necessary."""
    current_league = elo_ratings[entity_id].get("league")

    if new_league == current_league or new_league in CROSS_LEAGUE_COMPETITIONS:
        return

    if league_elo_dict and current_league and new_league:
        curr_elo = league_elo_dict.get(current_league, baseline_elo)
        new_elo = league_elo_dict.get(new_league, baseline_elo)
        diff = new_elo - curr_elo
        elo_ratings[entity_id]["elo"] += transfer_factor * diff
    else:
        elo_ratings[entity_id]["elo"] = baseline_elo

    elo_ratings[entity_id]["league"] = new_league


# ------------------------------------------------------------------------------
# 3. process_game & Tuning
# ------------------------------------------------------------------------------
def process_game(
    df: pd.DataFrame,
    game_group: pd.DataFrame,
    elo_ratings: Dict[Union[int, str], Dict[str, Any]],
    k_factor: float,
    entity: str,
    entity_key: str,
    baseline_elo: float,
    decay_factor: float,
    elo_divisor: float,
    league_elo_dict: Dict[str, float],
    transfer_factor: float,
    avg_league_elo: float,
    initial_elo_adjustment_factor: float,
) -> None:
    """
    Process a single game group, updating Elo ratings for both sides. Results
    are written to df in bulk (elo_before, elo_after, etc.).
    """
    current_season = game_group.iloc[0]["season"]
    linear_decay_reset(elo_ratings, baseline_elo, current_season, decay_factor)

    # Identify all entities in the group
    entity_ids = game_group[entity_key].unique()

    # Check/initialize each entity’s Elo rating
    for ent_id in entity_ids:
        if ent_id not in elo_ratings:
            # brand-new entity
            new_league = game_group.loc[game_group[entity_key] == ent_id, "league"].iloc[0]
            league_elo = league_elo_dict.get(new_league, baseline_elo)
            init_adjustment = (league_elo - avg_league_elo) * initial_elo_adjustment_factor
            elo_ratings[ent_id] = {
                "elo": baseline_elo + init_adjustment,
                "season": current_season,
                "league": new_league,
            }
        else:
            new_league = game_group.loc[game_group[entity_key] == ent_id, "league"].iloc[0]
            handle_entity_swap(ent_id, new_league, elo_ratings, league_elo_dict, baseline_elo, transfer_factor)

    # Split sides
    blue_rows = game_group[game_group["side"] == "Blue"].copy()
    red_rows = game_group[game_group["side"] == "Red"].copy()

    # If players, sort by position
    if entity.lower() == "player":
        blue_rows.sort_values(by="position", inplace=True)
        red_rows.sort_values(by="position", inplace=True)

    # Compute total Elo for each side
    blue_elo_sum = aggregate_team_elo(blue_rows, elo_ratings, entity_key)
    red_elo_sum = aggregate_team_elo(red_rows, elo_ratings, entity_key)

    # Expected outcome
    blue_expected = expected_outcome(blue_elo_sum, red_elo_sum, elo_divisor)
    blue_result = blue_rows.iloc[0]["result"]
    red_result = 1 - blue_result

    # Prepare arrays for old/new Elo
    blue_ids = blue_rows[entity_key].values
    red_ids = red_rows[entity_key].values

    blue_old_elos = [elo_ratings[bid]["elo"] for bid in blue_ids]
    red_old_elos = [elo_ratings[rid]["elo"] for rid in red_ids]

    # Vectorized updates
    blue_new_elos = [update_elo_rating(old, blue_expected, blue_result, k_factor) for old in blue_old_elos]
    red_new_elos = [update_elo_rating(old, 1 - blue_expected, red_result, k_factor) for old in red_old_elos]

    # Commit updates
    for i, bid in enumerate(blue_ids):
        elo_ratings[bid]["elo"] = blue_new_elos[i]
    for i, rid in enumerate(red_ids):
        elo_ratings[rid]["elo"] = red_new_elos[i]

    # Write results back to df
    df.loc[blue_rows.index, "elo_before"] = blue_old_elos
    df.loc[blue_rows.index, "opp_elo_before"] = red_old_elos
    df.loc[blue_rows.index, "elo_win_likelihood"] = blue_expected
    df.loc[blue_rows.index, "elo_after"] = blue_new_elos

    df.loc[red_rows.index, "elo_before"] = red_old_elos
    df.loc[red_rows.index, "opp_elo_before"] = blue_old_elos
    df.loc[red_rows.index, "elo_win_likelihood"] = 1 - blue_expected
    df.loc[red_rows.index, "elo_after"] = red_new_elos


# ------------------------------------------------------------------------------
# 3. Hyperparameter Tuning
# ------------------------------------------------------------------------------


def tune_elo_hyperparameters(
    df: pd.DataFrame,
    entity: str,
    hyperparameters_path: Path,
    league_elo_dict: Dict[str, float],
) -> Dict[str, float]:
    """
    Always either load hyperparameters if they exist, or compute them with Optuna.
    Returns best_params.
    """
    # If file already exists, load
    if os.path.exists(hyperparameters_path):
        logger.info(f"Loading hyperparameters from {hyperparameters_path}")
        with open(hyperparameters_path) as f:
            best_params = json.load(f)
        logger.info(f"Loaded hyperparameters: {best_params}")
        return best_params

    logger.info(f"No hyperparameters found at {hyperparameters_path}. Starting tuning process...")

    def objective(trial: optuna.trial.Trial) -> float:
        k_factor = trial.suggest_float("k_factor", 16, 96, step=8)
        initial_elo = trial.suggest_float("initial_elo", 1200, 1800, step=100)
        elo_divisor = trial.suggest_float("elo_divisor", 100, 500, step=50)
        decay_factor = trial.suggest_float("decay_factor", 0.5, 1.0, step=0.05)
        transfer_factor = trial.suggest_float("transfer_factor", 0.1, 1.0, step=0.1)
        init_adjust = trial.suggest_float("initial_elo_adjustment_factor", 0.0, 1.0, step=0.1)

        # Sort + minimal checks
        df_sorted = df.sort_values(by=["date", "gameid", "side"]).reset_index(drop=True)
        if df_sorted.empty:
            return float("inf")

        try:
            split_year = df_sorted["date"].dt.year.max()
        except AttributeError as e:
            logger.error(f"Error accessing 'date' column with .dt accessor: {e}")
            return float("inf")
        split_date = pd.to_datetime(f"{split_year}-01-01")

        df_train = df_sorted[df_sorted["date"] < split_date].reset_index(drop=True)
        df_valid = df_sorted[df_sorted["date"] >= split_date].reset_index(drop=True)

        # Quick group-size check
        expected_count = 10 if entity.lower() == "player" else 2
        if len(df_train) > 0:
            if not (df_train.groupby("gameid").size() == expected_count).all():
                logger.warning("Training data has gameids with incorrect number of entities.")
                return float("inf")
        if len(df_valid) > 0:
            if not (df_valid.groupby("gameid").size() == expected_count).all():
                logger.warning("Validation data has gameids with incorrect number of entities.")
                return float("inf")

        # 1) Train Elo
        try:
            df_train_res = run_elo_computation(
                df_train.copy(),
                entity=entity,
                initial_elo=initial_elo,
                k_factor=k_factor,
                decay_factor=decay_factor,
                elo_divisor=elo_divisor,
                transfer_factor=transfer_factor,
                initial_elo_adjustment_factor=init_adjust,
                league_elo_dict=league_elo_dict,
                show_progress=True,
            )
        except Exception as e:
            logger.error(f"Error during Elo calculation in trial: {e}")
            return float("inf")

        # 2) Initialize validation Elo from last training Elo
        entity_key = "teamid" if entity.lower() == "team" else "playerid"
        last_elo = df_train_res.groupby(entity_key)["elo_after"].last().to_dict()

        val_elo = defaultdict(lambda: {"elo": initial_elo})
        for e_id, rating in last_elo.items():
            val_elo[e_id]["elo"] = rating

        df_valid = df_valid.sort_values(by=["date", "gameid"]).reset_index(drop=True)
        expected_probs = []

        # 3) Iteratively update Elo in validation
        for _, grp in df_valid.groupby(["date", "gameid"]):
            blue_side = grp[grp["side"] == "Blue"]
            red_side = grp[grp["side"] == "Red"]
            blue_elo_sum = sum(val_elo[eid]["elo"] for eid in blue_side[entity_key])
            red_elo_sum = sum(val_elo[eid]["elo"] for eid in red_side[entity_key])

            blue_expected = expected_outcome(blue_elo_sum, red_elo_sum, elo_divisor)
            blue_result = blue_side.iloc[0]["result"]

            for eid in blue_side[entity_key]:
                val_elo[eid]["elo"] = update_elo_rating(val_elo[eid]["elo"], blue_expected, blue_result, k_factor)
            for eid in red_side[entity_key]:
                val_elo[eid]["elo"] = update_elo_rating(
                    val_elo[eid]["elo"], 1 - blue_expected, 1 - blue_result, k_factor
                )

            expected_probs.append(blue_expected)

        # 4) log_loss
        y_true = df_valid.loc[df_valid["side"] == "Blue", "result"]
        y_pred = pd.Series(expected_probs).clip(0.0001, 0.9999)
        return log_loss(y_true, y_pred)

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(objective, n_trials=TRIALS_NUM, show_progress_bar=True)
    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")

    # Save
    logger.info(f"Storing hyperparameters to {hyperparameters_path}")
    with open(hyperparameters_path, "w") as f:
        json.dump(best_params, f)

    return best_params


# ------------------------------------------------------------------------------
# 4. Main Elo Computation
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
    league_elo_dict: Dict[str, float],
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Main Elo update procedure:
      - Group matches by (date, gameid)
      - For each group, use process_game to update Elo ratings
      - Return df with columns: 'elo_before', 'opp_elo_before', 'elo_win_likelihood', 'elo_after'
    """
    entity_key = "teamid" if entity.lower() == "team" else "playerid"

    # 1) Elo store
    elo_ratings = defaultdict(
        lambda: {
            "elo": initial_elo,
            "season": df["season"].min(),
            "league": None,
        }
    )

    # 2) Pre-allocate columns
    for col in ["elo_before", "opp_elo_before", "elo_win_likelihood", "elo_after"]:
        df[col] = None

    # 3) Group by game
    group_obj = df.groupby(["date", "gameid"], sort=False)
    if show_progress:
        group_obj = tqdm(group_obj, desc="Processing games", total=group_obj.ngroups)

    # 4) Process each game
    for _, game_grp in group_obj:
        process_game(
            df,
            game_grp,
            elo_ratings=elo_ratings,
            k_factor=k_factor,
            entity=entity,
            entity_key=entity_key,
            baseline_elo=initial_elo,
            decay_factor=decay_factor,
            elo_divisor=elo_divisor,
            league_elo_dict=league_elo_dict,
            transfer_factor=transfer_factor,
            avg_league_elo=sum(league_elo_dict.values()) / len(league_elo_dict) if league_elo_dict else initial_elo,
            initial_elo_adjustment_factor=initial_elo_adjustment_factor,
        )

    return df


# ------------------------------------------------------------------------------
# 5. Orchestration
# ------------------------------------------------------------------------------
def calculate_elo(
    df: pd.DataFrame,
    entity: str,
    league_elo_dict: Dict[str, float] = None,
) -> pd.DataFrame:
    """Main entry point for Elo computation"""
    df_pre = preprocess_elo_dataframe(df, entity)

    # Attempt to load league Elo if not provided
    if league_elo_dict is None:
        league_elo_dict = {}
        if LEAGUE_ELO.exists():
            logger.info(f"Loading league Elo ratings from {LEAGUE_ELO}")
            league_elo_df = pd.read_parquet(LEAGUE_ELO)
            league_elo_dict = league_elo_df.set_index("league")["elo"].to_dict()

    # Load or compute hyperparams
    hyperparams_path = Path(str(ENTITY_ELO_HYPERPARAMETERS).replace("entity", entity))
    best_params = tune_elo_hyperparameters(df_pre, entity, hyperparams_path, league_elo_dict)

    # Compute Elo using those params
    df_final = run_elo_computation(
        df_pre.copy(),
        entity=entity,
        initial_elo=best_params["initial_elo"],
        k_factor=best_params["k_factor"],
        decay_factor=best_params["decay_factor"],
        elo_divisor=best_params["elo_divisor"],
        transfer_factor=best_params["transfer_factor"],
        initial_elo_adjustment_factor=best_params["initial_elo_adjustment_factor"],
        league_elo_dict=league_elo_dict,
        show_progress=True,
    )
    return df_final
