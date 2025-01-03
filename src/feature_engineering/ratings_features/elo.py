"""
Elo Rating System with Hyperparameter Tuning using Optuna

This module contains functions to calculate Elo ratings for teams or players based on match results.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Union

import optuna
import pandas as pd
from sklearn.metrics import log_loss
from tqdm import tqdm

from src.utils.logger import logger
from src.utils.paths import CONSIDERED_LEAGUES, DEFAULT_MODELS_PARAMETERS, ENTITY_ELO_HYPERPARAMETERS, LEAGUE_ELO
from src.utils.utils import get_sorting_keys, json_loader

# Load configuration parametersxq
config = json_loader(DEFAULT_MODELS_PARAMETERS)

# Load considered leagues configuration
considered_leagues_config = json_loader(CONSIDERED_LEAGUES)
MAJOR_LEAGUES = considered_leagues_config["major_leagues"]
CROSS_LEAGUE_COMPETITIONS = considered_leagues_config["cross_league_competitions"]


def expected_outcome(elo_a: float, elo_b: float, elo_divisor: float) -> float:
    """Calculate the expected match outcome between two Elo ratings."""
    exponent = (elo_b - elo_a) / elo_divisor
    return 1 / (1 + 10**exponent)


def update_elo_rating(old_elo: float, expected: float, actual_result: float, k_factor: float) -> float:
    """Update Elo rating based on match result."""
    adjustment = k_factor * (actual_result - expected)
    return old_elo + adjustment


def aggregate_team_elo(
    rows: pd.DataFrame, elo_ratings: Dict[Union[int, str], Dict[str, Any]], entity_key: str
) -> float:
    """Aggregate Elo ratings for a team or group of players."""
    return rows[entity_key].map(lambda x: elo_ratings[x]["elo"]).sum()


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
    """Handle entity swap between leagues and adjust Elo rating if necessary."""
    current_league = elo_ratings[entity_id].get("league")

    if new_league == current_league or new_league in CROSS_LEAGUE_COMPETITIONS:
        # No action needed if the entity remains in the same league or moves to a cross-competition league
        return

    # Adjust Elo rating based on league change
    if league_elo_dict and current_league and new_league:
        current_league_elo = league_elo_dict.get(current_league, baseline_elo)
        new_league_elo = league_elo_dict.get(new_league, baseline_elo)
        # Adjust the entity's Elo based on a proportion of the difference between the league Elos
        elo_difference = new_league_elo - current_league_elo
        adjusted_difference = transfer_factor * elo_difference
        elo_ratings[entity_id]["elo"] += adjusted_difference
    else:
        # If league Elo ratings are not available, reset to baseline
        elo_ratings[entity_id]["elo"] = baseline_elo

    # Update the league
    elo_ratings[entity_id]["league"] = new_league


def process_game(
    df_sorted: pd.DataFrame,
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
    """Process each game and update Elo ratings for both sides."""
    current_season = game_group.iloc[0]["season"]
    linear_decay_reset(elo_ratings, baseline_elo, current_season, decay_factor)

    # Get the entity IDs involved in the current game group
    entity_ids = game_group[entity_key].unique()

    # Handle entity swaps before processing game ratings
    for entity_id in entity_ids:
        if entity_id not in elo_ratings:
            # Initialize Elo rating for new entities
            new_league = game_group[game_group[entity_key] == entity_id]["league"].iloc[0]
            league_elo = league_elo_dict.get(new_league, baseline_elo)
            # Adjust initial Elo based on the difference between league Elo and average league Elo
            initial_elo_adjustment = (league_elo - avg_league_elo) * initial_elo_adjustment_factor
            initial_elo_entity = baseline_elo + initial_elo_adjustment
            elo_ratings[entity_id] = {
                "elo": initial_elo_entity,
                "season": current_season,
                "league": new_league,
            }
        else:
            new_league = game_group[game_group[entity_key] == entity_id]["league"].iloc[0]
            handle_entity_swap(entity_id, new_league, elo_ratings, league_elo_dict, baseline_elo, transfer_factor)

    blue_rows = game_group[game_group["side"] == "Blue"].copy()
    red_rows = game_group[game_group["side"] == "Red"].copy()

    if entity == "player":
        blue_rows.sort_values(by="position", inplace=True)
        red_rows.sort_values(by="position", inplace=True)

    blue_elo_sum = aggregate_team_elo(blue_rows, elo_ratings, entity_key)
    red_elo_sum = aggregate_team_elo(red_rows, elo_ratings, entity_key)

    blue_expected = expected_outcome(blue_elo_sum, red_elo_sum, elo_divisor)
    blue_result = blue_rows.iloc[0]["result"]
    red_result = 1 - blue_result

    for blue_row, red_row in zip(blue_rows.itertuples(), red_rows.itertuples()):
        blue_id = getattr(blue_row, entity_key)
        red_id = getattr(red_row, entity_key)
        blue_elo = elo_ratings[blue_id]["elo"]
        red_elo = elo_ratings[red_id]["elo"]

        blue_new_elo = update_elo_rating(blue_elo, blue_expected, blue_result, k_factor)
        red_new_elo = update_elo_rating(red_elo, 1 - blue_expected, red_result, k_factor)

        elo_ratings[blue_id]["elo"] = blue_new_elo
        elo_ratings[red_id]["elo"] = red_new_elo

        df_sorted.at[blue_row.Index, "elo_before"] = blue_elo
        df_sorted.at[blue_row.Index, "opp_elo_before"] = red_elo
        df_sorted.at[blue_row.Index, "elo_win_likelihood"] = blue_expected
        df_sorted.at[blue_row.Index, "elo_after"] = blue_new_elo

        df_sorted.at[red_row.Index, "elo_before"] = red_elo
        df_sorted.at[red_row.Index, "opp_elo_before"] = blue_elo
        df_sorted.at[red_row.Index, "elo_win_likelihood"] = 1 - blue_expected
        df_sorted.at[red_row.Index, "elo_after"] = red_new_elo


def tune_hyperparameters(
    df: pd.DataFrame,
    entity: str,
    hyperparameters_path: Path,
    league_elo_dict: Dict[str, float],
) -> Dict[str, float]:
    """
    Perform hyperparameter tuning using Optuna and return the best parameters.

    Check if hyperparameters exist at the specified path; if so, load them, otherwise compute and store.
    """
    # Check if the hyperparameters file exists
    if os.path.exists(hyperparameters_path):
        logger.info(f"Loading hyperparameters from {hyperparameters_path}")
        with open(hyperparameters_path) as file:
            best_params = json.load(file)
        logger.info(f"Loaded hyperparameters: {best_params}")
        return best_params

    logger.info(f"No hyperparameters found at {hyperparameters_path}. Starting tuning process...")

    def objective(trial: optuna.trial.Trial) -> float:
        # Suggest hyperparameters
        k_factor = trial.suggest_float("k_factor", 16, 96, step=8)
        initial_elo = trial.suggest_float("initial_elo", 1200, 1800, step=100)
        elo_divisor = trial.suggest_float("elo_divisor", 100, 500, step=50)
        decay_factor = trial.suggest_float("decay_factor", 0.5, 1.0, step=0.05)
        transfer_factor = trial.suggest_float("transfer_factor", 0.1, 1.0, step=0.1)
        initial_elo_adjustment_factor = trial.suggest_float("initial_elo_adjustment_factor", 0.0, 1.0, step=0.1)

        # Sort the DataFrame by date and gameid
        df_sorted = df.sort_values(by=["date", "gameid", "side"]).reset_index(drop=True)

        # Ensure 'date' column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_sorted["date"]):
            try:
                df_sorted["date"] = pd.to_datetime(df_sorted["date"], errors="coerce")
                if df_sorted["date"].isnull().any():
                    # Count and drop NaT entries
                    empty_dates = df_sorted["date"].isnull().sum()
                    logger.warning(
                        f"{empty_dates} 'dates' entries could not be converted and are NaT. Dropping these entries."
                    )
                    df_sorted = df_sorted.dropna(subset=["date"])
            except Exception as e:
                logger.error(f"Error converting 'date' to datetime: {e}")
                return float("inf")

        # Evaluate ratings on the last year of data
        try:
            split_year = df_sorted["date"].dt.year.max()
        except AttributeError as e:
            logger.error(f"Error accessing 'date' column with .dt accessor: {e}")
            return float("inf")
        split_date = pd.to_datetime(f"{split_year}-01-01")

        # Split data into training and validation sets based on the split date
        df_train = df_sorted[df_sorted["date"] < split_date].reset_index(drop=True)
        df_valid = df_sorted[df_sorted["date"] >= split_date].reset_index(drop=True)

        # Validate that each gameid has the correct number of entities
        expected_count = 10 if entity == "player" else 2
        if len(df_train) > 0:
            if not (df_train.groupby("gameid").size() == expected_count).all():
                logger.warning("Training data has gameids with incorrect number of entities.")
                return float("inf")
        if len(df_valid) > 0:
            if not (df_valid.groupby("gameid").size() == expected_count).all():
                logger.warning("Validation data has gameids with incorrect number of entities.")
                return float("inf")

        # Calculate Elo ratings on the training data
        try:
            df_with_elo_train = calculate_elo(
                df_train.copy(),
                entity=entity,
                initial_elo=initial_elo,
                k_factor=k_factor,
                decay_factor=decay_factor,
                elo_divisor=elo_divisor,
                perform_tuning=False,
                league_elo_dict=league_elo_dict,
                transfer_factor=transfer_factor,
                initial_elo_adjustment_factor=initial_elo_adjustment_factor,
            )
        except Exception as e:
            logger.error(f"Error during Elo calculation in trial: {e}")
            return float("inf")

        # Get the last Elo ratings from the training data
        entity_key = "teamid" if entity.lower() == "team" else "playerid"
        last_elo_ratings = df_with_elo_train.groupby(entity_key)["elo_after"].last().to_dict()

        # Prepare validation data
        df_valid = df_valid.copy()
        df_valid["elo_before"] = df_valid[entity_key].map(last_elo_ratings)
        df_valid["elo_before"].fillna(initial_elo, inplace=True)

        # Map team Elo ratings
        if entity == "player":
            # Sum up Elo ratings for each team
            team_elos = df_valid.groupby(["gameid", "side"])["elo_before"].sum().reset_index()
            df_valid = df_valid.merge(team_elos, on=["gameid", "side"], how="left", suffixes=("", "_team"))
            df_valid.rename(columns={"elo_before_team": "team_elo"}, inplace=True)
        else:
            # For teams, team_elo is just elo_before
            df_valid["team_elo"] = df_valid["elo_before"]

        # Map opponent team Elo
        opp_team_elos = df_valid[["gameid", "side", "team_elo"]].copy()
        opp_team_elos["side"] = opp_team_elos["side"].map({"Blue": "Red", "Red": "Blue"})
        opp_team_elos.rename(columns={"team_elo": "opp_team_elo"}, inplace=True)
        df_valid = df_valid.merge(opp_team_elos, on=["gameid", "side"], how="left")
        df_valid["opp_team_elo"].fillna(initial_elo, inplace=True)

        # Calculate expected outcomes for validation data
        df_valid["expected"] = expected_outcome(df_valid["team_elo"], df_valid["opp_team_elo"], elo_divisor)

        # Compute log loss on the validation set
        y_true = df_valid[df_valid["side"] == "Blue"]["result"]
        y_pred = df_valid[df_valid["side"] == "Blue"]["expected"].clip(0.0001, 0.9999)

        loss = log_loss(y_true, y_pred)

        return loss

    # Create and optimize the study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2, timeout=None, show_progress_bar=True)  # TODO: Change n_trials to 100

    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")

    # Store the best hyperparameters
    logger.info(f"Storing hyperparameters to {hyperparameters_path}")
    with open(hyperparameters_path, "w") as file:
        json.dump(best_params, file)

    return best_params


def calculate_elo(
    df: pd.DataFrame,
    entity: str,
    initial_elo: float = None,
    k_factor: float = None,
    decay_factor: float = None,
    elo_divisor: float = None,
    perform_tuning: bool = True,
    league_elo_dict: Dict[str, float] = None,
    transfer_factor: float = 0.5,
    initial_elo_adjustment_factor: float = 0.0,
) -> pd.DataFrame:
    """Calculate and update Elo ratings for entities within each team."""
    if entity.lower() not in ["team", "player"]:
        raise ValueError("Entity must be 'team' or 'player'")

    entity_key = "teamid" if entity.lower() == "team" else "playerid"

    required_columns = ["season", "date", "gameid", entity_key, "league", "side", "result"]
    if entity == "player":
        required_columns.append("position")

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_columns}")

    # Load league Elo ratings if not provided
    if league_elo_dict is None:
        league_elo_dict = {}
        if LEAGUE_ELO.exists():
            league_elo_df = pd.read_parquet(LEAGUE_ELO)
            league_elo_dict = league_elo_df.set_index("league")["elo"].to_dict()
            avg_league_elo = league_elo_df["elo"].mean()
        else:
            avg_league_elo = initial_elo  # If no league Elo ratings are available
    else:
        # Compute avg_league_elo from league_elo_dict
        avg_league_elo = sum(league_elo_dict.values()) / len(league_elo_dict) if league_elo_dict else initial_elo

    # Perform hyperparameter tuning if required
    if perform_tuning:
        hyperparameters_path = Path(str(ENTITY_ELO_HYPERPARAMETERS).replace("entity", entity))
        best_params = tune_hyperparameters(df, entity, hyperparameters_path, league_elo_dict)
        initial_elo = best_params["initial_elo"]
        k_factor = best_params["k_factor"]
        decay_factor = best_params["decay_factor"]
        elo_divisor = best_params["elo_divisor"]
        transfer_factor = best_params.get("transfer_factor", 0.5)
        initial_elo_adjustment_factor = best_params.get("initial_elo_adjustment_factor", 0.0)
    else:
        # Ensure all parameters are provided
        if initial_elo is None or k_factor is None or decay_factor is None or elo_divisor is None:
            raise ValueError("All hyperparameters must be provided if perform_tuning is False")

    # Sort the DataFrame by date and gameid
    df_sorted = df.sort_values(by=get_sorting_keys(entity)).reset_index(drop=True)

    # Ensure 'date' column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_sorted["date"]):
        try:
            df_sorted["date"] = pd.to_datetime(df_sorted["date"], errors="coerce")
            if df_sorted["date"].isnull().any():
                # Count and drop NaT entries
                empty_dates = df_sorted["date"].isnull().sum()
                logger.warning(
                    f"{empty_dates} 'date' entries could not be converted and are NaT. Dropping these entries."
                )
                df_sorted = df_sorted.dropna(subset=["date"])
        except Exception as e:
            logger.error(f"Error converting 'date' to datetime: {e}")
            raise

    # Initialize Elo ratings
    elo_ratings = defaultdict(lambda: {"elo": initial_elo, "season": df_sorted["season"].min(), "league": None})

    # Initialize columns for Elo ratings
    for col in ["elo_before", "opp_elo_before", "elo_win_likelihood", "elo_after"]:
        df_sorted[col] = None

    logger.info("Processing games to calculate Elo ratings...")
    games_iterator = df_sorted.groupby(["date", "gameid"])
    if perform_tuning:
        games_iterator = tqdm(games_iterator, desc="Processing games")

    for _, game_group in games_iterator:
        process_game(
            df_sorted,
            game_group,
            elo_ratings=elo_ratings,
            k_factor=k_factor,
            entity=entity,
            entity_key=entity_key,
            baseline_elo=initial_elo,
            decay_factor=decay_factor,
            elo_divisor=elo_divisor,
            league_elo_dict=league_elo_dict,
            transfer_factor=transfer_factor,
            avg_league_elo=avg_league_elo,
            initial_elo_adjustment_factor=initial_elo_adjustment_factor,
        )

    return df_sorted
