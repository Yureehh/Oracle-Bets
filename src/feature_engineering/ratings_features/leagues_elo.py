"""
League Elo Rating System with Hyperparameter Tuning using Optuna

This script calculates league Elo ratings and uses Optuna to optimize hyperparameters.
"""

import json
import os
from collections import defaultdict
from typing import Dict, Tuple, Union

import fireducks.pandas as pd
import optuna
from sklearn.metrics import log_loss
from tqdm import tqdm

from src.utils.logger import logger
from src.utils.paths import CONSIDERED_LEAGUES, LEAGUE_ELO, LEAGUES_ELO_HYPERPARAMETERS, TEAM_LEAGUES_MAPPING
from src.utils.utils import get_sorting_keys, json_loader, safe_store_df_as_parquet

# Load considered leagues configuration
considered_leagues_config = json_loader(CONSIDERED_LEAGUES)
CROSS_LEAGUE_COMPETITIONS = set(considered_leagues_config["cross_league_competitions"])


def map_team_to_league(df: pd.DataFrame, team_column: str) -> Dict[str, str]:
    """
    Map each team to the last league it played in, excluding cross-league competitions.

    Args:
        df (pd.DataFrame): DataFrame containing team and league information.
        team_column (str): Name of the column representing teams.

    Returns:
        Dict[str, str]: Dictionary mapping team IDs to league names.
    """
    # Filter out cross-league competitions
    df_filtered = df[~df["league"].isin(CROSS_LEAGUE_COMPETITIONS)].copy()

    # Ensure the DataFrame is sorted by date to get the last league played
    df_filtered.sort_values(by=["date"], inplace=True)

    # Group by team and get the last league they played in
    last_league = df_filtered.groupby(team_column)["league"].last().to_dict()

    return last_league


def expected_outcome(elo_a: float, elo_b: float, elo_divisor: float) -> float:
    """
    Calculate the expected match outcome between two aggregated Elo ratings.

    Args:
        elo_a (float): Elo rating of the first league.
        elo_b (float): Elo rating of the second league.
        elo_divisor (float): Divisor used in the Elo expected outcome calculation.

    Returns:
        float: Expected probability of the first league winning.
    """
    exponent = (elo_b - elo_a) / elo_divisor
    return 1 / (1 + 10**exponent)


def update_elo_rating(old_elo: float, expected: float, actual_result: float, k_factor: float) -> float:
    """
    Update Elo rating based on match result.

    Args:
        old_elo (float): Previous Elo rating.
        expected (float): Expected match outcome.
        actual_result (float): Actual match outcome (1 for win, 0 for loss).
        k_factor (float): K-factor for Elo rating adjustment.

    Returns:
        float: Updated Elo rating.
    """
    adjustment = k_factor * (actual_result - expected)
    return old_elo + adjustment


def linear_decay_reset_league_elo(
    elo_ratings: Dict[str, Dict[str, Union[float, int]]],
    baseline: float,
    current_season: int,
    decay_factor: float,
) -> None:
    """
    Apply linear decay reset to league Elo ratings at the beginning of a new season.

    Args:
        elo_ratings (Dict[str, Dict[str, Union[float, int]]]): Dictionary of league Elo ratings.
        baseline (float): Baseline Elo rating.
        current_season (int): Current season number.
        decay_factor (float): Decay factor for Elo rating adjustment.
    """
    for data in elo_ratings.values():
        if data["season"] < current_season:
            data["elo"] = baseline + (data["elo"] - baseline) * decay_factor
            data["season"] = current_season


def process_game(
    df_sorted: pd.DataFrame,
    game_group: pd.DataFrame,
    elo_ratings: Dict[str, Dict[str, Union[float, int]]],
    k_factor: float,
    entity_key: str,
    belonging_league: Dict[str, str],
    baseline_elo: float,
    decay_factor: float,
    elo_divisor: float,
) -> None:
    """
    Process each game and update Elo ratings for both sides.

    Args:
        df_sorted (pd.DataFrame): The main DataFrame being processed.
        game_group (pd.DataFrame): Grouped DataFrame for a single game.
        elo_ratings (Dict[str, Dict[str, Union[float, int]]]): Dictionary of league Elo ratings.
        k_factor (float): K-factor for Elo rating adjustment.
        entity_key (str): Column name representing the team ID.
        belonging_league (Dict[str, str]): Dictionary mapping team IDs to leagues.
        baseline_elo (float): Baseline Elo rating.
        decay_factor (float): Decay factor for Elo rating adjustment.
        elo_divisor (float): Divisor used in the Elo expected outcome calculation.
    """
    current_season = game_group.iloc[0]["season"]
    linear_decay_reset_league_elo(elo_ratings, baseline_elo, current_season, decay_factor)

    blue_rows = game_group[game_group["side"] == "Blue"]
    red_rows = game_group[game_group["side"] == "Red"]

    if blue_rows.empty or red_rows.empty:
        logger.warning(f"Skipping game {game_group.iloc[0]['gameid']} due to missing side information.")
        return

    blue_row = blue_rows.iloc[0]
    red_row = red_rows.iloc[0]

    blue_entity_id = blue_row[entity_key]
    red_entity_id = red_row[entity_key]

    blue_league = belonging_league.get(blue_entity_id)
    red_league = belonging_league.get(red_entity_id)

    if blue_league is None or red_league is None:
        return

    # Initialize leagues in elo_ratings if not already present
    if blue_league not in elo_ratings:
        elo_ratings[blue_league] = {"elo": baseline_elo, "season": current_season}
    if red_league not in elo_ratings:
        elo_ratings[red_league] = {"elo": baseline_elo, "season": current_season}

    blue_league_elo = elo_ratings[blue_league]["elo"]
    red_league_elo = elo_ratings[red_league]["elo"]

    if blue_league != red_league:
        blue_expected = expected_outcome(blue_league_elo, red_league_elo, elo_divisor)
        blue_result = blue_row["result"]
        red_result = 1 - blue_result

        blue_new_league_elo = update_elo_rating(blue_league_elo, blue_expected, blue_result, k_factor)
        red_new_league_elo = update_elo_rating(red_league_elo, 1 - blue_expected, red_result, k_factor)

        elo_ratings[blue_league]["elo"] = blue_new_league_elo
        elo_ratings[red_league]["elo"] = red_new_league_elo
    else:
        # No update to Elo ratings since both teams are from the same league
        blue_expected = 0.5  # Expected outcome is neutral

    # Update DataFrame with league Elo information
    df_sorted.at[blue_row.name, "league_elo_before"] = blue_league_elo
    df_sorted.at[blue_row.name, "opp_league_elo_before"] = red_league_elo
    df_sorted.at[blue_row.name, "league_elo_win_likelihood"] = blue_expected
    df_sorted.at[blue_row.name, "league_elo_after"] = elo_ratings[blue_league]["elo"]

    df_sorted.at[red_row.name, "league_elo_before"] = red_league_elo
    df_sorted.at[red_row.name, "opp_league_elo_before"] = blue_league_elo
    df_sorted.at[red_row.name, "league_elo_win_likelihood"] = 1 - blue_expected
    df_sorted.at[red_row.name, "league_elo_after"] = elo_ratings[red_league]["elo"]


def tune_hyperparameters(df: pd.DataFrame, entity: str, hyperparameters_path: str) -> Dict[str, float]:
    """
    Perform hyperparameter tuning using Optuna and return the best parameters.
    Check if hyperparameters exist at the specified path; if so, load them, otherwise compute and store.

    Args:
        df (pd.DataFrame): DataFrame containing match data.
        entity (str): The type of entity, e.g., 'player' or 'team'.
        hyperparameters_path (str): The path where best hyperparameters are stored.

    Returns:
        Dict[str, float]: Dictionary of best hyperparameters.
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

        # Sort the DataFrame by date, gameid, and side
        df_sorted = df.sort_values(by=["date", "gameid", "side"]).reset_index(drop=True)

        # Ensure 'date' column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_sorted["date"]):
            try:
                df_sorted["date"] = pd.to_datetime(df_sorted["date"], errors="coerce")
                if df_sorted["date"].isnull().any():
                    # Count and drop NaT entries
                    n_missing_dates = df_sorted["date"].isnull().sum()
                    logger.warning(
                        f"{n_missing_dates} 'date' entries could not be converted and are NaT. Dropping these entries."
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

        # Validate that each gameid has exactly two teams
        if len(df_train) > 0:
            if not (df_train.groupby("gameid").size() == 2).all():
                logger.warning("Training data has gameids with != 2 teams.")
                return float("inf")
        if len(df_valid) > 0:
            if not (df_valid.groupby("gameid").size() == 2).all():
                logger.warning("Validation data has gameids with != 2 teams.")
                return float("inf")

        # Calculate Elo ratings on the training data
        try:
            df_with_elo_train = calculate_leagues_elo(
                df_train.copy(),
                entity=entity,
                initial_elo=initial_elo,
                k_factor=k_factor,
                elo_divisor=elo_divisor,
                decay_factor=decay_factor,
                perform_tuning=False,  # Avoid recursive tuning
            )
        except Exception as e:
            logger.error(f"Error during Elo calculation in trial: {e}")
            return float("inf")

        # Use the last Elo ratings from the training data
        last_elo_ratings = df_with_elo_train.groupby("league")["league_elo_after"].last().to_dict()

        # Prepare validation data
        df_valid = df_valid.copy()
        df_valid["league_elo_before"] = df_valid["league"].map(last_elo_ratings)
        df_valid["league_elo_before"].fillna(initial_elo, inplace=True)

        # Map opponent league Elo ratings using shift
        opponent_league = df_valid.groupby("gameid")["league"].shift(-1)
        df_valid["opp_league"] = opponent_league
        df_valid["opp_league_elo_before"] = df_valid["opp_league"].map(last_elo_ratings)
        df_valid["opp_league_elo_before"].fillna(initial_elo, inplace=True)

        # Calculate expected outcomes for validation data
        df_valid["expected"] = expected_outcome(
            df_valid["league_elo_before"], df_valid["opp_league_elo_before"], elo_divisor
        )

        # Compute log loss on the validation set
        y_true = df_valid["result"]
        y_pred = df_valid["expected"].clip(0.0001, 0.9999)  # Avoid log loss errors

        loss = log_loss(y_true, y_pred)

        return loss

    # Create and optimize the study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=None, show_progress_bar=True)

    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")

    # Store the best hyperparameters
    logger.info(f"Storing hyperparameters to {hyperparameters_path}")
    with open(hyperparameters_path, "w") as file:
        json.dump(best_params, file)

    return best_params


def calculate_leagues_elo(
    df: pd.DataFrame,
    entity: str,
    initial_elo: float = None,
    k_factor: float = None,
    elo_divisor: float = None,
    decay_factor: float = None,
    perform_tuning: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate and update Elo ratings for leagues based on match results.
    Performs hyperparameter tuning if parameters are not provided.

    Args:
        df (pd.DataFrame): DataFrame containing match data.
        entity (str): The type of entity, e.g., 'player' or 'team'.
        initial_elo (float): Initial Elo rating for leagues.
        k_factor (float): K-factor for Elo rating adjustment.
        elo_divisor (float): Divisor used in the Elo expected outcome calculation.
        decay_factor (float): Decay factor for Elo rating adjustment.
        perform_tuning (bool): Whether to perform hyperparameter tuning.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Updated DataFrame with league Elo information,
        DataFrame mapping teams to leagues, and DataFrame of league Elo ratings.
    """
    if perform_tuning:
        # Perform hyperparameter tuning
        best_params = tune_hyperparameters(df, entity, LEAGUES_ELO_HYPERPARAMETERS)
        initial_elo = best_params["initial_elo"]
        k_factor = best_params["k_factor"]
        elo_divisor = best_params["elo_divisor"]
        decay_factor = best_params["decay_factor"]
    else:
        # Ensure all parameters are provided
        if initial_elo is None or k_factor is None or elo_divisor is None or decay_factor is None:
            raise ValueError("All hyperparameters must be provided if perform_tuning is False")

    required_columns = ["season", "date", "gameid", "teamid", "league", "side", "result"]

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_columns}")

    entity_key = "teamid"

    # Sort the DataFrame by sorting keys
    df_sorted = df.sort_values(by=get_sorting_keys(entity)).reset_index(drop=True)

    # Ensure 'date' column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_sorted["date"]):
        try:
            df_sorted["date"] = pd.to_datetime(df_sorted["date"], errors="coerce")
            if df_sorted["date"].isnull().any():
                # Count and drop NaT entries
                n_missing_dates = df_sorted["date"].isnull().sum()
                logger.warning(
                    f"{n_missing_dates} 'date' entries could not be converted and are NaT. Dropping these entries."
                )
                df_sorted = df_sorted.dropna(subset=["date"])
        except Exception as e:
            logger.error(f"Error converting 'date' to datetime: {e}")
            raise

    league_elo_ratings = defaultdict(lambda: {"elo": initial_elo, "season": df_sorted["season"].min()})
    belonging_league = map_team_to_league(df_sorted, entity_key)

    # Initialize columns for Elo ratings
    for col in ["league_elo_before", "opp_league_elo_before", "league_elo_win_likelihood", "league_elo_after"]:
        df_sorted[col] = None

    # Use tqdm only when perform_tuning is False
    games_iterator = df_sorted.groupby(["date", "gameid"])
    if perform_tuning:
        games_iterator = tqdm(games_iterator, desc="Processing games")

    for _, game_group in games_iterator:
        process_game(
            df_sorted,
            game_group,
            elo_ratings=league_elo_ratings,
            k_factor=k_factor,
            entity_key=entity_key,
            belonging_league=belonging_league,
            baseline_elo=initial_elo,
            decay_factor=decay_factor,
            elo_divisor=elo_divisor,
        )

    belonging_league_df = pd.DataFrame(belonging_league.items(), columns=[entity_key, "league"])

    # Convert league Elo ratings to DataFrame
    league_elo_df = pd.DataFrame(
        [(league, data["elo"]) for league, data in league_elo_ratings.items()],
        columns=["league", "elo"],
    )
    league_elo_df = league_elo_df.sort_values(by="elo", ascending=False).reset_index(drop=True)

    if perform_tuning:
        safe_store_df_as_parquet(belonging_league_df, TEAM_LEAGUES_MAPPING, logger)
        safe_store_df_as_parquet(league_elo_df, LEAGUE_ELO, logger)

    return df_sorted
