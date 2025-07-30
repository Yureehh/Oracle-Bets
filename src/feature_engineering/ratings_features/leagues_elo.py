"""
League Elo Rating System with Hyperparameter Tuning using Optuna

This script calculates league Elo ratings and uses Optuna to optimize hyperparameters.
"""

import json
import os
from collections import defaultdict

import optuna
import pandas as pd
from sklearn.metrics import log_loss
from tqdm import tqdm

from utils.logger import instantiate_conf_logger, logger
from utils.paths import (
    CONSIDERED_LEAGUES,
    LEAGUE_ELO,
    LEAGUES_ELO_HYPERPARAMETERS,
    TEAM_LEAGUES_MAPPING,
)
from utils.utils import get_sorting_keys, json_loader, safe_store_df_as_parquet

# Load considered leagues configuration
considered_leagues_config = json_loader(CONSIDERED_LEAGUES)
CROSS_LEAGUE_COMPETITIONS = set(considered_leagues_config["cross_league_competitions"])
data_pipeline_logger = instantiate_conf_logger("data_pipeline")


def preprocess_dataframe(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Perform all necessary preprocessing on the input DataFrame, ensuring it's clean and ready for tuning or computation.

    Args:
        df (pd.DataFrame): Input DataFrame with match data.
        entity (str): The type of entity, e.g., 'player' or 'team'.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.

    """
    required_columns = [
        "season",
        "date",
        "gameid",
        "teamid",
        "league",
        "side",
        "result",
    ]
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        msg = f"Input DataFrame is missing required columns: {missing_columns}"
        raise ValueError(msg)

    # Ensure 'date' is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if df["date"].isnull().any():
                n_missing_dates = df["date"].isnull().sum()
                logger.warning(
                    f"{n_missing_dates} 'date' entries could not be converted and are NaT. Dropping these."
                )
                df = df.dropna(subset=["date"])
        except Exception as e:
            logger.error(f"Error converting 'date' to datetime: {e}")
            data_pipeline_logger.exception(f"Error converting 'date' to datetime: {e}")
            raise

    if df["league"].isna().any() or df["result"].isna().any():
        n_missing_leagues = df["league"].isnull().sum()
        n_missing_results = df["result"].isnull().sum()
        logger.warning(
            f"{n_missing_leagues} 'league' and {n_missing_results} 'result' missing. Dropping."
        )
        data_pipeline_logger.warning(
            f"{n_missing_leagues} 'league' and {n_missing_results} 'result' missing. Dropping."
        )
        df = df.dropna(subset=["league", "result"]).reset_index(drop=True)

    return df.sort_values(by=get_sorting_keys(entity)).reset_index(drop=True)


def map_team_to_league(
    df: pd.DataFrame, team_column: str
) -> dict[str, list[tuple[pd.Timestamp, str]]]:
    """
    Map each team to the leagues it played in over time, excluding cross-league competitions.

    Args:
        df (pd.DataFrame): DataFrame containing team and league information.
        team_column (str): Name of the column representing teams.

    Returns:
        Dict[str, List[Tuple[pd.Timestamp, str]]]: Dictionary mapping team IDs to a list of (date, league).

    """
    # Filter out cross-league competitions
    df_filtered = df[~df["league"].isin(CROSS_LEAGUE_COMPETITIONS)].copy()

    # Sort by date to get the chronological order
    df_filtered = df_filtered.sort_values(by=["date"])

    # Initialize an empty dictionary to store league history for each team
    league_history = {}

    # Group by the team column and iterate through the groups
    for team, group in df_filtered.groupby(team_column):
        # Collect (date, league) pairs for the current team
        league_history[team] = list(zip(group["date"], group["league"], strict=False))

    return league_history


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


def update_elo_rating(
    old_elo: float, expected: float, actual_result: float, k_factor: float
) -> float:
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


def linear_decay_reset_leagues_elo(
    elo_ratings: dict[str, dict[str, float | int]],
    baseline: float,
    current_season: int,
    decay_factor: float,
) -> dict[str, dict[str, float | int]]:
    """
    Apply linear decay reset to league Elo ratings at the beginning of a new season.

    Returns a new dictionary with the updated Elo ratings.
    """
    updated_elo_ratings = {}
    for league, data in elo_ratings.items():
        updated_data = data.copy()
        if data["season"] < current_season:
            updated_data["elo"] = baseline + (data["elo"] - baseline) * decay_factor
            updated_data["season"] = current_season
        updated_elo_ratings[league] = updated_data
    return updated_elo_ratings


def pivot_games_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the original DataFrame so that each game is a single row:
      - 'Blue' columns => e.g. teamid_Blue, league_Blue, result_Blue
      - 'Red' columns => e.g. teamid_Red,  league_Red,  result_Red
    This cuts the row count in half (2 rows per game -> 1 row per game).
    """
    # Validate that each game has exactly 2 rows for pivot to work
    counts = df.groupby(["gameid"]).size()
    if not (counts == 2).all():
        logger.warning(
            "Some gameid groups do not have exactly 2 rows. Pivot will fail or skip those."
        )
        data_pipeline_logger.warning(
            "Some gameid groups do not have exactly 2 rows. Pivot will fail or skip those."
        )
        df = df[df["gameid"].isin(counts[counts == 2].index)]

    # Move side to columns; the pivoted columns become multi-index
    df_wide = df.pivot(
        index=["date", "gameid", "season"],
        columns="side",
        values=["teamid", "league", "result"],
    ).reset_index()

    # Flatten multi-index column names: (('teamid','Blue'), ...) -> teamid_Blue
    df_wide.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in df_wide.columns.to_flat_index()
    ]
    return df_wide


def merge_wide_results_back(
    df: pd.DataFrame,
    df_wide: pd.DataFrame,
    columns_to_add: dict[str, str],
) -> pd.DataFrame:
    """
    After computing Elo in the wide pivot, each row corresponds to a single game (blue vs red).
    We hold new columns for 'blue' or 'red' side Elo.
    This function merges those results back to the original shape (two rows per game).

    columns_to_add is a dict of { "league_elo_before_blue": "league_elo_before", ... }
    indicating how to rename wide columns back into standard columns in the tall shape.
    """
    # We’ll pivot df_wide back to tall shape so it lines up with the original df again.
    # Example approach:
    #   1) For each side in ['Blue', 'Red'], rename columns + side => base column name
    #   2) stack them or concatenate them
    #   3) combine with original
    #
    # A simpler route is to keep the pivoted df_wide as our final results if your pipeline
    # doesn’t absolutely need two rows per game. But here we show how to revert if required.

    # Re-split into a 'blue' sub-dataframe and a 'red' sub-dataframe
    # Then unify them with 'side' info
    df_blue = df_wide.copy()
    df_blue["side"] = "Blue"
    df_red = df_wide.copy()
    df_red["side"] = "Red"

    # Drop irrelevant columns from each side's DataFrame
    drop_cols_blue = [c for c in df_blue.columns if c.endswith(("_Red", "_red"))]
    drop_cols_red = [c for c in df_red.columns if c.endswith(("_Blue", "_blue"))]
    df_blue = df_blue.drop(columns=drop_cols_blue)
    df_red = df_red.drop(columns=drop_cols_red)

    # Rename Elo-related columns to their final names based on side
    for wide_col, final_col in columns_to_add.items():
        df_blue = df_blue.rename(columns={wide_col: final_col})
        df_red = df_red.rename(columns={wide_col: final_col})

    # Combine back into a single tall DataFrame
    df_tall = pd.concat([df_blue, df_red], ignore_index=True)

    # Merge back with the original DataFrame
    merge_cols = ["date", "gameid", "season", "side"]
    merged = pd.merge(
        df,
        df_tall[
            [
                *merge_cols,
                "league_elo_before",
                "opp_league_elo_before",
                "league_elo_win_likelihood",
                "league_elo_after",
            ]
        ],
        validate="many_to_many",
        on=merge_cols,
        how="left",
    )

    # Retain only original columns and Elo-related columns
    required_columns = set(df.columns).union(set(columns_to_add.values()))
    return merged[[col for col in merged.columns if col in required_columns]]


def tune_hyperparameters(
    df: pd.DataFrame,
    entity: str,
    belonging_league: dict[str, str],
    hyperparameters_path: str,
) -> dict[str, float]:
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
    df_cross = df.copy()
    df_cross = df[df["league"].isin(CROSS_LEAGUE_COMPETITIONS)].copy()

    def objective(trial: optuna.trial.Trial) -> float:
        k_factor = trial.suggest_float("k_factor", 16, 96, step=8)
        initial_elo = trial.suggest_float("initial_elo", 1200, 1800, step=100)
        elo_divisor = trial.suggest_float("elo_divisor", 100, 500, step=50)
        decay_factor = trial.suggest_float("decay_factor", 0.5, 1.0, step=0.05)

        # Sort DataFrame
        df_sorted = df_cross.sort_values(by=["date", "gameid"]).reset_index(drop=True)

        # Evaluate ratings on the last year of data
        try:
            split_year = df_sorted["date"].dt.year.max()
        except AttributeError as e:
            logger.error(f"Error accessing 'date' column with .dt accessor: {e}")
            data_pipeline_logger.exception(
                f"Error accessing 'date' column with .dt accessor: {e}"
            )
            return float("inf")
        split_date = pd.to_datetime(f"{split_year}-01-01")

        # Split data
        df_train = df_sorted[df_sorted["date"] < split_date].reset_index(drop=True)
        df_valid = df_sorted[df_sorted["date"] >= split_date].reset_index(drop=True)

        # Check if each game is present as 2 rows
        if df_train.empty or not (df_train.groupby("gameid").size() == 2).all():
            logger.warning(
                "Training data is insufficient or improperly structured. Skipping trial."
            )
            data_pipeline_logger.warning(
                "Training data is insufficient or improperly structured. Skipping trial."
            )
            return float("inf")

        try:
            # Compute Elo ratings on training data
            df_with_elo_train = leagues_elo_computation(
                df_train.copy(),
                entity=entity,
                belonging_league=belonging_league,
                initial_elo=initial_elo,
                k_factor=k_factor,
                elo_divisor=elo_divisor,
                decay_factor=decay_factor,
                performing_tuning=True,
            )
        except Exception as e:
            logger.error(f"Elo calculation error in trial: {e}")
            data_pipeline_logger.exception(f"Elo calculation error in trial: {e}")
            return float("inf")

        # Grab final league Elo from training
        last_elo_ratings = (
            df_with_elo_train.groupby("league")["league_elo_after"]
            .last()
            .dropna()
            .to_dict()
        )

        # Initialize Elo ratings for validation
        validation_elo_ratings = defaultdict(
            lambda: {"elo": initial_elo, "season": split_year}
        )
        validation_elo_ratings.update(
            last_elo_ratings
        )  # Use trained Elo ratings as the starting point

        # Map validation teams to their original leagues using belonging_league
        df_valid["league"] = df_valid["teamid"].map(belonging_league)
        df_valid["opp_league"] = df_valid.groupby("gameid")["league"].shift(-1)

        # Initialize Elo simulation for validation
        df_valid = df_valid.sort_values(by=["date", "gameid"]).reset_index(drop=True)
        expected_probabilities = []

        for _, row in df_valid.iterrows():
            blue_league = row["league"]
            red_league = row["opp_league"]
            blue_result = row["result"]

            # Retrieve current Elo ratings
            blue_elo = validation_elo_ratings[blue_league]["elo"]
            red_elo = validation_elo_ratings[red_league]["elo"]

            # Calculate expected outcome
            expected = expected_outcome(blue_elo, red_elo, elo_divisor)
            expected_probabilities.append(expected)

            # Update Elo ratings based on the actual result
            red_result = 1 - blue_result
            validation_elo_ratings[blue_league]["elo"] = update_elo_rating(
                blue_elo, expected, blue_result, k_factor
            )
            validation_elo_ratings[red_league]["elo"] = update_elo_rating(
                red_elo, 1 - expected, red_result, k_factor
            )

        # Compute log loss
        y_true = df_valid["result"]
        y_pred = pd.Series(expected_probabilities).clip(0.0001, 0.9999)
        return log_loss(y_true, y_pred)

    # Create and optimize study
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=None, show_progress_bar=True)
    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")
    data_pipeline_logger.info(f"Best hyperparameters: {best_params}")

    # Store best hyperparameters
    logger.info(f"Storing hyperparameters to {hyperparameters_path}")
    data_pipeline_logger.info(f"Storing hyperparameters to {hyperparameters_path}")
    with open(hyperparameters_path, "w") as f:
        json.dump(best_params, f)
    return best_params


def leagues_elo_computation(
    df: pd.DataFrame,
    entity: str,
    belonging_league: dict[str, str],
    initial_elo: float | None = None,
    k_factor: float | None = None,
    elo_divisor: float | None = None,
    decay_factor: float | None = None,
    performing_tuning: bool = False,
) -> pd.DataFrame:
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
        performing_tuning (bool): Whether hyperparameter tuning is being performed.

    Returns:
        pd.DataFrame: DataFrame with updated Elo ratings for leagues.

    """
    if any(p is None for p in [initial_elo, k_factor, elo_divisor, decay_factor]):
        msg = "All hyperparameters must be provided"
        raise ValueError(msg)

    df_wide = pivot_games_to_wide(df)
    league_elo_ratings = defaultdict(
        lambda: {"elo": initial_elo, "season": df_wide["season"].min()}
    )

    wide_columns = {
        "league_elo_before_blue": [],
        "league_elo_before_red": [],
        "opp_league_elo_before_blue": [],
        "opp_league_elo_before_red": [],
        "league_elo_win_likelihood_blue": [],
        "league_elo_win_likelihood_red": [],
        "league_elo_after_blue": [],
        "league_elo_after_red": [],
    }

    if not performing_tuning:
        logger.info("Calculating Leagues Elo")
        data_pipeline_logger.info("Calculating Leagues Elo")
    df_wide_iter = tqdm(df_wide.itertuples(index=True), total=len(df_wide))

    for row in df_wide_iter:
        wide_columns, league_elo_ratings = process_elo_for_row(
            row,
            league_elo_ratings,
            belonging_league,
            wide_columns,
            initial_elo,
            k_factor,
            elo_divisor,
            decay_factor,
        )

    update_wide_dataframe(df_wide, wide_columns)

    df_final = finalize_dataframe(df, df_wide, entity)

    if not performing_tuning:
        store_results(belonging_league, league_elo_ratings)

    return df_final


def process_elo_for_row(
    row,
    league_elo_ratings,
    belonging_league,
    wide_columns,
    initial_elo,
    k_factor,
    elo_divisor,
    decay_factor,
) -> tuple[dict[str, list[float | int]], dict[str, dict[str, float | int]]]:
    """Process a single row to calculate and update Elo ratings for leagues."""
    current_season = row.season
    league_elo_ratings = linear_decay_reset_leagues_elo(
        league_elo_ratings, initial_elo, current_season, decay_factor
    )

    # Replace cross-league entries using historical league data
    def resolve_league(team_id, match_date):
        history = belonging_league.get(team_id, [])
        # Find the most recent league up to the match_date
        for date, league in reversed(history):
            if date <= match_date:
                return league
        return None

    blue_league = resolve_league(row.teamid_Blue, row.date)
    red_league = resolve_league(row.teamid_Red, row.date)
    blue_result = getattr(row, "result_Blue", None)
    red_result = getattr(row, "result_Red", None)

    if blue_result is None or red_result is None:
        for c in wide_columns:
            logger.warning(
                f"Missing result for gameid {row.gameid}. Filling with None."
            )
            data_pipeline_logger.warning(
                f"Missing result for gameid {row.gameid}. Filling with None."
            )
            wide_columns[c].append(None)
            return wide_columns, league_elo_ratings

    # Initialize if not present
    if blue_league not in league_elo_ratings:
        league_elo_ratings[blue_league] = {"elo": initial_elo, "season": current_season}
    if red_league not in league_elo_ratings:
        league_elo_ratings[red_league] = {"elo": initial_elo, "season": current_season}

    blue_league_elo_before = league_elo_ratings[blue_league]["elo"]
    red_league_elo_before = league_elo_ratings[red_league]["elo"]

    if blue_league != red_league:
        blue_expected = expected_outcome(
            blue_league_elo_before, red_league_elo_before, elo_divisor
        )
        red_expected = 1 - blue_expected
        new_blue_elo = update_elo_rating(
            blue_league_elo_before, blue_expected, blue_result, k_factor
        )
        new_red_elo = update_elo_rating(
            red_league_elo_before, red_expected, red_result, k_factor
        )
    else:
        blue_expected = 0.5
        new_blue_elo = blue_league_elo_before
        new_red_elo = red_league_elo_before

    league_elo_ratings[blue_league]["elo"] = new_blue_elo
    league_elo_ratings[red_league]["elo"] = new_red_elo

    wide_columns["league_elo_before_blue"].append(blue_league_elo_before)
    wide_columns["league_elo_before_red"].append(red_league_elo_before)
    wide_columns["opp_league_elo_before_blue"].append(red_league_elo_before)
    wide_columns["opp_league_elo_before_red"].append(blue_league_elo_before)
    wide_columns["league_elo_win_likelihood_blue"].append(blue_expected)
    wide_columns["league_elo_win_likelihood_red"].append(1 - blue_expected)
    wide_columns["league_elo_after_blue"].append(new_blue_elo)
    wide_columns["league_elo_after_red"].append(new_red_elo)

    return wide_columns, league_elo_ratings


def update_wide_dataframe(df_wide, wide_columns):
    """Update the wide dataframe with the computed columns."""
    for col_name, col_values in wide_columns.items():
        df_wide[col_name] = col_values


def finalize_dataframe(df, df_wide, entity):
    """Finalize the dataframe by pivoting back and sorting."""
    # Now pivot back to tall form so it matches the original shape (one row per side).
    # We'll map columns from wide to final tall columns:
    columns_map = {
        "league_elo_before_blue": "league_elo_before",
        "league_elo_before_red": "league_elo_before",
        "opp_league_elo_before_blue": "opp_league_elo_before",
        "opp_league_elo_before_red": "opp_league_elo_before",
        "league_elo_win_likelihood_blue": "league_elo_win_likelihood",
        "league_elo_win_likelihood_red": "league_elo_win_likelihood",
        "league_elo_after_blue": "league_elo_after",
        "league_elo_after_red": "league_elo_after",
    }
    df_final = merge_wide_results_back(df, df_wide, columns_map)
    df_final = df_final.sort_values(by=get_sorting_keys(entity))
    return df_final.reset_index(drop=True)


def store_results(belonging_league, league_elo_ratings):
    """Store belonging leagues and league Elo ratings."""
    store_belonging_leagues(belonging_league)
    store_leagues_elo(league_elo_ratings)


def store_belonging_leagues(
    belonging_league: dict[str, list[tuple[pd.Timestamp, str]]],
) -> None:
    """
    Store the mapping of teams to their most recent league.

    Args:
        belonging_league (Dict[str, List[Tuple[pd.Timestamp, str]]]): Dictionary where each key is a team ID, and the value
        is a list of tuples (date, league), sorted by date.

    """
    # Extract the last league entry for each team
    latest_belonging_league = {
        team: history[-1][1] for team, history in belonging_league.items() if history
    }

    # Create a DataFrame from the latest belonging league mapping
    belonging_league_df = pd.DataFrame(
        latest_belonging_league.items(), columns=["teamid", "league"]
    )

    # Store the resulting DataFrame as parquet
    safe_store_df_as_parquet(
        belonging_league_df, TEAM_LEAGUES_MAPPING, [logger, data_pipeline_logger]
    )


def store_leagues_elo(league_elo_ratings: dict[str, dict[str, float | int]]) -> None:
    """Store the Elo ratings for leagues."""
    league_elo_df = (
        pd.DataFrame(
            [(lg, dat["elo"]) for lg, dat in league_elo_ratings.items()],
            columns=["league", "elo"],
        )
        .sort_values(by="elo", ascending=False)
        .reset_index(drop=True)
    )
    league_elo_df = league_elo_df.dropna(subset=["league"]).reset_index(drop=True)
    safe_store_df_as_parquet(league_elo_df, LEAGUE_ELO, [logger, data_pipeline_logger])


def calculate_leagues_elo(
    df: pd.DataFrame,
    entity: str,
    hyperparameters_path: str = LEAGUES_ELO_HYPERPARAMETERS,
) -> pd.DataFrame:
    """Run the Leagues Elo pipeline with hyperparameter tuning."""
    df_preprocessed = preprocess_dataframe(df, entity)

    # Precompute the belonging league mapping
    belonging_league = map_team_to_league(df_preprocessed, "teamid")

    if os.path.exists(hyperparameters_path):
        logger.info(
            f"Hyperparameters found at {os.path.basename(hyperparameters_path)}"
        )
        data_pipeline_logger.info(
            f"Hyperparameters found at {os.path.basename(hyperparameters_path)}"
        )
        with open(hyperparameters_path) as f:
            best_params = json.load(f)
    else:
        logger.info("No hyperparameters found; starting tuning.")
        data_pipeline_logger.info("No hyperparameters found; starting tuning.")
        best_params = tune_hyperparameters(
            df_preprocessed, entity, belonging_league, hyperparameters_path
        )

    return leagues_elo_computation(
        df_preprocessed,
        entity=entity,
        belonging_league=belonging_league,
        initial_elo=best_params["initial_elo"],
        k_factor=best_params["k_factor"],
        elo_divisor=best_params["elo_divisor"],
        decay_factor=best_params["decay_factor"],
    )
