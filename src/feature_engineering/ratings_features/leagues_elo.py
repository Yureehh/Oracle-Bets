"""
League Elo Rating System

This module contains functions to calculate Elo ratings for teams or players based on match results.
"""

import math
from collections import defaultdict
from typing import Dict, Tuple, Union

import pandas as pd
from tqdm import tqdm

from utils.paths import DEFAULT_MODELS_PARAMETERS
from utils.utils import get_sorting_keys, json_loader

# Load configuration parameters
config = json_loader(DEFAULT_MODELS_PARAMETERS)
DEFAULT_INITIAL_ELO = config["league_elo"]["initial"]
DEFAULT_K_FACTOR = config["league_elo"]["k_factor"]
ELO_DIVISOR = 400  # Constant for Elo rating calculation


def map_team_to_league(df: pd.DataFrame, team_column: str) -> Dict[str, str]:
    """
    Map each team to its corresponding league based on the highest frequency of league appearance.

    Parameters:
        df (pd.DataFrame): DataFrame containing at least columns for teams and leagues.
        team_column (str): Name of the column in the DataFrame that contains team names.

    Returns:
        Dict[str, str]: A dictionary mapping each team to its most frequently associated league.
    """
    belonging_league = df.groupby(team_column)["league"].agg(lambda x: x.value_counts().idxmax()).to_dict()
    return belonging_league


def expected_outcome(elo_a: float, elo_b: float) -> float:
    """
    Calculate the expected match outcome between two aggregated Elo ratings.

    Parameters:
        elo_a (float): Elo rating of team A.
        elo_b (float): Elo rating of team B.

    Returns:
        float: Expected outcome probability for team A.
    """
    exponent = (elo_b - elo_a) / ELO_DIVISOR
    return 1 / (1 + 10**exponent)


def update_elo_rating(old_elo: float, expected: float, actual_result: float, k_factor: int) -> float:
    """
    Update Elo rating based on match result.

    Parameters:
        old_elo (float): Previous Elo rating.
        expected (float): Expected match outcome.
        actual_result (float): Actual match result (1 for win, 0 for loss).
        k_factor (int): K-factor for Elo rating adjustment.

    Returns:
        float: Updated Elo rating.
    """
    adjustment = k_factor * (actual_result - expected)
    return old_elo + adjustment


def dynamic_percentage_reset_league_elo(
    elo_ratings: Dict[str, Dict[str, Union[float, int]]], baseline: float, current_season: int
) -> None:
    """
    Apply dynamic percentage reset to league Elo ratings at the beginning of a new season.

    Parameters:
        elo_ratings (Dict[str, Dict[str, Union[float, int]]]): Dictionary of current Elo ratings.
        baseline (float): Baseline Elo value.
        current_season (int): The current season.
    """
    for _, data in elo_ratings.items():
        if data["season"] < current_season:
            delta = abs(data["elo"] - baseline)
            reset_factor = 1 / (math.log2(delta + 1) + 1)
            data["elo"] = baseline + (data["elo"] - baseline) * reset_factor
            data["season"] = current_season


def process_game(
    df_sorted: pd.DataFrame,
    game_group: pd.DataFrame,
    elo_ratings: Dict[str, Dict[str, Union[float, int]]],
    k_factor: int,
    entity_key: str,
    belonging_league: Dict[str, str],
    baseline_elo: float,
) -> None:
    """
    Process each game and update Elo ratings for both sides.

    Parameters:
        df_sorted (pd.DataFrame): The sorted DataFrame containing match data.
        game_group (pd.DataFrame): The game group DataFrame.
        elo_ratings (Dict[str, Dict[str, Union[float, int]]]): Dictionary of current Elo ratings.
        k_factor (int): K-factor for Elo rating adjustment.
        entity_key (str): The column name for the entity identifier.
        belonging_league (Dict[str, str]): Dictionary mapping each team to its most frequently associated league.
        baseline_elo (float): Baseline Elo value.
    """
    current_season = game_group.iloc[0]["season"]
    dynamic_percentage_reset_league_elo(elo_ratings, baseline_elo, current_season)

    blue_row = game_group[game_group["side"] == "Blue"].iloc[0]
    red_row = game_group[game_group["side"] == "Red"].iloc[0]

    blue_entity_id = blue_row[entity_key]
    red_entity_id = red_row[entity_key]

    blue_league = belonging_league[blue_entity_id]
    red_league = belonging_league[red_entity_id]

    blue_league_elo = elo_ratings[blue_league]["elo"]
    red_league_elo = elo_ratings[red_league]["elo"]

    if blue_league != red_league:
        blue_expected = expected_outcome(blue_league_elo, red_league_elo)
        blue_result = blue_row["result"]
        red_result = 1 - blue_result

        blue_new_league_elo = update_elo_rating(blue_league_elo, blue_expected, blue_result, k_factor)
        red_new_league_elo = update_elo_rating(red_league_elo, 1 - blue_expected, red_result, k_factor)

        elo_ratings[blue_league]["elo"] = blue_new_league_elo
        elo_ratings[red_league]["elo"] = red_new_league_elo
    else:
        blue_expected = 0.5
        blue_new_league_elo = blue_league_elo
        red_new_league_elo = red_league_elo

    df_sorted.at[blue_row.name, "league_elo_before"] = blue_league_elo
    df_sorted.at[blue_row.name, "opp_league_elo_before"] = red_league_elo
    df_sorted.at[blue_row.name, "league_elo_win_likelihood"] = blue_expected
    df_sorted.at[blue_row.name, "league_elo_after"] = blue_new_league_elo

    df_sorted.at[red_row.name, "league_elo_before"] = red_league_elo
    df_sorted.at[red_row.name, "opp_league_elo_before"] = blue_league_elo
    df_sorted.at[red_row.name, "league_elo_win_likelihood"] = 1 - blue_expected
    df_sorted.at[red_row.name, "league_elo_after"] = red_new_league_elo


def calculate_leagues_elo(
    df: pd.DataFrame, entity: str, initial_elo: int = DEFAULT_INITIAL_ELO, k_factor: int = DEFAULT_K_FACTOR
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate and update Elo ratings for entities within each team, return DataFrame and ratings dictionary.

    Parameters:
        df (pd.DataFrame): DataFrame containing match data.
        entity (str): The type of entity, e.g., 'team'.
        initial_elo (int): Initial Elo rating.
        k_factor (int): K-factor for Elo rating adjustment.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Updated DataFrame, team-to-league mapping, and league Elo ratings.
    """
    entity_key = "teamid"
    df_sorted = df.sort_values(by=get_sorting_keys(entity)).reset_index(drop=True)
    league_elo_ratings = defaultdict(lambda: {"elo": initial_elo, "season": df_sorted["season"].min()})
    belonging_league = map_team_to_league(df_sorted, entity_key)

    for _, game_group in tqdm(df_sorted.groupby(["date", "gameid"])):
        process_game(df_sorted, game_group, league_elo_ratings, k_factor, entity_key, belonging_league, initial_elo)

    belonging_league_df = pd.DataFrame(belonging_league.items(), columns=[entity_key, "league"])

    league_elo_ratings_dict = {league: data["elo"] for league, data in league_elo_ratings.items()}
    league_elo_df = pd.DataFrame(league_elo_ratings_dict.items(), columns=["league", "elo"])
    league_elo_df.sort_values(by="elo", ascending=False, inplace=True)

    return df_sorted, belonging_league_df, league_elo_df
