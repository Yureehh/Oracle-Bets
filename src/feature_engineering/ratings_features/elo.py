"""
Elo rating system

This script contains functions to calculate Elo ratings for teams or players based on match results.
"""

import math
from collections import defaultdict
from typing import Any, Dict, Union

import pandas as pd
from tqdm import tqdm

from utils.paths import DEFAULT_MODELS_PARAMETERS
from utils.utils import get_sorting_keys, json_loader

# Load configuration parameters
config = json_loader(DEFAULT_MODELS_PARAMETERS)
DEFAULT_INITIAL_ELO = config["elo"]["initial"]
DEFAULT_K_FACTOR = config["elo"]["k_factor"]
ELO_DIVISOR = 400  # Constant for Elo rating calculation
RESET_BASE = 1  # Constant for reset factor calculation


def expected_outcome(elo_a: float, elo_b: float) -> float:
    """Calculate the expected match outcome between two Elo ratings."""
    exponent = (elo_b - elo_a) / ELO_DIVISOR
    return 1 / (1 + 10**exponent)


def update_elo_rating(old_elo: float, expected: float, actual_result: float, k_factor: int) -> float:
    """Update Elo rating based on match result."""
    adjustment = k_factor * (actual_result - expected)
    return old_elo + adjustment


def aggregate_team_elo(
    rows: pd.DataFrame, elo_ratings: Dict[Union[int, str], Dict[str, Any]], entity_key: str
) -> float:
    """Aggregate Elo ratings for a team or player set."""
    return sum(elo_ratings[getattr(row, entity_key)]["elo"] for row in rows.itertuples())


def dynamic_percentage_reset(
    elo_ratings: Dict[Union[int, str], Dict[str, Any]], baseline: float, current_season: int
) -> None:
    """Apply dynamic percentage reset to Elo ratings at the beginning of a new season."""
    for entity, data in elo_ratings.items():
        if data["season"] < current_season:
            delta = abs(data["elo"] - baseline)
            reset_factor = RESET_BASE / (math.log2(delta + RESET_BASE) + RESET_BASE)
            data["elo"] = baseline + (data["elo"] - baseline) * reset_factor
            data["season"] = current_season


def handle_player_swap(
    player_id: Union[int, str], new_league: str, elo_ratings: Dict[Union[int, str], Dict[str, Any]], baseline_elo: float
) -> None:
    """Handle player swap between leagues and reset Elo rating."""
    elo_ratings[player_id]["elo"] = baseline_elo
    elo_ratings[player_id]["league"] = new_league


def process_game(
    df_sorted: pd.DataFrame,
    game_group: pd.DataFrame,
    elo_ratings: Dict[Union[int, str], Dict[str, Any]],
    k_factor: int,
    entity: str,
    entity_key: str,
    baseline_elo: float,
) -> None:
    """Process each game and update Elo ratings for both sides."""
    current_season = game_group.iloc[0]["season"]
    dynamic_percentage_reset(elo_ratings, baseline_elo, current_season)

    # Get the player IDs involved in the current game group
    player_ids = game_group[entity_key].unique()

    # Handle player swaps before processing game ratings
    for player_id in player_ids:
        player_data = elo_ratings[player_id]
        new_league = game_group[game_group[entity_key] == player_id]["league"].iloc[0]
        if player_data["league"] != new_league:
            handle_player_swap(player_id, new_league, elo_ratings, baseline_elo)

    blue_rows = (
        game_group[game_group["side"] == "Blue"].sort_values(by="position")
        if entity == "player"
        else game_group[game_group["side"] == "Blue"]
    )
    red_rows = (
        game_group[game_group["side"] == "Red"].sort_values(by="position")
        if entity == "player"
        else game_group[game_group["side"] == "Red"]
    )

    blue_elo_sum = aggregate_team_elo(blue_rows, elo_ratings, entity_key)
    red_elo_sum = aggregate_team_elo(red_rows, elo_ratings, entity_key)

    blue_expected = expected_outcome(blue_elo_sum, red_elo_sum)
    blue_result = blue_rows.iloc[0]["result"]
    red_result = 1 - blue_result

    for blue_row, red_row in zip(blue_rows.itertuples(), red_rows.itertuples()):
        blue_elo = elo_ratings[getattr(blue_row, entity_key)]["elo"]
        red_elo = elo_ratings[getattr(red_row, entity_key)]["elo"]

        blue_new_elo = update_elo_rating(blue_elo, blue_expected, blue_result, k_factor)
        red_new_elo = update_elo_rating(red_elo, 1 - blue_expected, red_result, k_factor)

        elo_ratings[getattr(blue_row, entity_key)]["elo"] = blue_new_elo
        elo_ratings[getattr(red_row, entity_key)]["elo"] = red_new_elo

        df_sorted.at[blue_row.Index, "elo_before"] = blue_elo
        df_sorted.at[blue_row.Index, "opp_elo_before"] = red_elo
        df_sorted.at[blue_row.Index, "elo_win_likelihood"] = blue_expected
        df_sorted.at[blue_row.Index, "elo_after"] = blue_new_elo

        df_sorted.at[red_row.Index, "elo_before"] = red_elo
        df_sorted.at[red_row.Index, "opp_elo_before"] = blue_elo
        df_sorted.at[red_row.Index, "elo_win_likelihood"] = 1 - blue_expected
        df_sorted.at[red_row.Index, "elo_after"] = red_new_elo


def calculate_elo(
    df: pd.DataFrame, entity: str, initial_elo: int = DEFAULT_INITIAL_ELO, k_factor: int = DEFAULT_K_FACTOR
) -> pd.DataFrame:
    """
    Calculate and update Elo ratings for entities within each team.

    Parameters:
        df (pd.DataFrame): The DataFrame containing match data.
        entity (str): The type of entity, e.g., 'player' or 'team'.
        initial_elo (int): Initial Elo rating.
        k_factor (int): K-factor for Elo rating adjustment.

    Returns:
        pd.DataFrame: DataFrame with updated Elo ratings.
    """
    entity_key = "teamid" if entity.lower() == "team" else "playerid"
    df_sorted = df.sort_values(by=get_sorting_keys(entity)).reset_index(drop=True)
    elo_ratings = defaultdict(lambda: {"elo": initial_elo, "season": df_sorted["season"].min(), "league": None})

    for _, game_group in tqdm(df_sorted.groupby(["date", "gameid"])):
        process_game(df_sorted, game_group, elo_ratings, k_factor, entity, entity_key, initial_elo)

    return df_sorted
