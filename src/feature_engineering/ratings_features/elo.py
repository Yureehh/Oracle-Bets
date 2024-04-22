"""
Elo rating system

This script contains functions to calculate Elo ratings for teams or players based on match results.
"""

from collections import defaultdict
from typing import List, Tuple

import pandas as pd

from utils.paths import DEFAULT_PARAMETERS
from utils.utils import get_sorting_keys, json_loader

config = json_loader(DEFAULT_PARAMETERS)

# Default parameters for Elo rating system
DEFAULT_INITIAL_ELO = config["elo"]["initial"]
DEFAULT_K_FACTOR = config["elo"]["k_factor"]


def expected_outcome(elo_a: float, elo_b: float) -> float:
    """Calculate the expected match outcome between two Elo ratings."""
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def update_elo_rating(old_elo: float, expected: float, actual_result: float, k_factor: int) -> float:
    """Update Elo rating based on match result."""
    return old_elo + k_factor * (actual_result - expected)


def validate_entity(entity: str) -> Tuple[str, str, List[str]]:
    """Validate the entity and return configuration for entity and opponent."""
    entity = entity.lower()
    if entity == "team":
        return ("teamid", "opponentteamid", get_sorting_keys("team"))
    elif entity == "player":
        return ("playerid", "opponentplayerid", get_sorting_keys("player"))
    else:
        raise ValueError(f"Unsupported entity name: {entity}")


def determine_primary_perspective(side: str) -> bool:
    """Determine whether the current row is the primary perspective."""
    return side == "Blue"


def calculate_elo(
    df: pd.DataFrame,
    entity: str,
    initial_elo: int = DEFAULT_INITIAL_ELO,
    k: int = DEFAULT_K_FACTOR,
) -> pd.DataFrame:
    """
    Calculate Elo ratings for the specified entity based on match results.
    """
    entity_column, opponent_entity, sort_keys = validate_entity(entity)
    df_sorted = df.sort_values(by=sort_keys).reset_index(drop=True)

    elo_ratings = defaultdict(lambda: initial_elo)
    processed_games = set()

    # New dictionaries for storing pre-match and updated Elo ratings by game ID (and position)
    game_elos = defaultdict(lambda: defaultdict(lambda: (0, 0)))

    for index, row in df_sorted.iterrows():
        game_id = row["gameid"]
        side = row["side"]

        if side == "Red":
            continue

        processed_games.add(game_id)

        entity_elo = elo_ratings[row[entity_column]]
        opponent_elo = elo_ratings[row[opponent_entity]]

        expected_win_chance = expected_outcome(entity_elo, opponent_elo)
        new_entity_elo = update_elo_rating(entity_elo, expected_win_chance, row["result"], k)
        new_opponent_elo = update_elo_rating(opponent_elo, 1 - expected_win_chance, 1 - row["result"], k)

        # Update Elo ratings in the dictionary
        elo_ratings[row[entity_column]] = new_entity_elo
        elo_ratings[row[opponent_entity]] = new_opponent_elo

        # Store pre-match and updated Elo ratings
        game_elos[game_id][row[entity_column]] = (entity_elo, new_entity_elo)
        game_elos[game_id][row[opponent_entity]] = (opponent_elo, new_opponent_elo)

    # Apply Elo ratings to the DataFrame, adjusting for perspective
    for index, row in df_sorted.iterrows():
        game_id = row["gameid"]
        entity_id = row[entity_column]
        opponent_id = row[opponent_entity]

        if game_id in game_elos:
            # We need to check whether the current row is the primary perspective or the opponent perspective
            pre_entity_elo, post_entity_elo = game_elos[game_id][entity_id]
            pre_opponent_elo, post_opponent_elo = game_elos[game_id][opponent_id]

            df_sorted.at[index, "elo_pre_match"] = pre_entity_elo
            df_sorted.at[index, "elo_pre_match_opponent"] = pre_opponent_elo
            df_sorted.at[index, "elo_win_likelihood"] = expected_outcome(pre_entity_elo, pre_opponent_elo)
            df_sorted.at[index, "elo"] = post_entity_elo
            df_sorted.at[index, "elo_opponent"] = post_opponent_elo
    return df_sorted
