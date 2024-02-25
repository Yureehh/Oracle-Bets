# Imports
from collections import defaultdict
from typing import Dict

import pandas as pd


def calculate_elo(
    df: pd.DataFrame,
    entity: str,
    initial_elo: int = 1200,
    k: int = 20,
    name_prefix: str = "",
) -> pd.DataFrame:
    """
    Calculate Elo ratings_features for entities in a DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame containing match results.
    entity (str): The name of the entity for which to calculate Elo ratings_features.
    initial_elo (int, optional): The initial Elo rating for entities. Defaults to 1200.
    k (int, optional): The K-factor used in Elo rating updates. Defaults to 20.

    Returns:
    pd.DataFrame: The DataFrame with Elo ratings_features added.
    """

    def _expected(elo_a: float, elo_b: float) -> float:
        """
        Calculate the expected outcome of a match between two entities.

        Args:
        elo_a (float): The Elo rating of the first entity.
        elo_b (float): The Elo rating of the second entity.

        Returns:
        float: The expected outcome of the match for the first entity.
        """
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    def _elo(old: float, exp: float, score: float) -> float:
        """
        Update an Elo rating based on the outcome of a match.

        Args:
        old (float): The old Elo rating.
        exp (float): The expected outcome of the match.
        score (float): The actual outcome of the match.

        Returns:
        float: The updated Elo rating.
        """
        if old is None:
            old = initial_elo
        return old + k * (score - exp)

    if entity.lower() == "team":
        opponent_entity = "opponentteamid"
        entity_column = "teamid"
        sort_keys = ["date", "league", "gameid", "result"]
    elif entity.lower() == "player":
        opponent_entity = "opponentplayerid"
        entity_column = "playerid"
        sort_keys = ["date", "league", "gameid", "teamid", "position", "result"]
    else:
        raise ValueError(f"Unsupported entity name: {entity}")

    df = df.sort_values(sort_keys).reset_index(drop=True)

    elo_dict: Dict[str, float] = defaultdict(lambda: initial_elo)
    (
        entity_elo_array,
        opponent_elo_array,
        pre_elo_array,
        pre_opponent_elo_array,
        win_likelihood_array,
    ) = ([], [], [], [], [])

    for _, row in df.iterrows():
        entity_elo = elo_dict[row[entity_column]]
        opponent_elo = elo_dict[row[opponent_entity]]
        pre_elo_array.append(entity_elo)
        pre_opponent_elo_array.append(pre_opponent_elo_array)

        expected_outcome = _expected(entity_elo, opponent_elo)

        entity_new_elo = _elo(entity_elo, expected_outcome, row["result"])
        opponent_new_elo = _elo(opponent_elo, 1 - expected_outcome, 1 - row["result"])

        entity_elo_array.append(entity_new_elo)
        opponent_elo_array.append(opponent_new_elo)
        win_likelihood_array.append(expected_outcome)

        elo_dict[row[entity_column]] = entity_new_elo
        elo_dict[row[opponent_entity]] = opponent_new_elo

    if name_prefix and not name_prefix.endswith("_"):
        name_prefix += "_"

    pre_elo_colname = str(name_prefix) + "pre_match_elo"
    pre_opponent_elo_colname = str(name_prefix) + "pre_match_opponent_elo"
    elo_colname = str(name_prefix) + "elo"
    opponent_elo_colname = str(name_prefix) + "opponent_elo"
    elo_win_likelihood_colname = str(name_prefix) + "elo_win_likelihood"

    df[pre_elo_colname] = pre_elo_array
    df[pre_opponent_elo_colname] = pre_opponent_elo_array
    df[elo_colname] = entity_elo_array
    df[opponent_elo_colname] = opponent_elo_array
    df[elo_win_likelihood_colname] = win_likelihood_array

    return df
