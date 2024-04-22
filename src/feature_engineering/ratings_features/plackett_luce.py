"""
Plackett-Luce rating model

This module provides functionality to rate teams or players using the Plackett-Luce model.
"""

from copy import deepcopy
from typing import List, Tuple

import pandas as pd
from openskill.models import PlackettLuce

from utils.paths import DEFAULT_PARAMETERS
from utils.utils import get_sorting_keys, json_loader

config = json_loader(DEFAULT_PARAMETERS)

# Default parameters for Elo rating system
DEFAULT_MU = config["plackett_luce"]["mu"]
DEFAULT_SIGMA = config["plackett_luce"]["sigma"]


def rate_teams_or_players(
    model: PlackettLuce, entities_ratings: Tuple[List, List], ranks: List[int]
) -> Tuple[List, List]:
    """
    Rate two entities (teams or players) and update their ratings based on the match outcome.
    """
    # Create a deep copy of entities_ratings to avoid modifying the original
    old_entities_ratings = deepcopy(entities_ratings)

    # Update the ratings based on the match outcome
    updated_entities = model.rate(old_entities_ratings, ranks=ranks)
    return updated_entities


def calculate_plackett_luce(
    df: pd.DataFrame,
    entity: str,
    initial_mu: float = DEFAULT_MU,
    initial_sigma: float = DEFAULT_SIGMA,
) -> pd.DataFrame:
    """
    Calculate and update Plackett-Luce ratings for teams or players within a DataFrame.

    Parameters:
    - df: DataFrame containing match data.
    - initial_mu: Initial mean rating.
    - initial_sigma: Initial standard deviation of ratings.
    - entity: Flag to indicate whether the calculations are for teams or individual players.

    Returns:
    - DataFrame with updated Plackett-Luce ratings and win likelihoods.
    """

    # Sort the DataFrame
    df_sorted = df.sort_values(get_sorting_keys(entity)).reset_index(drop=True)

    # Initialize the Plackett-Luce model
    model = PlackettLuce(mu=initial_mu, sigma=initial_sigma)

    # Initialize ratings
    entity_key = "teamid" if entity == "team" else "playerid"
    entity_ratings = {
        entity: model.rating(mu=initial_mu, sigma=initial_sigma) for entity in df_sorted[entity_key].unique()
    }

    # Columns for updated ratings and pre-match info
    df_sorted["pl_pre_match_mu"] = pd.NA
    df_sorted["pl_pre_match_sigma"] = pd.NA
    df_sorted["pl_win_likelihood"] = pd.NA
    df_sorted["pl_mu"] = pd.NA
    df_sorted["pl_sigma"] = pd.NA
    df_sorted["pl_pre_match_mu_opponent"] = pd.NA

    for game_id in df_sorted["gameid"].unique():
        game_data = df_sorted[df_sorted["gameid"] == game_id]

        # Teams or players in the game
        blue_entities = game_data[game_data["side"] == "Blue"]
        red_entities = game_data[game_data["side"] == "Red"]

        # Extract entity IDs and current ratings
        blue_ratings = [entity_ratings[id_] for id_ in blue_entities[entity_key]]
        red_ratings = [entity_ratings[id_] for id_ in red_entities[entity_key]]

        # Calculate win likelihood
        win_probs = model.predict_win([blue_ratings, red_ratings])

        # Update pre-match ratings and win likelihood with game-specific filtering
        for side_data, ratings, win_prob in zip(
            [blue_entities, red_entities],
            [blue_ratings, red_ratings],
            [win_probs[0], win_probs[1]],
        ):
            for index, rating in zip(side_data.index, ratings):
                df_sorted.at[index, "pl_pre_match_mu"] = rating.mu
                df_sorted.at[index, "pl_pre_match_sigma"] = rating.sigma
                df_sorted.at[index, "pl_win_likelihood"] = win_prob
                df_sorted.at[index, "pl_pre_match_mu_opponent"] = (
                    red_ratings[0].mu if side_data.iloc[0]["side"] == "Blue" else blue_ratings[0].mu
                )

        # Determine the match outcome
        blue_win = blue_entities.iloc[0]["result"] == 1

        ranks = [0, 1] if blue_win else [1, 0]

        # Update ratings
        updated_ratings = rate_teams_or_players(model, [blue_ratings, red_ratings], ranks)

        # Assign updated ratings and calculate win likelihood
        for side_data, new_ratings in zip([blue_entities, red_entities], updated_ratings):
            for index, new_rating in zip(side_data.index, new_ratings):
                player_or_team_id = side_data.loc[index, entity_key]
                entity_ratings[player_or_team_id] = new_rating
                df_sorted.loc[index, ["pl_mu", "pl_sigma"]] = (
                    new_rating.mu,
                    new_rating.sigma,
                )

    return df_sorted
