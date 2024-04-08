"""
Wrapper class for rating models
"""

from dataclasses import dataclass

from src.ratings_features.elo import calculate_elo
from src.ratings_features.plackett_luce import calculate_plackett_luce
from src.ratings_features.trueskill import trueskill_model


@dataclass
class Ratings:
    """
    Wrapper class for rating models
    """

    @staticmethod
    def compute_elo(df, entity, initial_elo=None, k=None):
        args = {"df": df, "entity": entity}
        if initial_elo is not None:
            args["initial_elo"] = initial_elo
        if k is not None:
            args["k"] = k
        return calculate_elo(**args)

    @staticmethod
    def compute_plackett_luce(df, entity, initial_mu=None, initial_sigma=None):
        args = {"df": df, "entity": entity}
        if initial_mu is not None:
            args["initial_mu"] = initial_mu
        if initial_sigma is not None:
            args["initial_sigma"] = initial_sigma
        return calculate_plackett_luce(**args)

    @staticmethod
    def compute_trueskill(player_data, team_data, initial_sigma=None):
        args = {"player_data": player_data, "team_data": team_data}
        if initial_sigma is not None:
            args["initial_sigma"] = initial_sigma
        return trueskill_model(**args)
