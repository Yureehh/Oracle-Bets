"""
Wrapper Class for Rating Models

This module contains a class that serves as a wrapper for various rating models.
"""

from dataclasses import dataclass
from typing import Any

from src.feature_engineering.ratings_features.elo import calculate_elo
from src.feature_engineering.ratings_features.glicko import calculate_glicko2
from src.feature_engineering.ratings_features.leagues_elo import calculate_leagues_elo
from src.feature_engineering.ratings_features.plackett_luce import calculate_plackett_luce
from src.feature_engineering.ratings_features.trueskill import calculate_trueskill

# from src.feature_engineering.ratings_features.wh import calculate_whr


@dataclass
class Ratings:
    """A class that serves as a wrapper for various rating models."""

    def compute_elo(self, df: Any, entity: str) -> Any:
        """
        Computes the ELO rating for the given DataFrame and entity.

        Args:
            df (Any): The input DataFrame containing match data.
            entity (str): The type of entity, 'team' or 'player'.

        Returns:
            Any: The DataFrame with computed ELO ratings.
        """
        return calculate_elo(df, entity)

    def compute_glicko2(self, df: Any, entity: str) -> Any:
        """
        Computes the Glicko-2 rating for the given DataFrame and entity.

        Args:
            df (Any): The input DataFrame containing match data.
            entity (str): The type of entity, 'team' or 'player'.

        Returns:
            Any: The DataFrame with computed Glicko-2 ratings.
        """
        return calculate_glicko2(df, entity)

    def compute_plackett_luce(self, df: Any, entity: str) -> Any:
        """
        Computes the Plackett-Luce rating for the given DataFrame and entity.

        Args:
            df (Any): The input DataFrame containing match data.
            entity (str): The type of entity, 'team' or 'player'.

        Returns:
            Any: The DataFrame with computed Plackett-Luce ratings.
        """
        return calculate_plackett_luce(df, entity)

    def compute_trueskill(self, df: Any, entity: str) -> Any:
        """
        Computes the TrueSkill rating for the given DataFrame and entity.

        Args:
            df (Any): The input DataFrame containing match data.
            entity (str): The type of entity, 'team' or 'player'.

        Returns:
            Any: The DataFrame with computed TrueSkill ratings.
        """
        return calculate_trueskill(df, entity)

    def compute_whr(self, df: Any, entity: str) -> Any:
        """
        Computes the Whole History Rating for the given DataFrame and entity.

        Args:
            df (Any): The input DataFrame containing match data.
            entity (str): The type of entity, 'team' or 'player'.

        Returns:
            Any: The DataFrame with computed Whole History Ratings.
        """
        pass  # calculate_whr(df, entity)

    def compute_leagues_elo(self, df: Any) -> Any:
        """
        Computes the ELO rating for leagues based on the given DataFrame.

        Args:
            df (Any): The input DataFrame containing match data.

        Returns:
            Any: The DataFrame with computed league ELO ratings.
        """
        return calculate_leagues_elo(df, entity="team")
