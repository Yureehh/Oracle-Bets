"""
Wrapper class for rating models

This script contains a dataclass that serves as a wrapper for various rating models.
"""

from dataclasses import dataclass
from typing import Any, Dict

from feature_engineering.ratings_features.glicko import calculate_glicko2
from feature_engineering.ratings_features.leagues_elo import calculate_leagues_elo
from feature_engineering.ratings_features.trueskill import calculate_trueskill
from feature_engineering.ratings_features.wh import calculate_whr
from src.feature_engineering.ratings_features.elo import calculate_elo
from src.feature_engineering.ratings_features.plackett_luce import calculate_plackett_luce


@dataclass
class Ratings:
    """
    A dataclass that serves as a wrapper for various rating models.
    """

    def _prepare_args(self, df: Any, entity: str) -> Dict[str, Any]:
        """
        Prepare arguments for rating calculation functions.

        Parameters:
            df (Any): The input dataframe containing match data.
            entity (str): The type of entity, e.g., 'team' or 'player'.

        Returns:
            Dict[str, Any]: A dictionary of filtered arguments.
        """
        return {"df": df, "entity": entity}

    def compute_elo(self, df: Any, entity: str) -> Any:
        """
        Computes the ELO rating for the given dataframe and entity.

        Parameters:
            df (Any): The input dataframe containing match data.
            entity (str): The type of entity, e.g., 'team' or 'player'.

        Returns:
            Any: The dataframe with computed ELO ratings.
        """
        return calculate_elo(**self._prepare_args(df, entity))

    def compute_glicko2(self, df: Any, entity: str) -> Any:
        """
        Computes the Glicko-2 rating for the given dataframe and entity.

        Parameters:
            df (Any): The input dataframe containing match data.
            entity (str): The type of entity, e.g., 'team' or 'player'.

        Returns:
            Any: The dataframe with computed Glicko-2 ratings.
        """
        return calculate_glicko2(**self._prepare_args(df, entity))

    def compute_plackett_luce(self, df: Any, entity: str) -> Any:
        """
        Computes the Plackett-Luce rating for the given dataframe and entity.

        Parameters:
            df (Any): The input dataframe containing match data.
            entity (str): The type of entity, e.g., 'team' or 'player'.

        Returns:
            Any: The dataframe with computed Plackett-Luce ratings.
        """
        return calculate_plackett_luce(**self._prepare_args(df, entity))

    def compute_trueskill(self, df: Any, entity: str) -> Any:
        """
        Computes the TrueSkill rating for the given dataframe and entity.

        Parameters:
            df (Any): The input dataframe containing match data.
            entity (str): The type of entity, e.g., 'team' or 'player'.

        Returns:
            Any: The dataframe with computed TrueSkill ratings.
        """
        return calculate_trueskill(**self._prepare_args(df, entity))

    def compute_whr(self, df: Any, entity: str) -> Any:
        """
        Computes the Whole History Rating for the given dataframe and entity.

        Parameters:
            df (Any): The input dataframe containing match data.
            entity (str): The type of entity, e.g., 'team' or 'player'.

        Returns:
            Any: The dataframe with computed Whole History Ratings.
        """
        return calculate_whr(**self._prepare_args(df, entity))

    def compute_leagues_elo(self, df: Any, entity: str) -> Any:
        """
        Computes the ELO rating for leagues based on the given dataframe and entity.

        Parameters:
            df (Any): The input dataframe containing match data.
            entity (str): The type of entity, e.g., 'team' or 'player'.

        Returns:
            Any: The dataframe with computed league ELO ratings.
        """
        return calculate_leagues_elo(**self._prepare_args(df, entity))
