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

    def _prepare_args(self, match_data: Any, entity_type: str) -> Dict[str, Any]:
        """
        Prepare arguments for rating calculation functions.

        Parameters:
            match_data (Any): The input dataframe containing match data.
            entity_type (str): The type of entity, e.g., 'team' or 'player'.

        Returns:
            Dict[str, Any]: A dictionary of filtered arguments.
        """
        return {"df": match_data, "entity": entity_type}

    def _compute_rating(self, match_data: Any, entity_type: str, rating_function: Any) -> Any:
        """
        Generic method to compute ratings using the specified rating function.

        Parameters:
            match_data (Any): The input dataframe containing match data.
            entity_type (str): The type of entity, e.g., 'team' or 'player'.
            rating_function (Any): The rating calculation function to use.

        Returns:
            Any: The dataframe with computed ratings.
        """
        args = self._prepare_args(match_data, entity_type)
        return rating_function(**args)

    def compute_elo(self, match_data: Any, entity_type: str) -> Any:
        """
        Computes the ELO rating for the given dataframe and entity.

        Parameters:
            match_data (Any): The input dataframe containing match data.
            entity_type (str): The type of entity, e.g., 'team' or 'player'.

        Returns:
            Any: The dataframe with computed ELO ratings.
        """
        return self._compute_rating(match_data, entity_type, calculate_elo)

    def compute_glicko2(self, match_data: Any, entity_type: str) -> Any:
        """
        Computes the Glicko-2 rating for the given dataframe and entity.

        Parameters:
            match_data (Any): The input dataframe containing match data.
            entity_type (str): The type of entity, e.g., 'team' or 'player'.

        Returns:
            Any: The dataframe with computed Glicko-2 ratings.
        """
        return self._compute_rating(match_data, entity_type, calculate_glicko2)

    def compute_plackett_luce(self, match_data: Any, entity_type: str) -> Any:
        """
        Computes the Plackett-Luce rating for the given dataframe and entity.

        Parameters:
            match_data (Any): The input dataframe containing match data.
            entity_type (str): The type of entity, e.g., 'team' or 'player'.

        Returns:
            Any: The dataframe with computed Plackett-Luce ratings.
        """
        return self._compute_rating(match_data, entity_type, calculate_plackett_luce)

    def compute_trueskill(self, match_data: Any, entity_type: str) -> Any:
        """
        Computes the TrueSkill rating for the given dataframe and entity.

        Parameters:
            match_data (Any): The input dataframe containing match data.
            entity_type (str): The type of entity, e.g., 'team' or 'player'.

        Returns:
            Any: The dataframe with computed TrueSkill ratings.
        """
        return self._compute_rating(match_data, entity_type, calculate_trueskill)

    def compute_whr(self, match_data: Any, entity_type: str) -> Any:
        """
        Computes the Whole History Rating for the given dataframe and entity.

        Parameters:
            match_data (Any): The input dataframe containing match data.
            entity_type (str): The type of entity, e.g., 'team' or 'player'.

        Returns:
            Any: The dataframe with computed Whole History Ratings.
        """
        return self._compute_rating(match_data, entity_type, calculate_whr)

    def compute_leagues_elo(self, match_data: Any, entity_type: str) -> Any:
        """
        Computes the ELO rating for leagues based on the given dataframe and entity.

        Parameters:
            match_data (Any): The input dataframe containing match data.
            entity_type (str): The type of entity, e.g., 'team' or 'player'.

        Returns:
            Any: The dataframe with computed league ELO ratings.
        """
        return self._compute_rating(match_data, entity_type, calculate_leagues_elo)
