"""
Wrapper class for in-game performance metric models
"""

from dataclasses import dataclass

import pandas as pd

# Import necessary modules for performance metrics calculation.
from src.feature_engineering.performance_features.entity_stats import enrich_entity_ema_statistics
from src.feature_engineering.performance_features.patch_win_rate import patch_win_rate_ewm_performance
from src.feature_engineering.performance_features.season_win_rate import season_win_rate_ewm_performance
from src.feature_engineering.performance_features.side_win_rate import side_win_rate_ewm_performance


@dataclass
class PerformanceMetrics:
    """
    Wrapper class for performance metrics calculations related to various game statistics.
    This class does not maintain any state and serves purely as an organizational tool
    for related static methods that operate on game data.
    """

    @staticmethod
    def validate_input_data(df: pd.DataFrame, entity: str):
        """
        Validates the input DataFrame and entity type.

        Parameters:
            df (pd.DataFrame): The DataFrame containing game data.
            entity (str): The type of entity, e.g., 'player' or 'team'.

        Raises:
            ValueError: If the entity is not 'player' or 'team' or if the DataFrame is invalid.
        """
        if entity not in ["player", "team"]:
            raise ValueError("Entity must be either 'player' or 'team'.")
        if df.empty or not isinstance(df, pd.DataFrame):
            raise ValueError("Input data must be a non-empty DataFrame.")
        required_columns = ["result", "patch"]  # Example required columns
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"Input data must contain the following columns: {required_columns}")

    @staticmethod
    def add_entity_ema_statistics(df: pd.DataFrame, entity: str) -> pd.DataFrame:
        """
        Enriches the DataFrame with entity-specific exponentially weighted moving average statistics.

        Parameters:
            df (pd.DataFrame): The DataFrame containing game data.
            entity (str): The type of entity, e.g., 'player' or 'team'.

        Returns:
            pd.DataFrame: DataFrame enriched with EMA statistics.
        """
        PerformanceMetrics.validate_input_data(df, entity)
        return enrich_entity_ema_statistics(df, entity)

    @staticmethod
    def add_side_win_rate_ewm(df: pd.DataFrame, entity: str) -> pd.DataFrame:
        """
        Calculates and appends side win rate statistics using an exponentially weighted mean model.

        Parameters:
            df (pd.DataFrame): The DataFrame containing game data.
            entity (str): The type of entity, e.g., 'player' or 'team'.

        Returns:
            pd.DataFrame: DataFrame with side win rate EWM statistics.
        """
        PerformanceMetrics.validate_input_data(df, entity)
        return side_win_rate_ewm_performance(df, entity)

    @staticmethod
    def add_season_win_rate_ewm(df: pd.DataFrame, entity: str) -> pd.DataFrame:
        """
        Computes and integrates the season win rate EWM model into the DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame containing game data.
            entity (str): The type of entity, e.g., 'player' or 'team'.

        Returns:
            pd.DataFrame: DataFrame updated with season win rate EWM model calculations.
        """
        PerformanceMetrics.validate_input_data(df, entity)
        return season_win_rate_ewm_performance(df, entity)

    @staticmethod
    def add_patch_win_rate_ewm(df: pd.DataFrame, entity: str) -> pd.DataFrame:
        """
        Computes and integrates the patch win rate EWM model into the DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame containing game data.
            entity (str): The type of entity, e.g., 'player' or 'team'.

        Returns:
            pd.DataFrame: DataFrame updated with patch win rate EWM model calculations.
        """
        PerformanceMetrics.validate_input_data(df, entity)
        return patch_win_rate_ewm_performance(df, entity)
