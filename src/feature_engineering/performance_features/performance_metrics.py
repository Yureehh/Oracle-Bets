"""
Wrapper Class for In-Game Performance Metric Models

This module contains a wrapper class for performance metrics calculations related to various game statistics.
It provides static methods to enrich game data with statistical features such as exponentially weighted
moving averages (EMA) for entities like players or teams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from feature_engineering.performance_features.entity_stats import (
    enrich_entity_ema_statistics,
)
from feature_engineering.performance_features.patch_win_rate import (
    patch_win_rate_ewm_performance,
)
from feature_engineering.performance_features.season_win_rate import (
    season_win_rate_ewm_performance,
)
from feature_engineering.performance_features.side_win_rate import (
    side_win_rate_ewm_performance,
)
from utils.pd import pd

Entity = Literal["player", "team"]


@dataclass()
class PerformanceMetrics:
    """
    A stateless utility class for performance metrics calculations.
    Groups together static methods that enrich a DataFrame with
    various exponentially weighted statistical features.
    """

    # Common columns needed for all methods (customize as needed)
    REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
        {"result", "patch", "season", "side"}
    )

    @staticmethod
    def validate_input_data(df: pd.DataFrame, entity: Entity) -> None:
        """
        Validates the input DataFrame and entity type.

        :param df: A Pandas DataFrame containing game data
        :param entity: 'player' or 'team'
        :raises ValueError: if entity is invalid, df is empty, or required columns are missing
        """
        if entity not in {"player", "team"}:
            msg = f"Invalid entity '{entity}': must be either 'player' or 'team'."
            raise ValueError(msg)
        if not isinstance(df, pd.DataFrame):
            msg = "Input data must be a Pandas DataFrame."
            raise TypeError(msg)
        if df.empty:
            msg = "Input DataFrame is empty."
            raise ValueError(msg)
        missing = PerformanceMetrics.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            msg = f"DataFrame is missing required columns: {missing}"
            raise ValueError(msg)

    @staticmethod
    def add_entity_ema_statistics(df: pd.DataFrame, entity: Entity) -> pd.DataFrame:
        """
        Enriches the DataFrame with entity-specific exponentially weighted moving average statistics.

        :param df: DataFrame containing game data
        :param entity: 'player' or 'team'
        :return: DataFrame enriched with EMA statistics
        """
        PerformanceMetrics.validate_input_data(df, entity)
        return enrich_entity_ema_statistics(df, entity)

    @staticmethod
    def add_side_win_rate_ewm(df: pd.DataFrame, entity: Entity) -> pd.DataFrame:
        """
        Calculates and appends side win rate statistics using an exponentially weighted mean model.

        :param df: DataFrame containing game data
        :param entity: 'player' or 'team'
        :return: DataFrame with side win rate EWM statistics
        """
        PerformanceMetrics.validate_input_data(df, entity)
        return side_win_rate_ewm_performance(df, entity)

    @staticmethod
    def add_season_win_rate_ewm(df: pd.DataFrame, entity: Entity) -> pd.DataFrame:
        """
        Computes and integrates the season win rate EWM model into the DataFrame.

        :param df: DataFrame containing game data
        :param entity: 'player' or 'team'
        :return: DataFrame updated with season win rate EWM model calculations
        """
        PerformanceMetrics.validate_input_data(df, entity)
        return season_win_rate_ewm_performance(df, entity)

    @staticmethod
    def add_patch_win_rate_ewm(df: pd.DataFrame, entity: Entity) -> pd.DataFrame:
        """
        Computes and integrates the patch win rate EWM model into the DataFrame.

        :param df: DataFrame containing game data
        :param entity: 'player' or 'team'
        :return: DataFrame updated with patch win rate EWM model calculations
        """
        PerformanceMetrics.validate_input_data(df, entity)
        return patch_win_rate_ewm_performance(df, entity)
