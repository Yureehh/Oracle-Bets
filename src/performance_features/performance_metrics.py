"""
Wrapper class for performance models
"""

from dataclasses import dataclass

from src.performance_features.egpm import egpm_model
from src.performance_features.entity_stats import enrich_entity_ema_statistics
from src.performance_features.side_win_rate import side_win_rate_ewm_performance


@dataclass
class PerformanceMetrics:
    """
    Wrapper class for performance models
    """

    @staticmethod
    def add_entity_ema_statistics(df, entity):
        return enrich_entity_ema_statistics(df, entity)

    @staticmethod
    def add_side_win_rate_ewm(df, entity):
        return side_win_rate_ewm_performance(df, entity)

    @staticmethod
    def add_egpm_model(df, entity):
        return egpm_model(df, entity)
