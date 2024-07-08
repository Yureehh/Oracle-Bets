"""
Early Game Stats Imputer

Consists of the following classes:
- EarlyGameStatsImputer: Impute metrics for the 15m mark, `csat15`, `xpat15`, `goldat15` given end game values using an
    ensemble model composed of L2 Regression, k-NN, and a decision tree.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.utils.logger import early_game_metrics_logger, logger
from src.utils.paths import FEATURES_TO_IMPUTE
from src.utils.utils import json_loader


def load_features() -> List[str]:
    """Load features to be imputed from configuration file."""
    return json_loader(FEATURES_TO_IMPUTE)["features"]


@dataclass
class EarlyGameStatsImputer:
    """
    Class to impute early game statistics (15m mark) given end game values using an ensemble model.
    """

    test_size: float = 0.15
    involved_cols: List[str] = field(default_factory=load_features)

    def train_models(self, train_data: pd.DataFrame, target: str) -> Dict[str, RegressorMixin]:
        """
        Train base models for the ensemble.
        """
        models = {
            "KNN": KNeighborsRegressor(n_neighbors=5),
            "Ridge": Ridge(alpha=1.0),
            "Decision Tree": DecisionTreeRegressor(max_depth=10),
        }
        features = [col for col in train_data.columns if col in self.involved_cols and col != target]
        for model in models.values():
            model.fit(train_data[features], train_data[target])
        return models

    def fit_meta_model(self, predictions: pd.DataFrame, target_values: pd.Series) -> LinearRegression:
        """
        Train meta-model using predictions from base models.
        """
        meta_model = LinearRegression()
        meta_model.fit(predictions, target_values)
        return meta_model

    def generate_predictions(
        self, models: Dict[str, RegressorMixin], data: pd.DataFrame, features: List[str]
    ) -> pd.DataFrame:
        """
        Generate predictions using base models.
        """
        predictions = {name: model.predict(data[features]) for name, model in models.items()}
        return pd.DataFrame(predictions)

    @staticmethod
    def clean_data(data: pd.Series) -> pd.Series:
        """
        Replace NaN or infinite values with the mean.
        """
        data = data.replace([np.inf, -np.inf], np.nan)
        return data.fillna(data.mean())

    @staticmethod
    def safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Safely divide two arrays, replacing illegal division results with NaN.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.true_divide(a, b)
            c[~np.isfinite(c)] = np.nan  # -inf, inf, NaN will be replaced with NaN
        return c

    def log_performance(self, model_name: str, predictions: pd.Series, true_values: pd.Series) -> None:
        """
        Log performance metrics of the model.
        """
        true_values = self.clean_data(true_values)
        predictions = self.clean_data(pd.Series(predictions))

        # Compute performance metrics
        mav = np.mean(true_values)
        mae = mean_absolute_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)

        # Ensure no division by zero or small numbers causing high MPE
        epsilon = 1e-8  # Small number to stabilize division
        mpe_values = self.safe_divide((predictions - true_values), true_values + epsilon) * 100
        mpe = np.nanmean(mpe_values)

        # Check if MPE is NaN due to all-zero denominators or empty data
        mpe_message = "Indeterminate" if np.isnan(mpe) else f"{mpe:.2f}%"

        # Log the performance metrics
        early_game_metrics_logger.info(
            f"{model_name} - MAV: {mav:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}, MPE: {mpe_message}"
        )

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for imputation by filtering out rows with missing involved columns.
        """
        return data.dropna(subset=self.involved_cols)

    def impute_data(self, data: pd.DataFrame, entity: str) -> pd.DataFrame:
        """
        Impute missing early game stats.
        """
        logger.info(f"Starting data imputation for {entity}...")
        data = self.prepare_data(data)
        features = [col for col in data.columns if col in self.involved_cols]
        train_data, val_data = train_test_split(data, test_size=self.test_size)
        stacked_models = {}

        for target in self.involved_cols:
            features_copy = features.copy()
            features_copy.remove(target)
            base_models = self.train_models(train_data, target)
            val_predictions = self.generate_predictions(base_models, val_data, features_copy)
            meta_model = self.fit_meta_model(val_predictions, val_data[target])
            self.log_performance(
                f"{entity} {target.capitalize()}", meta_model.predict(val_predictions), val_data[target]
            )
            stacked_models[target] = {"base_models": base_models, "meta_model": meta_model}

        early_game_metrics_logger.info(f"Finished imputing metrics for {entity} data\n\n")
        logger.info(f"Finished data imputation for {entity}\n")
        return data
