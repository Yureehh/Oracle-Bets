"""
Consists of the following classes:
- EarlyGameStatsImputer: Impute metrics for the 15m mark, `csat15`, `xpat15`, `goldat15` given end game values using an ensemble model composed of L2 Regression, k-NN, and a decision tree.
- FutureGamesStatsImputer: Impute future game stats using the same ensemble model as EarlyGameStatsImputer.
"""

from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from utils.logger import logger


@dataclass
class EarlyGameStatsImputer:
    test_size: float = 0.2
    random_state: int = 42

    def _generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate required additional features for imputing the given data.
        """
        # Calculate teamkills before generating other features
        teamkills = data.groupby(["gameid", "teamid"])["kills"].transform("sum")

        # KDA ratio
        data["KDA"] = (data["kills"] + data["assists"]) / np.where(
            data["deaths"] == 0, 1, data["deaths"]
        )

        # Gold Efficiency
        data["gold_efficiency"] = np.where(
            data["gamelength"] == 0, 0, data["totalgold"] / data["gamelength"]
        )

        # XP Efficiency
        data["xp_efficiency"] = np.where(
            data["gamelength"] == 0, 0, data["total_cs"] / data["gamelength"]
        )

        # Kill Participation
        data["kill_participation"] = np.where(
            teamkills == 0, 0, (data["kills"] + data["assists"]) / teamkills
        )

        # Sort and group the data once
        data = data.sort_values(by=["playerid", "date"], ascending=True).reset_index(
            drop=True
        )
        grouped_data = data.groupby("playerid")

        # Calculate volatility metrics, can return NaNs
        data["kills_volatility"] = grouped_data["kills"].transform(
            lambda x: x.rolling(window=5).std()
        )
        data["deaths_volatility"] = grouped_data["deaths"].transform(
            lambda x: x.rolling(window=5).std()
        )

        # Calculate growth metrics, can return NaNs
        data["kills_growth"] = grouped_data["kills"].transform(
            lambda x: x.rolling(window=5).mean()
        )
        data["deaths_growth"] = grouped_data["deaths"].transform(
            lambda x: x.rolling(window=5).mean()
        )

        return data

    def _train_stacked_model(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        features: List[str],
        target: str,
    ) -> tuple:
        """
        Train stacked models for the given target using the specified features.
        """
        # TODO: Store logs in a file
        knn = KNeighborsRegressor(n_neighbors=5)
        ridge = Ridge(alpha=1.0)
        tree = DecisionTreeRegressor(max_depth=5)
        base_models = {"KNN": knn, "Ridge": ridge, "Decision Tree": tree}

        for model in base_models.values():
            model.fit(train_data[features], train_data[target])
        val_predictions_base = {
            name: model.predict(val_data[features])
            for name, model in base_models.items()
        }
        val_predictions_df = pd.DataFrame(val_predictions_base)
        meta_model = LinearRegression().fit(val_predictions_df, val_data[target])

        return base_models, meta_model

    def _impute_missing_values(
        self,
        data: pd.DataFrame,
        features: List[str],
        stacked_models: Dict[
            str,
            Dict[
                str,
                Union[
                    KNeighborsRegressor, Ridge, DecisionTreeRegressor, LinearRegression
                ],
            ],
        ],
    ) -> pd.DataFrame:
        """
        Impute missing values using the trained stacked models.
        """
        missing_data = data[
            data["goldat15"].isnull()
            | data["xpat15"].isnull()
            | data["csat15"].isnull()
        ]

        for target, models in stacked_models.items():
            base_predictions = {
                name: model.predict(missing_data[features])
                for name, model in models["base_models"].items()
            }
            base_predictions_df = pd.DataFrame(base_predictions)
            final_predictions = models["meta_model"].predict(base_predictions_df)
            data.loc[missing_data.index, target] = final_predictions

        return data

    def _log_performance_metrics(
        self,
        val_data: pd.DataFrame,
        features_extended: List[str],
        target: str,
        base_models: dict,
        meta_model: LinearRegression,
    ):
        """Log performance metrics for each model."""
        for model_name, model in base_models.items():
            predictions = model.predict(val_data[features_extended])
            mae = mean_absolute_error(val_data[target], predictions)
            r2 = r2_score(val_data[target], predictions)
            logger.info(f"{model_name} - MAE: {mae:.2f}, R2: {r2:.2f}")

        val_predictions_base = {
            name: model.predict(val_data[features_extended])
            for name, model in base_models.items()
        }
        val_predictions_df = pd.DataFrame(val_predictions_base)
        ensemble_predictions = meta_model.predict(val_predictions_df)
        mae = mean_absolute_error(val_data[target], ensemble_predictions)
        r2 = r2_score(val_data[target], ensemble_predictions)
        logger.info(f"Ensemble - MAE: {mae:.2f}, R2: {r2:.2f}\n{'=' * 60}")

    def _prepare_data_for_modeling(self, data: pd.DataFrame) -> tuple:
        """Prepare data for modeling by encoding categorical variables and filtering."""

        # One-hot encode the 'position' column
        position_dummies = pd.get_dummies(data["position"], prefix="position")
        data = pd.concat([data, position_dummies], axis=1)

        # Extend the feature list to include one-hot encoded position columns
        features_extended = [
            "egpm",
            "gamelength",
            "totalgold",
            "ckpm",
            "monsterkills",
            "minionkills",
            "kills",
            "deaths",
            "assists",
        ] + position_dummies.columns.tolist()

        return data, features_extended

    def _train_models(self, data: pd.DataFrame, features_extended: List[str]) -> Dict:
        """Train stacked models for each target."""
        logger.info(f"Stacked models training started...\n{'=' * 60}")
        # Drop rows with NaN target values for training and validation
        training_data = data.dropna(subset=["goldat15", "xpat15", "csat15"])

        # Split the data into training and validation sets
        train_data, val_data = train_test_split(
            training_data, test_size=self.test_size, random_state=self.random_state
        )

        # Check in train data the columns with NaN values
        # Train stacked models
        stacked_models = {}
        for target in ["goldat15", "xpat15", "csat15"]:
            base_models, meta_model = self._train_stacked_model(
                train_data, val_data, features_extended, target
            )
            self._log_performance_metrics(
                val_data, features_extended, target, base_models, meta_model
            )
            stacked_models[target] = {
                "base_models": base_models,
                "meta_model": meta_model,
            }
        logger.info("Stacked models trained successfully.\n")
        return stacked_models

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the given data to impute missing values.

        Parameters:
        - data (pd.DataFrame): The input dataframe.

        Returns:
        - pd.DataFrame: The processed dataframe.
        """

        logger.info("Imputing missing data for players.")
        data = self._generate_features(data)
        data, features_extended = self._prepare_data_for_modeling(data)
        stacked_models = self._train_models(data, features_extended)
        data = self._impute_missing_values(data, features_extended, stacked_models)
        return data


@dataclass
class FutureGamesStatsImputer:
    pass
