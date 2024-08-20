"""
LightGBM model and model factory classes.

This module contains the LightGBMModel class, a subclass of GradientBoostingModel.
It is used to train and evaluate a LightGBM model, optimize hyperparameters using Optuna,
and log feature importances and model metrics.

The ModelFactory class is used to create a LightGBMModel based on the problem type.
"""

from dataclasses import dataclass
from typing import List, Tuple

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error

from prediction_models.gbdt_model import GradientBoostingModel
from src.utils.logger import logger, models_logger
from src.utils.paths import GAMELENGTH_PREDICTION_BEST_HYPERPARAMETERS, OUTCOME_PREDICTION_BEST_HYPERPARAMETERS
from src.utils.utils import load_model

# Constants
VALIDATION_SIZE = 0.25
TEST_SIZE = 0.2


@dataclass
class LightGBMModel(GradientBoostingModel):
    """A class to train and evaluate a LightGBM model."""

    def train_model(self, target_col: str) -> Tuple[lgb.LGBMModel, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Train the LightGBM model using the training data."""
        X, y, categorical_cols = self._prepare_features_and_target(target_col)
        X_train, X_val, X_test, y_train, y_val, y_test, eval_gameids, eval_sides = self._split_data(X, y)
        X_train, X_val, X_test = self._preprocess_features(X_train, X_val, X_test)
        X_train, X_val, X_test = self._select_and_store_features(X_train, X_val, X_test, y_val, categorical_cols)
        best_params = self._get_best_hyperparameters(X_train, y_train, X_val, y_val, target_col)
        model = self._fit_model(X_train, y_train, X_val, y_val, best_params)
        return model, X_test, y_test, eval_gameids, eval_sides

    def train_and_validate_model(self, target_col: str = "result", validate: bool = True) -> lgb.LGBMModel:
        """Train and validate the LightGBM model."""
        model, X_test, y_test, eval_gameids, eval_sides = self.train_model(target_col)
        if validate:
            self.validate_model(model, X_test, y_test, eval_gameids, eval_sides)
        self._calculate_and_plot_feature_importances(model, X_test, y_test, X_test.columns)
        return model

    def _prepare_features_and_target(self, target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare the features and target for model training."""
        X = self.training_data.drop(columns=[target_col])
        y = self.training_data[target_col]
        X, categorical_features = self.preprocess_categorical_features(X, exclude_cols=["gameid", "side", "league"])
        return X, y, categorical_features

    def _split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Split the data into training, validation, and testing sets."""
        X_train, X_val, X_test, y_train, y_val, y_test = self.grouped_stratified_train_val_test_split(
            X, y, X["gameid"], X["league"], val_size=VALIDATION_SIZE, test_size=TEST_SIZE
        )

        eval_gameids, eval_sides = X_test["gameid"], X_test["side"]

        X_train = X_train.drop(columns=["gameid", "side", "league"], errors="ignore")
        X_val = X_val.drop(columns=["gameid", "side", "league"], errors="ignore")
        X_test = X_test.drop(columns=["gameid", "side", "league"], errors="ignore")
        return X_train, X_val, X_test, y_train, y_val, y_test, eval_gameids, eval_sides

    def _preprocess_features(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process and fuse features for training, validation, and testing sets."""
        for func in [self.process_players_likelihood_columns, self.fuse_opposing_team_features]:
            X_train = func(X_train)
            X_val = func(X_val)
            X_test = func(X_test)
        return X_train, X_val, X_test

    def _select_and_store_features(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_val: pd.Series,
        categorical_cols: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Select features, store them, and plot the correlation matrix.
        NOTE: I don't use RFE to further reduce features here, since the number of features is already reduced
        """
        X_train = self.remove_unnecessary_columns(X_train)
        selected_features = X_train.columns
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]

        categorical_features = [col for col in categorical_cols if col in selected_features]

        self.store_correlation(X_val, y_val)
        self.store_model_features(selected_features)
        self.store_categorical_features(categorical_features)

        return X_train, X_val, X_test

    def _get_best_hyperparameters(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, target_col: str
    ) -> dict:
        """Retrieve the best hyperparameters for the LightGBM model."""
        best_hyperparams_map = {
            "result": OUTCOME_PREDICTION_BEST_HYPERPARAMETERS,
            "gamelength": GAMELENGTH_PREDICTION_BEST_HYPERPARAMETERS,
        }
        if best_hyperparams_map[target_col].exists():
            best_params = load_model(best_hyperparams_map[target_col])
            logger.info(f"Found best hyperparameters: {best_params}\n")
            models_logger.info(f"Found best hyperparameters: {best_params}")
        else:
            best_params = self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
        return best_params

    def _fit_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, best_params: dict
    ) -> lgb.LGBMModel:
        """Fit the LightGBM model using the best hyperparameters."""
        model = (
            lgb.LGBMClassifier(**best_params, force_col_wise=True, verbosity=-1)
            if self.problem_type == "classification"
            else lgb.LGBMRegressor(**best_params, force_col_wise=True, verbosity=-1)
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        return model

    def _optimize_hyperparameters(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> dict:
        """Optimize hyperparameters for the LightGBM model using Optuna."""

        def objective(trial):
            params = {
                "objective": "binary" if self.problem_type == "classification" else "regression",
                "metric": "binary_logloss" if self.problem_type == "classification" else "mae",
                "bagging_freq": 1,
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 2, 1024),
                "max_depth": trial.suggest_int("max_depth", -1, 50),
                "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10),
            }

            clf = (
                lgb.LGBMClassifier(**params, force_col_wise=True, verbosity=-1)
                if self.problem_type == "classification"
                else lgb.LGBMRegressor(**params, force_col_wise=True, verbosity=-1)
            )
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            score = (
                log_loss(y_val, clf.predict_proba(X_val)[:, 1])
                if self.problem_type == "classification"
                else mean_absolute_error(y_val, clf.predict(X_val))
            )
            return score

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=self.trials)
        self.store_best_hyperparameters(study)
        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best score: {study.best_value:.4f}\n")
        models_logger.info(f"Best hyperparameters: {study.best_params}")
        models_logger.info(f"Best score: {study.best_value:.4f}")
        return study.best_params

    def _calculate_and_plot_feature_importances(
        self, model, X_test: pd.DataFrame, y_test: pd.Series, selected_features: List[str]
    ):
        """Calculate and plot feature importances."""
        logger.info("Calculating and plotting feature importances...")
        self.store_feature_importance(model, selected_features)
        self.calculate_permutation_importance(model, X_test, y_test, selected_features)
        self.calculate_and_plot_shap(model, X_test, selected_features)
        logger.info("Finished calculating and plotting feature importances\n")


class ModelFactory:
    @staticmethod
    def create_model(
        model_name: str, problem_type: str, training_team_data: pd.DataFrame, training_player_data: pd.DataFrame
    ) -> LightGBMModel:
        """Factory method to create a LightGBMModel based on the problem type."""
        if problem_type not in ["classification", "regression"]:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        return LightGBMModel(
            model_name=model_name,
            problem_type=problem_type,
            team_data=training_team_data,
            player_data=training_player_data,
        )
