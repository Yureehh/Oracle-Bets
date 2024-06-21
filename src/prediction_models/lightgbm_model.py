"""
LightGBM model class

This module contains the LightGBMModel class, a subclass of GradientBoostingModel.
It is used to train and evaluate a LightGBM model, optimize hyperparameters using Optuna,
and log feature importances and model metrics.
"""

from dataclasses import dataclass

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import log_loss

from prediction_models.gbdt_model import GradientBoostingModel
from utils.logger import logger, models_logger
from utils.paths import LIGHTGBM_BEST_HYPERPARAMETERS
from utils.utils import load_model

# Constants
VALIDATION_SIZE = 0.25
TEST_SIZE = 0.2


@dataclass
class LightGBMModel(GradientBoostingModel):
    def train_model(self):
        """Train the LightGBM model using the training data."""
        self.training_data.sort_values(by=["date", "gameid", "side"], inplace=True)
        X, y = self.training_data.drop(["date", "result"], axis=1), self.training_data["result"]

        # Handle categorical columns
        X, categorical_cols = self.preprocess_categorical_features(X, exclude_cols=["gameid", "side", "league"])

        # Split the data into training, validation, and testing sets
        X_train, X_val, X_test, y_train, y_val, y_test = self.grouped_stratified_train_val_test_split(
            X, y, X["gameid"], X["league"], val_size=VALIDATION_SIZE, test_size=TEST_SIZE
        )

        eval_gameids, eval_sides = X_test["gameid"], X_test["side"]

        # Drop specific columns
        X_train = X_train.drop(columns=["gameid", "side", "league"], errors="ignore")
        X_val = X_val.drop(columns=["gameid", "side", "league"], errors="ignore")
        X_test = X_test.drop(columns=["gameid", "side", "league"], errors="ignore")

        # Process likelihood columns and fuse opposing team features
        X_train = self.process_players_likelihood_columns(X_train)
        X_val = self.process_players_likelihood_columns(X_val)
        X_test = self.process_players_likelihood_columns(X_test)
        X_train = self.fuse_opposing_team_features(X_train)
        X_val = self.fuse_opposing_team_features(X_val)
        X_test = self.fuse_opposing_team_features(X_test)

        # Remove unnecessary columns and plot correlation matrix
        X_train = self.remove_unnecessary_columns(X_train)
        selected_features = X_train.columns
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]
        self.store_correlation(X_val, y_val)

        # Update and store categorical features
        categorical_features = [col for col in categorical_cols if col in selected_features]
        self.store_model_features(selected_features)
        self.store_categorical_features(categorical_features)

        # * NOTE: I don't use RFE to further reduce features here, since the number of features is already reduced

        # Get best hyperparameters and fit the model
        best_params = self.get_best_hyperparameters(X_train, y_train, X_val, y_val)
        model = lgb.LGBMClassifier(**best_params, force_col_wise=True, verbosity=-1)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        # Validate the model
        self.validate_model(model, X_test, y_test, eval_gameids, eval_sides)

        # Calculate and plot feature importances
        logger.info("Calculating and plotting feature importances...")
        self.store_feature_importance(model, selected_features)
        self.calculate_permutation_importance(model, X_test, y_test, selected_features)
        self.calculate_and_plot_shap(model, X_train, selected_features)
        logger.info("Finished calculating and plotting feature importances.\n")

        return model

    def get_best_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Retrieve the best hyperparameters for the LightGBM model."""
        if LIGHTGBM_BEST_HYPERPARAMETERS.exists():
            best_params = load_model(LIGHTGBM_BEST_HYPERPARAMETERS)
            logger.info(f"Found best hyperparameters: {best_params}\n")
            models_logger.info(f"Found best hyperparameters: {best_params}\n")
        else:
            best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
        return best_params

    def optimize_hyperparameters(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> dict:
        """Optimize hyperparameters for the LightGBM model using Optuna."""

        def objective(trial):
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
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

            clf = lgb.LGBMClassifier(**params, force_col_wise=True, verbosity=-1)
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            pred_proba = clf.predict_proba(X_val)[:, 1]
            logloss_score = log_loss(y_val, pred_proba)

            return logloss_score

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=self.trials)
        self.store_best_hyperparameters(study)
        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best log loss: {study.best_value:.4f}\n")
        models_logger.info(f"Best hyperparameters: {study.best_params}")
        models_logger.info(f"Best log loss: {study.best_value:.4f}\n")
        return study.best_params
