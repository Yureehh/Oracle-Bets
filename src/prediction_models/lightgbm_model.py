"""
LightGBM Model and Model Factory Classes.

This module contains the `LightGBMModel` class, a subclass of `GradientBoostingModel`.
It is used to train and evaluate a LightGBM model, optimize hyperparameters using Optuna,
and log feature importances and model metrics.

The `ModelFactory` class is used to create a `LightGBMModel` instance based on the problem type.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error

from prediction_models.gbdt_model import GradientBoostingModel
from src.utils.logger import logger
from src.utils.paths import GAMELENGTH_PREDICTION_BEST_HYPERPARAMETERS, OUTCOME_PREDICTION_BEST_HYPERPARAMETERS
from src.utils.utils import load_model

# Constants
VALIDATION_SIZE = 0.25
TEST_SIZE = 0.2


@dataclass
class LightGBMModel(GradientBoostingModel):
    """
    A class to train and evaluate a LightGBM model.

    Inherits from:
        GradientBoostingModel: An abstract base class providing common functionality for gradient boosting models.
    """

    validation_size: float = field(default=VALIDATION_SIZE)
    test_size: float = field(default=TEST_SIZE)

    def __post_init__(self):
        """Additional initialization after the object has been created."""
        super().__post_init__()
        # Additional initialization if needed

    def train_model(self, target_col: str) -> Tuple[lgb.LGBMModel, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Train the LightGBM model using the training data.

        Args:
            target_col (str): The target column name in the training data.

        Returns:
            Tuple containing the trained model, test features, test target, evaluation game IDs, and evaluation sides.

        Raises:
            Exception: If training fails.
        """
        try:
            X, y, categorical_cols = self._prepare_features_and_target(target_col)
            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                eval_gameids,
                eval_sides,
            ) = self._split_data(X, y)
            X_train, X_val, X_test = self._preprocess_features(X_train, X_val, X_test)
            X_train, X_val, X_test = self._select_and_store_features(X_train, X_val, X_test, y_val, categorical_cols)
            best_params = self._get_best_hyperparameters(X_train, y_train, X_val, y_val, target_col)
            model = self._fit_model(X_train, y_train, X_val, y_val, best_params)
            return model, X_test, y_test, eval_gameids, eval_sides
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            raise

    def train_and_validate_model(self, target_col: str = "result", validate: bool = True) -> lgb.LGBMModel:
        """
        Train and optionally validate the LightGBM model.

        Args:
            target_col (str): The target column name in the training data. Defaults to 'result'.
            validate (bool): Whether to validate the model after training. Defaults to True.

        Returns:
            lgb.LGBMModel: The trained LightGBM model.

        Raises:
            Exception: If training or validation fails.
        """
        try:
            model, X_test, y_test, eval_gameids, eval_sides = self.train_model(target_col)
            if validate:
                self.validate_model(model, X_test, y_test, eval_gameids, eval_sides)
            self._calculate_and_plot_feature_importances(model, X_test, y_test, X_test.columns.tolist())
            return model
        except Exception as e:
            logger.error(f"Failed to train and validate model: {e}")
            raise

    def _prepare_features_and_target(self, target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare the features and target for model training.

        Args:
            target_col (str): The target column name.

        Returns:
            Tuple containing the features DataFrame, target Series, and list of categorical feature names.

        Raises:
            ValueError: If the target column is not found in training data.
        """
        if target_col not in self.training_data.columns:
            logger.error(f"Target column '{target_col}' not found in training data.")
            raise ValueError(f"Target column '{target_col}' not found in training data.")
        X = self.training_data.drop(columns=[target_col])
        y = self.training_data[target_col]
        X, categorical_features = self.preprocess_categorical_features(X, exclude_cols=["gameid", "side", "league"])
        return X, y, categorical_features

    def _split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series,]:
        """
        Split the data into training, validation, and testing sets.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            Tuple containing the training, validation, and test splits for X and y,
            along with evaluation game IDs and sides.

        Raises:
            ValueError: If required columns are missing.
        """
        required_columns = ["gameid", "side", "league"]
        missing_columns = [col for col in required_columns if col not in X.columns]
        if missing_columns:
            logger.error(f"Missing required columns in X: {missing_columns}")
            raise ValueError(f"Missing required columns in X: {missing_columns}")
        try:
            (X_train, X_val, X_test, y_train, y_val, y_test,) = self.grouped_stratified_train_val_test_split(
                X,
                y,
                groups=X["gameid"],
                stratify=X["league"],
                val_size=self.validation_size,
                test_size=self.test_size,
            )

            eval_gameids = X_test["gameid"].reset_index(drop=True)
            eval_sides = X_test["side"].reset_index(drop=True)

            X_train = X_train.drop(columns=required_columns, errors="ignore")
            X_val = X_val.drop(columns=required_columns, errors="ignore")
            X_test = X_test.drop(columns=required_columns, errors="ignore")
            return (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                eval_gameids,
                eval_sides,
            )
        except Exception as e:
            logger.error(f"Error during data splitting: {e}")
            raise

    def _preprocess_features(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Process and fuse features for training, validation, and testing sets.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            X_val (pd.DataFrame): Validation feature matrix.
            X_test (pd.DataFrame): Test feature matrix.

        Returns:
            Tuple containing the processed training, validation, and test feature matrices.

        Raises:
            Exception: If preprocessing fails.
        """
        try:
            for func in [
                self.process_players_likelihood_columns,
                self.fuse_opposing_team_features,
            ]:
                X_train = func(X_train)
                X_val = func(X_val)
                X_test = func(X_test)
            return X_train, X_val, X_test
        except Exception as e:
            logger.error(f"Error during feature preprocessing: {e}")
            raise

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

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            X_val (pd.DataFrame): Validation feature matrix.
            X_test (pd.DataFrame): Test feature matrix.
            y_val (pd.Series): Validation target vector.
            categorical_cols (List[str]): List of categorical column names.

        Returns:
            Tuple containing the feature-selected training, validation, and test feature matrices.

        Raises:
            Exception: If feature selection fails.
        """
        try:
            X_train = self.remove_unnecessary_columns(X_train)
            selected_features = X_train.columns.tolist()
            selected_features = [feat for feat in selected_features if feat in X_val.columns and feat in X_test.columns]
            X_val = X_val[selected_features]
            X_test = X_test[selected_features]

            categorical_features = [col for col in categorical_cols if col in selected_features]

            self.store_correlation(X_val, y_val)
            self.store_model_features(selected_features)
            self.store_categorical_features(categorical_features)

            return X_train, X_val, X_test
        except Exception as e:
            logger.error(f"Error during feature selection and storage: {e}")
            raise

    def _get_best_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        target_col: str,
    ) -> Dict[str, Any]:
        """
        Retrieve the best hyperparameters for the LightGBM model.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Training target vector.
            X_val (pd.DataFrame): Validation feature matrix.
            y_val (pd.Series): Validation target vector.
            target_col (str): The target column name.

        Returns:
            Dict[str, Any]: Dictionary of the best hyperparameters.

        Raises:
            Exception: If hyperparameter optimization fails.
        """
        best_hyperparams_map = {
            "result": OUTCOME_PREDICTION_BEST_HYPERPARAMETERS,
            "gamelength": GAMELENGTH_PREDICTION_BEST_HYPERPARAMETERS,
        }
        hyperparams_path = best_hyperparams_map.get(target_col)
        if hyperparams_path and hyperparams_path.exists():
            try:
                best_params = load_model(hyperparams_path)
                logger.info(f"Loaded best hyperparameters from {hyperparams_path}")
            except Exception as e:
                logger.error(f"Failed to load hyperparameters from {hyperparams_path}: {e}")
                best_params = self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
                self.store_best_hyperparameters(best_params)
        else:
            best_params = self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
            self.store_best_hyperparameters(best_params)
        return best_params

    def _fit_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        best_params: Dict[str, Any],
    ) -> Union[lgb.LGBMClassifier, lgb.LGBMRegressor]:
        """
        Fit the LightGBM model using the best hyperparameters.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Training target vector.
            X_val (pd.DataFrame): Validation feature matrix.
            y_val (pd.Series): Validation target vector.
            best_params (Dict[str, Any]): Dictionary of best hyperparameters.

        Returns:
            Trained LightGBM model.

        Raises:
            Exception: If model fitting fails.
        """
        model_class = lgb.LGBMClassifier if self.problem_type == "classification" else lgb.LGBMRegressor
        try:
            model = model_class(**best_params, verbosity=-1)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False,
            )
            return model
        except Exception as e:
            logger.error(f"Failed to fit LightGBM model: {e}")
            raise

    def _optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for the LightGBM model using Optuna.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Training target vector.
            X_val (pd.DataFrame): Validation feature matrix.
            y_val (pd.Series): Validation target vector.

        Returns:
            Dict[str, Any]: Dictionary of the best hyperparameters.

        Raises:
            Exception: If hyperparameter optimization fails.
        """

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
                "verbosity": -1,
            }

            model_class = lgb.LGBMClassifier if self.problem_type == "classification" else lgb.LGBMRegressor
            clf = model_class(**params, verbosity=-1)
            clf.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False,
            )
            if self.problem_type == "classification":
                preds = clf.predict_proba(X_val)[:, 1]
                score = log_loss(y_val, preds)
            else:
                preds = clf.predict(X_val)
                score = mean_absolute_error(y_val, preds)
            return score

        try:
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
            study.optimize(objective, n_trials=self.trials)
            best_params = study.best_params
            logger.info(f"Best hyperparameters: {best_params}")
            logger.info(f"Best score: {study.best_value:.4f}")
            return best_params
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            raise

    def _calculate_and_plot_feature_importances(
        self,
        model: Union[lgb.LGBMClassifier, lgb.LGBMRegressor],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        selected_features: List[str],
    ) -> None:
        """
        Calculate and plot feature importances.

        Args:
            model: Trained LightGBM model.
            X_test (pd.DataFrame): Test feature matrix.
            y_test (pd.Series): Test target vector.
            selected_features (List[str]): List of selected feature names.

        Raises:
            Exception: If calculation or plotting fails.
        """
        logger.info("Calculating and plotting feature importances...")
        try:
            self.store_feature_importance(model, selected_features)
        except Exception as e:
            logger.error(f"Failed to store feature importance: {e}")

        try:
            self.calculate_permutation_importance(model, X_test, y_test, selected_features)
        except Exception as e:
            logger.error(f"Failed to calculate permutation importance: {e}")

        try:
            self.calculate_and_plot_shap(model, X_test, selected_features)
        except Exception as e:
            logger.error(f"Failed to calculate or plot SHAP values: {e}")

        logger.info("Finished calculating and plotting feature importances.\n")


class ModelFactory:
    """Factory class to create models based on the problem type."""

    @staticmethod
    def create_model(
        model_name: str,
        problem_type: str,
        training_team_data: pd.DataFrame,
        training_player_data: pd.DataFrame,
        model_type: str = "lightgbm",
    ) -> GradientBoostingModel:
        """
        Factory method to create a model based on the problem type and model type.

        Args:
            model_name (str): Name of the model.
            problem_type (str): Type of problem ('classification' or 'regression').
            training_team_data (pd.DataFrame): DataFrame containing team-level data.
            training_player_data (pd.DataFrame): DataFrame containing player-level data.
            model_type (str): Type of model to create ('lightgbm', 'xgboost', etc.).

        Returns:
            GradientBoostingModel: An instance of a model configured for the problem type.

        Raises:
            ValueError: If the problem type or model type is unsupported.
        """
        if problem_type not in ["classification", "regression"]:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        if model_type == "lightgbm":
            return LightGBMModel(
                model_name=model_name,
                problem_type=problem_type,
                team_data=training_team_data,
                player_data=training_player_data,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
