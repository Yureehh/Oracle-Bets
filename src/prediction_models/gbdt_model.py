"""
Gradient Boosting Decision Tree Model.

This module defines an abstract class for Gradient Boosting Decision Tree models to extend from.
It provides methods for preprocessing data, training the model, optimizing hyperparameters,
storing best hyperparameters, and storing categorical features.
"""

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from prediction_models.data_preprocessor import DataPreprocessor
from utils.logger import logger
from utils.paths import (
    FEATURE_IMP_DIR,
    FIGURES_DIR,
    INSIGHTS_DIR,
    MODELS_DIR,
    PROCESSED_TEAMS,
)

sns.set_style("darkgrid")

# Constants
DEFAULT_TRIALS = 2000
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
TOP_N_FEATURES = 30
LOW_STD_THRESHOLD = 0.05
HIGH_CORR_THRESHOLD = 0.90
FIGSIZE = (20, 16)
CMAP = "coolwarm"


@dataclass
class GradientBoostingModel(ABC):
    """
    Abstract base class for Gradient Boosting Decision Tree models.

    Attributes:
        model_name (str): Name of the model.
        problem_type (str): Type of problem ('classification' or 'regression').
        team_data (pd.DataFrame): DataFrame containing team-level data.
        player_data (pd.DataFrame): DataFrame containing player-level data.
        trials (int): Number of trials for hyperparameter optimization.
        directory (Path): Directory to store figures and outputs.
        training_data (pd.DataFrame): Preprocessed training data.

    """

    model_name: str
    problem_type: str
    team_data: pd.DataFrame
    player_data: pd.DataFrame
    trials: int = DEFAULT_TRIALS
    directory: Path = FIGURES_DIR
    training_data: pd.DataFrame = field(init=False)

    def __post_init__(self):
        """Post-initialization to set up the training_data attribute."""
        self.training_data = pd.DataFrame()

    def preprocess_data(self, target_col: str) -> pd.DataFrame:
        """
        Preprocess the data by pivoting player data and merging datasets.

        Args:
            target_col (str): The target column for the model.

        Returns:
            pd.DataFrame: The preprocessed training data.

        Raises:
            ValueError: If preprocessing fails.

        """
        try:
            preprocessor = DataPreprocessor(self.team_data, self.player_data)
            self.training_data = preprocessor.preprocess(target_col, self.problem_type)
            logger.info("Data preprocessing completed successfully.")
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            msg = f"Data preprocessing failed: {e}"
            raise ValueError(msg) from e
        return self.training_data

    @staticmethod
    def grouped_stratified_train_val_test_split(
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        stratify: pd.Series,
        val_size: float = VALIDATION_SIZE,
        test_size: float = TEST_SIZE,
    ) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """
        Splits the data into training, validation, and test sets, grouped by specified
        columnsand stratified by a specified column.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
            groups (pd.Series): Groups for splitting.
            stratify (pd.Series): Stratification column.
            val_size (float): Proportion of validation set.
            test_size (float): Proportion of test set.

        Returns:
            Tuple containing training, validation, and test splits for X and y.

        Raises:
            ValueError: If splitting fails or required columns are missing.

        """

        def stratified_split(temp_df, n_splits, groups_col, stratify_col):
            strat_split = StratifiedGroupKFold(n_splits=n_splits, shuffle=True)
            splits = strat_split.split(
                temp_df[groups_col], temp_df[stratify_col], groups=temp_df[groups_col]
            )
            return next(splits)

        required_columns = ["gameid", "league"]
        missing_columns = [col for col in required_columns if col not in X.columns]
        if missing_columns:
            logger.error(f"Missing required columns in X: {missing_columns}")
            msg = f"Missing required columns in X: {missing_columns}"
            raise ValueError(msg)

        try:
            temp_df = pd.DataFrame(
                {"group": groups, "stratify": stratify}
            ).drop_duplicates()

            n_splits_test = int(1 / test_size)
            train_val_idx, test_idx = stratified_split(
                temp_df, n_splits_test, "group", "stratify"
            )

            train_val_groups = temp_df.iloc[train_val_idx]["group"]
            test_groups = temp_df.iloc[test_idx]["group"]

            X_train_val = X[X["gameid"].isin(train_val_groups)].copy()
            X_test = X[X["gameid"].isin(test_groups)].copy()
            y_train_val = y[X["gameid"].isin(train_val_groups)]
            y_test = y[X["gameid"].isin(test_groups)]

            temp_df_train_val = pd.DataFrame(
                {"group": X_train_val["gameid"], "stratify": X_train_val["league"]}
            ).drop_duplicates()

            n_splits_val = int(1 / val_size)
            train_idx, val_idx = stratified_split(
                temp_df_train_val, n_splits_val, "group", "stratify"
            )

            train_groups = temp_df_train_val.iloc[train_idx]["group"]
            val_groups = temp_df_train_val.iloc[val_idx]["group"]

            X_train = X_train_val[X_train_val["gameid"].isin(train_groups)].copy()
            X_val = X_train_val[X_train_val["gameid"].isin(val_groups)].copy()
            y_train = y_train_val[X_train_val["gameid"].isin(train_groups)]
            y_val = y_train_val[X_train_val["gameid"].isin(val_groups)]

            logger.info("Data splitting completed successfully.")
            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            logger.error(f"Data splitting failed: {e}")
            msg = f"Data splitting failed: {e}"
            raise ValueError(msg) from e

    def fuse_opposing_team_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fuses opposing team features by subtracting the opposing stats from the original stats.

        Args:
            X (pd.DataFrame): DataFrame containing features.

        Returns:
            pd.DataFrame: Modified DataFrame with fused features.

        Raises:
            ValueError: If required columns are missing.

        """
        X = X.copy()
        try:
            # Process team-level 'opp_' prefixed columns
            opp_cols = [col for col in X.columns if col.startswith("opp_")]
            for col in opp_cols:
                original_col = col[4:]
                if original_col in X.columns:
                    X[original_col] = X[original_col] - X[col]
                    X = X.drop(columns=[col])
                else:
                    logger.warning(
                        f"Original column '{original_col}' not found for opposing feature '{col}'."
                    )
                    X = X.drop(columns=[col])

            # Process role-specific 'opp_' prefixed columns
            roles = ["top", "jng", "mid", "bot", "sup"]
            for role in roles:
                role_opp_cols = [
                    col for col in X.columns if col.startswith(f"{role}_opp_")
                ]
                for col in role_opp_cols:
                    original_col = col.replace(f"{role}_opp_", f"{role}_")
                    if original_col in X.columns:
                        X[original_col] = X[original_col] - X[col]
                        X = X.drop(columns=[col])
                    else:
                        logger.warning(
                            f"Original column '{original_col}' not found for opposing feature '{col}'."
                        )
                        X = X.drop(columns=[col])
            logger.info("Fused opposing team features successfully.")
        except Exception as e:
            logger.error(f"Error in fusing opposing team features: {e}")
            msg = f"Error in fusing opposing team features: {e}"
            raise ValueError(msg) from e

        return X

    def process_players_likelihood_columns(
        self, df: pd.DataFrame, target_role: str = "top"
    ) -> pd.DataFrame:
        """
        Processes player likelihood columns by renaming and dropping unnecessary ones.

        Args:
            df (pd.DataFrame): DataFrame containing player data.
            target_role (str): The role whose likelihood columns to retain. Defaults to 'top'.

        Returns:
            pd.DataFrame: Modified DataFrame with processed likelihood columns.

        Raises:
            ValueError: If likelihood columns are missing.

        """
        df = df.copy()
        try:
            likelihood_columns = df.columns[df.columns.str.contains("likelihood")]
            if likelihood_columns.empty:
                logger.warning("No likelihood columns found in the DataFrame.")
                return df

            modified_df = df[likelihood_columns].copy()
            modified_df.columns = modified_df.columns.str.replace(
                f"{target_role}_", "players_"
            )

            roles = ["top_", "jng_", "mid_", "bot_", "sup_"]
            roles.remove(f"{target_role}_")
            cols_to_drop = []
            for role in roles:
                cols_to_drop.extend(
                    [col for col in modified_df.columns if col.startswith(role)]
                )
            modified_df = modified_df.drop(columns=cols_to_drop)

            df = df.drop(columns=likelihood_columns, inplace=False)
            df = pd.concat([df, modified_df], axis=1)
            logger.info(f"Processed likelihood columns for role '{target_role}'.")
        except Exception as e:
            logger.error(f"Error processing player likelihood columns: {e}")
            msg = f"Error processing player likelihood columns: {e}"
            raise ValueError(msg) from e

        return df

    def remove_unnecessary_columns(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Removes columns that are unnecessary or degrade model performance.

        Args:
            df (pd.DataFrame): DataFrame from which to remove columns.
            std_threshold (float): Threshold for coefficient of variation.
            corr_threshold (float): Correlation threshold above which to drop features.
            patterns_to_remove (List[str], optional): List of regex patterns for columns to remove.

        Returns:
            pd.DataFrame: DataFrame after removing unnecessary columns.

        Raises:
            ValueError: If an error occurs during the process.

        """
        df = df.copy()
        try:
            df = df.loc[:, ~df.columns.str.contains("_std")]
            df = self.drop_low_std_columns(df)
            df = self.drop_highly_correlated_features(df)
            df = df.loc[:, ~df.columns.str.contains("ema_total_towers")]
            df = df.loc[:, ~df.columns.str.contains("ema_total_kills")]
            logger.info("Removed unnecessary columns successfully.")
        except Exception as e:
            logger.error(f"Error removing unnecessary columns: {e}")
            msg = f"Error removing unnecessary columns: {e}"
            raise ValueError(msg) from e

        return df

    def drop_low_std_columns(
        self, df: pd.DataFrame, threshold: float = LOW_STD_THRESHOLD
    ) -> pd.DataFrame:
        """
        Drops columns with low standard deviation.

        Args:
            df (pd.DataFrame): DataFrame from which to drop columns.
            threshold (float): Threshold for coefficient of variation.

        Returns:
            pd.DataFrame: DataFrame after dropping low standard deviation columns.

        Raises:
            ValueError: If an error occurs during the process.

        """
        df = df.copy()
        try:
            std_dev = df.std()
            mean_values = df.mean()
            # Avoid division by zero
            mean_values = mean_values.replace(0, np.finfo(float).eps)
            coef_of_variation = std_dev / mean_values

            columns_to_drop = coef_of_variation[coef_of_variation < threshold].index
            df = df.drop(columns=columns_to_drop, inplace=False)
            logger.info(
                f"Dropped columns with coefficient of variation below {threshold}: {list(columns_to_drop)}"
            )
        except Exception as e:
            logger.error(f"Error dropping low standard deviation columns: {e}")
            msg = f"Error dropping low standard deviation columns: {e}"
            raise ValueError(msg) from e

        return df

    def drop_highly_correlated_features(
        self,
        df: pd.DataFrame,
        threshold: float = HIGH_CORR_THRESHOLD,
        priority_features: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Drops one feature from each pair of highly correlated features.

        Args:
            df (pd.DataFrame): DataFrame from which to drop columns.
            threshold (float): Correlation threshold above which to drop features.
            priority_features (List[str], optional): List of features to prioritize for retention.

        Returns:
            pd.DataFrame: DataFrame after dropping highly correlated features.

        Raises:
            ValueError: If an error occurs during the process.

        """
        df = df.copy()
        try:
            corr_matrix = df.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            to_drop = set()

            for column in upper.columns:
                highly_correlated = [
                    index
                    for index in upper.index
                    if upper.loc[index, column] > threshold
                ]
                if highly_correlated and not any(
                    col in to_drop for col in highly_correlated
                ):
                    to_drop.add(column)

            df = df.drop(columns=list(to_drop), inplace=False)
            logger.info(f"Dropped columns due to high correlation: {list(to_drop)}")
        except Exception as e:
            logger.error(f"Error dropping highly correlated features: {e}")
            msg = f"Error dropping highly correlated features: {e}"
            raise ValueError(msg) from e

        return df

    def store_correlation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        figsize: tuple[int, int] = FIGSIZE,
        cmap: str = CMAP,
    ) -> None:
        """
        Stores the correlation matrix as a heatmap.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
            figsize (Tuple[int, int]): Figure size for the plot.
            cmap (str): Colormap for the heatmap.

        Raises:
            ValueError: If plotting fails.

        """
        try:
            df = pd.concat([X, y.rename("target")], axis=1)
            corr_matrix = df.corr()

            plt.figure(figsize=figsize)
            sns.heatmap(corr_matrix, cmap=cmap, cbar=True)
            plt.title("Correlation Matrix Heatmap")
            plt.tight_layout()
            plt.savefig(
                self.directory.joinpath(f"{self.model_name}_Correlation_Matrix.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            logger.info(f"Correlation matrix plot for {self.model_name} stored.")
        except Exception as e:
            logger.error(f"Failed to store correlation matrix: {e}")
            msg = f"Failed to store correlation matrix: {e}"
            raise ValueError(msg) from e

    @abstractmethod
    def train_model(self) -> None:
        """
        Abstract method to train the model.

        This method should be implemented by subclasses to define the model training process.
        """

    @abstractmethod
    def _optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, Any]:
        """
        Abstract method to optimize hyperparameters.

        This method should be implemented by subclasses to define the hyperparameter optimization process.

        Returns:
            Dict[str, Any]: A dictionary of optimized hyperparameters.

        """

    @staticmethod
    def preprocess_categorical_features(
        X: pd.DataFrame,
        exclude_cols: list[str] | None = None,
        categorical_columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Preprocess features by converting categorical columns to category dtype.

        Args:
            X (pd.DataFrame): DataFrame containing the features.
            exclude_cols (List[str], optional): List of columns to exclude.
            categorical_columns (List[str], optional): List of categorical columns.

        Returns:
            Tuple[pd.DataFrame, List[str]]: The processed DataFrame and the list of categorical columns.

        Raises:
            ValueError: If an error occurs during processing.

        """
        try:
            X = X.copy()
            if exclude_cols is None:
                exclude_cols = []
            if categorical_columns is None:
                categorical_columns = [
                    col for col in X.columns if X[col].dtype == "object"
                ]

            for col in categorical_columns:
                X[col] = X[col].astype("category")

            if exclude_cols:
                categorical_columns = [
                    col for col in categorical_columns if col not in exclude_cols
                ]

            consistent_cats = {
                col: X[col].cat.categories for col in categorical_columns
            }
            for col, cats in consistent_cats.items():
                X[col] = X[col].cat.set_categories(cats)

            return X, categorical_columns

        except Exception as e:
            logger.error(f"Error processing categorical features: {e}")
            msg = f"Error processing categorical features: {e}"
            raise ValueError(msg) from e

    def store_model_features(self, all_features: pd.Index) -> None:
        """
        Store all features to a pickle file.

        Args:
            all_features (pd.Index): Index of feature names.

        """
        self._store_pickle(
            f"{self.model_name}_final_features.pkl", all_features.tolist()
        )

    def store_categorical_features(self, categorical_features: list[str]) -> None:
        """
        Store categorical features to a pickle file.

        Args:
            categorical_features (List[str]): List of categorical feature names.

        """
        self._store_pickle(
            f"{self.model_name}_categorical_features.pkl", categorical_features
        )

    def _store_pickle(self, filename: str, data: Any) -> None:
        """
        Helper method to store data to a pickle file.

        Args:
            filename (str): Name of the file.
            data (Any): Data to store.

        Raises:
            ValueError: If storing fails.

        """
        try:
            with open(MODELS_DIR / filename, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Stored {filename} successfully.")
        except Exception as e:
            logger.error(f"Failed to store {filename}: {e}")
            msg = f"Failed to store {filename}: {e}"
            raise ValueError(msg) from e

    def store_best_hyperparameters(self, hyperparams: dict[str, Any]) -> None:
        """
        Store best hyperparameters to a pickle file.

        Args:
            hyperparams (Dict[str, Any]): Dictionary of best hyperparameters.

        """
        self._store_pickle(f"{self.model_name}_best_hyperparameters.pkl", hyperparams)

    def store_predictions(
        self, predictions: np.ndarray, eval_gameids: pd.Series, eval_sides: pd.Series
    ) -> None:
        """
        Store prediction insights to a parquet file.

        Args:
            predictions (np.ndarray): Array of predictions.
            eval_gameids (pd.Series): Series of game IDs.
            eval_sides (pd.Series): Series of sides.

        Raises:
            ValueError: If storing fails.

        """
        try:
            insights = pd.DataFrame(
                {"prediction": predictions, "gameid": eval_gameids, "side": eval_sides}
            )
            insights.to_parquet(
                INSIGHTS_DIR / f"{self.model_name}_predictions.parquet",
                index=False,
                compression="gzip",
            )
            logger.info(f"Prediction insights for {self.model_name} stored.")
        except Exception as e:
            logger.error(f"Failed to store prediction insights: {e}")
            msg = f"Failed to store prediction insights: {e}"
            raise ValueError(msg) from e

    def validate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        eval_gameids: pd.Series,
        eval_sides: pd.Series,
    ) -> None:
        """
        Validate the model and store evaluation metrics.

        Args:
            model: Trained model to validate.
            X_test (pd.DataFrame): Test feature matrix.
            y_test (pd.Series): Test target vector.
            eval_gameids (pd.Series): Series of game IDs.
            eval_sides (pd.Series): Series of sides.

        Raises:
            ValueError: If validation fails.

        """
        try:
            logger.info("Validating the model...")
            predictions = model.predict(X_test)

            if self.problem_type == "classification":
                metrics = self.compute_classification_metrics(y_test, predictions)
                self.plot_confusion_matrix(metrics["cm"])
                self.plot_accuracy_over_samples(y_test, predictions)
                self.plot_historical_accuracy(X_test, y_test, predictions, eval_gameids)
            elif self.problem_type == "regression":
                metrics = self.compute_regression_metrics(y_test, predictions)
                self.plot_regression_results(y_test, predictions)
                self.plot_regression_error_over_samples(y_test, predictions)
                self.plot_regression_error_over_time(
                    X_test, y_test, predictions, eval_gameids
                )
            else:
                logger.error(
                    f"Unknown problem type '{self.problem_type}' during validation."
                )
                msg = f"Unknown problem type '{self.problem_type}' during validation."
                raise ValueError(msg)

            self.log_evaluation_metrics(metrics)
            self.store_evaluation_metrics(metrics)
            self.store_predictions(predictions, eval_gameids, eval_sides)

            logger.info(f"Model {self.model_name} validated and insights stored.\n")
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            msg = f"Model validation failed: {e}"
            raise ValueError(msg) from e

    def compute_classification_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray
    ) -> dict[str, Any]:
        """
        Compute evaluation metrics for classification models.

        Args:
            y_true (pd.Series): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            Dict[str, Any]: Dictionary of computed metrics.

        """
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary"
        )
        cm = confusion_matrix(y_true, y_pred)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "cm": cm,
        }

    def compute_regression_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray
    ) -> dict[str, Any]:
        """
        Compute evaluation metrics for regression models.

        Args:
            y_true (pd.Series): True values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            Dict[str, Any]: Dictionary of computed metrics.

        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

    def store_evaluation_metrics(self, metrics: dict[str, Any]) -> None:
        """
        Store evaluation metrics to a JSON file.

        Args:
            metrics (Dict[str, Any]): Dictionary of evaluation metrics.

        Raises:
            ValueError: If storing fails.

        """
        metrics_to_store = {k: v for k, v in metrics.items() if k != "cm"}
        try:
            with open(INSIGHTS_DIR / f"{self.model_name}_metrics.json", "w") as f:
                json.dump(metrics_to_store, f)
            logger.info(f"Evaluation metrics for {self.model_name} stored.")
        except Exception as e:
            logger.error(f"Failed to store evaluation metrics: {e}")
            msg = f"Failed to store evaluation metrics: {e}"
            raise ValueError(msg) from e

    def log_evaluation_metrics(self, metrics: dict[str, Any]) -> None:
        """
        Log evaluation metrics.

        Args:
            metrics (Dict[str, Any]): Dictionary of evaluation metrics.

        """
        if self.problem_type == "classification":
            log_message = (
                f"Evaluation Metrics - Accuracy: {metrics['accuracy']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
                f"F1 Score: {metrics['f1']:.4f}"
            )
        elif self.problem_type == "regression":
            log_message = (
                f"Evaluation Metrics - MAE: {metrics['mae']:.4f}, MSE: {metrics['mse']:.4f}, "
                f"RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}"
            )
        else:
            log_message = "Unknown problem type."
        logger.info(log_message)

    def plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """
        Plot and save the confusion matrix.

        Args:
            cm (np.ndarray): Confusion matrix.

        Raises:
            ValueError: If plotting fails.

        """
        try:
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Negative:0", "Actual Positive:1"],
                columns=["Predict Negative:0", "Predict Positive:1"],
            )
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig(
                self.directory.joinpath(f"{self.model_name}_Confusion_Matrix.png"),
                dpi=300,
            )
            plt.close()
            logger.info(f"Confusion matrix plot for {self.model_name} stored.")
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {e}")
            msg = f"Failed to plot confusion matrix: {e}"
            raise ValueError(msg) from e

    def plot_regression_results(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        """
        Plot and save regression results.

        Args:
            y_true (pd.Series): True values.
            y_pred (np.ndarray): Predicted values.

        Raises:
            ValueError: If plotting fails.

        """
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot(
                [y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                color="red",
                linestyle="--",
            )
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Regression Model Results")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                self.directory.joinpath(f"{self.model_name}_Regression_Results.png"),
                dpi=300,
            )
            plt.close()
            logger.info(f"Regression results plot for {self.model_name} stored.")
        except Exception as e:
            logger.error(f"Failed to plot regression results: {e}")
            msg = f"Failed to plot regression results: {e}"
            raise ValueError(msg) from e

    def plot_accuracy_over_samples(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        """
        Plot and save accuracy over samples.

        Args:
            y_true (pd.Series): True labels.
            y_pred (np.ndarray): Predicted labels.

        Raises:
            ValueError: If plotting fails.

        """
        try:
            accuracy_timeline = [
                accuracy_score(y_true[:i], y_pred[:i])
                for i in range(1, len(y_true) + 1)
            ]
            plt.figure(figsize=(10, 5))
            sns.lineplot(
                x=range(1, len(y_true) + 1),
                y=accuracy_timeline,
                linestyle="--",
                color="#84C3FA",
            )
            plt.xlabel("Number of Samples")
            plt.ylabel("Accuracy")
            plt.grid(True, linestyle="--", alpha=0.6, axis="y")
            plt.grid(False, axis="x")
            plt.gca().set_facecolor("none")
            plt.legend(["Accuracy"], facecolor="white", edgecolor="none")
            plt.tight_layout()
            plt.gca().tick_params(axis="x", colors="white")
            plt.gca().tick_params(axis="y", colors="white")
            plt.savefig(
                self.directory.joinpath(f"{self.model_name}_Accuracy_Over_Samples.png"),
                dpi=300,
            )
            plt.close()
            logger.info(f"Accuracy over samples plot for {self.model_name} stored.")
        except Exception as e:
            logger.error(f"Failed to plot accuracy over samples: {e}")
            msg = f"Failed to plot accuracy over samples: {e}"
            raise ValueError(msg) from e

    def plot_regression_error_over_samples(
        self, y_true: pd.Series, y_pred: np.ndarray, metric: str = "mae"
    ) -> None:
        """
        Plots the regression error over samples using the specified error metric.

        Args:
            y_true (pd.Series): True values.
            y_pred (np.ndarray): Predicted values.
            metric (str): The error metric to use ('mae' or 'mse').

        Raises:
            ValueError: If plotting fails or invalid metric is specified.

        """
        try:
            if metric == "mae":
                errors = np.abs(y_true - y_pred)
            elif metric == "mse":
                errors = (y_true - y_pred) ** 2
            else:
                msg = "Invalid metric specified. Use 'mae' or 'mse'."
                raise ValueError(msg)

            cumulative_error_timeline = [
                np.mean(errors[:i]) for i in range(1, len(y_true) + 1)
            ]

            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(y_true) + 1), cumulative_error_timeline, marker="o")
            plt.title(f"Cumulative {metric.upper()} Over Samples")
            plt.xlabel("Number of Samples")
            plt.ylabel(f"Cumulative {metric.upper()}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                self.directory.joinpath(
                    f"{self.model_name}_Cumulative_{metric.upper()}_Over_Samples.png"
                ),
                dpi=300,
            )
            plt.close()
            logger.info(
                f"Cumulative {metric.upper()} over samples plot for {self.model_name} stored."
            )
        except Exception as e:
            logger.error(f"Failed to plot regression error over samples: {e}")
            msg = f"Failed to plot regression error over samples: {e}"
            raise ValueError(msg) from e

    def plot_historical_accuracy(
        self,
        X_val: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray,
        eval_gameids: pd.Series,
    ) -> None:
        """
        Plot and save historical accuracy over weekly timespans.

        Args:
            X_val (pd.DataFrame): Validation feature matrix.
            y_true (pd.Series): True labels.
            y_pred (np.ndarray): Predicted labels.
            eval_gameids (pd.Series): Series of game IDs.

        Raises:
            ValueError: If plotting fails.

        """
        try:
            X_val = X_val.copy()
            X_val["gameid"] = eval_gameids

            if "date" not in X_val.columns:
                team_data = pd.read_parquet(PROCESSED_TEAMS, engine="fastparquet")[
                    ["gameid", "date"]
                ].drop_duplicates()
                X_val = pd.merge(
                    X_val, team_data, on="gameid", how="left", validate="m2m"
                ).reset_index(drop=True)

            df = pd.DataFrame(
                {"date": X_val["date"], "correct": (y_true == y_pred).astype(int)}
            )
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            df["week_start"] = (
                df["date"].dt.to_period("W").apply(lambda r: r.start_time)
            )
            df = df.sort_values(by="week_start")
            df_grouped = (
                df.groupby("week_start")["correct"].mean().reset_index(name="accuracy")
            )
            df_grouped["week_start"] = pd.to_datetime(df_grouped["week_start"])

            _, ax = plt.subplots(figsize=(15, 8))
            sns.lineplot(
                data=df_grouped,
                x="week_start",
                y="accuracy",
                marker="o",
                linestyle="--",
                ax=ax,
                label="Weekly Accuracy",
                color="#84C3FA",
            )

            plt.axhline(y=0.5, color="gray", linestyle="--", label="50% Accuracy")

            polynomial_degree = 3
            z = np.polyfit(
                mdates.date2num(df_grouped["date"]),
                df_grouped["accuracy"],
                polynomial_degree,
            )
            p = np.poly1d(z)
            plt.plot(
                df_grouped["date"],
                p(mdates.date2num(df_grouped["date"])),
                "r--",
                label="Trend Line",
            )

            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.xticks(rotation=90, color="white")
            ax.set_xlabel("Date", color="white")
            ax.set_ylabel("Accuracy", color="white")
            ax.grid(True, linestyle="--", alpha=0.6, axis="y")
            ax.grid(False, axis="x")
            plt.gca().set_facecolor("none")
            plt.legend(facecolor="white", edgecolor="none")
            plt.tight_layout()
            ax.tick_params(axis="x", colors="white")
            ax.tick_params(axis="y", colors="white")

            plt.savefig(
                self.directory.joinpath(f"{self.model_name}_Historical_Accuracy.png"),
                dpi=300,
                transparent=True,
            )
            plt.close()
            logger.info(f"Historical accuracy plot for {self.model_name} stored.")
        except Exception as e:
            logger.error(f"Failed to plot historical accuracy: {e}")
            msg = f"Failed to plot historical accuracy: {e}"
            raise ValueError(msg) from e

    def plot_regression_error_over_time(
        self,
        X_val: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray,
        eval_gameids: pd.Series,
    ) -> None:
        """
        Plots the regression error (MAE) over time.

        Args:
            X_val (pd.DataFrame): Validation feature matrix.
            y_true (pd.Series): True values.
            y_pred (np.ndarray): Predicted values.
            eval_gameids (pd.Series): Series of game IDs.

        Raises:
            ValueError: If plotting fails.

        """
        try:
            X_val = X_val.copy()
            X_val["gameid"] = eval_gameids

            if "date" not in X_val.columns:
                team_data = pd.read_parquet(PROCESSED_TEAMS, engine="fastparquet")[
                    ["gameid", "date"]
                ].drop_duplicates()
                X_val = pd.merge(
                    X_val, team_data, on="gameid", how="left", validate="m2m"
                ).reset_index(drop=True)

            df = pd.DataFrame(
                {
                    "date": X_val["date"].values,
                    "true": y_true.values,
                    "predicted": y_pred,
                }
            )
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            df["week_start"] = (
                df["date"].dt.to_period("W").apply(lambda r: r.start_time)
            )
            df = df.sort_values(by="week_start")
            df_grouped = df.groupby("week_start").apply(
                lambda x: np.mean(np.abs(x["true"] - x["predicted"]))
            )
            df_grouped = df_grouped.reset_index(name="mae")
            df_grouped["week_start"] = pd.to_datetime(df_grouped["week_start"])

            # Plotting the MAE over time
            _, ax = plt.subplots(figsize=(15, 8))
            sns.lineplot(
                data=df_grouped,
                x="week_start",
                y="mae",
                marker="o",
                linestyle="--",
                ax=ax,
                label="Weekly MAE",
                color="#84C3FA",
            )

            plt.axhline(
                y=df_grouped["mae"].mean(),
                color="gray",
                linestyle="--",
                label="Average MAE",
            )

            polynomial_degree = 3
            z = np.polyfit(
                mdates.date2num(df_grouped["date"]),
                df_grouped["mae"],
                polynomial_degree,
            )
            p = np.poly1d(z)
            plt.plot(
                df_grouped["date"],
                p(mdates.date2num(df_grouped["date"])),
                "r--",
                label="Trend Line",
            )

            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.xticks(rotation=90, color="white")
            ax.set_xlabel("Date", color="white")
            ax.set_ylabel("MAE", color="white")
            ax.grid(True, linestyle="--", alpha=0.6, axis="y")
            ax.grid(False, axis="x")
            plt.gca().set_facecolor("none")
            plt.legend(facecolor="white", edgecolor="none")
            plt.tight_layout()
            ax.tick_params(axis="x", colors="white")
            ax.tick_params(axis="y", colors="white")

            plt.savefig(
                self.directory.joinpath(
                    f"{self.model_name}_Historical_MAE_Over_Time.png"
                ),
                dpi=300,
                transparent=True,
            )
            plt.close()
            logger.info(f"Historical MAE over time plot for {self.model_name} stored.")
        except Exception as e:
            logger.error(f"Failed to plot regression error over time: {e}")
            msg = f"Failed to plot regression error over time: {e}"
            raise ValueError(msg) from e

    def store_feature_importance(self, model, feature_names: list[str]) -> None:
        """
        Store feature importance to a parquet file and plot as a PNG file.

        Args:
            model: Trained model.
            feature_names (List[str]): List of feature names.

        Raises:
            ValueError: If storing fails.

        """
        try:
            importances = model.feature_importances_
            sorted_importances = sorted(
                zip(feature_names, importances, strict=False),
                key=lambda x: x[1],
                reverse=True,
            )

            data = [
                {"Feature": name, "Importance": importance}
                for name, importance in sorted_importances
            ]
            df = pd.DataFrame(data)
            df.to_parquet(
                INSIGHTS_DIR / f"{self.model_name}_feature_importances.parquet",
                index=False,
                compression="gzip",
            )

            self.plot_feature_importance(sorted_importances)
            logger.info(f"Feature importance for {self.model_name} stored.")
        except Exception as e:
            logger.error(f"Failed to store feature importance: {e}")
            msg = f"Failed to store feature importance: {e}"
            raise ValueError(msg) from e

    def plot_feature_importance(
        self, sorted_importances: list[tuple[str, float]], top_n: int = TOP_N_FEATURES
    ) -> None:
        """
        Plots the feature importances, showing only the top_n features.

        Args:
            sorted_importances (List[Tuple[str, float]]): List of tuples with feature names and importances.
            top_n (int): Number of top features to plot.

        Raises:
            ValueError: If plotting fails.

        """
        try:
            top_features = sorted_importances[:top_n]
            features, importances = zip(*top_features, strict=False)
            plt.figure(figsize=(10, 8))
            plt.barh(features, importances)
            plt.xlabel("Feature Importance")
            plt.title(f"Top {top_n} Feature Importances")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(
                FEATURE_IMP_DIR / f"{self.model_name}_feature_importance_plot.png"
            )
            plt.close()
            logger.info(
                f"Top {top_n} feature importances for {self.model_name} stored."
            )
        except Exception as e:
            logger.error(f"Failed to plot feature importance: {e}")
            msg = f"Failed to plot feature importance: {e}"
            raise ValueError(msg) from e

    def calculate_permutation_importance(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: list[str],
        top_n: int = TOP_N_FEATURES,
    ) -> None:
        """
        Calculates and plots permutation importances for the top_n features.

        Args:
            model: Trained model.
            X_test (pd.DataFrame): Test feature matrix.
            y_test (pd.Series): Test target vector.
            feature_names (List[str]): List of feature names.
            top_n (int): Number of top features to plot.

        Raises:
            ValueError: If calculation or plotting fails.

        """
        try:
            result = permutation_importance(
                model, X_test, y_test, n_repeats=10, n_jobs=-1, random_state=42
            )
            sorted_idx = result.importances_mean.argsort()[-top_n:]

            plt.figure(figsize=(10, 8))
            plt.boxplot(
                result.importances[sorted_idx].T,
                vert=False,
                labels=np.array(feature_names)[sorted_idx],
            )
            plt.title(f"Top {top_n} Permutation Importances (test set)")
            plt.tight_layout()
            plt.savefig(
                FEATURE_IMP_DIR / f"{self.model_name}_permutation_importance_plot.png"
            )
            plt.close()
            logger.info(f"Permutation importance plot for {self.model_name} stored.")
        except Exception as e:
            logger.error(f"Failed to calculate or plot permutation importance: {e}")
            msg = f"Failed to calculate or plot permutation importance: {e}"
            raise ValueError(msg) from e

    def calculate_and_plot_shap(
        self,
        model,
        X: pd.DataFrame,
        feature_names: list[str],
        top_n: int = TOP_N_FEATURES,
    ) -> None:
        """
        Plots SHAP values for the top_n features.

        Args:
            model: Trained model.
            X (pd.DataFrame): Feature matrix.
            feature_names (List[str]): List of feature names.
            top_n (int): Number of top features to plot.

        Raises:
            ValueError: If calculation or plotting fails.

        """
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            if self.problem_type == "classification":
                if isinstance(shap_values, list):
                    # For binary classification, select the second class
                    shap_values = shap_values[1]
            elif self.problem_type == "regression":
                pass  # shap_values is already correct
            else:
                logger.warning(
                    f"Unknown problem type '{self.problem_type}' for SHAP values."
                )
                shap_values = np.array([])

            if shap_values.size == 0:
                logger.warning("No SHAP values calculated.")
                return

            mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
            top_features_idx = np.argsort(mean_abs_shap_values)[-top_n:]
            top_feature_names = [feature_names[i] for i in top_features_idx]

            shap.summary_plot(
                shap_values[:, top_features_idx],
                X.iloc[:, top_features_idx],
                feature_names=top_feature_names,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(FEATURE_IMP_DIR / f"{self.model_name}_shap_summary_plot.png")
            plt.close()
            logger.info(f"SHAP summary plot for {self.model_name} stored.")
        except Exception as e:
            logger.error(f"Failed to calculate or plot SHAP values: {e}")
            msg = f"Failed to calculate or plot SHAP values: {e}"
            raise ValueError(msg) from e
