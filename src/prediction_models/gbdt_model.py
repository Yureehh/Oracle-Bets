"""
Gradient Boosting Decision Tree model.

This defines an abstract class for Gradient Boosting Decision Tree models to extend from.
It provides methods for preprocessing data, training the model, optimizing hyperparameters,
storing best hyperparameters, and storing categorical features.
"""

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from utils.logger import logger, models_logger
from utils.paths import FEATURE_IMP_DIR, FIGURES_DIR, INSIGHTS_DIR, MODELS_DIR, PROCESSED_TEAMS

sns.set_style("darkgrid")

# Constants
DEFAULT_TRIALS = 1000
VALIDATION_SIZE = 0.15
TOP_N_FEATURES = 25
LOW_STD_THRESHOLD = 0.05
HIGH_CORR_THRESHOLD = 0.90


@dataclass
class GradientBoostingModel(ABC):
    model_name: str = field(default="")
    team_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    training_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    trials: int = field(default=DEFAULT_TRIALS)
    directory: Path = field(default=FIGURES_DIR)

    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess the data by pivoting player data and merging datasets."""
        self.pivot_player_data()
        self.merge_datasets()
        return self.training_data

    def pivot_player_data(self) -> None:
        """Pivot player data to create features for each position."""
        numeric_cols = self.player_data.select_dtypes(include=["number"]).columns
        non_numeric_cols = self.player_data.columns.difference(numeric_cols)
        agg_funcs = {col: "mean" for col in numeric_cols}
        agg_funcs.update({col: "first" for col in non_numeric_cols if col not in ["position"]})
        self.player_data = self.player_data.pivot_table(
            index=["gameid", "side"], columns="position", aggfunc=agg_funcs, fill_value=0
        )
        self.player_data.columns = [f"{col[1]}_{col[0]}" for col in self.player_data.columns]
        self.player_data.reset_index(inplace=True)

    def merge_datasets(self) -> None:
        """Merge team and player datasets.
        Remember: I don't drop gameid here because I use it later for plotting daily accuracy
        """
        self.training_data = pd.merge(
            self.team_data, self.player_data, on=["gameid", "side"], how="inner", validate="many_to_many"
        )
        drop_cols = [f"{pos}_gameid" for pos in ["top", "jng", "mid", "bot", "sup"]] + [
            f"{pos}_side" for pos in ["top", "jng", "mid", "bot", "sup"]
        ]
        self.training_data.drop(columns=drop_cols, inplace=True)

    def store_model_features(self, all_features: pd.Index) -> None:
        """Store all features to a pickle file."""
        self._store_pickle(f"{self.model_name}_final_features.pkl", all_features)

    def store_categorical_features(self, categorical_features: list) -> None:
        """Store categorical features to a pickle file."""
        self._store_pickle(f"{self.model_name}_categorical_features.pkl", categorical_features)

    def _store_pickle(self, filename: str, data: Any) -> None:
        try:
            with open(MODELS_DIR / filename, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to store {filename}: {e}")

    @staticmethod
    def grouped_stratified_train_val_test_split(
        X: pd.DataFrame, y: pd.Series, groups: pd.Series, stratify: pd.Series, val_size: float, test_size: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Splits the data into training, validation, and test sets, grouped by specified columns and stratified by a specified column.
        """

        def stratified_split(temp_df, n_splits, groups_col, stratify_col):
            strat_split = StratifiedGroupKFold(n_splits=n_splits, shuffle=True)
            return next(strat_split.split(temp_df[groups_col], temp_df[stratify_col], groups=temp_df[groups_col]))

        temp_df = pd.DataFrame({"group": groups, "stratify": stratify}).drop_duplicates()

        train_val_idx, test_idx = stratified_split(temp_df, int(1 / test_size), "group", "stratify")

        train_val_groups = temp_df["group"].iloc[train_val_idx]
        test_groups = temp_df["group"].iloc[test_idx]

        X_train_val, X_test = X[X["gameid"].isin(train_val_groups)], X[X["gameid"].isin(test_groups)]
        y_train_val, y_test = y[X["gameid"].isin(train_val_groups)], y[X["gameid"].isin(test_groups)]

        temp_df_train_val = pd.DataFrame(
            {"group": X_train_val["gameid"], "stratify": X_train_val["league"]}
        ).drop_duplicates()

        train_idx, val_idx = stratified_split(temp_df_train_val, int(1 / val_size), "group", "stratify")

        train_groups = temp_df_train_val["group"].iloc[train_idx]
        val_groups = temp_df_train_val["group"].iloc[val_idx]

        X_train, X_val = (
            X_train_val[X_train_val["gameid"].isin(train_groups)],
            X_train_val[X_train_val["gameid"].isin(val_groups)],
        )
        y_train, y_val = (
            y_train_val[X_train_val["gameid"].isin(train_groups)],
            y_train_val[X_train_val["gameid"].isin(val_groups)],
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def fuse_opposing_team_features(X: pd.DataFrame) -> pd.DataFrame:
        """Fuses opposing team features by subtracting the opposing stats from the original stats."""
        for col in X.filter(like="opp_").columns:
            original_col = col[4:]
            if original_col in X.columns:
                X[original_col] -= X[col]
                X.drop(columns=[col], inplace=True)

        roles = ["top", "jng", "mid", "bot", "sup"]
        for role in roles:
            for col in X.filter(like=f"{role}_opp_").columns:
                original_col = col.replace(f"{role}_opp_", f"{role}_")
                if original_col in X.columns:
                    X[original_col] -= X[col]
                    X.drop(columns=[col], inplace=True)

        return X

    @staticmethod
    def process_players_likelihood_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Processes player likelihood columns by renaming and dropping unnecessary ones."""
        likelihood_columns = df.columns[df.columns.str.contains("likelihood")]
        modified_df = df[likelihood_columns].rename(columns=lambda x: x.replace("top_", "players_"))

        for role in ["jng_", "mid_", "bot_", "sup_"]:
            modified_df.drop(columns=[col for col in modified_df.columns if col.startswith(role)], inplace=True)

        df = df.drop(columns=likelihood_columns, inplace=False)
        return pd.concat([df, modified_df], axis=1)

    @staticmethod
    def remove_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Removes columns that are unnecessary or degrade model performance."""
        df = df.loc[:, ~df.columns.str.contains("_std")]
        df = df.loc[:, ~df.columns.str.contains("season_win_likelihood")]
        df = GradientBoostingModel.drop_low_std_columns(df, LOW_STD_THRESHOLD)
        df = GradientBoostingModel.drop_highly_correlated_features(df, HIGH_CORR_THRESHOLD)
        return df

    @staticmethod
    def drop_low_std_columns(df: pd.DataFrame, threshold: float = LOW_STD_THRESHOLD) -> pd.DataFrame:
        """Drops columns with low standard deviation."""
        mean_values = df.mean()
        std_dev = df.std()
        coef_of_variation = std_dev / mean_values

        columns_to_drop = coef_of_variation[coef_of_variation < threshold].index
        df_dropped = df.drop(columns=columns_to_drop, inplace=False)
        logger.info(f"Dropped columns with coefficient of variation below {threshold * 100}%: {list(columns_to_drop)}")
        models_logger.info(
            f"Dropped columns with coefficient of variation below {threshold * 100}%: {list(columns_to_drop)}"
        )
        return df_dropped

    @staticmethod
    def drop_highly_correlated_features(df: pd.DataFrame, threshold: float = HIGH_CORR_THRESHOLD) -> pd.DataFrame:
        """Drops highly correlated features."""
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        df_dropped = df.drop(columns=to_drop)
        logger.info(f"Dropped columns due to high correlation: {list(to_drop)}")
        models_logger.info(f"Dropped columns due to high correlation: {list(to_drop)}")
        return df_dropped

    def store_correlation(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Stores the correlation matrix as a heatmap."""
        df = pd.concat([X, y], axis=1)
        corr_matrix = df.corr()

        plt.figure(figsize=(20, 16))
        sns.heatmap(corr_matrix, cmap="coolwarm", cbar=True)
        plt.title("Correlation Matrix Heatmap")
        plt.savefig(self.directory.joinpath(f"{self.model_name}_Correlation_Matrix.png"), dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Correlation matrix plot for {self.model_name} stored.")

    @abstractmethod
    def train_model(self) -> None:
        """Abstract method to train the model."""
        pass

    @abstractmethod
    def optimize_hyperparameters(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        """Abstract method to optimize hyperparameters."""
        pass

    @staticmethod
    def preprocess_categorical_features(
        X: pd.DataFrame, exclude_cols: List[str] = None, categorical_columns: List[str] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Preprocess features by converting categorical columns to category dtype."""
        if categorical_columns is None:
            categorical_columns = [col for col in X.columns if X[col].dtype == "object"]

        for col in categorical_columns:
            X[col] = X[col].astype("category")

        if exclude_cols:
            for col in exclude_cols:
                if col in categorical_columns:
                    categorical_columns.remove(col)

        consistent_cats = {col: X[col].cat.categories for col in categorical_columns}
        for col, cats in consistent_cats.items():
            X[col] = X[col].cat.set_categories(cats)

        return X, categorical_columns

    def store_best_hyperparameters(self, study) -> None:
        """Store best hyperparameters to a pickle file."""
        self._store_pickle(f"{self.model_name}_best_hyperparameters.pkl", study.best_params)

    def store_predictions(self, predictions: np.ndarray, eval_gameids: pd.Series, eval_sides: pd.Series) -> None:
        """Store prediction insights to a CSV file."""
        insights = pd.DataFrame({"prediction": predictions, "gameid": eval_gameids, "side": eval_sides})
        try:
            insights.to_parquet(INSIGHTS_DIR / f"{self.model_name}_predictions.parquet", index=False)
            logger.info(f"Prediction insights for {self.model_name} stored.")
        except Exception as e:
            logger.error(f"Failed to store prediction insights: {e}")

    def validate_model(
        self, model, X_test: pd.DataFrame, y_test: pd.Series, eval_gameids: pd.Series, eval_sides: pd.Series
    ) -> None:
        """Validate the model and store evaluation metrics."""
        logger.info("Validating the model...")
        predictions = model.predict(X_test)
        metrics = self.compute_evaluation_metrics(y_test, predictions)
        self.log_evaluation_metrics(metrics)
        self.plot_confusion_matrix(metrics["cm"])
        self.store_evaluation_metrics(metrics)
        self.store_predictions(predictions, eval_gameids, eval_sides)
        self.plot_accuracy_over_samples(y_test, predictions)
        self.plot_historical_accuracy(X_test, y_test, predictions, eval_gameids)
        logger.info(f"Model {self.model_name} validated and insights stored.\n")

    def compute_evaluation_metrics(self, y_val: pd.Series, predictions: np.ndarray) -> Dict[str, Any]:
        """Compute evaluation metrics for the model."""
        accuracy = accuracy_score(y_val, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, predictions, average="binary")
        cm = confusion_matrix(y_val, predictions)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "cm": cm}

    def store_evaluation_metrics(self, metrics: Dict[str, Any]) -> None:
        """Store evaluation metrics to a JSON file."""
        metrics.pop("cm")
        try:
            with open(INSIGHTS_DIR / f"{self.model_name}_metrics.json", "w") as f:
                json.dump(metrics, f)
            logger.info(f"Evaluation metrics for {self.model_name} stored.")
        except OSError as e:
            logger.error(f"Failed to store evaluation metrics: {e}")

    def log_evaluation_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log evaluation metrics."""
        logger.info(
            f"Evaluation Metrics - Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1 Score: {metrics['f1']:.4f}\n"
        )
        models_logger.info(
            f"Evaluation Metrics - Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1 Score: {metrics['f1']:.4f}\n"
        )

    def plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """Plot and save the confusion matrix."""
        cm_df = pd.DataFrame(
            cm, index=["Actual Negative:0", "Actual Positive:1"], columns=["Predict Negative:0", "Predict Positive:1"]
        )
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.savefig(self.directory.joinpath(f"{self.model_name}_Confusion_Matrix.png"), dpi=300)
        plt.close()
        logger.info(f"Confusion matrix plot for {self.model_name} stored.")

    def plot_accuracy_over_samples(self, y_val: pd.Series, predictions: np.ndarray) -> None:
        """Plot and save accuracy over samples."""
        accuracy_timeline = [accuracy_score(y_val[:i], predictions[:i]) for i in range(1, len(y_val) + 1)]
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=range(1, len(y_val) + 1), y=accuracy_timeline, linestyle="--", color="#84C3FA")
        plt.xlabel("Number of Samples", color="white")
        plt.ylabel("Accuracy", color="white")
        plt.grid(True, linestyle="--", alpha=0.6, axis="y")
        plt.grid(False, axis="x")
        plt.gca().set_facecolor("none")
        plt.legend(["Accuracy"], facecolor="none", edgecolor="none")
        plt.tight_layout()
        plt.gca().tick_params(axis="x", colors="white")
        plt.gca().tick_params(axis="y", colors="white")
        plt.savefig(self.directory.joinpath(f"{self.model_name}_Accuracy_Over_Samples.png"), dpi=300, transparent=True)
        plt.close()
        logger.info(f"Accuracy over samples plot for {self.model_name} stored.")

    def plot_historical_accuracy(
        self, X_val: pd.DataFrame, y_val: pd.Series, predictions: np.ndarray, eval_gameids: pd.Series
    ) -> None:
        """Plot and save historical accuracy over weekly timespans and analyze league distribution."""
        X_val = pd.DataFrame(X_val.copy())
        X_val["gameid"] = eval_gameids

        if "date" not in X_val.columns:
            team_data = pd.read_parquet(PROCESSED_TEAMS)[["gameid", "date"]].drop_duplicates()
            X_val = (
                pd.merge(X_val, team_data, on="gameid", how="left", validate="many_to_many")
                .drop("gameid", axis=1)
                .reset_index(drop=True)
            )

        df = pd.DataFrame({"date": X_val["date"].values, "correct": (y_val == predictions).astype(int)})
        df["date"] = pd.to_datetime(df["date"]).dt.to_period("W").apply(lambda r: r.start_time)
        df = df.sort_values(by="date")
        df_grouped = df.groupby("date")["correct"].mean().reset_index(name="accuracy")
        df_grouped["date"] = pd.to_datetime(df_grouped["date"])

        _, ax = plt.subplots(figsize=(15, 8))
        sns.lineplot(
            data=df_grouped,
            x="date",
            y="accuracy",
            marker="o",
            linestyle="--",
            ax=ax,
            label="Weekly Accuracy",
            color="#84C3FA",
        )

        plt.axhline(y=0.5, color="gray", linestyle="--", label="50% Accuracy")

        polynomial_degree = 3
        z = np.polyfit(mdates.date2num(df_grouped["date"]), df_grouped["accuracy"], polynomial_degree)
        p = np.poly1d(z)
        plt.plot(df_grouped["date"], p(mdates.date2num(df_grouped["date"])), "r--", label="Trend Line")

        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=90, color="white")
        ax.set_xlabel("Date", color="white")
        ax.set_ylabel("Accuracy", color="white")
        ax.grid(True, linestyle="--", alpha=0.6, axis="y")
        ax.grid(False, axis="x")
        plt.gca().set_facecolor("none")
        plt.legend(facecolor="none", edgecolor="none")
        plt.tight_layout()
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")

        plt.savefig(self.directory.joinpath(f"{self.model_name}_Historical_Accuracy.png"), dpi=300, transparent=True)
        plt.close()
        logger.info(f"Historical accuracy plot for {self.model_name} stored.")

    def store_feature_importance(self, model, feature_names: List[str]) -> None:
        """Store feature importance to a CSV and plot as a PNG file."""
        importances = model.feature_importances_
        sorted_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

        data = [{"Feature": name, "Importance": importance} for name, importance in sorted_importances]
        df = pd.DataFrame(data)
        df.to_parquet(INSIGHTS_DIR / f"{self.model_name}_feature_importances.parquet", index=False)

        self.plot_feature_importance(sorted_importances)

    def plot_feature_importance(self, sorted_importances: List[Tuple[str, float]], top_n: int = TOP_N_FEATURES) -> None:
        """Plots the feature importances, showing only the top_n features."""
        sorted_importances = sorted_importances[:top_n]
        features, importances = zip(*sorted_importances)
        plt.figure(figsize=(10, 8))
        plt.barh(features, importances)
        plt.xlabel("Feature Importance")
        plt.title(f"Top {top_n} Feature Importances")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(FEATURE_IMP_DIR / f"{self.model_name}_feature_importance_plot.png")
        plt.close()
        logger.info(f"Top {top_n} feature importances for {self.model_name} stored.")

    def calculate_permutation_importance(
        self, model, X_test: pd.DataFrame, y_test: pd.Series, feature_names: List[str], top_n: int = TOP_N_FEATURES
    ) -> None:
        """Calculates and plots permutation importances for the top_n features."""
        result = permutation_importance(model, X_test, y_test, n_repeats=10, n_jobs=-1)
        sorted_idx = result.importances_mean.argsort()[-top_n:]

        plt.figure(figsize=(10, 8))
        plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(feature_names)[sorted_idx])
        plt.title(f"Top {top_n} Permutation Importances (test set)")
        plt.tight_layout()
        plt.savefig(FEATURE_IMP_DIR / f"{self.model_name}_permutation_importance_plot.png")
        plt.close()

    def calculate_and_plot_shap(
        self, model, df: pd.DataFrame, feature_names: List[str], top_n: int = TOP_N_FEATURES
    ) -> None:
        """Plots SHAP values for the top_n features."""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df, from_call=True)

        mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
        top_features = np.argsort(mean_abs_shap_values)[-top_n:]
        top_feature_names = [feature_names[i] for i in top_features]

        shap.summary_plot(
            shap_values[:, top_features], df.iloc[:, top_features], feature_names=top_feature_names, show=False
        )
        plt.savefig(FEATURE_IMP_DIR / f"{self.model_name}_shap_summary_plot.png")
        plt.close()
