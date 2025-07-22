"""A module for preprocessing team and player data for training models."""

from typing import Any

import pandas as pd

from src.utils.logger import logger
from src.utils.paths import TARGET_FEATURES
from src.utils.utils import json_loader


class DataPreprocessor:
    """A class to preprocess team and player data for training models."""

    def __init__(self, team_data: pd.DataFrame, player_data: pd.DataFrame):
        """
        Initialize the DataPreprocessor with team and player data.

        Args:
            team_data (pd.DataFrame): DataFrame containing team-level data.
            player_data (pd.DataFrame): DataFrame containing player-level data.

        """
        self.team_data = team_data.copy()
        self.player_data = player_data.copy()
        self.training_data: pd.DataFrame | None = None

        self.merge_keys = ["gameid", "side"]

    def preprocess(
        self, target_col: str, problem_type: str = "classification"
    ) -> pd.DataFrame:
        """
        Preprocess data by removing unnecessary targets, pivoting player data, and merging datasets.

        Args:
            target_col (str): The target column to retain in the team data.
            problem_type (str): The type of problem ('classification' or 'regression'). Defaults to 'classification'.

        Returns:
            pd.DataFrame: The preprocessed training data.

        """
        self._remove_other_targets(target_col)
        self._pivot_player_data()
        self._merge_datasets()

        if problem_type == "regression":
            self._handle_regression_specifics(target_col)

        return self.training_data

    def _remove_other_targets(self, target_col: str) -> None:
        """
        Remove other target columns from the team data, retaining only the specified target column.

        Args:
            target_col (str): The target column to retain.

        """
        try:
            targets_config: dict[str, Any] = json_loader(TARGET_FEATURES)
            target_cols = targets_config.get("targets", [])
            if target_col in target_cols:
                target_cols.remove(target_col)
            self.team_data = self.team_data.drop(columns=target_cols, errors="ignore")
            logger.info(f"Removed target columns except for '{target_col}'.")
        except FileNotFoundError as e:
            logger.error(f"Target features configuration file not found: {e}")
            msg = f"Configuration file '{TARGET_FEATURES}' not found."
            raise FileNotFoundError(msg) from e
        except Exception as e:
            logger.error(f"Error removing target columns: {e}")
            raise

    def _pivot_player_data(self) -> None:
        """Pivot player data to create features for each position."""
        if "position" not in self.player_data.columns:
            logger.error("Column 'position' not found in player data.")
            msg = "Column 'position' is required in player data."
            raise ValueError(msg)

        numeric_cols = self.player_data.select_dtypes(
            include=["number"]
        ).columns.tolist()
        non_numeric_cols = self.player_data.columns.difference(numeric_cols).tolist()
        if "position" in non_numeric_cols:
            non_numeric_cols.remove(
                "position"
            )  # Exclude 'position' from non-numeric columns

        agg_funcs = {col: "mean" for col in numeric_cols}
        agg_funcs.update(
            {col: "first" for col in non_numeric_cols if col != "position"}
        )

        self.player_data = self.player_data.pivot_table(
            index=self.merge_keys, columns="position", aggfunc=agg_funcs, fill_value=0
        )
        self.player_data.columns = [
            f"{position}_{col}" for col, position in self.player_data.columns
        ]
        self.player_data = self.player_data.reset_index()
        logger.info("Pivoted player data to create position-based features.")

    def _merge_datasets(self) -> None:
        """
        Merge team and player datasets to create the training data.
        Reminder: 'gameid' is retained for plotting purposes.
        """
        try:
            self.training_data = pd.merge(
                self.team_data,
                self.player_data,
                on=self.merge_keys,
                how="inner",
                validate="m:1",
            )
            if "date" in self.training_data.columns:
                self.training_data = self.training_data.sort_values(
                    by=["date", *self.merge_keys]
                )
            logger.info("Merged team and player data successfully.")
        except Exception as e:
            logger.error(f"Error merging datasets: {e}")
            raise

        drop_cols = [
            f"{pos}_{key}"
            for pos in ["top", "jng", "mid", "bot", "sup"]
            for key in self.merge_keys
        ]
        drop_cols.append("date")

        self.training_data = self.training_data.drop(columns=drop_cols, errors="ignore")
        logger.info(f"Dropped unnecessary columns: {drop_cols}")

    def _handle_regression_specifics(self, target_col: str) -> None:
        """
        Handle preprocessing steps specific to regression tasks.

        Args:
            target_col (str): The target column for regression.

        """
        if target_col not in self.training_data.columns:
            logger.error(f"Target column '{target_col}' not found in training data.")
            msg = f"Target column '{target_col}' is required in training data."
            raise ValueError(msg)

        # Ensure the target variable is numerical
        self.training_data[target_col] = pd.to_numeric(
            self.training_data[target_col], errors="coerce"
        )

        # Drop rows with NaN in the target column
        initial_row_count = len(self.training_data)
        self.training_data = self.training_data.dropna(subset=[target_col])
        dropped_rows = initial_row_count - len(self.training_data)
        if dropped_rows > 0:
            logger.warning(
                f"Dropped {dropped_rows} rows due to NaN in target column '{target_col}'."
            )

        # Handle outliers by removing values below the 0.1 quantile and above the 0.99 quantile
        lower_bound = self.training_data[target_col].quantile(0.01)
        upper_bound = self.training_data[target_col].quantile(0.99)

        # Filter the data to keep only the rows where the target column is within the specified bounds
        self.training_data = self.training_data[
            (self.training_data[target_col] >= lower_bound)
            & (self.training_data[target_col] <= upper_bound)
        ]

        logger.info(
            f"Applied regression-specific preprocessing for target: '{target_col}'."
        )
