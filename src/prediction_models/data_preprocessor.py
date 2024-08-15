import pandas as pd

from src.utils.logger import logger
from src.utils.paths import TARGET_FEATURES
from src.utils.utils import json_loader


class DataPreprocessor:
    def __init__(self, team_data: pd.DataFrame, player_data: pd.DataFrame):
        self.team_data = team_data
        self.player_data = player_data
        self.training_data = None

    def preprocess(self, target_col: str, problem_type: str = "classification") -> pd.DataFrame:
        """Preprocess data by removing unnecessary targets, pivoting player data, and merging datasets."""
        self._remove_other_targets(target_col)
        self._pivot_player_data()
        self._merge_datasets()

        if problem_type == "regression":
            self._handle_regression_specifics(target_col)

        return self.training_data

    def _remove_other_targets(self, target_col: str) -> None:
        """Remove other target columns from the dataset."""
        try:
            target_cols = json_loader(TARGET_FEATURES)["targets"]
            if target_col in target_cols:
                target_cols.remove(target_col)
            self.team_data.drop(columns=target_cols, inplace=True)
        except KeyError as e:
            logger.error(f"Error removing target columns: {e}")

    def _pivot_player_data(self) -> None:
        """Pivot player data to create features for each position."""
        numeric_cols = self.player_data.select_dtypes(include=["number"]).columns
        non_numeric_cols = self.player_data.columns.difference(numeric_cols)
        agg_funcs = dict.fromkeys(numeric_cols, "mean")
        agg_funcs.update({col: "first" for col in non_numeric_cols if col not in ["position"]})

        self.player_data = self.player_data.pivot_table(
            index=["gameid", "side"], columns="position", aggfunc=agg_funcs, fill_value=0
        )
        self.player_data.columns = [f"{col[1]}_{col[0]}" for col in self.player_data.columns]
        self.player_data.reset_index(inplace=True)

    def _merge_datasets(self) -> None:
        """
        Merge team and player datasets.
        Reminder: I don't drop gameid here because it's needed for plotting later.
        """
        drop_cols = (
            [f"{pos}_gameid" for pos in ["top", "jng", "mid", "bot", "sup"]]
            + [f"{pos}_side" for pos in ["top", "jng", "mid", "bot", "sup"]]
            + ["date"]
        )
        self.training_data = pd.merge(
            self.team_data, self.player_data, on=["gameid", "side"], how="inner", validate="many_to_many"
        )
        self.training_data.sort_values(by=["date", "gameid", "side"], inplace=True)
        self.training_data.drop(columns=drop_cols, inplace=True)

    def _handle_regression_specifics(self, target_col: str) -> None:
        """Handle preprocessing steps specific to regression tasks."""
        # Example: Ensure the target variable is numerical and handle outliers.
        if not pd.api.types.is_numeric_dtype(self.training_data[target_col]):
            self.training_data[target_col] = pd.to_numeric(self.training_data[target_col], errors="coerce")
        # Example: Handling outliers by clipping.
        self.training_data[target_col] = self.training_data[target_col].clip(
            lower=self.training_data[target_col].quantile(0.01), upper=self.training_data[target_col].quantile(0.99)
        )

        logger.info(f"Applied regression-specific preprocessing for target: {target_col}")
