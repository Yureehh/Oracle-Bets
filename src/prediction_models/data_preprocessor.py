"""Preprocessing utilities for joining team & player training tables."""

from __future__ import annotations

from typing import Any

import pandas as pd

from utils.io_utils import json_loader
from utils.logger import logger
from utils.paths import TARGET_FEATURES


class DataPreprocessor:
    """
    Prepare a single training table by:
      1) removing non-active targets from team data,
      2) pivoting player rows to position-wide numeric features,
      3) merging team + pivoted player tables on (gameid, side),
      4) (optional) regression-only target cleaning.
    """

    def __init__(self, team_data: pd.DataFrame, player_data: pd.DataFrame) -> None:
        self.team_data = team_data.copy()
        self.player_data = player_data.copy()
        self.training_data: pd.DataFrame | None = None

        # We always merge on these; keep naming aligned with upstream pipeline.
        self.merge_keys: list[str] = ["gameid", "side"]

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────
    def preprocess(
        self, target_col: str, problem_type: str = "classification"
    ) -> pd.DataFrame:
        """
        Produce a model-ready table.

        Parameters
        ----------
        target_col : str
            The target to keep in the team table (others removed via config).
        problem_type : {"classification","regression"}
            Applies a small, opt-in clean-up for regression targets.

        Returns
        -------
        pd.DataFrame
            Sorted by date (if present), one row per (gameid, side).

        """
        self._drop_inactive_targets(target_col)
        self._pivot_player_numeric()
        self._merge_team_and_players()

        if problem_type == "regression":
            self._handle_regression_specifics(target_col)

        return self.training_data  # type: ignore[return-value]

    # ──────────────────────────────────────────────────────────────────────
    # Steps
    # ──────────────────────────────────────────────────────────────────────
    def _drop_inactive_targets(self, target_col: str) -> None:
        """Remove all configured targets except the one we’re modeling."""
        try:
            cfg: dict[str, Any] = json_loader(TARGET_FEATURES)
        except FileNotFoundError:
            logger.error("Target features configuration file not found.")
            raise

        all_targets = set(map(str, cfg.get("targets", [])))
        to_drop = list(all_targets - {target_col})
        # Drop quietly if any are already absent
        self.team_data = self.team_data.drop(columns=to_drop, errors="ignore")
        logger.info(
            "Dropped inactive target columns: %s", ", ".join(sorted(to_drop)) or "none"
        )

        if target_col not in self.team_data.columns:
            msg = (
                f"Active target '{target_col}' not found in team data. "
                f"Available columns: {len(self.team_data.columns)}"
            )
            raise ValueError(msg)

    def _pivot_player_numeric(self) -> None:
        """
        Pivot player rows into wide features per position.

        Only numeric columns are aggregated (mean). This avoids mixing
        strings/categoricals into the pivot and keeps the table compact.
        """
        for key in (*self.merge_keys, "position"):
            if key not in self.player_data.columns:
                msg = f"Column '{key}' required in player data."
                raise ValueError(msg)

        # Keep only numeric columns for aggregation
        numeric_cols = self.player_data.select_dtypes(
            include=["number"]
        ).columns.tolist()
        if not numeric_cols:
            msg = "No numeric columns found in player data to pivot."
            raise ValueError(msg)

        # If multiple rows exist per (gameid, side, position), average them beforehand
        grp_keys = [*self.merge_keys, "position"]
        base = (
            self.player_data[grp_keys + numeric_cols]
            .groupby(grp_keys, observed=True, sort=False)
            .mean()
        )

        # Wide pivot via unstack on 'position'
        wide = base.unstack(  # noqa: PD010
            "position"
        )  # MultiIndex columns: (col, position)
        # Normalize columns to "<pos>_<col>"
        wide.columns = [f"{pos}_{col}" for col, pos in wide.columns.to_flat_index()]
        self.player_data = wide.reset_index()

        logger.info(
            "Pivoted player data into wide format: %s columns.",
            f"{len(self.player_data.columns):,}",
        )

    def _merge_team_and_players(self) -> None:
        """
        Merge team_data with pivoted player_data.

        Ensures one row per (gameid, side) and sorts by date if present.
        """
        if any(k not in self.team_data.columns for k in self.merge_keys):
            missing = [k for k in self.merge_keys if k not in self.team_data.columns]
            msg = f"Team data missing merge keys: {missing}"
            raise ValueError(msg)

        merged = self.team_data.merge(
            self.player_data,
            on=self.merge_keys,
            how="inner",
            validate="m:1",  # team rows m, one player-wide row per (gameid, side)
        )

        # Sort deterministically if date provided
        if "date" in merged.columns:
            merged = merged.sort_values(by=["date", *self.merge_keys], kind="mergesort")

        # Sanity: unique (gameid, side)  # noqa: ERA001
        dup_mask = merged.duplicated(self.merge_keys, keep=False)
        if dup_mask.any():
            dup_count = dup_mask.sum()
            logger.warning(
                "Found %d duplicated (gameid, side) rows after merge; keeping all. "
                "Downstream splitter should handle this.",
                dup_count,
            )

        self.training_data = merged
        logger.info(
            "Merged team + player features: %s rows, %s columns.",
            f"{len(self.training_data):,}",
            f"{self.training_data.shape[1]:,}",
        )

        # Drop position-propagated merge keys or date if any slipped through during pivot
        # (Mostly a no-op with the numeric-only pivot, but harmless.)
        drop_cols = [
            f"{pos}_{key}"
            for pos in ["top", "jng", "mid", "bot", "sup"]
            for key in self.merge_keys
        ]
        self.training_data = self.training_data.drop(columns=drop_cols, errors="ignore")
        # keep 'date' so temporal split works without reattach
        self.training_data = self.training_data.drop(columns=drop_cols, errors="ignore")

    def _handle_regression_specifics(self, target_col: str) -> None:
        """
        Light regression cleaning:
          - numeric cast with coercion
          - drop NaNs in target
          - keep central 1–99% quantile range (remove extreme outliers)
        """
        if self.training_data is None:
            msg = "Call preprocess() before regression handling."
            raise RuntimeError(msg)

        if target_col not in self.training_data.columns:
            msg = f"Target column '{target_col}' not found after merge."
            raise ValueError(msg)

        # Force numeric target
        self.training_data[target_col] = pd.to_numeric(
            self.training_data[target_col], errors="coerce"
        )

        # Drop NaN targets
        before = len(self.training_data)
        self.training_data = self.training_data.dropna(subset=[target_col])
        removed_nan = before - len(self.training_data)
        if removed_nan:
            logger.warning(
                "Dropped %d rows with NaN in target '%s'.", removed_nan, target_col
            )

        # Remove extreme outliers (1st–99th percentiles)
        q_low, q_high = self.training_data[target_col].quantile([0.01, 0.99])
        keep_mask = self.training_data[target_col].between(
            q_low, q_high, inclusive="both"
        )
        kept = int(keep_mask.sum())
        dropped = len(self.training_data) - kept
        if dropped:
            logger.warning(
                "Removed %d outlier rows outside [%.3f, %.3f] for '%s'.",
                dropped,
                q_low,
                q_high,
                target_col,
            )
        self.training_data = self.training_data.loc[keep_mask].reset_index(drop=True)

        logger.info("Regression-specific preprocessing complete for '%s'.", target_col)
