"""
Abstract GBM base: preprocessing, splits, feature fusion, pruning, metrics & validation.
Subclasses (e.g., LightGBMModel) only need to implement:
  - train_model(X_train, y_train, X_val, y_val, categorical_features) -> fitted model
  - _optimize_hyperparameters(X_train, y_train, X_val, y_val) -> dict[str, Any]

This base also provides a full train_and_validate_model(...) orchestration that:
  • builds X/y from the preprocessed table
  • performs grouped+stratified splits (gameid, league [+ season if present]) or temporal split
  • prunes features (missingness, low variance, high correlation) using TRAIN ONLY
  • casts categoricals, imputes missing values (median / "Unknown")
  • trains the subclass model, then runs evaluation + observability artifacts
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import pickle
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from prediction_models.data_preprocessor import DataPreprocessor
from prediction_models.observability import MLObservabilityMixin
from utils.logger import logger
from utils.paths import FIGURES_DIR, MODEL_ARTIFACTS, PROCESSED_TEAMS
from utils.pd import pd

if TYPE_CHECKING:
    from pathlib import Path

# Defaults
DEFAULT_TRIALS = 100
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
LOW_STD_THRESHOLD = 0.03  # coefficient of variation threshold (less aggressive)
HIGH_CORR_THRESHOLD = 0.95  # Pearson correlation threshold (keep more features)
MAX_MISSING_FRAC = 0.40  # drop features missing >40% on TRAIN
RANDOM_STATE = 42
MIN_SHAPE_FOR_CORR = 2  # min numeric cols to run high-corr pruning
BINARY_CLASS_UNIQUE_VALUES = 2


@dataclass
class FeaturePipeline:
    """
    Train-time feature decisions (drops, categories, imputations) that can be
    reapplied verbatim to validation/test/inference data to guarantee parity.
    """

    train_columns: list[str]
    categorical_features: list[str]
    categorical_levels: dict[str, list[str]]
    numeric_medians: dict[str, float]
    drop_high_missing: list[str]
    drop_low_variance: list[str]
    drop_high_correlation: list[str]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the stored pipeline to any dataframe (no label leakage)."""
        Xp = X.drop(
            columns=[
                *self.drop_high_missing,
                *self.drop_low_variance,
                *self.drop_high_correlation,
            ],
            errors="ignore",
        ).copy()

        # Ensure every categorical column exists, then coerce unseen levels to "Unknown"
        for col in self.categorical_features:
            if col not in Xp.columns:
                Xp[col] = "Unknown"
        for col, levels in self.categorical_levels.items():
            if col not in Xp.columns:
                Xp[col] = "Unknown"
            vals = pd.Series(Xp[col], index=Xp.index, dtype="object")
            known = pd.Index(levels, dtype="object")
            vals = vals.where(vals.isin(known), "Unknown")
            Xp[col] = pd.Categorical(vals, categories=levels)

        # Add any entirely-missing numeric features with their train medians
        for col, median in self.numeric_medians.items():
            if col not in Xp.columns:
                Xp[col] = median

        Xp = GradientBoostingModel._impute_apply_numeric(Xp, self.numeric_medians)
        Xp = GradientBoostingModel._impute_categorical(Xp, self.categorical_features)
        return GradientBoostingModel._align_like_train(
            self.train_columns,
            Xp,
            categorical_features=self.categorical_features,
        )


@dataclass
class GradientBoostingModel(MLObservabilityMixin, ABC):
    model_name: str
    problem_type: str  # "classification" | "regression"
    team_data: pd.DataFrame
    player_data: pd.DataFrame
    trials: int = DEFAULT_TRIALS
    directory: Path = FIGURES_DIR
    run_id: str = field(
        default_factory=lambda: dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    training_data: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.training_data = pd.DataFrame()

    # ───────────────────────────── Preprocessing ───────────────────────────── #

    def preprocess_data(self, target_col: str) -> pd.DataFrame:
        """Create a model-ready table via DataPreprocessor."""
        try:
            pre = DataPreprocessor(self.team_data, self.player_data)
            self.training_data = pre.preprocess(target_col, self.problem_type)
            logger.info(
                "Preprocessing done: %s rows x %s cols.",
                f"{len(self.training_data):,}",
                f"{self.training_data.shape[1]:,}",
            )
            return self.training_data
        except Exception as e:
            logger.error("Data preprocessing failed: %s", e)
            raise

    # ─────────────────────────── Feature utilities ─────────────────────────── #

    @staticmethod
    def _meta_columns() -> list[str]:
        """Columns that must never be used as features."""
        return [
            "gameid",
            "side",
            "league",
            "season",
            "patch",
            "date",
            "playoffs",
            "game",
            "result",  # typical classification target name
        ]

    @staticmethod
    def fuse_opposing_team_features(df: pd.DataFrame) -> pd.DataFrame:
        """Turn opp_* columns into in-situ diffs (base - opp_base), then drop the opp_* columns."""
        X = df.copy()

        # Team-level opp_* → base - opp_base  (numeric only)
        opp_cols = [c for c in X.columns if c.startswith("opp_")]
        for col in opp_cols:
            base = col[4:]
            if (
                base in X.columns
                and is_numeric_dtype(X[base])
                and is_numeric_dtype(X[col])
            ):
                X[base] = X[base] - X[col]
            # Drop opp_* regardless (strings like opp_side shouldn’t survive)
            X = X.drop(columns=[col], errors="ignore")

        # Role-specific "<role>_opp_*" → "<role>_*" (numeric only)
        roles = ("top", "jng", "mid", "bot", "sup")
        for role in roles:
            prefix = f"{role}_opp_"
            for col in [c for c in X.columns if c.startswith(prefix)]:
                base = f"{role}_{col.split(prefix, 1)[1]}"
                if (
                    base in X.columns
                    and is_numeric_dtype(X[base])
                    and is_numeric_dtype(X[col])
                ):
                    X[base] = X[base] - X[col]
                X = X.drop(columns=[col], errors="ignore")

        return X

    @staticmethod
    def process_players_likelihood_columns(
        df: pd.DataFrame, agg: str = "mean"
    ) -> pd.DataFrame:
        """
        Aggregate per-role *_likelihood columns into players_*_likelihood.
        """
        df = df.copy()
        role_pat = re.compile(r"^(top|jng|mid|bot|sup)_(.+_likelihood)$")
        role_cols = [c for c in df.columns if role_pat.match(c)]
        if not role_cols:
            logger.info(
                "No role-based *_likelihood columns found; skipping aggregation."
            )
            return df

        stems: dict[str, list[str]] = {}
        for c in role_cols:
            m = role_pat.match(c)
            stem = m[2]
            stems.setdefault(stem, []).append(c)

        out = df.drop(columns=role_cols, errors="ignore")
        for stem, cols in stems.items():
            out[f"players_{stem}"] = (
                df[cols].max(axis=1) if agg == "max" else df[cols].mean(axis=1)
            )

        return out

    def _drop_high_missing(
        self, X: pd.DataFrame, threshold: float
    ) -> tuple[pd.DataFrame, list[str]]:
        """Drop columns with missing rate > threshold (computed on TRAIN later)."""
        miss = X.isna().mean()
        drop = miss[miss > threshold].index.tolist()
        X2 = X.drop(columns=drop, errors="ignore")
        if drop:
            logger.info(
                "Dropped %d high-missing columns (>%.0f%%).", len(drop), threshold * 100
            )
        return X2, drop

    def drop_low_std_columns(
        self, df: pd.DataFrame, threshold: float = LOW_STD_THRESHOLD
    ) -> tuple[pd.DataFrame, list[str]]:
        """Remove near-constant numeric columns using coefficient of variation (train only)."""
        df = df.copy()
        num = df.select_dtypes("number")
        if num.empty:
            return df, []
        std = num.std()
        mean = num.mean().replace(0, np.finfo(float).eps)
        cov = std / mean
        drop = cov[cov < threshold].index.tolist()
        df = df.drop(columns=drop, errors="ignore")
        if drop:
            logger.info(
                "Dropped low-variance columns (<%.3f): %d", threshold, len(drop)
            )
        return df, drop

    def drop_highly_correlated_features(
        self, df: pd.DataFrame, threshold: float = HIGH_CORR_THRESHOLD
    ) -> tuple[pd.DataFrame, list[str]]:
        """Greedy drop among highly-correlated numeric features (Pearson)."""
        df = df.copy()
        num = df.select_dtypes("number")
        if num.shape[1] < MIN_SHAPE_FOR_CORR:
            return df, []
        corr = num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop: set[str] = set()
        for col in upper.columns:
            if col in to_drop:
                continue
            hits = [idx for idx in upper.index if upper.loc[idx, col] > threshold]
            to_drop.update(hits)
        df = df.drop(columns=list(to_drop), errors="ignore")
        if to_drop:
            logger.info(
                "Dropped highly-correlated columns (>|%.2f|): %d",
                threshold,
                len(to_drop),
            )
        return df, list(to_drop)

    # ─────────────────────── Grouped, stratified splits ────────────────────── #

    @staticmethod
    def grouped_stratified_train_val_test_split(
        X_with_meta: pd.DataFrame,
        y: pd.Series,
        val_size: float = VALIDATION_SIZE,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        def _one_split(
            groups_df: pd.DataFrame, n_splits: int
        ) -> tuple[np.ndarray, np.ndarray]:
            cv = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            return next(
                cv.split(
                    groups_df[["gameid"]],
                    groups_df["stratify"],
                    groups=groups_df["gameid"],
                )
            )

        required = {"gameid", "league"}
        if not required.issubset(X_with_meta.columns):
            missing = sorted(required - set(X_with_meta.columns))
            msg = f"Missing required columns in X: {missing}"
            raise ValueError(msg)

        has_season = "season" in X_with_meta.columns

        # unique groups + stratify label
        if has_season:
            gdf = X_with_meta[["gameid", "league", "season"]].drop_duplicates()
            gdf["stratify"] = (
                gdf["league"].astype(str) + "_" + gdf["season"].astype(str)
            )
        else:
            gdf = X_with_meta[["gameid", "league"]].drop_duplicates()
            gdf["stratify"] = gdf["league"].astype(str)

        # test split
        n_test = max(2, round(1 / test_size))
        trainval_idx, test_idx = _one_split(gdf, n_test)
        trainval_gids = set(gdf.iloc[trainval_idx]["gameid"])
        test_gids = set(gdf.iloc[test_idx]["gameid"])

        X_trainval = X_with_meta[X_with_meta["gameid"].isin(trainval_gids)]
        y_trainval = y.loc[X_trainval.index]
        X_test = X_with_meta[X_with_meta["gameid"].isin(test_gids)]
        y_test = y.loc[X_test.index]

        # validation split within trainval
        if has_season:
            gdf_tv = X_trainval[["gameid", "league", "season"]].drop_duplicates()
            gdf_tv["stratify"] = (
                gdf_tv["league"].astype(str) + "_" + gdf_tv["season"].astype(str)
            )
        else:
            gdf_tv = X_trainval[["gameid", "league"]].drop_duplicates()
            gdf_tv["stratify"] = gdf_tv["league"].astype(str)

        n_val = max(2, round(1 / val_size))
        tv_train_idx, tv_val_idx = _one_split(gdf_tv, n_val)
        train_gids = set(gdf_tv.iloc[tv_train_idx]["gameid"])
        val_gids = set(gdf_tv.iloc[tv_val_idx]["gameid"])

        X_train = X_trainval[X_trainval["gameid"].isin(train_gids)]
        X_val = X_trainval[X_trainval["gameid"].isin(val_gids)]
        y_train = y_trainval.loc[X_train.index]
        y_val = y_trainval.loc[X_val.index]

        logger.info(
            "Split -> train: %d, val: %d, test: %d rows",
            len(X_train),
            len(X_val),
            len(X_test),
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ─────────────────────── Temporal (time-ordered) split ─────────────────── #

    @staticmethod
    def temporal_train_val_test_split(
        X_with_meta: pd.DataFrame,
        y: pd.Series,
        date_col: str = "date",
        group_col: str = "gameid",
        val_size: float = VALIDATION_SIZE,
        test_size: float = TEST_SIZE,
    ) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """
        Time-ordered split by unique groups (gameid). Keeps both sides together.
        Uses the group's min(date) to order games.
        """
        required = {group_col, date_col}
        if not required.issubset(X_with_meta.columns):
            raise ValueError(f"Missing required columns {sorted(required)}")

        g = X_with_meta[[group_col, date_col]].drop_duplicates(subset=group_col).copy()
        g[date_col] = pd.to_datetime(g[date_col], errors="coerce")
        g = g.sort_values(date_col).dropna(subset=[date_col])
        if g.empty:
            raise ValueError("No valid dates for temporal split.")

        n = len(g)
        n_test = max(1, int(round(n * test_size)))
        n_val = max(1, int(round((n - n_test) * val_size)))
        n_train = max(1, n - n_val - n_test)

        gids_train = set(g.iloc[:n_train][group_col])
        gids_val = set(g.iloc[n_train : n_train + n_val][group_col])
        gids_test = set(g.iloc[n_train + n_val :][group_col])

        def _sel(gids: set[Any]) -> tuple[pd.DataFrame, pd.Series]:
            Xp = X_with_meta[X_with_meta[group_col].isin(gids)]
            yp = y.loc[Xp.index]
            return Xp, yp

        X_train, y_train = _sel(gids_train)
        X_val, y_val = _sel(gids_val)
        X_test, y_test = _sel(gids_test)

        logger.info(
            "Temporal split -> train: %d, val: %d, test: %d rows",
            len(X_train),
            len(X_val),
            len(X_test),
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ─────────────────────── Categorical / imputation ─────────────────────── #

    @staticmethod
    def _impute_train_numeric(
        X: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        medians = X.median(numeric_only=True).to_dict()
        return X.fillna(value=medians), medians

    @staticmethod
    def _impute_apply_numeric(
        X: pd.DataFrame, medians: dict[str, float]
    ) -> pd.DataFrame:
        return X.fillna(value=medians)

    @staticmethod
    def _impute_categorical(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
        X = X.copy()
        for c in cat_cols:
            if X[c].isna().any():
                # ensure "Unknown" in categories
                if X[c].dtype.name == "category":
                    new_cats = list(X[c].cat.categories)
                    if "Unknown" not in new_cats:
                        new_cats.append("Unknown")
                    X[c] = X[c].cat.set_categories(new_cats)
                X[c] = X[c].fillna("Unknown")
        return X

    @staticmethod
    def _align_like_train(
        train_cols: list[str],
        X: pd.DataFrame,
        *,
        categorical_features: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Align columns to the training set:
          - add missing numeric columns as 0,
          - add missing categorical columns as "Unknown",
          - drop extras, and order identically to train_cols.
        """
        X = X.copy()
        categorical_features = categorical_features or []
        missing = [c for c in train_cols if c not in X.columns]
        for c in missing:
            X[c] = "Unknown" if c in categorical_features else 0
        extras = [c for c in X.columns if c not in train_cols]
        if extras:
            X = X.drop(columns=extras)
        for c in categorical_features:
            if c in X.columns and X[c].dtype.name != "category":
                X[c] = X[c].astype("category")
        return X[train_cols]

    @staticmethod
    def preprocess_categorical_features(
        X: pd.DataFrame,
        exclude_cols: list[str] | None = None,
        categorical_columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Cast object columns to category; respect exclude list."""
        X = X.copy()
        exclude_cols = exclude_cols or []
        if categorical_columns is None:
            categorical_columns = [
                c for c in X.columns if X[c].dtype == "object" and c not in exclude_cols
            ]
        for c in categorical_columns:
            X[c] = X[c].astype("category")
            X[c] = X[c].cat.set_categories(X[c].cat.categories)  # freeze categories
        return X, categorical_columns

    # ─────────────────────────── Feature pipeline ─────────────────────────── #

    def _fit_feature_pipeline(
        self,
        X_train: pd.DataFrame,
        *,
        drop_missing_threshold: float,
        drop_low_std_threshold: float,
        drop_high_corr_threshold: float,
    ) -> tuple[pd.DataFrame, FeaturePipeline]:
        """
        Fit all train-only feature decisions (drops, categories, imputations)
        and return the transformed train set + a reusable pipeline.
        """
        Xp = X_train.copy()

        Xp, drop_high_miss = self._drop_high_missing(Xp, drop_missing_threshold)
        Xp, drop_low_var = self.drop_low_std_columns(Xp, drop_low_std_threshold)
        Xp, drop_corr = self.drop_highly_correlated_features(
            Xp, drop_high_corr_threshold
        )

        Xp, categorical_features = self.preprocess_categorical_features(Xp)

        cat_levels = {c: list(Xp[c].cat.categories) for c in categorical_features}
        for c in categorical_features:
            if "Unknown" not in cat_levels[c]:
                cat_levels[c].append("Unknown")
            Xp[c] = Xp[c].cat.set_categories(cat_levels[c])

        Xp, medians = self._impute_train_numeric(Xp)
        Xp = self._impute_categorical(Xp, categorical_features)

        pipeline = FeaturePipeline(
            train_columns=list(Xp.columns),
            categorical_features=categorical_features,
            categorical_levels=cat_levels,
            numeric_medians=medians,
            drop_high_missing=drop_high_miss,
            drop_low_variance=drop_low_var,
            drop_high_correlation=drop_corr,
        )
        return Xp, pipeline

    # ─────────────────────────────── Storage ──────────────────────────────── #

    def _store_pickle(self, filename: str, data: Any) -> None:
        try:
            out_dir = MODEL_ARTIFACTS / self.model_name
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / filename).open("wb") as f:
                pickle.dump(data, f)
            logger.info("Stored %s", filename)
        except Exception as e:
            logger.error("Storing %s failed: %s", filename, e)
            raise

    def store_model_features(self, all_features: pd.Index) -> None:
        self._store_pickle(
            f"{self.model_name}_final_features.pkl", all_features.tolist()
        )

    def store_categorical_features(self, categorical_features: list[str]) -> None:
        self._store_pickle(
            f"{self.model_name}_categorical_features.pkl", categorical_features
        )

    def store_feature_pipeline(self, pipeline: FeaturePipeline) -> None:
        self._store_pickle(
            f"{self.model_name}_feature_pipeline.pkl",
            pipeline,
        )

    def store_best_hyperparameters(self, hyperparams: dict[str, Any]) -> None:
        self._store_pickle(f"{self.model_name}_best_hyperparameters.pkl", hyperparams)

    # ─────────────────────────────── Metrics ──────────────────────────────── #

    def compute_classification_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None
    ) -> dict[str, Any]:
        """Accuracy/Precision/Recall/F1 + ROC-AUC + Brier score if proba available."""
        accuracy = float(accuracy_score(y_true, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        metrics: dict[str, Any] = {
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        if y_proba is not None:
            with contextlib.suppress(ValueError, IndexError):
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
                metrics["brier"] = float(brier_score_loss(y_true, y_proba))
        metrics["cm"] = confusion_matrix(y_true, y_pred)
        return metrics

    def compute_regression_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray
    ) -> dict[str, Any]:
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))
        return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

    def store_evaluation_metrics(self, metrics: dict[str, Any]) -> None:
        """Persist metrics to JSON (drop non-serializable arrays like CM)."""
        payload = {k: v for k, v in metrics.items() if k != "cm"}
        try:
            self.insight_path("metrics.json").write_text(json.dumps(payload))
            logger.info("Stored metrics for %s.", self.model_name)
        except Exception as e:
            logger.error("Storing metrics failed: %s", e)
            raise

    def log_evaluation_metrics(self, metrics: dict[str, Any]) -> None:
        if self.problem_type == "classification":
            base = (
                f"Acc {metrics.get('accuracy', np.nan):.4f} | "
                f"Prec {metrics.get('precision', np.nan):.4f} | "
                f"Rec {metrics.get('recall', np.nan):.4f} | "
                f"F1 {metrics.get('f1', np.nan):.4f}"
            )
            if "roc_auc" in metrics:
                base += f" | AUC {metrics['roc_auc']:.4f}"
            if "brier" in metrics:
                base += f" | Brier {metrics['brier']:.4f}"
            logger.info("Eval: %s", base)
        else:
            logger.info(
                "Eval: MAE %.4f | MSE %.4f | RMSE %.4f | R2 %.4f",
                metrics.get("mae", np.nan),
                metrics.get("mse", np.nan),
                metrics.get("rmse", np.nan),
                metrics.get("r2", np.nan),
            )

    # ─────────────────────────────── Validation ─────────────────────────────── #

    def store_predictions(
        self,
        predictions: np.ndarray,
        eval_gameids: pd.Series,
        eval_sides: pd.Series,
        proba: np.ndarray | None = None,
    ) -> None:
        """Persist per-row predictions (+probabilities for classification)."""
        try:
            frame = {
                "gameid": eval_gameids.to_numpy(),
                "side": eval_sides.to_numpy(),
                "prediction": predictions,
            }
            if proba is not None:
                frame["proba"] = proba
            pd.DataFrame(frame).to_parquet(
                self.insight_path("predictions.parquet"),
                index=False,
                compression="gzip",
            )
            logger.info("Stored predictions for %s.", self.model_name)
        except Exception as e:
            logger.error("Storing predictions failed: %s", e)
            raise

    def validate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        eval_gameids: pd.Series,
        eval_sides: pd.Series,
    ) -> None:
        """Validate & store: metrics, predictions, and observability artifacts."""
        try:
            logger.info("Validating %s ...", self.model_name)
            y_pred = model.predict(X_test)
            y_proba = None
            if self.problem_type == "classification" and hasattr(
                model, "predict_proba"
            ):
                y_proba = model.predict_proba(X_test)[:, 1]

            if self.problem_type == "classification":
                metrics = self.compute_classification_metrics(y_test, y_pred, y_proba)
                self.plot_confusion_matrix(y_test, y_pred)
                self.plot_accuracy_over_samples(y_test, y_pred)
                if y_proba is not None:
                    self.plot_roc_pr_calibration(y_test, y_proba)
                # historical accuracy uses PROCESSED_TEAMS join by gameid internally
                self.plot_historical_accuracy(X_test, y_test, y_pred, eval_gameids)
            else:
                metrics = self.compute_regression_metrics(y_test, y_pred)
                self.plot_regression_results(y_test, y_pred)
                self.plot_regression_error_over_samples(y_test, y_pred, metric="mae")
                self.plot_regression_error_over_time(
                    X_test, y_test, y_pred, eval_gameids
                )

            self.log_evaluation_metrics(metrics)
            self.store_evaluation_metrics(metrics)
            self.store_predictions(y_pred, eval_gameids, eval_sides, proba=y_proba)

            logger.info("Validation complete for %s.", self.model_name)
        except Exception as e:
            logger.error("Validation failed: %s", e)
            raise

    # ─────────────────────────── Helpers (meta drop with logging) ─────────────────────────── #

    def _strip_meta_from_features(self, X: pd.DataFrame, name: str) -> pd.DataFrame:
        """
        Drop all meta columns (including 'date') from a feature frame.
        Logs exactly what was removed to avoid any ambiguity.
        """
        meta = [c for c in self._meta_columns() if c in X.columns]
        if not meta:
            logger.debug("No meta columns to drop from %s features.", name)
            return X
        before = X.shape[1]
        X2 = X.drop(columns=meta, errors="ignore")
        removed = [c for c in meta if c not in X2.columns or c not in X.columns]
        logger.debug(
            "Dropped %d meta columns from %s features: %s",
            before - X2.shape[1],
            name,
            ", ".join(removed),
        )
        return X2

    # ─────────────────────────── Public orchestrator ────────────────────────── #

    def train_and_validate_model(  # noqa: PLR0912, PLR0915
        self,
        target_col: str,
        *,
        validate: bool = True,
        fuse_opponents: bool = True,
        process_player_likelihoods: bool = True,
        drop_missing_threshold: float = MAX_MISSING_FRAC,
        drop_low_std_threshold: float = LOW_STD_THRESHOLD,
        drop_high_corr_threshold: float = HIGH_CORR_THRESHOLD,
        compute_perm_importance: bool = False,
        compute_shap: bool = True,
        store_cohorts: bool = True,
        temporal_split: bool = True,
    ):  # sourcery skip: low-code-quality
        """
        Main entrypoint used by the training script.

        Returns
        -------
        fitted_model

        """
        if self.training_data.empty:
            msg = "Call preprocess_data(target_col=...) before training."
            raise RuntimeError(msg)

        # Separate y and the feature/meta table
        y = self.training_data[target_col]
        X_full = self.training_data.drop(columns=[target_col], errors="ignore")

        # Keep meta for splits & later evaluation artifacts
        meta_cols = [c for c in self._meta_columns() if c in X_full.columns]
        meta_df = X_full[meta_cols].copy()
        X = X_full.drop(columns=meta_cols, errors="ignore")

        # Optional feature transformations (pre-split; these do not use labels)
        if fuse_opponents:
            X = self.fuse_opposing_team_features(X)
        if process_player_likelihoods:
            X = self.process_players_likelihood_columns(X, agg="mean")

        # If 'date' is missing (preprocessor may drop it), reattach via PROCESSED_TEAMS
        if "date" not in meta_df.columns:
            try:
                team_dates = self._safe_read_parquet(PROCESSED_TEAMS)[
                    ["gameid", "date"]
                ].drop_duplicates()
                meta_df = meta_df.merge(
                    team_dates, on="gameid", how="left", validate="m:1"
                )
            except (FileNotFoundError, KeyError, ValueError) as e:
                logger.warning("Could not reattach 'date' for temporal split: %s", e)

        # Attach back split keys (includes season if present)
        split_keys = ["gameid", "league"] + (
            ["season"] if "season" in meta_df.columns else []
        )
        extra_split_cols = ["date"] if "date" in meta_df.columns else []
        X_for_split = pd.concat([X, meta_df[split_keys + extra_split_cols]], axis=1)

        # Choose splitter (temporal if 'date' is available)
        if temporal_split and "date" in X_for_split.columns:
            X_train, X_val, X_test, y_train, y_val, y_test = (
                self.temporal_train_val_test_split(
                    X_for_split,
                    y,
                    date_col="date",
                    group_col="gameid",
                    val_size=VALIDATION_SIZE,
                    test_size=TEST_SIZE,
                )
            )
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = (
                self.grouped_stratified_train_val_test_split(
                    X_for_split,
                    y,
                    val_size=VALIDATION_SIZE,
                    test_size=TEST_SIZE,
                    random_state=RANDOM_STATE,
                )
            )

        # ── CRITICAL: drop meta (incl. 'date') from the actual feature matrices, with logs ── #
        X_train = self._strip_meta_from_features(X_train, "train")
        X_val = self._strip_meta_from_features(X_val, "val")
        X_test = self._strip_meta_from_features(X_test, "test")

        # Record eval identifiers for artifacts (sourced from meta_df, not features)
        eval_gameids = (
            meta_df.loc[X_test.index, "gameid"]
            if "gameid" in meta_df
            else pd.Series(index=X_test.index, dtype=object)
        )
        eval_sides = (
            meta_df.loc[X_test.index, "side"]
            if "side" in meta_df
            else pd.Series(index=X_test.index, dtype=object)
        )

        # Fit feature pipeline on TRAIN (drop/missing/variance/corr/cats/impute)
        # and reapply it to val/test to guarantee feature parity.
        X_train, feature_pipeline = self._fit_feature_pipeline(
            X_train,
            drop_missing_threshold=drop_missing_threshold,
            drop_low_std_threshold=drop_low_std_threshold,
            drop_high_corr_threshold=drop_high_corr_threshold,
        )
        X_val = feature_pipeline.transform(X_val)
        X_test = feature_pipeline.transform(X_test)
        categorical_features = feature_pipeline.categorical_features
        train_cols = feature_pipeline.train_columns

        # Explicit guardrail: eval splits must perfectly mirror train features
        for split_name, X_split in {"val": X_val, "test": X_test}.items():
            if list(X_split.columns) != train_cols:
                msg = f"{split_name} columns misaligned with train features."
                raise ValueError(msg)

        # ── Operator checks (#6): detect suspicious single-feature leakage signals ── #
        try:
            # 6a) Top single-feature AUCs (TRAIN ONLY, numeric cols)
            if self.problem_type == "classification":
                num = X_train.select_dtypes("number")
                if not num.empty and y_train.nunique() == BINARY_CLASS_UNIQUE_VALUES:
                    aucs = num.apply(
                        lambda s: roc_auc_score(
                            y_train, pd.Series(s).fillna(s.median())
                        ),
                        axis=0,
                    )
                    HIGH_SINGLE_FEATURE_AUC = 0.95
                    high = aucs[aucs > HIGH_SINGLE_FEATURE_AUC].sort_values(
                        ascending=False
                    )
                    if len(high):
                        logger.warning(
                            "Single-feature AUC > %.2f on TRAIN: %s",
                            HIGH_SINGLE_FEATURE_AUC,
                            ", ".join(f"{k}={v:.3f}" for k, v in high.items()),
                        )
        except (ValueError, IndexError) as e:
            logger.warning("Single-feature AUC check failed: %s", e)

        try:
            # 6b) Pure (single-class) categorical buckets on TRAIN
            if self.problem_type == "classification":
                obj = X_train.select_dtypes("category")
                hits: list[str] = []
                for c in obj.columns:
                    tab = pd.crosstab(X_train[c], y_train)
                    pure = (tab.max(axis=1) == tab.sum(axis=1)).sum()
                    if pure > 0:
                        hits.append(c)
                if hits:
                    logger.warning(
                        "Categoricals with pure (single-class) buckets on TRAIN: %s",
                        ", ".join(hits),
                    )
        except (ValueError, KeyError) as e:
            logger.warning("Pure-bucket categorical check failed: %s", e)

        # Persist features metadata
        self.store_model_features(pd.Index(train_cols))
        self.store_categorical_features(categorical_features)
        self.store_feature_pipeline(feature_pipeline)

        # ───────────────────────────── Train ───────────────────────────── #
        logger.info("Training %s on %d features …", self.model_name, len(train_cols))
        model = self.train_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            categorical_features=categorical_features,
        )

        # ─────────────────────────── Validate ──────────────────────────── #
        if validate:
            self.validate_model(model, X_test, y_test, eval_gameids, eval_sides)

            # Observability extras
            try:
                self.store_feature_importance(model, train_cols)
            except (ValueError, AttributeError) as e:
                logger.warning("Feature importance failed: %s", e)

            if compute_perm_importance:
                try:
                    self.calculate_permutation_importance(
                        model, X_test, y_test, train_cols
                    )
                except (ValueError, AttributeError) as e:
                    logger.warning("Permutation importance failed: %s", e)

            if compute_shap:
                try:
                    # SHAP can be expensive; sample if very large
                    max_rows = 5000
                    if len(X_test) > max_rows:
                        Xs = X_test.sample(n=max_rows, random_state=RANDOM_STATE)
                    else:
                        Xs = X_test
                    self.calculate_and_plot_shap(model, Xs, train_cols)
                except (ValueError, AttributeError, ImportError) as e:
                    logger.warning("SHAP computation failed: %s", e)

            if store_cohorts:
                try:
                    eval_meta = meta_df.loc[
                        X_test.index,
                        [
                            c
                            for c in ["league", "patch", "side", "gameid"]
                            if c in meta_df.columns
                        ],
                    ]
                    # classification: also persist cohort metrics with probabilities if available
                    y_proba = None
                    if self.problem_type == "classification" and hasattr(
                        model, "predict_proba"
                    ):
                        y_proba = model.predict_proba(X_test)[:, 1]
                    self.store_cohort_metrics(
                        eval_meta, y_test, model.predict(X_test), y_proba
                    )
                except (ValueError, AttributeError) as e:
                    logger.warning("Cohort metrics failed: %s", e)

        # ───────────────────────── Traceability (model card) ───────────────────── #
        try:
            data_window = None
            if "date" in meta_df.columns:
                dates = pd.to_datetime(meta_df["date"], errors="coerce")
                if dates.notna().any():
                    data_window = {
                        "min_date": str(dates.min().date()),
                        "max_date": str(dates.max().date()),
                    }
            self.store_model_card(
                run_id=self.run_id,
                data_window=data_window,
                n_rows_train=len(X_train),
                n_rows_val=len(X_val),
                n_rows_test=len(X_test),
                features=train_cols,
                hyperparams=None,
                code_version=None,
                data_hash=None,
            )
        except Exception as e:
            logger.warning("Model card storage failed: %s", e)

        return model

    # ───────────────────────────── Abstract hooks ───────────────────────────── #

    @abstractmethod
    def train_model(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        categorical_features: list[str] | None,
    ):
        """Implement training flow (and optionally early stopping / logging)."""

    @abstractmethod
    def _optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, Any]:
        """Return best hyperparameters."""
