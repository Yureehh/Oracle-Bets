"""
ML observability & explainability utilities for GBM-based models.

Drop-in module. Import and mix into your model class:

    from lol_bets.prediction_models.ml_observability import MLObservabilityMixin

Your concrete model must define the following attributes:
    - self.model_name: str
    - self.problem_type: str  # "classification" | "regression"
    - self.directory: pathlib.Path  # figures output dir (usually FIGURES_DIR)
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from oracle_bets_core.logger import logger
from oracle_bets_core.paths import FIGURES_DIR, INSIGHTS_DIR, PROCESSED_TEAMS
from oracle_bets_core.pd import pd
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

sns.set_style("darkgrid")

TOP_N_FEATURES = 30
MIN_WEEKS_FOR_TREND = 4


def _expected_calibration_error(
    actual: np.ndarray,
    probability: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Return equal-width ECE for one explicit cohort."""
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.minimum(np.digitize(probability, boundaries[1:-1]), bins - 1)
    error = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if np.any(mask):
            error += float(
                np.mean(mask) * abs(np.mean(probability[mask]) - np.mean(actual[mask]))
            )
    return error


class MLObservabilityMixin:
    """
    Reusable plotting & logging helpers for training/validation.

    Expects subclasses to define:
      - self.model_name: str
      - self.problem_type: str
      - self.directory: Path
      - self.run_id: str  (optional; defaults to "default" if absent)
    """

    model_name: str
    problem_type: str
    run_id: str

    # ────────────────────────── internal helpers ────────────────────────── #

    @staticmethod
    def _ensure_dir(p: Path | str) -> None:
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_read_parquet(path: Path | str) -> pd.DataFrame:
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except (ImportError, ValueError):
            return pd.read_parquet(path)

    # Unified artifact locations (per-model / per-run)
    @property
    def fig_dir(self) -> Path:
        run_id = getattr(self, "run_id", "default")
        report_root = getattr(self, "report_root", None)
        if report_root is not None:
            return Path(report_root) / self.model_name / "figures"
        return FIGURES_DIR / self.model_name / run_id

    @property
    def insight_dir(self) -> Path:
        run_id = getattr(self, "run_id", "default")
        report_root = getattr(self, "report_root", None)
        if report_root is not None:
            return Path(report_root) / self.model_name
        return INSIGHTS_DIR / self.model_name / run_id

    def fig_path(self, filename: str) -> Path:
        path = self.fig_dir / filename
        self._ensure_dir(path)
        return path

    def insight_path(self, filename: str) -> Path:
        path = self.insight_dir / filename
        self._ensure_dir(path)
        return path

    # ───────────────────── classification diagnostics ───────────────────── #

    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        """Confusion matrix."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay.from_predictions(
                y_true, y_pred, ax=ax, colorbar=False
            )
            ax.set_title("Confusion Matrix")
            fig.tight_layout()
            out = self.fig_path("confusion_matrix.png")
            fig.savefig(out, dpi=300)
            plt.close(fig)
            logger.info("Stored confusion matrix for %s.", self.model_name)
        except Exception as e:
            logger.error("Confusion matrix plot failed: %s", e)
            raise

    def plot_accuracy_over_samples(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        """Cumulative accuracy vs. sample count."""
        try:
            acc = [
                (y_true.iloc[:i] == y_pred[:i]).mean()
                for i in range(1, len(y_true) + 1)
            ]
            fig = plt.figure(figsize=(10, 5))
            sns.lineplot(
                x=range(1, len(y_true) + 1), y=acc, linestyle="--", color="#84C3FA"
            )
            plt.xlabel("Number of Samples")
            plt.ylabel("Accuracy")
            plt.grid(True, linestyle="--", alpha=0.6, axis="y")
            plt.grid(False, axis="x")
            plt.tight_layout()
            out = self.fig_path("accuracy_over_samples.png")
            plt.savefig(out, dpi=300)
            plt.close(fig)
            logger.info("Stored accuracy-over-samples for %s.", self.model_name)
        except Exception as e:
            logger.error("Accuracy-over-samples plot failed: %s", e)
            raise

    def plot_roc_pr_calibration(self, y_true: pd.Series, y_proba: np.ndarray) -> None:
        """ROC, PR, and calibration curves for classification."""
        try:
            fig, axs = plt.subplots(1, 3, figsize=(18, 5))
            RocCurveDisplay.from_predictions(y_true, y_proba, ax=axs[0])
            axs[0].set_title("ROC Curve")
            PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=axs[1])
            axs[1].set_title("Precision-Recall")
            CalibrationDisplay.from_predictions(y_true, y_proba, n_bins=10, ax=axs[2])
            axs[2].set_title("Calibration")
            fig.tight_layout()
            out = self.fig_path("roc_pr_calibration.png")
            fig.savefig(out, dpi=300)
            plt.close(fig)
            logger.info("Stored ROC/PR/Calibration for %s.", self.model_name)
        except Exception as e:
            logger.error("ROC/PR/Calibration failed: %s", e)
            raise

    # ────────────────────── time-based diagnostics ─────────────────────── #

    def _attach_dates_if_missing(
        self, X_like: pd.DataFrame, eval_gameids: pd.Series
    ) -> pd.DataFrame:
        """Ensure 'date' column is present by joining PROCESSED_TEAMS if missing."""
        Xv = X_like.copy()
        Xv["gameid"] = eval_gameids.to_numpy()
        if "date" not in Xv.columns:
            team_dates = self._safe_read_parquet(PROCESSED_TEAMS)[
                ["gameid", "date"]
            ].drop_duplicates()
            Xv = Xv.merge(team_dates, on="gameid", how="left", validate="m:1")
        return Xv

    def plot_historical_accuracy(
        self,
        X_val: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray,
        eval_gameids: pd.Series,
    ) -> None:
        """Weekly accuracy trajectory to catch drift or seasonal effects."""
        try:
            Xv = self._attach_dates_if_missing(X_val, eval_gameids)
            dates = pd.to_datetime(Xv["date"], errors="coerce")
            periods = dates.dt.to_period("W")
            mask = periods.notna()
            if mask.sum() == 0:
                logger.warning("No valid dates for historical accuracy; skipping.")
                return

            df = pd.DataFrame(
                {
                    "week": periods[mask].dt.start_time,
                    "correct": (
                        y_true.iloc[mask.to_numpy()] == y_pred[mask.to_numpy()]
                    ).astype(int),
                }
            )
            weekly = df.groupby("week", as_index=False)["correct"].mean()

            fig, ax = plt.subplots(figsize=(15, 8))
            sns.lineplot(
                data=weekly,
                x="week",
                y="correct",
                marker="o",
                linestyle="--",
                color="#84C3FA",
                ax=ax,
            )
            if len(weekly) >= MIN_WEEKS_FOR_TREND:
                z = np.polyfit(mdates.date2num(weekly["week"]), weekly["correct"], 3)
                p = np.poly1d(z)
                ax.plot(
                    weekly["week"],
                    p(mdates.date2num(weekly["week"])),
                    "r--",
                    label="Trend",
                )
            ax.set_xlabel("Week")
            ax.set_ylabel("Accuracy")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6, axis="y")
            ax.grid(False, axis="x")
            fig.tight_layout()
            out = self.fig_path("historical_accuracy.png")
            fig.savefig(out, dpi=300, transparent=True)
            plt.close(fig)
            logger.info("Stored historical accuracy for %s.", self.model_name)
        except Exception as e:
            logger.error("Historical accuracy plot failed: %s", e)
            raise

    def plot_regression_error_over_time(
        self,
        X_val: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray,
        eval_gameids: pd.Series,
    ) -> None:
        """Weekly MAE, to track over-time behavior."""
        try:
            Xv = self._attach_dates_if_missing(X_val, eval_gameids)
            dates = pd.to_datetime(Xv["date"], errors="coerce")
            periods = dates.dt.to_period("W")
            mask = periods.notna()
            if mask.sum() == 0:
                logger.warning(
                    "No valid dates for regression error-over-time; skipping."
                )
                return

            df = pd.DataFrame(
                {
                    "week": periods[mask].dt.start_time,
                    "true": y_true.iloc[mask.to_numpy()].to_numpy(),
                    "pred": y_pred[mask.to_numpy()],
                }
            )
            weekly = (
                df.groupby("week", as_index=False)
                .apply(
                    lambda t: float(np.mean(np.abs(t["true"] - t["pred"]))),
                    include_groups=False,
                )
                .rename(columns={None: "mae"})
            )

            fig, ax = plt.subplots(figsize=(15, 8))
            sns.lineplot(
                data=weekly,
                x="week",
                y="mae",
                marker="o",
                linestyle="--",
                color="#84C3FA",
                ax=ax,
            )
            ax.axhline(
                float(weekly["mae"].mean()),
                color="gray",
                linestyle="--",
                label="Average MAE",
            )
            if len(weekly) >= MIN_WEEKS_FOR_TREND:
                z = np.polyfit(mdates.date2num(weekly["week"]), weekly["mae"], 3)
                p = np.poly1d(z)
                ax.plot(
                    weekly["week"],
                    p(mdates.date2num(weekly["week"])),
                    "r--",
                    label="Trend",
                )
            ax.set_xlabel("Week")
            ax.set_ylabel("MAE")
            ax.legend()
            fig.tight_layout()
            out = self.fig_path("historical_mae_over_time.png")
            fig.savefig(out, dpi=300, transparent=True)
            plt.close(fig)
            logger.info("Stored historical MAE for %s.", self.model_name)
        except Exception as e:
            logger.error("Regression error-over-time plot failed: %s", e)
            raise

    # ───────────────────────── regression plots ────────────────────────── #

    def plot_regression_results(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        """Pred vs. actual with y=x guideline."""
        try:
            fig = plt.figure(figsize=(10, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            lo, hi = float(np.min(y_true)), float(np.max(y_true))
            plt.plot([lo, hi], [lo, hi], "r--")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Regression Results")
            plt.tight_layout()
            out = self.fig_path("regression_results.png")
            plt.savefig(out, dpi=300)
            plt.close(fig)
            logger.info("Stored regression results for %s.", self.model_name)
        except Exception as e:
            logger.error("Regression results plot failed: %s", e)
            raise

    def plot_regression_error_over_samples(
        self, y_true: pd.Series, y_pred: np.ndarray, metric: str = "mae"
    ) -> None:
        """Cumulative error (MAE/MSE) across samples."""
        try:
            if metric == "mae":
                errors = np.abs(y_true - y_pred)
            elif metric == "mse":
                errors = (y_true - y_pred) ** 2
            else:
                raise ValueError("metric must be 'mae' or 'mse'")  # noqa: TRY301
            curve = [float(np.mean(errors[:i])) for i in range(1, len(errors) + 1)]
            fig = plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(curve) + 1), curve, marker="o")
            plt.title(f"Cumulative {metric.upper()} Over Samples")
            plt.xlabel("Samples")
            plt.ylabel(metric.upper())
            plt.tight_layout()
            out = self.fig_path(f"cumulative_{metric.lower()}_over_samples.png")
            plt.savefig(out, dpi=300)
            plt.close(fig)
            logger.info("Stored regression error-over-samples for %s.", self.model_name)
        except Exception as e:
            logger.error("Regression error-over-samples plot failed: %s", e)
            raise

    # ───────────────────── feature importance / SHAP ───────────────────── #

    def store_feature_importance(self, model, feature_names: list[str]) -> None:
        """Persist LightGBM gain/split importance and a top-gain plot."""
        try:
            raw_model = getattr(model, "raw_model", model)
            booster = getattr(raw_model, "booster_", None)
            if booster is None:
                logger.warning("Model has no feature_importances_; skipping.")
                return
            gain = np.asarray(booster.feature_importance(importance_type="gain"))
            split = np.asarray(booster.feature_importance(importance_type="split"))
            if len(gain) != len(feature_names) or len(split) != len(feature_names):
                self._raise_importance_shape_error()
            gain_total = float(gain.sum())
            split_total = float(split.sum())
            ranks = sorted(
                zip(feature_names, gain, strict=True),
                key=lambda x: float(x[1]),
                reverse=True,
            )
            df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "gain": gain.astype(float),
                    "gain_fraction": (
                        gain.astype(float) / gain_total if gain_total else 0.0
                    ),
                    "split": split.astype(float),
                    "split_fraction": (
                        split.astype(float) / split_total if split_total else 0.0
                    ),
                }
            ).sort_values(["gain", "split"], ascending=False)
            out_tbl = self.insight_path("feature_importances.parquet")
            df.to_parquet(out_tbl, index=False, compression="gzip")
            self.plot_feature_importance(ranks)
            logger.info(
                "Stored gain/split feature importances for %s.", self.model_name
            )
        except Exception as e:
            logger.error("Storing feature importances failed: %s", e)
            raise

    @staticmethod
    def _raise_importance_shape_error() -> None:
        raise ValueError("Feature names do not match LightGBM importance vectors.")

    def plot_feature_importance(
        self, sorted_importances: list[tuple[str, float]], top_n: int = TOP_N_FEATURES
    ) -> None:
        """Barh of top-N importances."""
        try:
            top = sorted_importances[:top_n]
            if not top:
                logger.warning("No importances to plot.")
                return
            features, values = zip(*top, strict=False)
            fig = plt.figure(figsize=(10, 8))
            plt.barh(features, values)
            plt.xlabel("Feature Importance")
            plt.title(f"Top {top_n} Feature Importances")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            out = self.fig_path("feature_importance.png")
            plt.savefig(out)
            plt.close(fig)
        except Exception as e:
            logger.error("Feature importance plot failed: %s", e)
            raise

    def calculate_permutation_importance(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: list[str],
        top_n: int = TOP_N_FEATURES,
        max_samples: int | None = 5000,
    ) -> None:
        """Permutation importance on a (possibly) downsampled test set."""
        try:
            Xpi, ypi = X_test, y_test
            if max_samples and len(X_test) > max_samples:
                Xpi = X_test.sample(n=max_samples, random_state=42)
                ypi = y_test.loc[Xpi.index]
            result = permutation_importance(
                model, Xpi, ypi, n_repeats=10, n_jobs=-1, random_state=42
            )
            table = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance_mean": result.importances_mean.astype(float),
                    "importance_std": result.importances_std.astype(float),
                }
            ).sort_values("importance_mean", ascending=False)
            table.to_parquet(
                self.insight_path("permutation_importances.parquet"),
                index=False,
                compression="gzip",
            )
            idx = result.importances_mean.argsort()[-top_n:]
            fig = plt.figure(figsize=(10, 8))
            plt.boxplot(
                result.importances[idx].T,
                vert=False,
                tick_labels=np.array(feature_names)[idx],
            )
            plt.title(f"Top {top_n} Permutation Importances")
            plt.tight_layout()
            out = self.fig_path("permutation_importance.png")
            plt.savefig(out)
            plt.close(fig)
            logger.info("Stored permutation importance for %s.", self.model_name)
        except Exception as e:
            logger.error("Permutation importance failed: %s", e)
            raise

    def calculate_and_plot_shap(
        self,
        model,
        X: pd.DataFrame,
        feature_names: list[str],
        top_n: int = TOP_N_FEATURES,
    ) -> None:
        """SHAP summary for top-N mean |SHAP| features; also stores per-row reasons."""
        try:
            base_model = getattr(model, "raw_model", model)
            explainer = shap.TreeExplainer(base_model)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        "LightGBM binary classifier with TreeExplainer shap values "
                        "output has changed to a list of ndarray"
                    ),
                    category=UserWarning,
                    module=r"shap\.explainers\._tree",
                )
                shap_values = explainer.shap_values(X)
            # Binary-class LightGBM returns [shap_class0, shap_class1]
            if self.problem_type == "classification" and isinstance(shap_values, list):
                shap_values = shap_values[1]
            if getattr(shap_values, "size", 0) == 0:
                logger.warning("No SHAP values; skipping.")
                return

            mean_abs = np.abs(shap_values).mean(axis=0)
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "mean_absolute_shap": mean_abs.astype(float),
                }
            ).sort_values("mean_absolute_shap", ascending=False).to_parquet(
                self.insight_path("shap_importances.parquet"),
                index=False,
                compression="gzip",
            )
            top_idx = np.argsort(mean_abs)[-top_n:]
            top_names = [feature_names[i] for i in top_idx]

            shap.summary_plot(
                shap_values[:, top_idx],
                X.iloc[:, top_idx],
                feature_names=top_names,
                show=False,
            )
            plt.tight_layout()
            out = self.fig_path("shap_summary.png")
            plt.savefig(out)
            plt.close()

            # Store per-row top contributors (reason codes) for auditability
            topk = 5
            rows = []
            limit = min(10000, len(X))  # keep artifacts light
            for r in range(limit):
                row_vals = shap_values[r]
                order = np.argsort(np.abs(row_vals))[-topk:][::-1]
                rows.append(
                    {
                        "row_index": int(X.index[r]),
                        "reasons": [
                            {"feature": feature_names[i], "shap": float(row_vals[i])}
                            for i in order
                        ],
                    }
                )
            out_tbl = self.insight_path("top_shap_reasons.parquet")
            pd.DataFrame(rows).to_parquet(out_tbl, index=False, compression="gzip")
            logger.info(
                "Stored SHAP summary and per-row top reasons for %s.", self.model_name
            )
        except Exception as e:
            logger.error("SHAP computation failed: %s", e)
            raise

    # ───────────────────────── cohort / calibration ────────────────────── #

    def store_cohort_metrics(  # noqa: PLR0912, PLR0915
        self,
        df_eval: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray | pd.Series,
        y_proba: np.ndarray | pd.Series | None,
        cohorts: list[str] | None = None,
    ) -> None:  # sourcery skip: low-code-quality
        """
        Compute metrics by cohort and persist parquet + heatmaps.

        Note: df_eval, y_true must share the same index (X_test.index).
        We convert predictions to Series with that index to avoid position/label mismatches.
        """
        if cohorts is None:
            cohorts = [
                "actionable",
                "league",
                "league_region",
                "league_tier",
                "strength_pool",
                "patch",
            ]

        try:
            # Ensure identical index across all evaluation vectors
            y_true = y_true.copy()
            if not isinstance(y_pred, pd.Series):
                y_pred = pd.Series(y_pred, index=y_true.index, name="pred")
            else:
                y_pred = y_pred.reindex(y_true.index)

            if y_proba is not None:
                if not isinstance(y_proba, pd.Series):
                    y_proba = pd.Series(y_proba, index=y_true.index, name="proba")
                else:
                    y_proba = y_proba.reindex(y_true.index)

            # Restrict df_eval to the same index just in case
            df_eval = df_eval.reindex(y_true.index)

            out_rows = []
            for col in [c for c in cohorts if c in df_eval.columns]:
                for lvl, idx in df_eval.groupby(col).groups.items():
                    # idx is a DatetimeIndex/Index of labels; use .loc on Series
                    yt = y_true.loc[idx]
                    yp = y_pred.loc[idx]

                    row = {"cohort": col, "level": str(lvl), "count": len(idx)}
                    if self.problem_type == "classification":
                        row["accuracy"] = float(accuracy_score(yt, yp))
                        row["f1"] = float(f1_score(yt, yp, zero_division=0))
                        if y_proba is not None:
                            yp_prob = y_proba.loc[idx]
                            row["log_loss"] = float(
                                log_loss(yt, yp_prob, labels=[0, 1])
                            )
                            row["brier"] = float(brier_score_loss(yt, yp_prob))
                            row["calibration_ece"] = _expected_calibration_error(
                                yt.to_numpy(dtype=float),
                                yp_prob.to_numpy(dtype=float),
                            )
                            # Only compute AUC if both classes are present
                            if yt.nunique() > 1:
                                row["roc_auc"] = float(roc_auc_score(yt, yp_prob))
                    else:
                        yp_f = yp.astype(float)
                        row["mae"] = float(mean_absolute_error(yt, yp_f))
                        row["r2"] = float(r2_score(yt, yp_f))
                    out_rows.append(row)

            df = pd.DataFrame(out_rows)
            out_tbl = self.insight_path("cohort_metrics.parquet")
            df.to_parquet(out_tbl, index=False, compression="gzip")

            # Quick heatmap per cohort (classification: accuracy; regression: MAE)
            metric = "accuracy" if self.problem_type == "classification" else "mae"
            if metric in df.columns and not df.empty:
                for coh in df["cohort"].unique():
                    pivot = df[df["cohort"] == coh].pivot_table(
                        index="level", values=metric
                    )
                    if pivot.empty:
                        continue
                    fig = plt.figure(figsize=(8, max(4, 0.4 * max(1, len(pivot)))))
                    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
                    plt.title(f"{self.model_name} – {metric} by {coh}")
                    plt.tight_layout()
                    out = self.fig_path(f"{metric}_by_{coh}.png")
                    plt.savefig(out, dpi=300)
                    plt.close(fig)

            logger.info("Stored cohort metrics for %s.", self.model_name)

        except Exception as e:
            logger.error("Cohort metrics failed: %s", e)
            raise

    def store_calibration_table(
        self, y_true: pd.Series, y_proba: np.ndarray, n_bins: int = 15
    ) -> pd.DataFrame:
        """Reliability table + Expected Calibration Error."""
        try:
            df = pd.DataFrame({"y": y_true.to_numpy(), "p": y_proba})
            df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
            tbl = (
                df.groupby("bin", observed=False)
                .agg(
                    bin_mean_p=("p", "mean"),
                    bin_emp_rate=("y", "mean"),
                    n=("y", "size"),
                )
                .reset_index()
            )
            ece = (
                tbl["n"] * (tbl["bin_mean_p"] - tbl["bin_emp_rate"]).abs()
            ).sum() / tbl["n"].sum()
            tbl["ece"] = ece
            tbl["bin"] = tbl["bin"].astype(str)
            out_tbl = self.insight_path("calibration_table.parquet")
            tbl.to_parquet(out_tbl, index=False, compression="gzip")
            logger.info(
                "Stored quantile-bin calibration table (ECE=%.4f) for %s.",
                ece,
                self.model_name,
            )
            return tbl
        except Exception as e:
            logger.error("Calibration table failed: %s", e)
            raise

    # ─────────────────────────── model card JSON ───────────────────────── #

    def store_model_card(
        self,
        run_id: str,
        data_window: dict[str, str] | None,
        n_rows_train: int,
        n_rows_val: int,
        n_rows_test: int,
        features: list[str],
        hyperparams: dict[str, Any] | None = None,
        code_version: str | None = None,
        data_hash: str | None = None,
    ) -> None:
        """Persist a single JSON with everything needed to audit a run."""
        try:
            card = {
                "run_id": run_id,
                "model_name": self.model_name,
                "problem_type": self.problem_type,
                "data_window": data_window or {},
                "rows": {"train": n_rows_train, "val": n_rows_val, "test": n_rows_test},
                "n_features": len(features),
                "feature_hash": hashlib.sha256(
                    json.dumps(
                        sorted(features),
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "hyperparameters": hyperparams or {},
                "calibration": {
                    "mode": getattr(self, "calibration", None),
                    "method": getattr(self, "calibration_method", None),
                    "selected_probability_method": getattr(
                        getattr(self, "probability_calibrator", None), "method", None
                    ),
                    "selected_prop_method": getattr(
                        getattr(self, "prop_calibrator", None), "method", None
                    ),
                    "calibration_size": getattr(self, "calibration_size", None),
                    "tune_size": getattr(self, "tune_size", None),
                    "test_size": getattr(self, "test_size", None),
                },
                "code_version": code_version,
                "data_hash": data_hash,
            }
            out = self.insight_path(f"model_card_{run_id}.json")
            out.write_text(json.dumps(card, indent=2))
            logger.info(
                "Stored model card for %s (run_id=%s).", self.model_name, run_id
            )
        except Exception as e:
            logger.error("Model card failed: %s", e)
            raise
