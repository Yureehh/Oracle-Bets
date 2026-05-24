"""
Automated feature selection methods for ML models.

Provides multiple strategies to reduce feature dimensionality:
- Importance-based: Keep features above relative importance threshold
- RFECV: Recursive Feature Elimination with Cross-Validation
- Boruta: All-relevant feature selection algorithm

Usage:
    selector = FeatureSelector()
    selected = selector.select_by_importance(model, X_train, threshold=0.001)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from oracle_bets_core.logger import logger
from oracle_bets_core.paths import FEATURE_REPORTS_DIR

if TYPE_CHECKING:
    from oracle_bets_core.pd import pd

CORRELATION_REDUNDANCY_THRESHOLD = 0.95


class FeatureSelector:
    """Automated feature selection methods for dimensionality reduction."""

    @staticmethod
    def select_by_importance(
        model: Any,
        X: pd.DataFrame,
        threshold: float = 0.001,
        min_features: int = 15,
    ) -> list[str]:
        """
        Keep features above relative importance threshold.

        Parameters
        ----------
        model : fitted model with feature_importances_ attribute
        X : DataFrame with feature columns
        threshold : minimum relative importance to keep (default 0.1% of total)
        min_features : minimum number of features to keep regardless of threshold

        Returns
        -------
        list[str] : Selected feature names

        """
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            logger.warning("Model has no feature_importances_; returning all features.")
            return list(X.columns)

        total = importances.sum()
        if total == 0:
            logger.warning("Total importance is zero; returning all features.")
            return list(X.columns)

        # Calculate relative importance and sort
        rel_importance = importances / total
        if len(X.columns) != len(rel_importance):
            msg = f"Column/importance mismatch: {len(X.columns)} cols vs {len(rel_importance)} importances"
            raise ValueError(msg)
        feature_imp = sorted(
            zip(X.columns, rel_importance, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )

        # Select features above threshold, ensuring minimum count
        selected = [f for f, imp in feature_imp if imp >= threshold]
        if len(selected) < min_features:
            selected = [f for f, _ in feature_imp[:min_features]]

        logger.info(
            "Importance selection: %d -> %d features (threshold=%.4f)",
            len(X.columns),
            len(selected),
            threshold,
        )
        return selected

    @staticmethod
    def select_by_cumulative_importance(
        model: Any,
        X: pd.DataFrame,
        cumulative_threshold: float = 0.95,
        min_features: int = 15,
    ) -> list[str]:
        """
        Keep top features that account for cumulative_threshold of total importance.

        Parameters
        ----------
        model : fitted model with feature_importances_ attribute
        X : DataFrame with feature columns
        cumulative_threshold : cumulative importance to capture (default 95%)
        min_features : minimum number of features to keep

        Returns
        -------
        list[str] : Selected feature names

        """
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            logger.warning("Model has no feature_importances_; returning all features.")
            return list(X.columns)

        total = importances.sum()
        if total == 0:
            return list(X.columns)

        # Sort by importance descending
        idx_sorted = np.argsort(importances)[::-1]
        cumsum = np.cumsum(importances[idx_sorted]) / total

        # Find cutoff index
        n_keep = max(min_features, int((cumsum <= cumulative_threshold).sum()) + 1)
        selected_idx = idx_sorted[:n_keep]
        selected = [X.columns[i] for i in selected_idx]

        logger.info(
            "Cumulative importance selection: %d -> %d features (threshold=%.2f)",
            len(X.columns),
            len(selected),
            cumulative_threshold,
        )
        return selected

    @staticmethod
    def select_by_rfecv(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model: Any = None,
        cv: int = 3,
        min_features: int = 20,
        step: int = 5,
    ) -> list[str]:
        """
        Recursive Feature Elimination with Cross-Validation.

        Parameters
        ----------
        X_train : Training features
        y_train : Training target
        model : Estimator to use (default: LightGBM)
        cv : Number of CV folds
        min_features : Minimum features to select
        step : Number of features to remove per iteration

        Returns
        -------
        list[str] : Selected feature names

        """
        from sklearn.feature_selection import RFECV

        if model is None:
            import lightgbm as lgb

            model = lgb.LGBMClassifier(
                n_estimators=100, verbosity=-1, n_jobs=-1, random_state=42
            )

        logger.info("Running RFECV (cv=%d, min_features=%d)...", cv, min_features)

        # Handle categorical features for sklearn
        X_numeric = X_train.select_dtypes(include=["number"]).copy()
        if X_numeric.empty:
            logger.warning("No numeric features for RFECV; returning all features.")
            return list(X_train.columns)

        selector = RFECV(
            estimator=model,
            step=step,
            cv=cv,
            scoring="roc_auc",
            min_features_to_select=min_features,
            n_jobs=-1,
        )
        selector.fit(X_numeric.fillna(0), y_train)

        selected = X_numeric.columns[selector.support_].tolist()
        logger.info(
            "RFECV selection: %d -> %d features", len(X_numeric.columns), len(selected)
        )
        return selected

    @staticmethod
    def select_by_boruta(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        max_features: int = 50,
        max_iter: int = 100,
    ) -> list[str]:
        """
        Boruta algorithm for all-relevant feature selection.

        Requires: pip install boruta

        Parameters
        ----------
        X_train : Training features
        y_train : Training target
        max_features : Maximum features to return
        max_iter : Maximum Boruta iterations

        Returns
        -------
        list[str] : Selected feature names

        """
        try:
            from boruta import BorutaPy
        except ImportError:
            logger.warning(
                "Boruta not installed; falling back to importance selection."
            )
            return list(X_train.columns)[:max_features]

        from sklearn.ensemble import RandomForestClassifier

        logger.info("Running Boruta (max_iter=%d)...", max_iter)

        # Handle categorical features
        X_numeric = X_train.select_dtypes(include=["number"]).copy().fillna(0)
        if X_numeric.empty:
            logger.warning("No numeric features for Boruta; returning all features.")
            return list(X_train.columns)

        rf = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=42, max_depth=7
        )
        boruta = BorutaPy(
            rf, n_estimators="auto", max_iter=max_iter, random_state=42, verbose=0
        )

        try:
            boruta.fit(X_numeric.values, y_train.values)
            # Get confirmed features
            confirmed = X_numeric.columns[boruta.support_].tolist()
            # Also include tentative if we need more
            tentative = X_numeric.columns[boruta.support_weak_].tolist()
            selected = (confirmed + tentative)[:max_features]
        except Exception as e:
            logger.warning("Boruta failed: %s; returning top features by variance.", e)
            var = X_numeric.var()
            selected = var.nlargest(max_features).index.tolist()

        logger.info(
            "Boruta selection: %d -> %d features", len(X_numeric.columns), len(selected)
        )
        return selected

    @classmethod
    def select_features(
        cls,
        method: Literal[
            "none", "importance", "cumulative", "rfecv", "boruta", "report"
        ],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model: Any = None,
        **kwargs,
    ) -> list[str]:
        """
        Unified interface for feature selection.

        Parameters
        ----------
        method : Selection method to use
        X_train : Training features
        y_train : Training target
        model : Fitted model (required for importance-based methods)
        **kwargs : Additional arguments passed to the specific method

        Returns
        -------
        list[str] : Selected feature names

        """
        if method == "none":
            return list(X_train.columns)

        if method == "importance":
            if model is None:
                raise ValueError("Model required for importance-based selection")
            return cls.select_by_importance(model, X_train, **kwargs)

        if method == "cumulative":
            if model is None:
                raise ValueError("Model required for cumulative importance selection")
            return cls.select_by_cumulative_importance(model, X_train, **kwargs)

        if method == "rfecv":
            return cls.select_by_rfecv(X_train, y_train, model=model, **kwargs)

        if method == "boruta":
            return cls.select_by_boruta(X_train, y_train, **kwargs)

        raise ValueError(f"Unknown selection method: {method}")

    @staticmethod
    def write_temporal_recommendation_report(
        *,
        model: Any,
        X_train: pd.DataFrame,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
        model_name: str,
        max_features: int = 120,
        feature_counts: tuple[int, ...] = (60, 90, 120, 160),
        problem_type: str = "classification",
    ) -> dict[str, Any]:
        """Write a compact-feature recommendation report from temporal validation."""
        importances = getattr(model, "feature_importances_", None)
        if importances is None or len(importances) != len(X_train.columns):
            logger.warning(
                "Cannot write feature recommendation report: no importances."
            )
            return {}

        total_importance = float(np.sum(importances))
        rel_importance = (
            np.asarray(importances, dtype=float) / total_importance
            if total_importance > 0
            else np.zeros(len(importances), dtype=float)
        )
        missingness = X_train.isna().mean(numeric_only=False).fillna(0.0)
        numeric = X_train.select_dtypes("number")
        corr_hits = dict.fromkeys(X_train.columns, 0)
        if numeric.shape[1] > 1:
            corr = numeric.corr().abs()
            for col in corr.columns:
                corr_hits[col] = int(
                    (
                        corr[col].drop(labels=[col]) > CORRELATION_REDUNDANCY_THRESHOLD
                    ).sum()
                )

        scorer_model = getattr(model, "raw_model", model)
        permutation_scores = _permutation_scores(
            model=scorer_model,
            X_validation=X_validation,
            y_validation=y_validation,
            problem_type=problem_type,
        )

        rows = []
        for idx, feature in enumerate(X_train.columns):
            perm = permutation_scores.get(feature, 0.0)
            miss_penalty = float(missingness.get(feature, 0.0))
            corr_penalty = min(corr_hits.get(feature, 0) * 0.02, 0.2)
            score = float(rel_importance[idx]) + max(perm, 0.0) - miss_penalty * 0.1
            score -= corr_penalty
            rows.append(
                {
                    "feature": feature,
                    "importance": float(importances[idx]),
                    "relative_importance": float(rel_importance[idx]),
                    "permutation_importance": float(perm),
                    "missing_rate": miss_penalty,
                    "correlated_feature_count": int(corr_hits.get(feature, 0)),
                    "recommendation_score": score,
                }
            )

        ranked = sorted(rows, key=lambda row: row["recommendation_score"], reverse=True)
        recommended = [row["feature"] for row in ranked[:max_features]]
        recommendations_by_count = {
            str(count): [row["feature"] for row in ranked[:count]]
            for count in feature_counts
        }
        payload = {
            "model_name": model_name,
            "selection_basis": "temporal validation feature report",
            "problem_type": problem_type,
            "max_features": max_features,
            "recommended_features": recommended,
            "recommendations_by_count": recommendations_by_count,
            "features": ranked,
        }

        FEATURE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = (
            FEATURE_REPORTS_DIR / f"{model_name}_recommended_compact_features.json"
        )
        output_path.write_text(json.dumps(payload, indent=2) + "\n")
        legacy_path = FEATURE_REPORTS_DIR / "recommended_compact_features.json"
        legacy_path.write_text(json.dumps(payload, indent=2) + "\n")
        logger.info("Stored feature recommendation report: %s", output_path)
        return payload


def _permutation_scores(
    *,
    model: Any,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    problem_type: str,
) -> dict[str, float]:
    try:
        from sklearn.inspection import permutation_importance
    except ImportError:
        return {}

    if X_validation.empty:
        return {}
    sample = X_validation
    max_rows = 2000
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=42)
        y_validation = y_validation.loc[sample.index]

    try:
        scoring = (
            "roc_auc"
            if problem_type == "classification"
            else "neg_root_mean_squared_error"
        )
        result = permutation_importance(
            model,
            sample,
            y_validation,
            scoring=scoring,
            n_repeats=3,
            random_state=42,
            n_jobs=-1,
        )
    except Exception as exc:
        logger.warning("Permutation scoring for feature report failed: %s", exc)
        return {}

    return {
        feature: float(score)
        for feature, score in zip(sample.columns, result.importances_mean, strict=False)
    }


def get_top_features_report(
    model: Any, feature_names: list[str], top_n: int = 30
) -> str:
    """Generate a human-readable report of top features."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return "Model has no feature importances."

    total = importances.sum()
    if len(feature_names) != len(importances):
        return (
            f"Feature/importance mismatch: {len(feature_names)} vs {len(importances)}"
        )
    ranked = sorted(
        zip(feature_names, importances, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )

    lines = ["Top Features by Importance:", "=" * 50]
    cumsum = 0.0
    for i, (name, imp) in enumerate(ranked[:top_n], 1):
        rel = imp / total if total > 0 else 0
        cumsum += rel
        lines.append(f"{i:3d}. {name:40s} {imp:8.0f} ({rel:6.2%}) [cum: {cumsum:6.2%}]")

    return "\n".join(lines)
