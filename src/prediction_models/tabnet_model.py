"""
TabNet Model implementation for tabular deep learning.

TabNet is an attention-based deep learning architecture specifically designed
for tabular data. It provides:
- Built-in feature selection via attention masks
- Interpretable feature importance
- Competitive performance with gradient boosting

Requires: pip install pytorch-tabnet torch
"""

from __future__ import annotations

# Must set environment variables BEFORE importing torch (via pytorch-tabnet)
# to avoid OpenMP/MKL threading issues and segfaults on macOS
import os
import platform

if platform.system() in ["Darwin", "Tahoe", "Ventura"]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna

from prediction_models.gbdt_model import DEFAULT_TRIALS as _TRIALS_CAP
from prediction_models.gbdt_model import GradientBoostingModel
from utils.logger import logger

if TYPE_CHECKING:
    from utils.pd import pd

# TabNet defaults
_DEFAULT_EPOCHS = 200
_PATIENCE = 30
_BATCH_SIZE = 1024
_RANDOM_STATE = 42


class _TabNetWithThreshold:
    """
    Wrapper for TabNet classifier that uses a tuned decision threshold.
    Provides consistent interface with LightGBM wrapper.
    """

    def __init__(self, raw_model: Any, threshold: float, feature_names: list[str]):
        self.raw_model = raw_model
        self.decision_threshold_ = threshold
        self.problem_type = "classification"
        self._feature_names = feature_names

    @property
    def feature_importances_(self) -> np.ndarray | None:
        """Get feature importance from TabNet's attention masks."""
        try:
            # TabNet stores feature importances as attribute after fitting
            if hasattr(self.raw_model, "feature_importances_"):
                return np.array(self.raw_model.feature_importances_)
            return None
        except Exception:
            return None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions."""
        X_values = X.values if hasattr(X, "values") else X
        return self.raw_model.predict_proba(X_values)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Get class predictions using tuned threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.decision_threshold_).astype(int)


@dataclass
class TabNetModel(GradientBoostingModel):
    """
    TabNet deep learning model for tabular classification/regression.

    Implements the abstract methods from GradientBoostingModel:
    - train_model(...)
    - _optimize_hyperparameters(...)

    Uses the base class orchestration for splits, pruning, and observability.
    """

    def train_model(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        categorical_features: list[str] | None,
    ) -> Any:
        """
        Fit TabNet model with early stopping and threshold tuning.

        Parameters
        ----------
        X_train, X_val : Feature DataFrames
        y_train, y_val : Target Series
        categorical_features : List of categorical column names

        Returns
        -------
        Fitted model (wrapped with threshold for classification)

        """
        try:
            import torch
            from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

            # Configure torch for stability on macOS
            if platform.system() == "Darwin":
                torch.set_num_threads(1)
        except ImportError as e:
            msg = "pytorch-tabnet not installed. Run: pip install pytorch-tabnet"
            raise ImportError(msg) from e

        # Load cached hyperparameters or run HPO
        best_params = self._maybe_load_cached_hparams()
        if best_params is None:
            logger.info("No cached TabNet hyperparameters; running Optuna...")
            best_params = self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
            self.store_best_hyperparameters(best_params)
        else:
            logger.info("Loaded cached TabNet hyperparameters.")

        # Encode categorical columns to integers (TabNet needs all numeric data)
        # Note: We disable TabNet's built-in categorical embeddings to avoid segfaults
        # and treat encoded categories as numeric features instead
        X_train = X_train.copy()
        X_val = X_val.copy()
        if categorical_features:
            for col in categorical_features:
                is_string_col = (
                    X_train[col].dtype == "object"
                    or str(X_train[col].dtype) == "category"
                )
                if col in X_train.columns and is_string_col:
                    all_categories = list(
                        set(X_train[col].dropna().unique())
                        | set(X_val[col].dropna().unique())
                    )
                    cat_mapping = {cat: i for i, cat in enumerate(all_categories)}
                    X_train[col] = (
                        X_train[col].map(cat_mapping).fillna(-1).astype(float)
                    )
                    X_val[col] = X_val[col].map(cat_mapping).fillna(-1).astype(float)

        # Build model parameters (no categorical embeddings - treated as numeric)
        # Force CPU to avoid MPS/GPU segfaults on macOS
        params = {
            "n_d": best_params.get("n_d", 24),
            "n_a": best_params.get("n_a", 24),
            "n_steps": best_params.get("n_steps", 4),
            "gamma": best_params.get("gamma", 1.3),
            "n_independent": best_params.get("n_independent", 2),
            "n_shared": best_params.get("n_shared", 2),
            "lambda_sparse": best_params.get("lambda_sparse", 1e-3),
            "momentum": best_params.get("momentum", 0.02),
            "clip_value": best_params.get("clip_value", 1.0),
            "verbose": 0,
            "seed": _RANDOM_STATE,
            "device_name": "cpu",
        }

        # Create model based on problem type
        if self.problem_type == "classification":
            model = TabNetClassifier(**params)
        else:
            model = TabNetRegressor(**params)

        # Convert to numpy arrays
        X_train_np = X_train.values.astype(np.float32)
        X_val_np = X_val.values.astype(np.float32)
        y_train_np = y_train.values.astype(
            np.float32 if self.problem_type == "regression" else np.int64
        )
        y_val_np = y_val.values.astype(
            np.float32 if self.problem_type == "regression" else np.int64
        )

        # Handle NaN and Inf values (can cause segfaults)
        train_nan_count = np.isnan(X_train_np).sum()
        val_nan_count = np.isnan(X_val_np).sum()
        train_inf_count = np.isinf(X_train_np).sum()
        val_inf_count = np.isinf(X_val_np).sum()
        if (
            train_nan_count > 0
            or val_nan_count > 0
            or train_inf_count > 0
            or val_inf_count > 0
        ):
            logger.info(
                "TabNet: replacing NaN/Inf values (train: nan=%d, inf=%d; val: nan=%d, inf=%d)",
                train_nan_count,
                train_inf_count,
                val_nan_count,
                val_inf_count,
            )
        X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_np = np.nan_to_num(X_val_np, nan=0.0, posinf=0.0, neginf=0.0)

        # Fit model
        logger.info(
            "Training TabNet (epochs=%d, patience=%d)...", _DEFAULT_EPOCHS, _PATIENCE
        )
        model.fit(
            X_train_np,
            y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            eval_name=["val"],
            eval_metric=["logloss"]
            if self.problem_type == "classification"
            else ["mae"],
            max_epochs=_DEFAULT_EPOCHS,
            patience=_PATIENCE,
            batch_size=_BATCH_SIZE,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
        )

        # For classification, tune decision threshold
        if self.problem_type == "classification":
            proba_val = model.predict_proba(X_val_np)[:, 1]
            thr = self._choose_threshold(y_val, proba_val)
            logger.info("TabNet decision threshold: %.4f", thr)
            return _TabNetWithThreshold(model, thr, list(X_train.columns))

        return model

    def _optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, Any]:
        """
        Optuna hyperparameter search for TabNet.

        Search space covers key TabNet architecture parameters.
        """
        try:
            import torch
            from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

            if platform.system() == "Darwin":
                torch.set_num_threads(1)
        except ImportError as e:
            msg = "pytorch-tabnet not installed"
            raise ImportError(msg) from e

        # Encode categorical columns to integers before converting to numpy
        X_train = X_train.copy()
        X_val = X_val.copy()
        for col in X_train.columns:
            if X_train[col].dtype == "object" or str(X_train[col].dtype) == "category":
                all_categories = list(
                    set(X_train[col].dropna().unique())
                    | set(X_val[col].dropna().unique())
                )
                # Start from 1, reserve 0 for unknown/NaN (TabNet needs non-negative indices)
                cat_mapping = {cat: i + 1 for i, cat in enumerate(all_categories)}
                X_train[col] = X_train[col].map(cat_mapping).fillna(0).astype(int)
                X_val[col] = X_val[col].map(cat_mapping).fillna(0).astype(int)

        # Convert data and handle NaN/Inf
        X_train_np = np.nan_to_num(
            X_train.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        X_val_np = np.nan_to_num(
            X_val.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        y_train_np = y_train.values.astype(
            np.int64 if self.problem_type == "classification" else np.float32
        )
        y_val_np = y_val.values.astype(
            np.int64 if self.problem_type == "classification" else np.float32
        )

        def _objective(trial: optuna.Trial) -> float:
            params = {
                "n_d": trial.suggest_int("n_d", 8, 64),
                "n_a": trial.suggest_int("n_a", 8, 64),
                "n_steps": trial.suggest_int("n_steps", 3, 8),
                "gamma": trial.suggest_float("gamma", 1.0, 2.0),
                "n_independent": trial.suggest_int("n_independent", 1, 4),
                "n_shared": trial.suggest_int("n_shared", 1, 4),
                "lambda_sparse": trial.suggest_float(
                    "lambda_sparse", 1e-5, 1e-2, log=True
                ),
                "momentum": trial.suggest_float("momentum", 0.01, 0.4),
                "verbose": 0,
                "seed": _RANDOM_STATE,
                "device_name": "cpu",
            }

            if self.problem_type == "classification":
                model = TabNetClassifier(**params)
            else:
                model = TabNetRegressor(**params)

            model.fit(
                X_train_np,
                y_train_np,
                eval_set=[(X_val_np, y_val_np)],
                eval_name=["val"],
                eval_metric=["logloss"]
                if self.problem_type == "classification"
                else ["mae"],
                max_epochs=100,  # Reduced for HPO
                patience=15,
                batch_size=_BATCH_SIZE,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False,
            )

            if self.problem_type == "classification":
                from sklearn.metrics import log_loss

                preds = model.predict_proba(X_val_np)[:, 1]
                preds = np.clip(preds, 1e-15, 1 - 1e-15)
                return float(log_loss(y_val_np, preds))
            from sklearn.metrics import mean_absolute_error

            preds = model.predict(X_val_np)
            return float(mean_absolute_error(y_val_np, preds))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=_RANDOM_STATE),
        )
        n_trials = min(self.trials, _TRIALS_CAP // 2)  # TabNet HPO is slower
        logger.info("TabNet Optuna: running %d trials...", n_trials)

        study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
        logger.info(
            "TabNet Optuna best: %.6f | params: %s",
            study.best_value,
            study.best_params,
        )
        return study.best_params

    def _maybe_load_cached_hparams(self) -> dict[str, Any] | None:
        """Load cached TabNet hyperparameters if available."""
        from utils.io_utils import load_model
        from utils.paths import MODEL_ARTIFACTS

        path = (
            MODEL_ARTIFACTS
            / self.model_name
            / f"{self.model_name}_best_hyperparameters.pkl"
        )
        try:
            if path.exists():
                hp = load_model(path)
                if isinstance(hp, dict) and hp:
                    return hp
        except Exception as e:
            logger.warning("Failed to load TabNet hyperparameters: %s", e)
        return None

    @staticmethod
    def _choose_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
        """Pick F1-maximizing threshold on validation."""
        from sklearn.metrics import f1_score

        best_thr, best_score = 0.5, -1.0
        for t in np.linspace(0.02, 0.98, 193):  # Match LightGBM resolution
            score = f1_score(y_true, proba >= t, zero_division=0)
            if score > best_score:
                best_score, best_thr = score, float(t)
        return best_thr
