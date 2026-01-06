# prediction_models/lightgbm_model.py
"""
LightGBM Model and Model Factory (lean, laptop-friendly).

- Strong defaults + early stopping
- Optuna HPO (trial cap)
- Auto class-imbalance handling (scale_pos_weight)
- Validation-optimized decision threshold for classification
- Drop-in compatible with GradientBoostingModel.train_and_validate_model(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import f1_score, log_loss, mean_absolute_error

from prediction_models.gbdt_model import DEFAULT_TRIALS as _TRIALS_CAP
from prediction_models.gbdt_model import GradientBoostingModel
from utils.io_utils import load_model
from utils.logger import logger
from utils.paths import MODEL_ARTIFACTS  # <-- artifacts root

if TYPE_CHECKING:
    from utils.pd import pd


# Lightweight, good defaults for laptop runs
_DEFAULT_N_ESTIMATORS = 3000
_EARLY_STOP_ROUNDS = 50
_RANDOM_STATE = 42


class _LGBWithThreshold:
    """Wraps an LGBMClassifier so .predict uses a tuned threshold; passes through .predict_proba and importances."""

    def __init__(self, raw_model: lgb.LGBMClassifier, threshold: float):
        # sourcery skip: remove-unnecessary-cast
        self.raw_model = raw_model
        self.decision_threshold_ = float(threshold)
        self.problem_type = "classification"

    @property
    def feature_importances_(self):
        return getattr(self.raw_model, "feature_importances_", None)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.raw_model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self.predict_proba(X)[:, 1]
        return (p >= self.decision_threshold_).astype(int)


@dataclass
class LightGBMModel(GradientBoostingModel):
    """
    Minimal subclass that implements:
      - train_model(...)
      - _optimize_hyperparameters(...)
    Uses the base class orchestration, splits, pruning, imputations, and observability.
    """

    # ─────────────────────────── training hook ─────────────────────────── #

    def train_model(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        categorical_features: list[str] | None,
    ) -> Any:  # sourcery skip: move-assign
        """Fit a single LightGBM model with early stopping and a tuned decision threshold (classification)."""
        # Load cached best hparams if present; otherwise run HPO (clamped for laptops)
        best_params = self._maybe_load_cached_hparams()
        if best_params is None:
            logger.info(
                "No cached hyperparameters found; running Optuna (cap=%d).", _TRIALS_CAP
            )
            best_params = self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
            self.store_best_hyperparameters(best_params)
        else:
            logger.info("Loaded cached hyperparameters for %s.", self.model_name)

        params = dict(best_params)
        params.setdefault("n_estimators", _DEFAULT_N_ESTIMATORS)
        params.setdefault("verbosity", -1)
        params.setdefault("random_state", _RANDOM_STATE)
        params.setdefault("n_jobs", -1)
        params.setdefault(
            "objective",
            "binary" if self.problem_type == "classification" else "regression",
        )
        params.setdefault(
            "metric",
            "binary_logloss" if self.problem_type == "classification" else "mae",
        )

        # Auto class-imbalance handling (always recomputed for current data)
        if self.problem_type == "classification":
            pos = float((y_train == 1).sum())
            neg = float((y_train == 0).sum())
            if pos > 0:
                params["scale_pos_weight"] = max(1.0, neg / pos)

        model_cls = (
            lgb.LGBMClassifier
            if self.problem_type == "classification"
            else lgb.LGBMRegressor
        )
        model = model_cls(**params)

        fit_kwargs: dict[str, Any] = {
            "X": X_train,
            "y": y_train,
            "eval_set": [(X_val, y_val)],
            "eval_metric": params.get("metric"),
        }
        if categorical_features:
            fit_kwargs["categorical_feature"] = categorical_features

        # Prefer callbacks (newer LightGBM); fallback otherwise
        try:
            callbacks = [
                lgb.early_stopping(_EARLY_STOP_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ]
            fit_kwargs["callbacks"] = callbacks
        except Exception:
            fit_kwargs["early_stopping_rounds"] = _EARLY_STOP_ROUNDS

        model.fit(**fit_kwargs)

        # For classification, tune a simple F1-based threshold on validation
        if self.problem_type == "classification":
            proba_val = model.predict_proba(X_val)[:, 1]
            thr = self._choose_threshold(y_val, proba_val)
            logger.info("Chosen decision threshold on validation: %.4f (F1)", thr)
            return _LGBWithThreshold(model, thr)

        return model

    # ─────────────────────── hyperparameter search ─────────────────────── #

    def _optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, Any]:
        """Optuna search with a compact space + early stopping. Good perf / runtime tradeoff for laptops."""

        def _store_if_best(study: optuna.Study, trial: optuna.Trial) -> None:
            """Persist hyperparameters whenever Optuna finds a new best."""
            try:
                if study.best_trial == trial:
                    self.store_best_hyperparameters(trial.params)
            except Exception as e:
                logger.debug("Failed to store interim best hyperparameters: %s", e)

        def _objective(trial: optuna.Trial) -> float:
            params: dict[str, Any] = {
                "boosting_type": trial.suggest_categorical(
                    "boosting_type", ["gbdt", "dart"]
                ),
                "objective": "binary"
                if self.problem_type == "classification"
                else "regression",
                "metric": "binary_logloss"
                if self.problem_type == "classification"
                else "mae",
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 0.1, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 31, 256),
                "max_depth": trial.suggest_int("max_depth", -1, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
                "n_estimators": _DEFAULT_N_ESTIMATORS,
                "verbosity": -1,
                "random_state": _RANDOM_STATE,
                "n_jobs": -1,
            }

            # auto imbalance heuristic
            if self.problem_type == "classification":
                pos = float((y_train == 1).sum())
                neg = float((y_train == 0).sum())
                if pos > 0:
                    params["scale_pos_weight"] = max(1.0, neg / pos)

            model_cls = (
                lgb.LGBMClassifier
                if self.problem_type == "classification"
                else lgb.LGBMRegressor
            )
            clf = model_cls(**params)

            fit_kwargs: dict[str, Any] = {
                "X": X_train,
                "y": y_train,
                "eval_set": [(X_val, y_val)],
                "eval_metric": params.get("metric"),
            }
            cat_cols = [
                c for c in X_train.columns if str(X_train[c].dtype) == "category"
            ]
            if cat_cols:
                fit_kwargs["categorical_feature"] = cat_cols
            try:
                callbacks = [
                    lgb.early_stopping(_EARLY_STOP_ROUNDS, verbose=False),
                    lgb.log_evaluation(0),
                ]
                fit_kwargs["callbacks"] = callbacks
            except Exception:
                fit_kwargs["early_stopping_rounds"] = _EARLY_STOP_ROUNDS

            clf.fit(**fit_kwargs)

            if self.problem_type == "classification":
                preds = clf.predict_proba(X_val)[:, 1]
                # sklearn >= 1.5: no 'eps' kwarg -> clip manually to avoid log(0)
                preds = np.clip(preds, 1e-15, 1 - 1e-15)
                return float(log_loss(y_val, preds))
            preds = clf.predict(X_val)
            return float(mean_absolute_error(y_val, preds))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=_RANDOM_STATE),
        )
        n_trials = min(self.trials, _TRIALS_CAP)
        logger.info("Optuna: running %d trials (cap).", n_trials)
        study.optimize(
            _objective,
            n_trials=n_trials,
            show_progress_bar=False,
            callbacks=[_store_if_best],
        )
        logger.info(
            "Optuna best value: %.6f | params: %s", study.best_value, study.best_params
        )
        # Ensure final best is stored even if callback failed silently
        self.store_best_hyperparameters(study.best_params)
        return study.best_params

    # ─────────────────────────── utilities ─────────────────────────── #

    def _maybe_load_cached_hparams(self) -> dict[str, Any] | None:
        """Load best_hyperparameters from models/artifacts/<ModelName>/"""
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
            logger.warning("Failed to load cached hyperparameters from %s: %s", path, e)
        return None

    @staticmethod
    def _choose_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
        """Pick F1-maximizing threshold on validation with a dense sweep."""
        best_thr, best_score = 0.5, -1.0
        for t in np.linspace(0.02, 0.98, 193):
            score = f1_score(y_true, proba >= t, zero_division=0)
            if score > best_score:
                best_score, best_thr = score, float(t)
        return best_thr


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
        if problem_type not in ["classification", "regression"]:
            msg = f"Unsupported problem type: {problem_type}"
            raise ValueError(msg)
        if model_type != "lightgbm":
            msg = f"Unsupported model type: {model_type}"
            raise ValueError(msg)

        return LightGBMModel(
            model_name=model_name,
            problem_type=problem_type,
            team_data=training_team_data,
            player_data=training_player_data,
        )
