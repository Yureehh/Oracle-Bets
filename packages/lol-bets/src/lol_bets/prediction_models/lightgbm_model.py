# prediction_models/lightgbm_model.py
"""
LightGBM Model and Model Factory (lean, laptop-friendly).

- Strong defaults + early stopping
- Optuna HPO (trial cap)
- Auto class-imbalance handling (scale_pos_weight)
- Probability-first classification with a fixed 0.5 decision threshold
- Drop-in compatible with GradientBoostingModel.train_and_validate_model(...)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np
import optuna
from oracle_bets_core.logger import logger
from oracle_bets_core.paths import TUNED_LIGHTGBM_HYPERPARAMETERS
from sklearn.metrics import log_loss

from lol_bets.prediction_models.gbdt_model import (
    DEFAULT_TRIALS as _TRIALS_CAP,
)
from lol_bets.prediction_models.gbdt_model import (
    GradientBoostingModel,
)

if TYPE_CHECKING:
    from oracle_bets_core.pd import pd


# Lightweight, good defaults for laptop runs
_DEFAULT_N_ESTIMATORS = 3000
_EARLY_STOP_ROUNDS = 50
_RANDOM_STATE = 42
_ALLOWED_BOOSTING_TYPES = ("gbdt",)
_HPARAM_CACHE_VERSION = 2
_HPARAM_PARAMS_KEY = "params"
_HPARAM_METADATA_KEY = "metadata"


class _LGBWithThreshold:
    """Wraps an LGBMClassifier so .predict uses a fixed threshold."""

    def __init__(self, raw_model: lgb.LGBMClassifier, threshold: float):
        # sourcery skip: remove-unnecessary-cast
        self.raw_model = raw_model
        self.decision_threshold_ = float(threshold)
        self.problem_type = "classification"

    @property
    def feature_importances_(self):
        return getattr(self.raw_model, "feature_importances_", None)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.raw_model.predict_proba(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        if proba.ndim == 1 or proba.shape[1] < 2:  # noqa: PLR2004
            msg = f"Expected binary classification proba, got shape {proba.shape}"
            raise ValueError(msg)
        return (proba[:, 1] >= self.decision_threshold_).astype(int)


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
        """Fit a single LightGBM model with early stopping."""
        # Retuning is explicit; routine retraining must never launch Optuna silently.
        best_params = self._maybe_load_cached_hparams()
        if best_params is None:
            if not self.force_retune:
                msg = (
                    f"No compatible production hyperparameters for {self.model_name}. "
                    "Run `oracle-bets lol retune`, review its report, then run "
                    "`oracle-bets lol promote-tuning <run-id>`."
                )
                raise RuntimeError(msg)
            logger.info("Running explicit Optuna retune (cap=%d).", _TRIALS_CAP)
            best_params = self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            logger.info("Loaded reviewed hyperparameters for %s.", self.model_name)

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
            "binary_logloss" if self.problem_type == "classification" else "rmse",
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
            "eval_X": X_val,
            "eval_y": y_val,
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
        except (AttributeError, TypeError):
            fit_kwargs["early_stopping_rounds"] = _EARLY_STOP_ROUNDS

        model.fit(**fit_kwargs)

        if self.problem_type == "classification":
            logger.info(
                "Using fixed 0.5000 decision threshold for probability-first classification."
            )
            return _LGBWithThreshold(model, 0.5)

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

        def _store_if_best(
            study: optuna.Study,
            trial: optuna.trial.FrozenTrial,
        ) -> None:
            """Persist hyperparameters whenever Optuna finds a new best."""
            try:
                if study.best_trial == trial:
                    if trial.value is None:
                        return
                    self.store_best_hyperparameters(
                        trial.params,
                        score=float(trial.value),
                    )
            except Exception as e:
                logger.debug("Failed to store interim best hyperparameters: %s", e)

        def _objective(trial: optuna.Trial) -> float:
            params: dict[str, Any] = {
                "boosting_type": trial.suggest_categorical(
                    "boosting_type", list(_ALLOWED_BOOSTING_TYPES)
                ),
                "objective": "binary"
                if self.problem_type == "classification"
                else "regression",
                "metric": "binary_logloss"
                if self.problem_type == "classification"
                else "rmse",
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.15, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.3),
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
                "eval_X": X_val,
                "eval_y": y_val,
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
            except (AttributeError, TypeError):
                fit_kwargs["early_stopping_rounds"] = _EARLY_STOP_ROUNDS

            clf.fit(**fit_kwargs)

            if self.problem_type == "classification":
                preds = np.asarray(clf.predict_proba(X_val))[:, 1]
                # sklearn >= 1.5: no 'eps' kwarg -> clip manually to avoid log(0)
                preds = np.clip(preds, 1e-15, 1 - 1e-15)
                return float(log_loss(y_val, preds))
            preds = clf.predict(X_val)
            residuals = np.asarray(y_val, dtype=float) - np.asarray(preds, dtype=float)
            return float(np.sqrt(np.mean(np.square(residuals))))

        n_trials = min(self.trials, _TRIALS_CAP)
        logger.info("Optuna: running %d trials (cap).", n_trials)
        previous_verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        try:
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=_RANDOM_STATE),
            )
            study.optimize(
                _objective,
                n_trials=n_trials,
                show_progress_bar=False,
                callbacks=[_store_if_best],
            )
        finally:
            optuna.logging.set_verbosity(previous_verbosity)
        logger.info(
            "Optuna best value: %.6f | params: %s", study.best_value, study.best_params
        )
        # Ensure final best is stored even if callback failed silently
        self.store_best_hyperparameters(
            study.best_params,
            score=float(study.best_value),
        )
        return study.best_params

    # ─────────────────────────── utilities ─────────────────────────── #

    def _hparams_path(self):
        if self.force_retune:
            if self.report_root is None:
                raise RuntimeError(
                    "Research retuning requires a run-scoped report root."
                )
            return self.report_root / self.model_name / "tuned_hyperparameters.json"
        return TUNED_LIGHTGBM_HYPERPARAMETERS / f"{self.model_name}.json"

    def _feature_schema_fingerprint(self) -> str:
        payload = json.dumps(
            {
                "team_columns": sorted(map(str, self.team_data.columns)),
                "player_columns": sorted(map(str, self.player_data.columns)),
                "feature_set": self.feature_set,
                "max_features": self.max_features,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _hparam_cache_metadata(self) -> dict[str, Any]:
        return {
            "version": _HPARAM_CACHE_VERSION,
            "model_type": "lightgbm",
            "model_name": self.model_name,
            "problem_type": self.problem_type,
            "feature_set": self.feature_set,
            "max_features": self.max_features,
            "calibration": self.calibration,
            "calibration_method": self.calibration_method,
            "calibration_size": self.calibration_size,
            "tune_size": self.tune_size,
            "test_size": self.test_size,
            "feature_schema_fingerprint": self._feature_schema_fingerprint(),
            "random_seed": _RANDOM_STATE,
            "objective": (
                "binary_logloss" if self.problem_type == "classification" else "rmse"
            ),
            "code_version": self.git_code_version(),
            "search_space_version": 1,
            "dataset_fingerprint": self.dataset_fingerprint,
        }

    def store_best_hyperparameters(
        self,
        hyperparams: dict[str, Any],
        *,
        score: float | None = None,
    ) -> None:
        """Persist tuned params with provenance needed for safe cache reuse."""
        path = self._hparams_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            _HPARAM_PARAMS_KEY: dict(hyperparams),
            _HPARAM_METADATA_KEY: self._hparam_cache_metadata()
            | {"validation_score": score},
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        logger.debug("Stored %s", path.name)

    def _validate_cached_hparams(self, hp: Any) -> dict[str, Any] | None:
        if not isinstance(hp, dict) or not hp:
            return None
        if hp.get("boosting_type", "gbdt") not in _ALLOWED_BOOSTING_TYPES:
            logger.info(
                "Ignoring cached hyperparameters for %s: unsupported boosting_type=%s.",
                self.model_name,
                hp.get("boosting_type"),
            )
            return None
        return hp

    def _params_from_cache_payload(self, payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, dict) or not payload:
            return None
        if {
            _HPARAM_PARAMS_KEY,
            _HPARAM_METADATA_KEY,
        } <= set(payload):
            metadata = payload[_HPARAM_METADATA_KEY]
            current = self._hparam_cache_metadata()
            identity_keys = (
                "version",
                "model_type",
                "model_name",
                "problem_type",
                "feature_set",
                "max_features",
            )
            if any(metadata.get(key) != current[key] for key in identity_keys):
                logger.info(
                    "Ignoring cached hyperparameters for %s: cache model identity does not match.",
                    self.model_name,
                )
                return None
            schema_matches = (
                metadata.get("feature_schema_fingerprint")
                == current["feature_schema_fingerprint"]
            )
            if not schema_matches and not self.allow_hparam_schema_drift:
                logger.info(
                    "Ignoring cached hyperparameters for %s: feature schema changed.",
                    self.model_name,
                )
                return None
            if not schema_matches:
                logger.warning(
                    "Reusing reviewed hyperparameters for shadow model %s across "
                    "feature-schema drift; retune before treating it as actionable.",
                    self.model_name,
                )
            if metadata != current:
                logger.info(
                    "Reusing reviewed hyperparameters for %s across non-schema run settings.",
                    self.model_name,
                )
            return self._validate_cached_hparams(payload[_HPARAM_PARAMS_KEY])
        logger.info(
            "Ignoring cached hyperparameters for %s: cache has no provenance metadata.",
            self.model_name,
        )
        return None

    def _maybe_load_cached_hparams(self) -> dict[str, Any] | None:
        """Load reviewed production hyperparameters from tracked JSON."""
        if self.force_retune:
            logger.info(
                "Ignoring production hyperparameters for explicit retuning of %s.",
                self.model_name,
            )
            return None
        path = self._hparams_path()
        try:
            if path.exists():
                return self._params_from_cache_payload(
                    json.loads(path.read_text(encoding="utf-8"))
                )
        except Exception as e:
            logger.warning("Failed to load cached hyperparameters from %s: %s", path, e)
        return None
