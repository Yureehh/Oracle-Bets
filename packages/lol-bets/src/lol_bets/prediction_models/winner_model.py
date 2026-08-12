"""Predeclared rating baseline, LightGBM ensemble, and logit blend for Winner V2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
from oracle_bets_core.logger import logger
from oracle_bets_core.pd import pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lol_bets.prediction_models.gbdt_model import (
    BINARY_CLASS_UNIQUE_VALUES,
    MIN_CALIBRATION_SAMPLES,
)
from lol_bets.prediction_models.lightgbm_model import LightGBMModel

ENSEMBLE_MEMBERS = 10
BLEND_GRID = np.linspace(0.0, 1.0, 21)
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 50
DIRECT_RATING_LIKELIHOOD_FRAGMENTS = (
    "elo_win_likelihood",
    "glicko2_win_likelihood",
    "pl_win_likelihood",
    "trueskill_win_likelihood",
)
DIRECT_RATING_STATE_COLUMNS = (
    "delta_elo",
    "delta_glicko2_mu",
    "delta_glicko2_phi",
    "delta_pl_mu",
    "delta_pl_sigma",
    "delta_trueskill_mu",
    "delta_trueskill_sigma",
)
PROBABILITY_EPSILON = 1e-6


def is_direct_rating_feature(column: str) -> bool:
    """Return whether one canonical matchup column belongs in the baseline."""
    return column in DIRECT_RATING_STATE_COLUMNS or any(
        fragment in column for fragment in DIRECT_RATING_LIKELIHOOD_FRAGMENTS
    )


def has_complete_direct_rating_contract(columns: tuple[str, ...]) -> bool:
    """Require raw rating state and every independently derived likelihood."""
    return set(DIRECT_RATING_STATE_COLUMNS).issubset(columns) and all(
        any(fragment in column for column in columns)
        for fragment in DIRECT_RATING_LIKELIHOOD_FRAGMENTS
    )


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _expit(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


@dataclass
class WinnerBlendClassifier:
    """Serializable convex logit blend of an independent baseline and ensemble."""

    baseline: Any
    members: tuple[Any, ...]
    rating_columns: tuple[str, ...]
    blend_weight: float = 1.0
    calibration_bias_bound: float = 0.0
    baseline_calibrator: Any | None = None
    decision_threshold_: float = 0.5
    problem_type: str = "classification"

    @property
    def raw_model(self) -> Any:
        """Expose one ensemble member for compatible local-attribution tooling."""
        return self.members[0]

    @property
    def feature_importances_(self) -> np.ndarray:
        values = [
            np.asarray(member.feature_importances_, dtype=float)
            for member in self.members
        ]
        return np.mean(values, axis=0)

    def component_probabilities(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        baseline = np.asarray(
            self.baseline.predict_proba(X.loc[:, self.rating_columns])[:, 1],
            dtype=float,
        )
        full = np.mean(self.member_probabilities(X), axis=0)
        return baseline, full

    def rating_baseline_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Return the independently calibrated rating-only comparator."""
        baseline = np.asarray(
            self.baseline.predict_proba(X.loc[:, self.rating_columns])[:, 1],
            dtype=float,
        )
        if self.baseline_calibrator is not None:
            baseline = self.baseline_calibrator.predict(baseline)
        return np.asarray(baseline, dtype=float)

    def member_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        return np.vstack(
            [
                np.asarray(member.predict_proba(X)[:, 1], dtype=float)
                for member in self.members
            ]
        )

    def blended_member_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the selected component blend to every bootstrap member."""
        baseline = np.asarray(
            self.baseline.predict_proba(X.loc[:, self.rating_columns])[:, 1],
            dtype=float,
        )
        members = self.member_probabilities(X)
        return _expit(
            (1.0 - self.blend_weight) * _logit(baseline)[None, :]
            + self.blend_weight * _logit(members)
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        baseline, full = self.component_probabilities(X)
        point = _expit(
            (1.0 - self.blend_weight) * _logit(baseline)
            + self.blend_weight * _logit(full)
        )
        return np.column_stack([1.0 - point, point])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.decision_threshold_).astype(int)

    def conservative_probability(
        self,
        X: pd.DataFrame,
        *,
        calibrator: Any | None = None,
        metadata: pd.DataFrame | None = None,
    ) -> np.ndarray:
        members = self.blended_member_probabilities(X)
        if calibrator is not None:
            members = np.vstack(
                [calibrator.predict(values, metadata=metadata) for values in members]
            )
        lower = np.quantile(members, 0.10, axis=0) - self.calibration_bias_bound
        return np.clip(lower, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)

    def conservative_interval(
        self,
        X: pd.DataFrame,
        *,
        calibrator: Any | None = None,
        metadata: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        members = self.blended_member_probabilities(X)
        if calibrator is not None:
            members = np.vstack(
                [calibrator.predict(values, metadata=metadata) for values in members]
            )
        lower = np.quantile(members, 0.10, axis=0) - self.calibration_bias_bound
        upper = np.quantile(members, 0.90, axis=0) + self.calibration_bias_bound
        return (
            np.clip(lower, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON),
            np.clip(upper, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON),
        )


@dataclass
class WinnerLightGBMModel(LightGBMModel):
    """Train the fixed Winner V2 component family without market information."""

    def train_model(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        categorical_features: list[str] | None,
    ) -> WinnerBlendClassifier:
        params = self._maybe_load_cached_hparams()
        if params is None:
            if not self.force_retune:
                raise RuntimeError(
                    f"No reviewed Winner V2 parameters for {self.model_name}; "
                    "run an explicit research retune first."
                )
            params = self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
        rating_columns = tuple(
            column for column in X_train.columns if is_direct_rating_feature(column)
        )
        if not has_complete_direct_rating_contract(rating_columns):
            raise ValueError("Winner V2 is missing direct rating state or likelihoods")

        baseline = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=2000, random_state=RANDOM_STATE),
        )
        baseline.fit(X_train.loc[:, rating_columns], y_train)
        weeks = self._training_weeks(X_train)
        rng = np.random.default_rng(RANDOM_STATE)
        unique_weeks = np.asarray(sorted(weeks.unique()))
        members: list[lgb.LGBMClassifier] = []
        for member_index in range(ENSEMBLE_MEMBERS):
            sampled_weeks = rng.choice(
                unique_weeks, size=len(unique_weeks), replace=True
            )
            sampled_indices = [
                index
                for week in sampled_weeks
                for index in weeks.index[weeks == week].tolist()
            ]
            X_bootstrap = X_train.loc[sampled_indices].reset_index(drop=True)
            y_bootstrap = y_train.loc[sampled_indices].reset_index(drop=True)
            members.append(
                self._fit_member(
                    params,
                    X_bootstrap,
                    y_bootstrap,
                    X_val,
                    y_val,
                    categorical_features,
                    seed=RANDOM_STATE + member_index,
                )
            )
        return WinnerBlendClassifier(
            baseline=baseline,
            members=tuple(members),
            rating_columns=rating_columns,
        )

    def _optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, Any]:
        """Search only expanding rolling-origin folds inside the 60% dev window."""
        X_development = pd.concat([X_train, X_val])
        y_development = pd.concat([y_train, y_val])
        metadata = pd.concat(
            [
                self.fit_partition_metadata["train"],
                self.fit_partition_metadata["tune"],
            ]
        )
        weeks = pd.to_datetime(metadata.loc[X_development.index, "date"]).dt.strftime(
            "%G-W%V"
        )
        unique_weeks = np.asarray(sorted(weeks.unique()))
        if len(unique_weeks) < 4:  # noqa: PLR2004
            raise ValueError(
                "Winner V2 retuning requires at least four historical weeks"
            )
        blocks = np.array_split(unique_weeks, 4)
        categorical = [
            column
            for column in X_development
            if str(X_development[column].dtype) == "category"
        ]

        def objective(trial: optuna.Trial) -> float:
            params: dict[str, Any] = {
                "boosting_type": "gbdt",
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.12, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 20, 120),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 30, 150),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
                "n_estimators": 3000,
                "verbosity": -1,
                "n_jobs": -1,
            }
            losses: list[float] = []
            for fold_index in range(1, len(blocks)):
                train_weeks = set(np.concatenate(blocks[:fold_index]))
                validation_weeks = set(blocks[fold_index])
                train_index = weeks.index[weeks.isin(train_weeks)]
                validation_index = weeks.index[weeks.isin(validation_weeks)]
                model = self._fit_member(
                    params,
                    X_development.loc[train_index],
                    y_development.loc[train_index],
                    X_development.loc[validation_index],
                    y_development.loc[validation_index],
                    categorical,
                    seed=RANDOM_STATE + fold_index,
                )
                probability = np.asarray(
                    model.predict_proba(X_development.loc[validation_index]),
                    dtype=float,
                )[:, 1]
                losses.append(
                    float(log_loss(y_development.loc[validation_index], probability))
                )
            return float(np.mean(losses))

        logger.info(
            "Winner V2 Optuna: %d rolling-origin trials inside development only.",
            self.trials,
        )
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(objective, n_trials=self.trials, show_progress_bar=False)
        self.store_best_hyperparameters(
            study.best_params, score=float(study.best_value)
        )
        return dict(study.best_params)

    def fit_probability_calibrator(self, model, *args, **kwargs):
        X_select = args[2]
        y_select = args[3]
        losses: list[tuple[float, float]] = []
        for weight in BLEND_GRID:
            model.blend_weight = float(weight)
            probability = model.predict_proba(X_select)[:, 1]
            losses.append((float(log_loss(y_select, probability)), float(weight)))
        model.blend_weight = min(losses)[1]
        self._store_winner_report(
            {
                "blend_weight": model.blend_weight,
                "selection_split": "calibration_select",
                "selection_log_loss": min(losses)[0],
                "ensemble_members": len(model.members),
                "rating_columns": list(model.rating_columns),
            }
        )
        calibrator = super().fit_probability_calibrator(model, *args, **kwargs)
        model.baseline_calibrator = self._fit_rating_baseline_calibrator(model, *args)
        self._store_winner_report(
            {
                "rating_baseline_calibration": (
                    model.baseline_calibrator.method
                    if model.baseline_calibrator is not None
                    else "unavailable"
                )
            },
            merge=True,
        )
        return calibrator

    def _fit_rating_baseline_calibrator(self, model, *args):
        X_fit, y_fit, X_select, y_select, X_full, y_full = args[:6]
        y_fit = pd.to_numeric(y_fit, errors="coerce").dropna()
        y_select = pd.to_numeric(y_select, errors="coerce").dropna()
        y_full = pd.to_numeric(y_full, errors="coerce").dropna()
        if (
            self.calibration == "none"
            or len(y_fit) < MIN_CALIBRATION_SAMPLES
            or y_fit.nunique() != BINARY_CLASS_UNIQUE_VALUES
            or y_select.nunique() != BINARY_CLASS_UNIQUE_VALUES
            or y_full.nunique() != BINARY_CLASS_UNIQUE_VALUES
        ):
            return None
        methods = ["raw", "sigmoid", "isotonic"]
        if self.calibration_method != "auto":
            methods = [self.calibration_method]
        p_fit = model.baseline.predict_proba(
            X_fit.loc[y_fit.index, model.rating_columns]
        )[:, 1]
        p_select = model.baseline.predict_proba(
            X_select.loc[y_select.index, model.rating_columns]
        )[:, 1]
        p_full = model.baseline.predict_proba(
            X_full.loc[y_full.index, model.rating_columns]
        )[:, 1]
        candidates: list[tuple[float, float, str]] = []
        for method in methods:
            candidate = self._fit_probability_candidate(method, p_fit, y_fit)
            metrics = self._probability_quality_metrics(
                y_select, candidate.predict(p_select)
            )
            candidates.append((metrics["log_loss"], metrics["brier"], method))
        selected_method = min(candidates)[2]
        return self._fit_probability_candidate(selected_method, p_full, y_full)

    def fit_probability_uncertainty(
        self,
        model,
        X_uncertainty: pd.DataFrame,
        y_uncertainty: pd.Series,
        metadata: pd.DataFrame | None = None,
    ):
        uncertainty = super().fit_probability_uncertainty(
            model, X_uncertainty, y_uncertainty, metadata
        )
        point = model.predict_proba(X_uncertainty)[:, 1]
        if self.probability_calibrator is not None:
            point = self.probability_calibrator.predict(point, metadata=metadata)
        residual = point - np.asarray(y_uncertainty, dtype=float)
        model.calibration_bias_bound = max(0.0, float(np.quantile(residual, 0.90)))
        self._store_winner_report(
            {
                "calibration_bias_bound": model.calibration_bias_bound,
                "bias_split": "uncertainty_fit",
            },
            merge=True,
        )
        return uncertainty

    def _training_weeks(self, X_train: pd.DataFrame) -> pd.Series:
        metadata = self.fit_partition_metadata.get("train")
        if metadata is None or "date" not in metadata:
            raise ValueError("Winner V2 week-block bootstrap requires training dates")
        dates = pd.to_datetime(metadata.loc[X_train.index, "date"], errors="coerce")
        if dates.isna().any():
            raise ValueError("Winner V2 training dates contain missing values")
        return dates.dt.strftime("%G-W%V")

    @staticmethod
    def _fit_member(
        params: dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        categorical_features: list[str] | None,
        *,
        seed: int,
    ) -> lgb.LGBMClassifier:
        member_params = dict(params)
        member_params |= {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "random_state": seed,
            "n_jobs": -1,
        }
        member_params.setdefault("n_estimators", 3000)
        model = lgb.LGBMClassifier(**member_params)
        fit_kwargs: dict[str, Any] = {
            "X": X_train,
            "y": y_train,
            "eval_X": X_val,
            "eval_y": y_val,
            "eval_metric": "binary_logloss",
            "callbacks": [
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
        }
        if categorical_features:
            fit_kwargs["categorical_feature"] = categorical_features
        model.fit(**fit_kwargs)
        return model

    def _store_winner_report(
        self, payload: dict[str, Any], *, merge: bool = False
    ) -> None:
        path = self.insight_path("winner_model_report.json")
        if merge and path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8")) | payload
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
