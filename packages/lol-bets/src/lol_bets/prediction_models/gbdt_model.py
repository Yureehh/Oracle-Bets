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
import hashlib
import json
import pickle
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from oracle_bets_core.io_utils import json_loader
from oracle_bets_core.logger import logger
from oracle_bets_core.paths import (
    FEATURE_REPORTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_TEAMS,
    SUITE_ROOT,
    TRAINING_COMPACT_PLAYER_CONFIG,
    TRAINING_COMPACT_TEAM_CONFIG,
)
from oracle_bets_core.pd import pd
from oracle_bets_core.probabilities import PropLineProbability, probability_over_under
from pandas.api.types import is_numeric_dtype
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from lol_bets.prediction_models.data_preprocessor import DataPreprocessor
from lol_bets.prediction_models.feature_selector import FeatureSelector
from lol_bets.prediction_models.observability import MLObservabilityMixin
from lol_bets.prediction_models.prop_features import (
    build_game_level_outcome_features,
    build_game_level_prop_features,
    is_prop_target,
)

if TYPE_CHECKING:
    from pathlib import Path

# Feature selection method type
FeatureSelectionMethod = Literal["none", "importance", "cumulative", "report"]
TrainingFeatureSet = Literal["full", "compact", "selected", "auto"]
CalibrationMode = Literal["auto", "none"]
CalibrationMethod = Literal["raw", "sigmoid", "isotonic", "auto"]

# Defaults
DEFAULT_TRIALS = 100
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
TUNE_SIZE = 0.10
CALIBRATION_SIZE = 0.15
LOW_STD_THRESHOLD = 0.03  # coefficient of variation threshold (less aggressive)
HIGH_CORR_THRESHOLD = 0.95  # Pearson correlation threshold (keep more features)
MAX_MISSING_FRAC = 0.40  # drop features missing >40% on TRAIN
RANDOM_STATE = 42
MIN_TEMPORAL_SPLITS = 3
MIN_CALIBRATION_SPLITS = 6
WINNER_V2_MODEL_NAMES = frozenset(
    {"SeriesWinnerPrediction_LightGBM", "NextMapWinnerPrediction_LightGBM"}
)
BINARY_CLASS_UNIQUE_VALUES = 2
ROWS_PER_GAME = 2
POSITIVE_RESULT_SUM_PER_GAME = 1
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5
MIN_CALIBRATION_SAMPLES = 40
MIN_UNCERTAINTY_SAMPLES = 30
MIN_UNCERTAINTY_BIN_SAMPLES = 20
MIN_UNCERTAINTY_WEEK_BLOCKS = 4
CONSERVATIVE_COVERAGE_TARGET = 0.90
MAX_CONSERVATIVE_COVERAGE_SHORTFALL = 0.02
MIN_SEGMENT_SIGMOID_SAMPLES = 80
MIN_SEGMENT_ISOTONIC_SAMPLES = 150
MIN_PROP_COHORT_SIZE = 30
PROBABILITY_EPSILON = 1e-6
CALIBRATION_BINS = 10
CALIBRATION_SEGMENT_SHRINKAGE = 120
CALIBRATION_VERSION = 3
PROBABILITY_UNCERTAINTY_VERSION = 2
MIN_CALIBRATION_SLOPE = 0.8
MAX_CALIBRATION_SLOPE = 1.2
MAX_ABSOLUTE_CALIBRATION_INTERCEPT = 0.10
CONFIDENCE_BAND_LOW = 0.55
CONFIDENCE_BAND_HIGH = 0.70
PATCH_FAMILY_PARTS = 2
COMPACT_ROLE_PREFIXES = ("top", "jng", "mid", "bot", "sup")
DEFAULT_SELECTED_MAX_FEATURES = 120
SELECTED_FEATURE_COUNTS = (60, 90, 120, 160)
MANDATORY_ANCHOR_PATTERNS = (
    "win_likelihood",
    "rating_consensus",
    "rating_disagreement",
    "strength_pool",
    "league_elo",
    "first_pick",
    "side_win_likelihood",
    "season_win_likelihood",
    "h2h_",
    "days_since_last_game",
    "roster_continuity",
    "uncertainty",
    "diff_ema_golddiff",
    "diff_ema_xpdiff",
    "diff_ema_csdiff",
    "diff_ema_team_vspm",
    "diff_ema_team_wcpm",
    "diff_ema_kda",
    "diff_ema_kill_participation",
    "diff_ema_damageshare",
    "diff_ema_earnedgoldshare",
)
DIRECT_RATING_LIKELIHOODS = (
    "elo_win_likelihood",
    "glicko2_win_likelihood",
    "pl_win_likelihood",
    "trueskill_win_likelihood",
)
PLAYER_RATING_LIKELIHOODS = tuple(f"players_{col}" for col in DIRECT_RATING_LIKELIHOODS)
DERIVED_STRENGTH_FEATURES = (
    "team_rating_consensus",
    "team_rating_disagreement",
    "players_rating_consensus",
    "players_rating_disagreement",
    "rating_consensus",
    "rating_disagreement",
)


def _model_feature_lineage(
    feature: str,
    *,
    next_map: bool = False,
) -> dict[str, Any]:
    """Describe one final matchup feature without claiming post-start availability."""
    source = feature
    swap_behavior = "canonical_team_a_minus_team_b"
    for prefix, behavior in (
        ("delta_", "canonical_team_a_minus_team_b"),
        ("context_", "invariant_context"),
        ("pair_", "ordered_pair_invariant"),
    ):
        if source.startswith(prefix):
            source = source.removeprefix(prefix)
            swap_behavior = behavior
            break
    lowered = source.casefold()
    if any(token in lowered for token in ("elo", "glicko", "pl_", "trueskill")):
        family = "ratings"
    elif "roster" in lowered or any(
        lowered.startswith(f"{role}_") for role in COMPACT_ROLE_PREFIXES
    ):
        family = "roster_and_players"
    elif any(token in lowered for token in ("ema_", "win_rate", "h2h", "season")):
        family = "historical_form"
    else:
        family = "prematch_context"
    next_map_state = any(
        token in lowered
        for token in (
            "maps_completed",
            "next_map_number",
            "series_wins_before",
            "series_losses_before",
            "series_score",
        )
    )
    return {
        "feature": feature,
        "source": source,
        "availability_timestamp": (
            "after_previous_map_before_target_map"
            if next_map and next_map_state
            else "strictly_before_fixture_start"
        ),
        "family": family,
        "swap_behavior": swap_behavior,
        "model_eligible": True,
    }


@dataclass
class ProbabilityCalibrator:
    """Validation-fitted probability calibration wrapper."""

    method: str
    model: Any

    def predict(
        self, probabilities: np.ndarray, metadata: pd.DataFrame | None = None
    ) -> np.ndarray:
        _ = metadata
        values = np.asarray(probabilities, dtype=float)
        if self.method == "raw" or self.model is None:
            calibrated = values
        elif self.method == "sigmoid":
            calibrated = self.model.predict_proba(values.reshape(-1, 1))[:, 1]
        else:
            calibrated = self.model.predict(values)
        return np.clip(calibrated, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)


@dataclass(frozen=True)
class ProbabilityResidualBand:
    """One probability region's held-out calibration-residual uncertainty."""

    probability_lower: float
    probability_upper: float
    residual_lower: float
    residual_upper: float
    sample_count: int

    def contains(self, probability: float) -> bool:
        return self.probability_lower <= probability and (
            probability < self.probability_upper
            or (self.probability_upper >= 1.0 and probability <= 1.0)
        )


@dataclass(frozen=True)
class ProbabilityUncertaintyModel:
    """
    Uncertainty in a calibrated probability estimate, fitted out of sample.

    This is not an outcome interval and must not be presented as a guarantee.
    """

    global_residual_lower: float
    global_residual_upper: float
    sample_count: int
    calibration_units: int
    confidence: float = 0.90
    bins: tuple[ProbabilityResidualBand, ...] = ()
    method: str = "week_block_q10_plus_one_sided_calibration_bias"
    fit_split: str = "uncertainty_fit"
    unit: str = "iso_week"
    version: int = PROBABILITY_UNCERTAINTY_VERSION

    @classmethod
    def fit(
        cls,
        y_true: pd.Series | np.ndarray,
        probabilities: np.ndarray,
        *,
        timestamps: pd.Series | np.ndarray,
        confidence: float = 0.90,
        bin_count: int = 5,
    ) -> ProbabilityUncertaintyModel:
        """Fit one-sided week-block calibration-bias bounds on a temporal holdout."""
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        if bin_count < 1:
            raise ValueError("bin_count must be positive")

        frame = pd.DataFrame(
            {
                "actual": pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy(),
                "probability": np.asarray(probabilities, dtype=float),
                "timestamp": pd.to_datetime(
                    pd.Series(timestamps), errors="coerce", utc=True
                ).to_numpy(),
            }
        ).replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna(subset=["actual", "probability", "timestamp"])
        frame = frame[
            frame["actual"].isin([0, 1])
            & frame["probability"].between(0.0, 1.0, inclusive="both")
        ].copy()
        if len(frame) < MIN_UNCERTAINTY_SAMPLES:
            raise ValueError(
                "Probability uncertainty requires at least "
                f"{MIN_UNCERTAINTY_SAMPLES} held-out rows."
            )

        frame["residual"] = frame["actual"] - frame["probability"]
        frame["week"] = frame["timestamp"].dt.strftime("%G-W%V")
        weekly_residuals = frame.groupby("week", sort=True)["residual"].mean()
        if len(weekly_residuals) < MIN_UNCERTAINTY_WEEK_BLOCKS:
            raise ValueError(
                "Probability uncertainty requires at least "
                f"{MIN_UNCERTAINTY_WEEK_BLOCKS} held-out ISO weeks."
            )
        global_lower, global_upper = cls._one_sided_residual_bounds(
            weekly_residuals.to_numpy(dtype=float), confidence
        )

        fitted_bins: list[ProbabilityResidualBand] = []
        edges = np.linspace(0.0, 1.0, bin_count + 1)
        for index, (lower, upper) in enumerate(pairwise(edges)):
            mask = frame["probability"].ge(lower) & (
                frame["probability"].le(upper)
                if index == bin_count - 1
                else frame["probability"].lt(upper)
            )
            selected = frame.loc[mask]
            if len(selected) < MIN_UNCERTAINTY_BIN_SAMPLES:
                continue
            residuals = selected.groupby("week", sort=True)["residual"].mean()
            if len(residuals) < MIN_UNCERTAINTY_WEEK_BLOCKS:
                continue
            residual_lower, residual_upper = cls._one_sided_residual_bounds(
                residuals.to_numpy(dtype=float), confidence
            )
            fitted_bins.append(
                ProbabilityResidualBand(
                    probability_lower=float(lower),
                    probability_upper=float(upper),
                    residual_lower=residual_lower,
                    residual_upper=residual_upper,
                    sample_count=int(len(selected)),
                )
            )
        return cls(
            global_residual_lower=global_lower,
            global_residual_upper=global_upper,
            sample_count=int(len(frame)),
            calibration_units=int(len(weekly_residuals)),
            confidence=confidence,
            bins=tuple(fitted_bins),
        )

    @staticmethod
    def _one_sided_residual_bounds(
        residuals: np.ndarray, confidence: float
    ) -> tuple[float, float]:
        lower = float(np.quantile(residuals, 1.0 - confidence))
        upper = float(np.quantile(residuals, confidence))
        # A conservative adjustment must never raise the lower bound or lower
        # the complementary upper bound past the calibrated point.
        return min(lower, 0.0), max(upper, 0.0)

    def interval(self, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return clipped lower/upper ranges for calibrated probabilities."""
        values = np.asarray(probabilities, dtype=float)
        lower = np.empty_like(values)
        upper = np.empty_like(values)
        for index, probability in enumerate(values):
            band = next(
                (
                    candidate
                    for candidate in self.bins
                    if candidate.contains(probability)
                ),
                None,
            )
            residual_lower = (
                band.residual_lower if band is not None else self.global_residual_lower
            )
            residual_upper = (
                band.residual_upper if band is not None else self.global_residual_upper
            )
            lower[index] = np.clip(
                probability + residual_lower, PROBABILITY_EPSILON, probability
            )
            upper[index] = np.clip(
                probability + residual_upper,
                probability,
                1.0 - PROBABILITY_EPSILON,
            )
        return lower, upper


def valid_probability_calibration_artifacts(
    calibrator: Any,
    uncertainty: Any,
) -> bool:
    """Return whether actionability-grade probability artifacts are complete."""
    return bool(
        callable(getattr(calibrator, "predict", None))
        and callable(getattr(uncertainty, "interval", None))
        and getattr(calibrator, "version", None) == CALIBRATION_VERSION
        and getattr(uncertainty, "version", None) == PROBABILITY_UNCERTAINTY_VERSION
        and getattr(uncertainty, "fit_split", None) == "uncertainty_fit"
        and getattr(uncertainty, "unit", None) == "iso_week"
        and getattr(uncertainty, "method", None)
        == "week_block_q10_plus_one_sided_calibration_bias"
        and int(getattr(uncertainty, "sample_count", 0)) >= MIN_UNCERTAINTY_SAMPLES
        and int(getattr(uncertainty, "calibration_units", 0))
        >= MIN_UNCERTAINTY_WEEK_BLOCKS
    )


def conservative_probability_coverage(
    y_true: pd.Series | np.ndarray,
    lower_probability: np.ndarray,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    """Validate lower-bound calibration over preregistered ISO-week cohorts."""
    if "date" not in metadata:
        raise ValueError("Conservative coverage requires fixture timestamps.")
    frame = pd.DataFrame(
        {
            "actual": pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy(),
            "lower": np.asarray(lower_probability, dtype=float),
            "date": pd.to_datetime(metadata["date"], errors="coerce", utc=True),
        }
    )
    for column in ("actionable", "league"):
        if column in metadata:
            frame[column] = metadata[column].reset_index(drop=True)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["actual", "lower", "date"]
    )
    frame = frame[
        frame["actual"].isin([0, 1])
        & frame["lower"].between(0.0, 1.0, inclusive="both")
    ].copy()
    frame["week"] = frame["date"].dt.strftime("%G-W%V")
    masks: dict[str, pd.Series] = {
        "aggregate": pd.Series(True, index=frame.index),
    }
    if "actionable" in frame:
        actionable = frame["actionable"].fillna(False).astype(bool)
        masks["actionable"] = actionable
        if "league" in frame:
            for league in sorted(
                frame.loc[actionable, "league"].dropna().astype(str).unique()
            ):
                masks[f"league:{league}"] = actionable & frame["league"].eq(league)

    cohorts: dict[str, Any] = {}
    for name, mask in masks.items():
        selected = frame.loc[mask]
        weekly = selected.groupby("week", sort=True).agg(
            actual=("actual", "mean"),
            lower=("lower", "mean"),
        )
        eligible = (
            len(selected) >= MIN_UNCERTAINTY_SAMPLES
            and len(weekly) >= MIN_UNCERTAINTY_WEEK_BLOCKS
        )
        coverage = (
            float(weekly["actual"].ge(weekly["lower"] - 1e-12).mean())
            if len(weekly)
            else 0.0
        )
        shortfall = CONSERVATIVE_COVERAGE_TARGET - coverage
        cohorts[name] = {
            "rows": int(len(selected)),
            "week_blocks": int(len(weekly)),
            "coverage": coverage,
            "shortfall": shortfall,
            "eligible": eligible,
            "passed": eligible and shortfall <= MAX_CONSERVATIVE_COVERAGE_SHORTFALL,
        }
    required = [
        payload
        for name, payload in cohorts.items()
        if name == "aggregate" or payload["eligible"]
    ]
    return {
        "schema_version": 1,
        "method": "iso_week_observed_rate_at_or_above_mean_lower_bound",
        "target": CONSERVATIVE_COVERAGE_TARGET,
        "maximum_shortfall": MAX_CONSERVATIVE_COVERAGE_SHORTFALL,
        "cohorts": cohorts,
        "passed": bool(required) and all(item["passed"] for item in required),
    }


def conservative_probability_lower_bound(
    model: Any,
    X: pd.DataFrame,
    calibrator: Any,
    uncertainty: ProbabilityUncertaintyModel,
    metadata: pd.DataFrame,
) -> np.ndarray:
    """Combine the week-block ensemble quantile and held-out bias adjustment."""
    point = np.asarray(
        calibrator.predict(model.predict_proba(X)[:, 1], metadata=metadata),
        dtype=float,
    )
    ensemble_lower = np.asarray(
        model.conservative_probability(X, calibrator=calibrator, metadata=metadata),
        dtype=float,
    )
    bias_lower, _ = uncertainty.interval(point)
    return np.clip(
        ensemble_lower + bias_lower - point,
        PROBABILITY_EPSILON,
        point,
    )


def conservative_probability_report(
    model: Any,
    X: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    calibrator: Any,
    uncertainty: ProbabilityUncertaintyModel,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    """Describe and validate the versioned direct-series lower bound."""
    lower = conservative_probability_lower_bound(
        model, X, calibrator, uncertainty, metadata
    )
    return {
        "version": 1,
        "method": uncertainty.method,
        "ensemble_quantile": 0.10,
        "coverage": conservative_probability_coverage(y_true, lower, metadata),
    }


@dataclass
class SegmentProbabilityCalibrator:
    """One metadata segment's fitted calibration rule and shrinkage metadata."""

    kind: str
    value: str
    method: str
    model: Any
    sample_count: int
    shrinkage_weight: float
    selection_metrics: dict[str, Any] = field(default_factory=dict)

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return ProbabilityCalibrator(method=self.method, model=self.model).predict(
            probabilities
        )


@dataclass
class MetadataAwareProbabilityCalibrator:
    """Global probability calibrator plus conservative metadata-segment overrides."""

    global_calibrator: ProbabilityCalibrator
    segments: dict[tuple[str, str], SegmentProbabilityCalibrator] = field(
        default_factory=dict
    )
    version: int = CALIBRATION_VERSION
    segment_order: tuple[str, ...] = (
        "league_bo_format",
        "league",
        "strength_pool",
        "bo_format",
        "patch_family",
        "confidence_band",
    )
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def method(self) -> str:
        return (
            f"metadata_aware:{self.global_calibrator.method}"
            if self.segments
            else self.global_calibrator.method
        )

    def predict(
        self, probabilities: np.ndarray, metadata: pd.DataFrame | None = None
    ) -> np.ndarray:
        values = np.asarray(probabilities, dtype=float)
        calibrated = self.global_calibrator.predict(values)
        if metadata is None or metadata.empty or not self.segments:
            return calibrated

        meta = calibration_metadata_frame(metadata, probabilities=values)
        for idx in range(len(values)):
            for key in self._candidate_keys(meta.iloc[idx]):
                segment = self.segments.get(key)
                if segment is None:
                    continue
                segment_prob = segment.predict(np.asarray([values[idx]], dtype=float))[
                    0
                ]
                weight = segment.shrinkage_weight
                calibrated[idx] = (
                    weight * segment_prob + (1.0 - weight) * calibrated[idx]
                )
                break
        return np.clip(calibrated, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)

    def _candidate_keys(self, row: pd.Series) -> list[tuple[str, str]]:
        keys: list[tuple[str, str]] = []
        for kind in self.segment_order:
            value = row.get(kind)
            if pd.notna(value):
                keys.append((kind, str(value)))
        return keys


@dataclass
class PropDistributionCalibrator:
    """Residual-distribution calibrator for regression over/under pricing."""

    method: str
    global_residuals: np.ndarray
    segment_residuals: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    min_league_samples: int = MIN_PROP_COHORT_SIZE
    shrinkage_samples: int = 50
    version: int = CALIBRATION_VERSION

    def _empirical_under(self, threshold: float, residuals: np.ndarray) -> float:
        values = np.asarray(residuals, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.5
        return float(np.mean(values <= threshold))

    def _global_under(self, threshold: float) -> float:
        if self.method == "normal_global":
            sigma = float(np.std(self.global_residuals, ddof=1))
            sigma = max(sigma, PROBABILITY_EPSILON)
            return probability_over_under(
                mean=0.0, line=threshold, sigma=sigma
            ).under_probability
        return self._empirical_under(threshold, self.global_residuals)

    def price(
        self,
        *,
        mean: float,
        line: float,
        league: str | None = None,
        metadata: pd.DataFrame | pd.Series | dict[str, Any] | None = None,
        strength_pool: str | None = None,
        bo_format: str | None = None,
        patch: str | None = None,
    ) -> PropLineProbability:
        threshold = float(line) - float(mean)
        under_probability = self._global_under(threshold)
        if self.method in {"league_shrunk", "metadata_shrunk"}:
            meta = calibration_metadata_frame(
                _metadata_to_frame(
                    metadata,
                    league=league,
                    strength_pool=strength_pool,
                    bo_format=bo_format,
                    patch=patch,
                )
            )
            for key in _calibration_segment_keys(meta.iloc[0]):
                residuals = self.segment_residuals.get(key)
                if residuals is None or len(residuals) < self.min_league_samples:
                    continue
                segment_under = self._empirical_under(threshold, residuals)
                weight = len(residuals) / (len(residuals) + self.shrinkage_samples)
                under_probability = (
                    weight * segment_under + (1.0 - weight) * under_probability
                )
                break

        under_probability = float(
            np.clip(under_probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
        )
        over_probability = 1.0 - under_probability
        sigma = float(np.std(self.global_residuals, ddof=1))
        return PropLineProbability(
            mean=mean,
            line=line,
            sigma=max(sigma, PROBABILITY_EPSILON),
            over_probability=over_probability,
            under_probability=under_probability,
        )


def _metadata_to_frame(
    metadata: pd.DataFrame | pd.Series | dict[str, Any] | None,
    **overrides: Any,
) -> pd.DataFrame:
    if metadata is None:
        frame = pd.DataFrame([{}])
    elif isinstance(metadata, pd.DataFrame):
        frame = metadata.copy()
    elif isinstance(metadata, pd.Series):
        frame = metadata.to_frame().T
    else:
        frame = pd.DataFrame([metadata])

    if frame.empty:
        frame = pd.DataFrame([{}])
    for key, value in overrides.items():
        if value is not None:
            frame[key] = value
    return frame


def _bo_format_from_row(row: pd.Series) -> str:
    if str(row.get("bo_format", "")).strip():
        return str(row["bo_format"]).strip().casefold()
    match_type = str(row.get("match_type", "")).casefold()
    for token in ("bo1", "bo3", "bo5"):
        if token in match_type:
            return token
    best_of = pd.to_numeric(pd.Series([row.get("best_of")]), errors="coerce").iloc[0]
    if pd.notna(best_of) and int(best_of) in {1, 3, 5}:
        return f"bo{int(best_of)}"
    for value in (1, 3, 5):
        flag = (
            pd.to_numeric(pd.Series([row.get(f"is_bo{value}")]), errors="coerce")
            .fillna(0)
            .iloc[0]
        )
        if flag == 1:
            return f"bo{value}"
    return "unknown"


def _patch_family(value: Any) -> str | None:
    if pd.isna(value):
        return None
    parts = str(value).split(".")
    if len(parts) >= PATCH_FAMILY_PARTS:
        return ".".join(parts[:PATCH_FAMILY_PARTS])
    return str(value) or None


def _confidence_band(probability: float) -> str:
    confidence = max(float(probability), 1.0 - float(probability))
    if confidence < CONFIDENCE_BAND_LOW:
        return "low"
    if confidence < CONFIDENCE_BAND_HIGH:
        return "medium"
    return "high"


def calibration_metadata_frame(
    metadata: pd.DataFrame | pd.Series | dict[str, Any] | None,
    *,
    probabilities: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = _metadata_to_frame(metadata).reset_index(drop=True)
    if probabilities is not None and len(frame) != len(probabilities):
        frame = frame.reindex(range(len(probabilities))).ffill().bfill()

    out = pd.DataFrame(index=frame.index)
    for col in ("league", "strength_pool", "season"):
        if col in frame.columns:
            out[col] = frame[col].astype("object").where(frame[col].notna())

    if "patch" in frame.columns:
        out["patch_family"] = frame["patch"].map(_patch_family)
    elif "patch_family" in frame.columns:
        out["patch_family"] = frame["patch_family"].astype("object")

    out["bo_format"] = frame.apply(_bo_format_from_row, axis=1)
    if "league" in out.columns:
        out["league_bo_format"] = (
            out["league"].astype(str) + "|" + out["bo_format"].astype(str)
        )

    if probabilities is not None:
        out["confidence_band"] = [_confidence_band(p) for p in probabilities]
    elif "confidence_band" in frame.columns:
        out["confidence_band"] = frame["confidence_band"].astype("object")

    return out


def _calibration_segment_keys(row: pd.Series) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for kind in (
        "league_bo_format",
        "league",
        "strength_pool",
        "bo_format",
        "patch_family",
        "confidence_band",
    ):
        value = row.get(kind)
        if pd.notna(value):
            keys.append((kind, str(value)))
    return keys


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

        # Add absent inference columns in bulk. Repeated ``Xp[col] = value``
        # fragments wide feature frames and makes routine prediction noisy/slow.
        categorical_columns = dict.fromkeys(
            [*self.categorical_features, *self.categorical_levels]
        )
        missing_categoricals = {
            col: "Unknown" for col in categorical_columns if col not in Xp.columns
        }
        if missing_categoricals:
            Xp = pd.concat(
                [Xp, pd.DataFrame(missing_categoricals, index=Xp.index)], axis=1
            )

        # Coerce unseen categorical levels to "Unknown".
        for col, levels in self.categorical_levels.items():
            vals = pd.Series(Xp[col], index=Xp.index, dtype="object")
            known = pd.Index(levels, dtype="object")
            vals = vals.where(vals.isin(known), "Unknown")
            Xp[col] = pd.Categorical(vals, categories=levels)

        # Add any entirely-missing numeric features with their train medians
        missing_numerics = {
            col: median
            for col, median in self.numeric_medians.items()
            if col not in Xp.columns
        }
        if missing_numerics:
            Xp = pd.concat([Xp, pd.DataFrame(missing_numerics, index=Xp.index)], axis=1)

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
    feature_set: TrainingFeatureSet = "full"
    max_features: int = DEFAULT_SELECTED_MAX_FEATURES
    force_retune: bool = False
    allow_hparam_schema_drift: bool = False
    calibration: CalibrationMode = "auto"
    calibration_method: CalibrationMethod = "auto"
    calibration_size: float = CALIBRATION_SIZE
    tune_size: float = TUNE_SIZE
    test_size: float = TEST_SIZE
    directory: Path = FIGURES_DIR
    artifact_root: Path = MODELS_DIR
    report_root: Path | None = None
    dataset_fingerprint: str | None = None
    run_id: str = field(
        default_factory=lambda: dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    training_data: pd.DataFrame = field(init=False)
    probability_calibrator: Any | None = field(default=None, init=False, repr=False)
    probability_uncertainty: ProbabilityUncertaintyModel | None = field(
        default=None, init=False, repr=False
    )
    prop_calibrator: PropDistributionCalibrator | None = field(
        default=None, init=False, repr=False
    )
    fit_partition_metadata: dict[str, pd.DataFrame] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.training_data = pd.DataFrame()

    # ─────────────────────────── Traceability ─────────────────────────── #

    @staticmethod
    def stable_dataframe_hash(df: pd.DataFrame) -> str:
        """Hash dataframe content deterministically for model-card traceability."""
        normalized = df.reindex(sorted(df.columns), axis=1)
        payload = normalized.to_json(
            orient="split",
            date_format="iso",
            default_handler=str,
        )
        if payload is None:
            msg = "Pandas did not produce a JSON payload for dataframe hashing."
            raise RuntimeError(msg)
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def git_code_version() -> str | None:
        """Best-effort current git revision without shelling out."""
        revision = None
        git_path = SUITE_ROOT / ".git"
        if git_path.exists():
            git_dir = git_path
            if git_path.is_file():
                content = git_path.read_text().strip()
                if content.startswith("gitdir:"):
                    git_dir = (SUITE_ROOT / content.split(":", 1)[1].strip()).resolve()

            head = git_dir / "HEAD"
            if head.exists():
                head_value = head.read_text().strip()
                if not head_value.startswith("ref:"):
                    revision = head_value or None
                else:
                    ref_name = head_value.split(" ", 1)[1]
                    ref_path = git_dir / ref_name
                    if ref_path.exists():
                        revision = ref_path.read_text().strip() or None
                    else:
                        packed_refs = git_dir / "packed-refs"
                        if packed_refs.exists():
                            for line in packed_refs.read_text().splitlines():
                                if line.startswith("#") or not line.strip():
                                    continue
                                packed_revision, _, packed_ref = line.partition(" ")
                                if packed_ref == ref_name:
                                    revision = packed_revision
                                    break
        return revision

    @classmethod
    def model_hyperparameters(cls, model) -> dict[str, Any]:
        """Extract JSON-safe fitted model parameters for model-card traceability."""
        raw_model = getattr(model, "raw_model", model)
        get_params = getattr(raw_model, "get_params", None)
        if not callable(get_params):
            return {}
        try:
            params = get_params()
        except Exception as e:
            logger.warning("Could not extract fitted model parameters: %s", e)
            return {}
        if not isinstance(params, dict):
            return {}
        return {
            str(key): cls._json_safe_param(value)
            for key, value in sorted(params.items(), key=lambda item: str(item[0]))
        }

    @classmethod
    def _json_safe_param(cls, value):
        if value is None or isinstance(value, str | int | float | bool):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, list | tuple):
            return [cls._json_safe_param(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): cls._json_safe_param(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        return str(value)

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
            "series_id",
            "source_gameid",
            "target_gameid",
            "teamid",
            "teamname",
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
        """Fuse non-EMA opponent columns; explicit EMA diffs are handled separately."""
        X = df.copy()
        X = GradientBoostingModel.add_explicit_ema_diffs(X, drop_opponents=False)

        # Team-level opp_* → base - opp_base  (numeric only)
        opp_cols = [c for c in X.columns if c.startswith("opp_")]
        for col in opp_cols:
            base = col[4:]
            if base.startswith("ema_"):
                X = X.drop(columns=[col], errors="ignore")
                continue
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
                if col.split(prefix, 1)[1].startswith("ema_"):
                    X = X.drop(columns=[col], errors="ignore")
                    continue
                if (
                    base in X.columns
                    and is_numeric_dtype(X[base])
                    and is_numeric_dtype(X[col])
                ):
                    X[base] = X[base] - X[col]
                X = X.drop(columns=[col], errors="ignore")

        return X

    @staticmethod
    def add_explicit_ema_diffs(
        df: pd.DataFrame, *, drop_opponents: bool = False
    ) -> pd.DataFrame:
        """Recompute current-opponent EMA deltas without overwriting own state."""
        X = df.copy()

        for col in [c for c in X.columns if c.startswith("opp_ema_")]:
            base = col[4:]
            diff = f"diff_{base}"
            if (
                base in X.columns
                and is_numeric_dtype(X[base])
                and is_numeric_dtype(X[col])
            ):
                X[diff] = X[base] - X[col]
            if drop_opponents:
                X = X.drop(columns=[col], errors="ignore")

        roles = ("top", "jng", "mid", "bot", "sup")
        for role in roles:
            prefix = f"{role}_opp_ema_"
            for col in [c for c in X.columns if c.startswith(prefix)]:
                metric = col.split(f"{role}_opp_", 1)[1]
                base = f"{role}_{metric}"
                diff = f"{role}_diff_{metric}"
                if (
                    base in X.columns
                    and is_numeric_dtype(X[base])
                    and is_numeric_dtype(X[col])
                ):
                    X[diff] = X[base] - X[col]
                if drop_opponents:
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
            if m is None:
                continue
            stem = m[2]
            stems.setdefault(stem, []).append(c)

        out = df.drop(columns=role_cols, errors="ignore")
        for stem, cols in stems.items():
            out[f"players_{stem}"] = (
                df[cols].max(axis=1) if agg == "max" else df[cols].mean(axis=1)
            )

        return out

    @staticmethod
    def add_rating_consensus_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine correlated rating-family probabilities into compact strength signals.

        The source rating systems are still available to the full model. These
        consensus features let compact/selected runs test a smaller feature
        surface without hard-coding one rating family as the winner.
        """
        out = df.copy()

        def _add_group(
            source_cols: tuple[str, ...], consensus_col: str, disagreement_col: str
        ) -> list[str]:
            cols = [col for col in source_cols if col in out.columns]
            if not cols:
                return []
            values = out[cols].apply(pd.to_numeric, errors="coerce")
            out[consensus_col] = values.mean(axis=1)
            out[disagreement_col] = values.std(axis=1).fillna(0.0)
            return [consensus_col, disagreement_col]

        consensus_cols = _add_group(
            DIRECT_RATING_LIKELIHOODS,
            "team_rating_consensus",
            "team_rating_disagreement",
        )
        consensus_cols += _add_group(
            PLAYER_RATING_LIKELIHOODS,
            "players_rating_consensus",
            "players_rating_disagreement",
        )

        if consensus_cols:
            consensus_values = out[
                [col for col in consensus_cols if col.endswith("_consensus")]
            ].apply(pd.to_numeric, errors="coerce")
            disagreement_values = out[
                [col for col in consensus_cols if col.endswith("_disagreement")]
            ].apply(pd.to_numeric, errors="coerce")
            out["rating_consensus"] = consensus_values.mean(axis=1)
            out["rating_disagreement"] = disagreement_values.mean(axis=1)

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
        """
        Remove exact numeric constants on train data.

        Matchup deltas are signed, so coefficient-of-variation pruning is invalid:
        every useful feature with a negative mean can otherwise look "low variance".
        ``threshold`` remains in the signature so older call sites and artifacts can
        be read, but production pruning intentionally ignores it.
        """
        del threshold
        df = df.copy()
        num = df.select_dtypes("number")
        if num.empty:
            return df, []
        drop = [
            col for col in num.columns if num[col].dropna().drop_duplicates().size <= 1
        ]
        df = df.drop(columns=drop, errors="ignore")
        if drop:
            logger.info("Dropped exact-constant numeric columns: %d", len(drop))
        return df, drop

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

        timestamps = pd.Index(g[date_col].drop_duplicates().sort_values())
        n = len(timestamps)
        if n < MIN_TEMPORAL_SPLITS:
            raise ValueError("At least three unique timestamps are required.")
        n_test = max(1, int(round(n * test_size)))
        n_val = max(1, int(round((n - n_test) * val_size)))
        n_train = n - n_val - n_test
        if n_train < 1:
            raise ValueError("Not enough timestamps for train/validation/test split.")

        train_dates = set(timestamps[:n_train])
        val_dates = set(timestamps[n_train : n_train + n_val])
        test_dates = set(timestamps[n_train + n_val :])
        gids_train = set(g.loc[g[date_col].isin(train_dates), group_col])
        gids_val = set(g.loc[g[date_col].isin(val_dates), group_col])
        gids_test = set(g.loc[g[date_col].isin(test_dates), group_col])

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

    @staticmethod
    def temporal_train_tune_cal_test_split(  # noqa: PLR0915
        X_with_meta: pd.DataFrame,
        y: pd.Series,
        date_col: str = "date",
        group_col: str = "gameid",
        tune_size: float = TUNE_SIZE,
        calibration_size: float = CALIBRATION_SIZE,
        test_size: float = TEST_SIZE,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
    ]:
        """Time-ordered split for train/tune/cal_fit/cal_select/uncertainty/test."""
        required = {group_col, date_col}
        if not required.issubset(X_with_meta.columns):
            raise ValueError(f"Missing required columns {sorted(required)}")
        if tune_size <= 0 or calibration_size <= 0 or test_size <= 0:
            raise ValueError("Tune, calibration, and test fractions must be positive.")
        if tune_size + calibration_size + test_size >= 1:
            raise ValueError("Tune + calibration + test fractions must be < 1.")

        g = X_with_meta[[group_col, date_col]].drop_duplicates(subset=group_col).copy()
        g[date_col] = pd.to_datetime(g[date_col], errors="coerce")
        g = g.sort_values(date_col).dropna(subset=[date_col])
        if g.empty:
            raise ValueError("No valid dates for temporal split.")

        timestamps = pd.Index(g[date_col].drop_duplicates().sort_values())
        n = len(timestamps)
        if n < MIN_CALIBRATION_SPLITS:
            raise ValueError(
                "At least six unique timestamps are required for calibration splits."
            )
        n_test = max(1, int(round(n * test_size)))
        n_cal = max(3, int(round(n * calibration_size)))
        n_tune = max(1, int(round(n * tune_size)))
        n_train = n - n_tune - n_cal - n_test
        if n_train < 1:
            raise ValueError("Not enough games for train/tune/calibration/test split.")

        n_cal_fit = max(1, int(round(n_cal * 0.50)))
        n_cal_select = max(1, int(round(n_cal * 0.25)))
        n_uncertainty = n_cal - n_cal_fit - n_cal_select
        if n_uncertainty < 1:
            n_uncertainty = 1
            n_cal_fit = max(1, n_cal_fit - 1)

        train_dates = timestamps[:n_train]
        tune_dates = timestamps[n_train : n_train + n_tune]
        cal_fit_dates = timestamps[n_train + n_tune : n_train + n_tune + n_cal_fit]
        cal_select_dates = timestamps[
            n_train + n_tune + n_cal_fit : n_train + n_tune + n_cal_fit + n_cal_select
        ]
        uncertainty_dates = timestamps[
            n_train + n_tune + n_cal_fit + n_cal_select : n_train
            + n_tune
            + n_cal_fit
            + n_cal_select
            + n_uncertainty
        ]
        test_dates = timestamps[n_train + n_tune + n_cal :]

        def _groups_for_dates(values: pd.Index) -> pd.DataFrame:
            return g.loc[g[date_col].isin(set(values))]

        train = _groups_for_dates(train_dates)
        tune = _groups_for_dates(tune_dates)
        cal_fit = _groups_for_dates(cal_fit_dates)
        cal_select = _groups_for_dates(cal_select_dates)
        uncertainty = _groups_for_dates(uncertainty_dates)
        test = _groups_for_dates(test_dates)

        def _sel(group_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
            gids = set(group_frame[group_col])
            Xp = X_with_meta[X_with_meta[group_col].isin(gids)]
            yp = y.loc[Xp.index]
            return Xp, yp

        X_train, y_train = _sel(train)
        X_tune, y_tune = _sel(tune)
        X_cal_fit, y_cal_fit = _sel(cal_fit)
        X_cal_select, y_cal_select = _sel(cal_select)
        X_uncertainty, y_uncertainty = _sel(uncertainty)
        X_test, y_test = _sel(test)
        logger.info(
            "Temporal calibration split -> train: %d, tune: %d, cal_fit: %d, "
            "cal_select: %d, uncertainty: %d, test: %d rows",
            len(X_train),
            len(X_tune),
            len(X_cal_fit),
            len(X_cal_select),
            len(X_uncertainty),
            len(X_test),
        )
        return (
            X_train,
            X_tune,
            X_cal_fit,
            X_cal_select,
            X_uncertainty,
            X_test,
            y_train,
            y_tune,
            y_cal_fit,
            y_cal_select,
            y_uncertainty,
            y_test,
        )

    @staticmethod
    def temporal_winner_v2_split(
        X_with_meta: pd.DataFrame,
        y: pd.Series,
        *,
        date_col: str = "date",
        group_col: str = "gameid",
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
    ]:
        """Create timestamp-atomic 55/5/10/10/5/15 Winner V2 partitions."""
        required = {group_col, date_col}
        if not required.issubset(X_with_meta.columns):
            raise ValueError(f"Missing required columns {sorted(required)}")
        groups = X_with_meta[[group_col, date_col]].drop_duplicates(group_col).copy()
        groups[date_col] = pd.to_datetime(groups[date_col], errors="coerce")
        groups = groups.dropna(subset=[date_col]).sort_values(date_col)
        timestamps = pd.Index(groups[date_col].drop_duplicates())
        if len(timestamps) < MIN_CALIBRATION_SPLITS:
            raise ValueError("Winner V2 requires at least six unique timestamps.")
        fractions = (0.55, 0.05, 0.10, 0.10, 0.05)
        boundaries = [0]
        cumulative = 0.0
        for fraction in fractions:
            cumulative += fraction
            boundary = max(boundaries[-1] + 1, int(round(len(timestamps) * cumulative)))
            boundaries.append(boundary)
        boundaries[-1] = min(boundaries[-1], len(timestamps) - 1)
        date_partitions = [
            timestamps[start:stop]
            for start, stop in pairwise([*boundaries, len(timestamps)])
        ]
        if any(len(values) == 0 for values in date_partitions):
            raise ValueError("Winner V2 temporal partitions must all be non-empty.")

        def _select(values: pd.Index) -> tuple[pd.DataFrame, pd.Series]:
            group_ids = set(groups.loc[groups[date_col].isin(set(values)), group_col])
            frame = X_with_meta.loc[X_with_meta[group_col].isin(group_ids)]
            return frame, y.loc[frame.index]

        selected = [_select(values) for values in date_partitions]
        return (
            selected[0][0],
            selected[1][0],
            selected[2][0],
            selected[3][0],
            selected[4][0],
            selected[5][0],
            selected[0][1],
            selected[1][1],
            selected[2][1],
            selected[3][1],
            selected[4][1],
            selected[5][1],
        )

    # ─────────────────────── Categorical / imputation ─────────────────────── #

    @staticmethod
    def _impute_train_numeric(
        X: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        medians = X.median(numeric_only=True).to_dict()
        return X.fillna(value=medians).infer_objects(copy=False), medians

    @staticmethod
    def _impute_apply_numeric(
        X: pd.DataFrame, medians: dict[str, float]
    ) -> pd.DataFrame:
        # Pandas will stop implicitly downcasting object columns during fillna.
        # Opt in now, then retain the existing explicit inference step.
        with pd.option_context("future.no_silent_downcasting", True):
            return X.fillna(value=medians).infer_objects(copy=False)

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
        # Correlation is useful research evidence, but is not a production drop
        # rule. Redundant predictors are handled by regularisation/tree fitting.
        del drop_high_corr_threshold
        drop_corr: list[str] = []

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

    # ─────────────────────────── Feature-set filters ─────────────────────────── #

    @staticmethod
    def _normalize_config_feature(feature: str) -> str:
        return feature.removesuffix("_before")

    @classmethod
    def compact_feature_candidates(cls) -> set[str]:
        """Return curated compact feature names after train-time renaming/pivoting."""
        candidates: set[str] = set()
        for path, key in (
            (TRAINING_COMPACT_TEAM_CONFIG, "team_features"),
            (TRAINING_COMPACT_PLAYER_CONFIG, "player_features"),
        ):
            payload = json_loader(path)
            values = payload.get(key, []) if isinstance(payload, dict) else []
            normalized = [cls._normalize_config_feature(str(v)) for v in values]
            if key == "team_features":
                candidates.update(normalized)
                continue
            for feature in normalized:
                if feature in cls._meta_columns() or feature == "position":
                    continue
                candidates.update(f"{role}_{feature}" for role in COMPACT_ROLE_PREFIXES)
                if feature.endswith("_win_likelihood"):
                    candidates.add(f"players_{feature}")
        candidates.update(DERIVED_STRENGTH_FEATURES)
        return candidates

    @staticmethod
    def _mandatory_anchor_features(columns: pd.Index | list[str]) -> list[str]:
        return [
            col
            for col in columns
            if any(pattern in col for pattern in MANDATORY_ANCHOR_PATTERNS)
        ]

    def _selected_feature_report_path(self) -> Path:
        return (
            FEATURE_REPORTS_DIR / f"{self.model_name}_recommended_compact_features.json"
        )

    def _load_selected_feature_candidates(
        self, columns: pd.Index | list[str]
    ) -> list[str]:
        """Load prior report recommendations, preserving mandatory anchors."""
        columns_list = list(columns)
        available = set(columns_list)
        paths = [
            self._selected_feature_report_path(),
            FEATURE_REPORTS_DIR / "recommended_compact_features.json",
        ]
        ranked: list[str] = []
        for path in paths:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Could not parse selected feature report %s: %s", path, exc
                )
                continue
            by_count = payload.get("recommendations_by_count", {})
            preferred = by_count.get(str(self.max_features)) or by_count.get(
                self.max_features
            )
            ranked = list(preferred or payload.get("recommended_features", []))
            break
        if not ranked:
            msg = (
                f"Feature set 'selected' requires a recommendation report for "
                f"{self.model_name}. Run a full or compact research training with "
                "--feature-selection report first."
            )
            raise FileNotFoundError(msg)

        anchors = [
            col
            for col in self._mandatory_anchor_features(columns_list)
            if col in available
        ]
        selected: list[str] = []
        for col in [*anchors, *ranked]:
            if col in available and col not in selected:
                selected.append(col)
            if len(selected) >= self.max_features:
                break
        if not selected:
            msg = (
                f"Selected feature report for {self.model_name} contains no "
                "available features."
            )
            raise ValueError(msg)
        return selected

    def _apply_feature_set_filter(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply full/compact/selected feature-set policy to a feature matrix."""
        if self.feature_set == "full":
            return X
        if self.feature_set == "compact":
            allowed = self.compact_feature_candidates()
            keep = [
                col
                for col in X.columns
                if col in allowed
                or col.removeprefix("delta_") in allowed
                or col in self._mandatory_anchor_features(X.columns)
            ]
        elif self.feature_set == "selected":
            keep = self._load_selected_feature_candidates(list(X.columns))
        else:
            raise ValueError(f"Unknown feature_set: {self.feature_set}")

        if not keep:
            msg = (
                f"Feature set '{self.feature_set}' matched no columns for "
                f"{self.model_name}."
            )
            raise ValueError(msg)
        logger.info(
            "Feature set '%s' keeps %d/%d features for %s.",
            self.feature_set,
            len(keep),
            X.shape[1],
            self.model_name,
        )
        return X.loc[:, keep]

    def _store_split_report(
        self,
        splits: dict[str, tuple[pd.DataFrame, pd.Series]],
        *,
        target_col: str,
        require_temporal_order: bool,
    ) -> None:
        report: dict[str, Any] = {
            "model_name": self.model_name,
            "target": target_col,
            "checks": self.validate_split_integrity(
                splits,
                require_temporal_order=require_temporal_order,
            ),
        }
        for name, (X_split, y_split) in splits.items():
            payload: dict[str, Any] = {
                "rows": int(len(X_split)),
                "games": int(X_split["gameid"].nunique())
                if "gameid" in X_split
                else None,
            }
            if "date" in X_split:
                dates = pd.to_datetime(
                    X_split["date"], errors="coerce", utc=True
                ).dropna()
                payload["date_min"] = (
                    dates.min().isoformat() if not dates.empty else None
                )
                payload["date_max"] = (
                    dates.max().isoformat() if not dates.empty else None
                )
            if "league" in X_split:
                payload["league_counts"] = {
                    str(k): int(v) for k, v in X_split["league"].value_counts().items()
                }
            missingness = X_split.isna().mean().sort_values(ascending=False)
            payload["feature_availability"] = {
                "columns": int(len(missingness)),
                "fully_unavailable": int(missingness.eq(1.0).sum()),
                "worst_missingness": {
                    str(feature): float(fraction)
                    for feature, fraction in missingness.head(20).items()
                },
            }
            y_num = pd.to_numeric(y_split, errors="coerce")
            if self.problem_type == "classification":
                payload["target_counts"] = {
                    str(k): int(v) for k, v in y_num.value_counts(dropna=False).items()
                }
            else:
                payload["target_mean"] = float(y_num.mean())
                payload["target_std"] = float(y_num.std())
            report[name] = payload
        self.insight_path("split_report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )

    @staticmethod
    def validate_split_integrity(
        splits: dict[str, tuple[pd.DataFrame, pd.Series]],
        *,
        require_temporal_order: bool,
    ) -> dict[str, Any]:
        """Validate split boundaries before storing metrics or training artifacts."""
        seen: dict[Any, str] = {}
        overlaps: list[dict[str, str]] = []
        for split_name, (X_split, _) in splits.items():
            if "gameid" not in X_split.columns:
                continue
            for gameid in X_split["gameid"].dropna().unique():
                if gameid in seen:
                    overlaps.append(
                        {
                            "gameid": str(gameid),
                            "first_split": seen[gameid],
                            "second_split": split_name,
                        }
                    )
                else:
                    seen[gameid] = split_name
        if overlaps:
            msg = f"Split gameids must be disjoint; sample={overlaps[:5]}"
            raise ValueError(msg)

        checks: dict[str, Any] = {"gameids_disjoint": True}
        if not require_temporal_order:
            checks["temporal_ordered"] = None
            return checks

        previous_name: str | None = None
        previous_max: pd.Timestamp | None = None
        date_ranges: dict[str, dict[str, str | None]] = {}
        for split_name, (X_split, _) in splits.items():
            if "date" not in X_split.columns:
                checks["temporal_ordered"] = None
                checks["temporal_order_reason"] = "date column unavailable"
                return checks
            dates = pd.to_datetime(X_split["date"], errors="coerce").dropna()
            if dates.empty:
                date_ranges[split_name] = {"min": None, "max": None}
                continue
            current_min = dates.min()
            current_max = dates.max()
            date_ranges[split_name] = {
                "min": current_min.isoformat(),
                "max": current_max.isoformat(),
            }
            if previous_max is not None and previous_max >= current_min:
                msg = (
                    "Temporal split order is invalid; "
                    f"{previous_name} max date {previous_max.date()} is not before "
                    f"{split_name} min date {current_min.date()}."
                )
                raise ValueError(msg)
            previous_name = split_name
            previous_max = current_max

        checks["temporal_ordered"] = True
        checks["date_ranges"] = date_ranges
        return checks

    # ─────────────────────────────── Storage ──────────────────────────────── #

    def _store_pickle(self, filename: str, data: Any) -> None:
        try:
            out_dir = self.artifact_root / self.model_name
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

    def store_feature_lineage(self, all_features: pd.Index) -> None:
        """Persist the train/serve availability contract for every final feature."""
        lineage = [
            _model_feature_lineage(
                str(feature),
                next_map=self.model_name == "NextMapWinnerPrediction_LightGBM",
            )
            for feature in all_features
        ]
        payload = json.dumps(lineage, indent=2, sort_keys=True) + "\n"
        artifact = self.artifact_root / self.model_name
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / f"{self.model_name}_feature_lineage.json").write_text(payload)
        self.insight_path("feature_lineage.json").write_text(payload)

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

    def store_probability_calibrator(self, calibrator: Any) -> None:
        self._store_pickle(f"{self.model_name}_probability_calibrator.pkl", calibrator)

    def store_probability_uncertainty(
        self, uncertainty: ProbabilityUncertaintyModel
    ) -> None:
        self._store_pickle(
            f"{self.model_name}_probability_uncertainty.pkl", uncertainty
        )

    def store_prop_calibrator(self, calibrator: PropDistributionCalibrator) -> None:
        self._store_pickle(f"{self.model_name}_prop_calibrator.pkl", calibrator)

    def store_calibration_report(self, report: dict[str, Any]) -> None:
        payload = json.dumps(report, indent=2, default=str) + "\n"
        self.insight_path("calibration_report.json").write_text(payload)
        out_dir = self.artifact_root / self.model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{self.model_name}_calibration_report.json").write_text(payload)

    def store_residual_summary(self, summary: dict[str, Any]) -> None:
        self._store_pickle(f"{self.model_name}_residual_summary.pkl", summary)
        self.insight_path("residual_summary.json").write_text(json.dumps(summary))

    def store_sealed_evaluation(
        self,
        raw_features: pd.DataFrame,
        actuals: pd.Series,
        metadata: pd.DataFrame,
    ) -> None:
        """Persist untouched holdout rows so champion and candidate replay identically."""
        root = self.artifact_root / "_evaluation" / self.model_name
        root.mkdir(parents=True, exist_ok=True)
        raw_features.reset_index(drop=True).to_parquet(
            root / "features.parquet",
            index=False,
            compression="gzip",
        )
        labels = metadata.reset_index(drop=True).copy()
        labels.insert(0, "actual", actuals.reset_index(drop=True))
        labels.to_parquet(
            root / "labels.parquet",
            index=False,
            compression="gzip",
        )

    def store_prop_evaluation_report(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        eval_meta: pd.DataFrame,
    ) -> None:
        actual = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
        pred = np.asarray(y_pred, dtype=float)
        frame = eval_meta.reset_index(drop=True).copy()
        frame["actual"] = actual
        frame["prediction"] = pred
        frame["residual"] = actual - pred
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["actual", "prediction", "residual"]
        )
        baseline_value = float(self.regression_baseline_value_)
        baseline_prediction = np.full(len(frame), baseline_value, dtype=float)
        report: dict[str, Any] = {
            "model_name": self.model_name,
            "n": int(len(frame)),
            "metrics": self.compute_regression_metrics(
                pd.Series(frame["actual"]), frame["prediction"].to_numpy()
            ),
            "line_backtest": {
                "available": False,
                "reason": "Historical market lines/odds are not stored in the training set yet.",
                "required_fields": ["line", "over_odds", "under_odds", "placed_at"],
            },
            "constant_baseline": {
                "fit_split": "train",
                "value": baseline_value,
                "metrics": self.compute_regression_metrics(
                    pd.Series(frame["actual"]), baseline_prediction
                ),
            },
            "line_semantics": {
                "version": 1,
                "supported": (
                    "continuous_threshold"
                    if self.model_name.startswith("Gamelength")
                    else "half_lines_only"
                ),
                "push_model": False,
            },
            "cohort_dimensions": ["league", "map_number"],
            "residual_cohorts": {},
        }
        for output_column, source_column in (
            ("league", "league"),
            ("patch", "patch"),
            ("map_number", "game"),
        ):
            if source_column not in frame.columns:
                continue
            cohorts: dict[str, Any] = {}
            for value, group in frame.groupby(source_column, dropna=True):
                if len(group) < MIN_PROP_COHORT_SIZE:
                    continue
                residuals = group["residual"].to_numpy(dtype=float)
                cohorts[str(value)] = {
                    "n": int(len(group)),
                    "mae": float(np.mean(np.abs(residuals))),
                    "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
                    "residual_sigma": float(np.std(residuals, ddof=1))
                    if len(group) > 1
                    else 0.0,
                    "bias": float(np.mean(residuals)),
                }
            if cohorts:
                report["residual_cohorts"][output_column] = cohorts
        self.insight_path("prop_evaluation_report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )

    # ─────────────────────────────── Metrics ──────────────────────────────── #

    def compute_classification_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None
    ) -> dict[str, Any]:
        """Threshold metrics plus probability-quality metrics when probabilities exist."""
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
                clipped = np.clip(y_proba, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
                metrics["log_loss"] = float(log_loss(y_true, clipped))
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
                metrics["brier"] = float(brier_score_loss(y_true, y_proba))
                metrics.update(
                    self.compute_probability_calibration_metrics(y_true, y_proba)
                )
        metrics["cm"] = confusion_matrix(y_true, y_pred)
        return metrics

    @staticmethod
    def compute_probability_calibration_metrics(
        y_true: pd.Series,
        y_proba: np.ndarray,
        *,
        n_bins: int = CALIBRATION_BINS,
    ) -> dict[str, float]:
        """Return ECE plus logistic calibration slope/intercept."""
        y = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
        p = np.clip(
            np.asarray(y_proba, dtype=float),
            PROBABILITY_EPSILON,
            1.0 - PROBABILITY_EPSILON,
        )
        mask = np.isfinite(y) & np.isfinite(p)
        y = y[mask]
        p = p[mask]
        if y.size == 0 or p.size == 0:
            return {}

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.clip(np.digitize(p, bins, right=True) - 1, 0, n_bins - 1)
        ece = 0.0
        for bin_id in range(n_bins):
            in_bin = bin_ids == bin_id
            if not in_bin.any():
                continue
            weight = float(in_bin.mean())
            ece += weight * float(abs(p[in_bin].mean() - y[in_bin].mean()))

        metrics = {"calibration_ece": float(ece)}
        if np.unique(p).size > 1 and np.unique(y).size == BINARY_CLASS_UNIQUE_VALUES:
            logits = np.log(p / (1.0 - p)).reshape(-1, 1)
            calibration_model = LogisticRegression(
                C=np.inf,
                solver="lbfgs",
                max_iter=1000,
            )
            calibration_model.fit(logits, y.astype(int))
            metrics["calibration_slope"] = float(calibration_model.coef_[0, 0])
            metrics["calibration_intercept"] = float(calibration_model.intercept_[0])
        return metrics

    @staticmethod
    def compute_pairwise_classification_metrics(
        *,
        y_true: pd.Series,
        y_proba: np.ndarray,
        eval_gameids: pd.Series,
    ) -> dict[str, Any]:
        """Evaluate two-row games as one market: pick the side with higher probability."""
        frame = pd.DataFrame(
            {
                "gameid": eval_gameids.to_numpy(),
                "actual": pd.to_numeric(y_true, errors="coerce").to_numpy(),
                "proba": np.asarray(y_proba, dtype=float),
            },
            index=y_true.index,
        ).dropna(subset=["gameid", "actual", "proba"])
        if frame.empty:
            return {}

        row_prediction_05 = (frame["proba"] >= DEFAULT_CLASSIFICATION_THRESHOLD).astype(
            int
        )
        row_accuracy_05 = float((row_prediction_05 == frame["actual"]).mean())

        pair_rows: list[dict[str, Any]] = []
        for gameid, group in frame.groupby("gameid", sort=False):
            if (
                len(group) != ROWS_PER_GAME
                or group["actual"].sum() != POSITIVE_RESULT_SUM_PER_GAME
            ):
                continue
            favorite_idx = group["proba"].idxmax()
            prob_sum = float(group["proba"].sum())
            normalized = (
                group["proba"] / prob_sum
                if prob_sum > 0
                else pd.Series([0.5, 0.5], index=group.index)
            )
            winner_probability = float(normalized[group["actual"] == 1].iloc[0])
            pred05_wins = int(
                (group["proba"] >= DEFAULT_CLASSIFICATION_THRESHOLD).sum()
            )
            pair_rows.append(
                {
                    "gameid": gameid,
                    "correct": int(frame.loc[favorite_idx, "actual"] == 1),
                    "favorite_probability": float(group["proba"].max()),
                    "winner_probability": winner_probability,
                    "probability_sum": prob_sum,
                    "both_predicted_win_05": int(pred05_wins == ROWS_PER_GAME),
                    "both_predicted_loss_05": int(pred05_wins == 0),
                }
            )
        if not pair_rows:
            return {"row_accuracy_at_0_5": row_accuracy_05}

        pair_df = pd.DataFrame(pair_rows)
        winner_probability = np.clip(
            pair_df["winner_probability"].to_numpy(dtype=float),
            PROBABILITY_EPSILON,
            1.0 - PROBABILITY_EPSILON,
        )
        return {
            "row_accuracy_at_0_5": row_accuracy_05,
            "pairwise_game_count": int(len(pair_df)),
            "pairwise_argmax_accuracy": float(pair_df["correct"].mean()),
            "pairwise_log_loss": float(-np.log(winner_probability).mean()),
            "pairwise_brier": float(np.mean(np.square(1.0 - winner_probability))),
            "pairwise_favorite_probability_mean": float(
                pair_df["favorite_probability"].mean()
            ),
            "pairwise_probability_sum_mean": float(pair_df["probability_sum"].mean()),
            "pairwise_both_predicted_win_at_0_5": int(
                pair_df["both_predicted_win_05"].sum()
            ),
            "pairwise_both_predicted_loss_at_0_5": int(
                pair_df["both_predicted_loss_05"].sum()
            ),
        }

    def compute_regression_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray
    ) -> dict[str, Any]:
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))
        return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

    @staticmethod
    def build_residual_summary(
        y_true: pd.Series, y_pred: np.ndarray, *, model_name: str
    ) -> dict[str, Any]:
        actual = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
        pred = np.asarray(y_pred, dtype=float)
        residuals = actual - pred
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size == 0:
            msg = "No finite residuals available."
            raise ValueError(msg)
        return {
            "model_name": model_name,
            "n": int(residuals.size),
            "residual_mean": float(np.mean(residuals)),
            "residual_sigma": float(np.std(residuals, ddof=1))
            if residuals.size > 1
            else float(np.std(residuals)),
            "mae": float(np.mean(np.abs(residuals))),
            "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
            "percentiles": {
                str(q): float(np.percentile(residuals, q))
                for q in (5, 10, 25, 50, 75, 90, 95)
            },
        }

    @staticmethod
    def _fit_prop_candidate(
        method: str,
        y_true: pd.Series,
        y_pred: np.ndarray,
        eval_meta: pd.DataFrame,
    ) -> PropDistributionCalibrator:
        actual = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
        pred = np.asarray(y_pred, dtype=float)
        residuals = actual - pred
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size == 0:
            raise ValueError("No finite residuals available for prop calibration.")
        segment_residuals: dict[tuple[str, str], np.ndarray] = {}
        if method in {"league_shrunk", "metadata_shrunk"}:
            frame = calibration_metadata_frame(eval_meta).reset_index(drop=True)
            frame["residual"] = actual - pred
            for kind in (
                ["league"]
                if method == "league_shrunk"
                else [
                    "league_bo_format",
                    "league",
                    "strength_pool",
                    "bo_format",
                    "patch_family",
                ]
            ):
                if kind not in frame.columns:
                    continue
                for value, group in frame.groupby(kind, dropna=True):
                    values = group["residual"].to_numpy(dtype=float)
                    values = values[np.isfinite(values)]
                    if values.size:
                        segment_residuals[(kind, str(value))] = values
        return PropDistributionCalibrator(
            method=method,
            global_residuals=residuals,
            segment_residuals=segment_residuals,
        )

    @staticmethod
    def _prop_calibration_log_loss(
        calibrator: PropDistributionCalibrator,
        y_true: pd.Series,
        y_pred: np.ndarray,
        eval_meta: pd.DataFrame,
    ) -> dict[str, Any]:
        actual = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
        pred = np.asarray(y_pred, dtype=float)
        sigma = float(np.std(calibrator.global_residuals, ddof=1))
        sigma = max(sigma, PROBABILITY_EPSILON)
        rows: list[dict[str, float]] = []
        meta = eval_meta.reset_index(drop=True)
        for idx, (actual_value, mean_value) in enumerate(
            zip(actual, pred, strict=False)
        ):
            if not np.isfinite(actual_value) or not np.isfinite(mean_value):
                continue
            league = None
            row_metadata = None
            if "league" in meta.columns and idx < len(meta):
                league = str(meta.iloc[idx]["league"])
            if idx < len(meta):
                row_metadata = meta.iloc[idx]
            for offset in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5):
                line = float(mean_value + offset * sigma)
                over_actual = int(actual_value > line)
                prob = calibrator.price(
                    mean=float(mean_value),
                    line=line,
                    league=league,
                    metadata=row_metadata,
                ).over_probability
                rows.append({"actual": over_actual, "probability": prob})
        if not rows:
            return {"log_loss": float("inf"), "brier": float("inf"), "n": 0}
        frame = pd.DataFrame(rows)
        p = np.clip(
            frame["probability"].to_numpy(dtype=float),
            PROBABILITY_EPSILON,
            1.0 - PROBABILITY_EPSILON,
        )
        y = frame["actual"].to_numpy(dtype=int)
        return {
            "log_loss": float(log_loss(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "n": int(len(frame)),
        }

    def fit_prop_calibrator(
        self,
        y_cal_fit: pd.Series,
        pred_cal_fit: np.ndarray,
        meta_cal_fit: pd.DataFrame,
        y_cal_select: pd.Series,
        pred_cal_select: np.ndarray,
        meta_cal_select: pd.DataFrame,
        y_cal_full: pd.Series,
        pred_cal_full: np.ndarray,
        meta_cal_full: pd.DataFrame,
    ) -> PropDistributionCalibrator | None:
        """Select residual-distribution prop calibrator, then refit on full calibration."""
        if self.calibration == "none":
            return None
        candidates: list[dict[str, Any]] = []
        for method in (
            "normal_global",
            "empirical_global",
            "league_shrunk",
            "metadata_shrunk",
        ):
            try:
                candidate = self._fit_prop_candidate(
                    method, y_cal_fit, pred_cal_fit, meta_cal_fit
                )
                metrics = self._prop_calibration_log_loss(
                    candidate, y_cal_select, pred_cal_select, meta_cal_select
                )
                candidates.append(
                    {
                        "method": method,
                        "metrics": metrics,
                    }
                )
            except Exception as e:
                logger.warning("Prop calibration candidate '%s' failed: %s", method, e)
        if not candidates:
            return None
        best = sorted(
            candidates,
            key=lambda item: (
                item["metrics"].get("log_loss", float("inf")),
                item["metrics"].get("brier", float("inf")),
            ),
        )[0]
        final_calibrator = self._fit_prop_candidate(
            best["method"], y_cal_full, pred_cal_full, meta_cal_full
        )
        self.store_prop_calibrator(final_calibrator)
        report = {
            "model_name": self.model_name,
            "selected_method": final_calibrator.method,
            "selection_metric": "synthetic_over_under_log_loss",
            "tie_breaker": "brier",
            "samples": {
                "cal_fit": int(len(y_cal_fit)),
                "cal_select": int(len(y_cal_select)),
                "cal_full": int(len(y_cal_full)),
            },
            "candidates": candidates,
        }
        self.store_calibration_report(report)
        return final_calibrator

    @staticmethod
    def _probability_quality_metrics(
        y_true: pd.Series,
        probabilities: np.ndarray,
        eval_gameids: pd.Series | None = None,
    ) -> dict[str, Any]:
        y = pd.to_numeric(y_true, errors="coerce")
        p = np.clip(
            np.asarray(probabilities, dtype=float),
            PROBABILITY_EPSILON,
            1.0 - PROBABILITY_EPSILON,
        )
        metrics: dict[str, Any] = {
            "log_loss": float(log_loss(y, p)),
            "brier": float(brier_score_loss(y, p)),
        }
        metrics.update(
            GradientBoostingModel.compute_probability_calibration_metrics(y, p)
        )
        with contextlib.suppress(ValueError):
            metrics["roc_auc"] = float(roc_auc_score(y, p))
        if eval_gameids is not None:
            metrics.update(
                GradientBoostingModel.compute_pairwise_classification_metrics(
                    y_true=y,
                    y_proba=p,
                    eval_gameids=eval_gameids,
                )
            )
        return metrics

    @staticmethod
    def _fit_probability_candidate(
        method: str,
        probabilities: np.ndarray,
        y: pd.Series,
    ) -> ProbabilityCalibrator:
        if method == "raw":
            return ProbabilityCalibrator(method="raw", model=None)
        if method == "sigmoid":
            model = LogisticRegression(solver="lbfgs")
            model.fit(probabilities.reshape(-1, 1), y.to_numpy(dtype=int))
            return ProbabilityCalibrator(method="sigmoid", model=model)
        if method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(probabilities, y.to_numpy(dtype=int))
            return ProbabilityCalibrator(method="isotonic", model=model)
        raise ValueError(f"Unknown calibration method: {method}")

    @staticmethod
    def _select_probability_calibration_candidate(
        candidates: list[dict[str, Any]],
        *,
        require_safe_calibration: bool,
    ) -> dict[str, Any]:
        """Choose on selection evidence; raw is the fail-safe identity fallback."""
        for candidate in candidates:
            metrics = candidate["metrics"]
            reasons: list[str] = []
            slope = metrics.get("calibration_slope")
            intercept = metrics.get("calibration_intercept")
            if slope is None or not (
                MIN_CALIBRATION_SLOPE <= float(slope) <= MAX_CALIBRATION_SLOPE
            ):
                reasons.append("calibration_slope_outside_0.8_1.2")
            if (
                intercept is None
                or abs(float(intercept)) > MAX_ABSOLUTE_CALIBRATION_INTERCEPT
            ):
                reasons.append("calibration_intercept_above_0.10")
            candidate["selection_reasons"] = reasons
            candidate["selection_safe"] = not reasons
            candidate["selection_eligible"] = not reasons
            candidate["selection_fallback"] = False

        eligible = candidates
        if require_safe_calibration:
            eligible = [
                candidate for candidate in candidates if candidate["selection_safe"]
            ]
            if not eligible:
                eligible = [
                    candidate
                    for candidate in candidates
                    if candidate["method"] == "raw"
                ]
                for candidate in eligible:
                    candidate["selection_eligible"] = True
                    candidate["selection_fallback"] = True
        if not eligible:
            raise RuntimeError(
                "No safe fitted probability calibrator or raw identity fallback "
                "was available on the Winner V2 selection split."
            )
        return min(
            eligible,
            key=lambda item: (
                item["metrics"].get("log_loss", float("inf")),
                item["metrics"].get("brier", float("inf")),
            ),
        )

    def fit_probability_calibrator(
        self,
        model,
        X_cal_fit: pd.DataFrame,
        y_cal_fit: pd.Series,
        X_cal_select: pd.DataFrame,
        y_cal_select: pd.Series,
        X_cal_full: pd.DataFrame,
        y_cal_full: pd.Series,
        meta_cal_fit: pd.DataFrame | None = None,
        meta_cal_select: pd.DataFrame | None = None,
        meta_cal_full: pd.DataFrame | None = None,
    ) -> MetadataAwareProbabilityCalibrator | None:
        """Select probability calibration on cal_select, then refit on full calibration."""
        del meta_cal_fit, meta_cal_select, meta_cal_full
        if (
            self.calibration == "none"
            or self.problem_type != "classification"
            or not hasattr(model, "predict_proba")
        ):
            return None

        y_fit = pd.to_numeric(y_cal_fit, errors="coerce").dropna()
        y_select = pd.to_numeric(y_cal_select, errors="coerce").dropna()
        y_full = pd.to_numeric(y_cal_full, errors="coerce").dropna()
        if (
            len(y_fit) < MIN_CALIBRATION_SAMPLES
            or y_fit.nunique() != BINARY_CLASS_UNIQUE_VALUES
            or y_select.nunique() != BINARY_CLASS_UNIQUE_VALUES
            or y_full.nunique() != BINARY_CLASS_UNIQUE_VALUES
        ):
            logger.warning(
                "Skipping probability calibration for %s: insufficient calibration diversity.",
                self.model_name,
            )
            return None

        methods = ["raw", "sigmoid", "isotonic"]
        if self.calibration_method != "auto":
            methods = [self.calibration_method]

        p_fit = model.predict_proba(X_cal_fit.loc[y_fit.index])[:, 1]
        p_select = model.predict_proba(X_cal_select.loc[y_select.index])[:, 1]
        p_full = model.predict_proba(X_cal_full.loc[y_full.index])[:, 1]
        candidates: list[dict[str, Any]] = []
        for method in methods:
            try:
                candidate = self._fit_probability_candidate(method, p_fit, y_fit)
                selected_prob = candidate.predict(p_select)
                metrics = self._probability_quality_metrics(y_select, selected_prob)
                candidates.append(
                    {
                        "method": method,
                        "metrics": metrics,
                        "calibrator": candidate,
                    }
                )
            except Exception as e:
                logger.warning("Calibration candidate '%s' failed: %s", method, e)

        if not candidates:
            return None

        best = self._select_probability_calibration_candidate(
            candidates,
            require_safe_calibration=self.model_name in WINNER_V2_MODEL_NAMES,
        )
        final_global = self._fit_probability_candidate(best["method"], p_full, y_full)
        global_selection_metrics = best["metrics"]
        segmented_selection_metrics = global_selection_metrics
        segments_accepted = False
        segments: dict[tuple[str, str], SegmentProbabilityCalibrator] = {}
        segment_reports: list[dict[str, Any]] = []
        final_calibrator = MetadataAwareProbabilityCalibrator(
            global_calibrator=final_global,
            segments=segments,
            report={
                "segment_count": len(segments),
                "segment_reports": segment_reports,
                "segments_accepted": segments_accepted,
                "segment_status": "disabled_without_independent_composite_gate",
                "global_selection_metrics": global_selection_metrics,
                "segmented_selection_metrics": segmented_selection_metrics,
            },
        )
        self.store_probability_calibrator(final_calibrator)
        report = {
            "model_name": self.model_name,
            "selected_method": final_calibrator.method,
            "global_method": final_global.method,
            "calibration_version": CALIBRATION_VERSION,
            "selection_metric": "log_loss",
            "tie_breaker": "brier",
            "segment_count": len(segments),
            "segments_accepted": segments_accepted,
            "segment_status": "disabled_without_independent_composite_gate",
            "segment_min_samples": {
                "sigmoid": MIN_SEGMENT_SIGMOID_SAMPLES,
                "isotonic": MIN_SEGMENT_ISOTONIC_SAMPLES,
            },
            "segment_shrinkage": CALIBRATION_SEGMENT_SHRINKAGE,
            "samples": {
                "cal_fit": int(len(y_fit)),
                "cal_select": int(len(y_select)),
                "cal_full": int(len(y_full)),
            },
            "candidates": [
                {
                    "method": item["method"],
                    "metrics": item["metrics"],
                    "selection_eligible": item["selection_eligible"],
                    "selection_reasons": item["selection_reasons"],
                }
                for item in candidates
            ],
            "segments": segment_reports,
            "global_selection_metrics": global_selection_metrics,
            "segmented_selection_metrics": segmented_selection_metrics,
        }
        self.store_calibration_report(report)
        return final_calibrator

    def _predict_positive_probability(
        self,
        model,
        X: pd.DataFrame,
        metadata: pd.DataFrame | None = None,
    ) -> np.ndarray | None:
        if self.problem_type != "classification" or not hasattr(model, "predict_proba"):
            return None
        y_proba = model.predict_proba(X)[:, 1]
        if self.probability_calibrator is not None:
            y_proba = self.probability_calibrator.predict(y_proba, metadata=metadata)
        return y_proba

    def fit_probability_uncertainty(
        self,
        model,
        X_uncertainty: pd.DataFrame,
        y_uncertainty: pd.Series,
        metadata: pd.DataFrame | None = None,
    ) -> ProbabilityUncertaintyModel | None:
        """Fit and store probability-estimate uncertainty on its own holdout."""
        if self.problem_type != "classification" or not hasattr(model, "predict_proba"):
            return None
        raw = model.predict_proba(X_uncertainty)[:, 1]
        probabilities = (
            self.probability_calibrator.predict(raw, metadata=metadata)
            if self.probability_calibrator is not None
            else raw
        )
        if metadata is None or "date" not in metadata:
            logger.warning(
                "Skipping probability uncertainty for %s: holdout timestamps missing",
                self.model_name,
            )
            return None
        try:
            uncertainty = ProbabilityUncertaintyModel.fit(
                y_uncertainty,
                probabilities,
                timestamps=metadata["date"],
            )
        except ValueError as exc:
            logger.warning(
                "Skipping probability uncertainty for %s: %s",
                self.model_name,
                exc,
            )
            return None
        self.store_probability_uncertainty(uncertainty)
        return uncertainty

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
            if "log_loss" in metrics:
                base += f" | LogLoss {metrics['log_loss']:.4f}"
            if "pairwise_argmax_accuracy" in metrics:
                base += f" | PairAcc {metrics['pairwise_argmax_accuracy']:.4f}"
            if "pairwise_log_loss" in metrics:
                base += f" | PairLogLoss {metrics['pairwise_log_loss']:.4f}"
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
        actuals: pd.Series,
        eval_meta: pd.DataFrame | None = None,
        proba: np.ndarray | None = None,
    ) -> None:
        """Persist per-row predictions (+probabilities for classification)."""
        try:
            frame = {
                "gameid": eval_gameids.to_numpy(),
                "side": eval_sides.to_numpy(),
                "actual": actuals.to_numpy(),
                "prediction": predictions,
            }
            for column in (
                "league",
                "league_region",
                "league_tier",
                "strength_pool",
                "actionable",
                "date",
            ):
                if eval_meta is not None and column in eval_meta:
                    frame[column] = eval_meta[column].to_numpy()
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
        eval_meta: pd.DataFrame | None = None,
    ) -> None:
        """Validate & store: metrics, predictions, and observability artifacts."""
        try:
            logger.info("Validating %s ...", self.model_name)
            y_pred = model.predict(X_test)
            y_proba = self._predict_positive_probability(model, X_test, eval_meta)

            if self.problem_type == "classification":
                metrics = self.compute_classification_metrics(y_test, y_pred, y_proba)
                if y_proba is not None:
                    metrics.update(
                        self.compute_pairwise_classification_metrics(
                            y_true=y_test,
                            y_proba=y_proba,
                            eval_gameids=eval_gameids,
                        )
                    )
                self.plot_confusion_matrix(y_test, y_pred)
                self.plot_accuracy_over_samples(y_test, y_pred)
                if y_proba is not None:
                    self.plot_roc_pr_calibration(y_test, y_proba)
                    self.store_calibration_table(y_test, y_proba)
                # historical accuracy uses PROCESSED_TEAMS join by gameid internally
                self.plot_historical_accuracy(X_test, y_test, y_pred, eval_gameids)
            else:
                metrics = self.compute_regression_metrics(y_test, y_pred)
                self.store_residual_summary(
                    self.build_residual_summary(
                        y_test, y_pred, model_name=self.model_name
                    )
                )
                if eval_meta is not None:
                    self.store_prop_evaluation_report(y_test, y_pred, eval_meta)
                self.plot_regression_results(y_test, y_pred)
                self.plot_regression_error_over_samples(y_test, y_pred, metric="mae")
                self.plot_regression_error_over_time(
                    X_test, y_test, y_pred, eval_gameids
                )

            self.log_evaluation_metrics(metrics)
            self.store_evaluation_metrics(metrics)
            self.store_predictions(
                y_pred,
                eval_gameids,
                eval_sides,
                y_test,
                eval_meta,
                proba=y_proba,
            )

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
        removed = [c for c in meta if c not in X2.columns]
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
        feature_selection: FeatureSelectionMethod = "none",
        feature_selection_threshold: float = 0.001,
    ):  # sourcery skip: low-code-quality
        """
        Main entrypoint used by the training script.

        Parameters
        ----------
        target_col : str
            Name of target column
        validate : bool
            Whether to run validation and store metrics
        feature_selection : str
            Feature selection method: "none", "importance", "cumulative", "report"
        feature_selection_threshold : float
            Threshold for importance-based selection (default 0.1% of total importance)

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
        X = self.add_explicit_ema_diffs(X, drop_opponents=True)
        if fuse_opponents:
            X = self.fuse_opposing_team_features(X)
        if process_player_likelihoods:
            X = self.process_players_likelihood_columns(X, agg="mean")
        X = self.add_rating_consensus_features(X)

        # Curated compact configs are written for side-POV rows, so apply them
        # before prop targets are collapsed into one row per game.
        if (
            self.feature_set == "compact"
            and self.model_name not in WINNER_V2_MODEL_NAMES
        ):
            X = self._apply_feature_set_filter(X)

        if self.problem_type == "regression" and is_prop_target(target_col):
            X, y_game, meta_game = build_game_level_prop_features(
                X,
                meta_df,
                y,
                target_col=target_col,
            )
            if y_game is None:
                msg = f"Could not build game-level target for {target_col}."
                raise ValueError(msg)
            y = y_game
            meta_df = meta_game
            logger.info(
                "Collapsed prop target '%s' to one row per game: %d rows.",
                target_col,
                len(X),
            )
        elif self.problem_type == "classification" and target_col == "result":
            X, y_match, meta_match = build_game_level_outcome_features(X, meta_df, y)
            if y_match is None:
                msg = "Could not build matchup outcome targets."
                raise ValueError(msg)
            y = y_match
            meta_df = meta_match
            logger.info("Collapsed outcome target to %d canonical matchups.", len(X))

        if self.feature_set == "selected":
            X = self._apply_feature_set_filter(X)

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
        extra_split_cols = [
            col
            for col in ("date", "patch")
            if col in meta_df.columns and col not in split_keys
        ]
        X_for_split = pd.concat([X, meta_df[split_keys + extra_split_cols]], axis=1)

        X_cal_fit = X_cal_select = X_cal_full = X_uncertainty = None
        y_cal_fit = y_cal_select = y_cal_full = y_uncertainty = None
        has_dedicated_calibration_split = False
        used_temporal_split = False
        # Choose splitter (temporal if 'date' is available)
        if (
            temporal_split
            and "date" in X_for_split.columns
            and self.calibration != "none"
        ):
            has_dedicated_calibration_split = True
            used_temporal_split = True
            (
                X_train,
                X_val,
                X_cal_fit,
                X_cal_select,
                X_uncertainty,
                X_test,
                y_train,
                y_val,
                y_cal_fit,
                y_cal_select,
                y_uncertainty,
                y_test,
            ) = (
                self.temporal_winner_v2_split(X_for_split, y)
                if self.model_name in WINNER_V2_MODEL_NAMES
                else self.temporal_train_tune_cal_test_split(
                    X_for_split,
                    y,
                    date_col="date",
                    group_col="gameid",
                    tune_size=self.tune_size,
                    calibration_size=self.calibration_size,
                    test_size=self.test_size,
                )
            )
            X_cal_full = pd.concat([X_cal_fit, X_cal_select], axis=0)
            y_cal_full = pd.concat([y_cal_fit, y_cal_select], axis=0)
        elif temporal_split and "date" in X_for_split.columns:
            used_temporal_split = True
            X_train, X_val, X_test, y_train, y_val, y_test = (
                self.temporal_train_val_test_split(
                    X_for_split,
                    y,
                    date_col="date",
                    group_col="gameid",
                    val_size=VALIDATION_SIZE,
                    test_size=self.test_size,
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
        if (
            X_cal_fit is None
            or X_cal_select is None
            or X_cal_full is None
            or y_cal_fit is None
            or y_cal_select is None
            or y_cal_full is None
        ):
            X_cal_fit = X_val
            X_cal_select = X_val
            X_cal_full = X_val
            y_cal_fit = y_val
            y_cal_select = y_val
            y_cal_full = y_val
        if X_uncertainty is None or y_uncertainty is None:
            X_uncertainty = X_val
            y_uncertainty = y_val
        if self.problem_type == "regression":
            self.regression_baseline_value_ = float(
                pd.to_numeric(y_train, errors="coerce").mean()
            )

        split_report_entries: dict[str, tuple[pd.DataFrame, pd.Series]] = {
            "train": (X_train, y_train),
            "tune": (X_val, y_val),
            "test": (X_test, y_test),
        }
        if has_dedicated_calibration_split:
            split_report_entries = {
                "train": (X_train, y_train),
                "tune": (X_val, y_val),
                "calibration_fit": (X_cal_fit, y_cal_fit),
                "calibration_select": (X_cal_select, y_cal_select),
                "uncertainty_fit": (X_uncertainty, y_uncertainty),
                "test": (X_test, y_test),
            }
        self._store_split_report(
            split_report_entries,
            target_col=target_col,
            require_temporal_order=used_temporal_split,
        )

        calibration_meta_cols = [
            col
            for col in (
                "gameid",
                "league",
                "strength_pool",
                "patch",
                "season",
                "game",
                "match_type",
                "is_bo1",
                "is_bo3",
                "is_bo5",
            )
            if col in X_for_split.columns
        ]
        meta_cal_fit_for_cal = X_cal_fit[calibration_meta_cols].copy()
        meta_cal_select_for_cal = X_cal_select[calibration_meta_cols].copy()
        meta_cal_full_for_cal = X_cal_full[calibration_meta_cols].copy()
        meta_uncertainty_for_cal = X_uncertainty[calibration_meta_cols].copy()
        meta_test_for_cal = X_test[calibration_meta_cols].copy()

        self.fit_partition_metadata = {
            name: frame[[col for col in ("gameid", "date") if col in frame]].copy()
            for name, frame in {
                "train": X_train,
                "tune": X_val,
                "calibration_fit": X_cal_fit,
                "calibration_select": X_cal_select,
                "uncertainty_fit": X_uncertainty,
                "test": X_test,
            }.items()
        }

        # ── CRITICAL: drop meta (incl. 'date') from the actual feature matrices, with logs ── #
        X_train = self._strip_meta_from_features(X_train, "train")
        X_val = self._strip_meta_from_features(X_val, "val")
        X_cal_fit = self._strip_meta_from_features(X_cal_fit, "cal_fit")
        X_cal_select = self._strip_meta_from_features(X_cal_select, "cal_select")
        X_cal_full = self._strip_meta_from_features(X_cal_full, "cal_full")
        X_uncertainty = self._strip_meta_from_features(X_uncertainty, "uncertainty")
        X_test = self._strip_meta_from_features(X_test, "test")
        if self.model_name in WINNER_V2_MODEL_NAMES:
            if self.feature_set == "auto":
                selector = getattr(self, "select_development_feature_schema", None)
                if not callable(selector):
                    raise RuntimeError("Winner research schema selector is unavailable")
                self.feature_set = selector(X_train, y_train, X_val, y_val)
            if self.feature_set == "compact":
                splits = [
                    X_train,
                    X_val,
                    X_cal_fit,
                    X_cal_select,
                    X_cal_full,
                    X_uncertainty,
                    X_test,
                ]
                (
                    X_train,
                    X_val,
                    X_cal_fit,
                    X_cal_select,
                    X_cal_full,
                    X_uncertainty,
                    X_test,
                ) = tuple(self._apply_feature_set_filter(frame) for frame in splits)
        sealed_test_features = X_test.copy()

        # Record eval identifiers for artifacts (sourced from meta_df, not features)
        def _safe_meta_col(col: str) -> pd.Series:
            """Safely extract meta column with index alignment handling."""
            if col not in meta_df.columns:
                return pd.Series(index=X_test.index, dtype=object)
            try:
                return meta_df.loc[X_test.index, col]
            except KeyError:
                logger.warning("Index mismatch extracting %s from meta_df", col)
                return pd.Series(index=X_test.index, dtype=object)

        eval_gameids = _safe_meta_col("gameid")
        eval_sides = _safe_meta_col("side")

        # Fit feature pipeline on TRAIN (drop/missing/variance/corr/cats/impute)
        # and reapply it to val/test to guarantee feature parity.
        X_train, feature_pipeline = self._fit_feature_pipeline(
            X_train,
            drop_missing_threshold=drop_missing_threshold,
            drop_low_std_threshold=drop_low_std_threshold,
            drop_high_corr_threshold=drop_high_corr_threshold,
        )
        X_val = feature_pipeline.transform(X_val)
        X_cal_fit = feature_pipeline.transform(X_cal_fit)
        X_cal_select = feature_pipeline.transform(X_cal_select)
        X_cal_full = feature_pipeline.transform(X_cal_full)
        X_uncertainty = feature_pipeline.transform(X_uncertainty)
        X_test = feature_pipeline.transform(X_test)
        categorical_features = feature_pipeline.categorical_features
        train_cols = feature_pipeline.train_columns

        # Explicit guardrail: eval splits must perfectly mirror train features
        for split_name, X_split in {
            "val": X_val,
            "cal_fit": X_cal_fit,
            "cal_select": X_cal_select,
            "cal_full": X_cal_full,
            "uncertainty": X_uncertainty,
            "test": X_test,
        }.items():
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
        self.store_feature_lineage(pd.Index(train_cols))
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
        if self.problem_type == "classification":
            self.probability_calibrator = self.fit_probability_calibrator(
                model,
                X_cal_fit,
                y_cal_fit,
                X_cal_select,
                y_cal_select,
                X_cal_full,
                y_cal_full,
                meta_cal_fit_for_cal,
                meta_cal_select_for_cal,
                meta_cal_full_for_cal,
            )
            if has_dedicated_calibration_split:
                self.probability_uncertainty = self.fit_probability_uncertainty(
                    model,
                    X_uncertainty,
                    y_uncertainty,
                    meta_uncertainty_for_cal,
                )
            if self.probability_calibrator is not None:
                raw_test = model.predict_proba(X_test)[:, 1]
                calibrated_test = self.probability_calibrator.predict(
                    raw_test, metadata=meta_test_for_cal
                )
                report_path = self.insight_path("calibration_report.json")
                report: dict[str, Any] = (
                    json.loads(report_path.read_text())
                    if report_path.exists()
                    else {
                        "model_name": self.model_name,
                        "selected_method": self.probability_calibrator.method,
                    }
                )
                raw_metrics = self._probability_quality_metrics(
                    y_test, raw_test, eval_gameids
                )
                calibrated_metrics = self._probability_quality_metrics(
                    y_test, calibrated_test, eval_gameids
                )
                report["test_metrics"] = {
                    "raw": raw_metrics,
                    "calibrated": calibrated_metrics,
                }
                if self.probability_uncertainty is not None:
                    test_lower, test_upper = self.probability_uncertainty.interval(
                        calibrated_test
                    )
                    report["uncertainty"] = {
                        "method": self.probability_uncertainty.method,
                        "fit_split": self.probability_uncertainty.fit_split,
                        "confidence": self.probability_uncertainty.confidence,
                        "fit_samples": self.probability_uncertainty.sample_count,
                        "calibration_units": (
                            self.probability_uncertainty.calibration_units
                        ),
                        "unit": self.probability_uncertainty.unit,
                        "version": self.probability_uncertainty.version,
                        "local_bins": len(self.probability_uncertainty.bins),
                        "test_mean_width": float(np.mean(test_upper - test_lower)),
                        "test_rows": int(len(calibrated_test)),
                        "test_used_for_fit": False,
                    }
                    if callable(getattr(model, "conservative_probability", None)):
                        report["conservative_probability"] = (
                            conservative_probability_report(
                                model,
                                X_test,
                                y_test,
                                self.probability_calibrator,
                                self.probability_uncertainty,
                                meta_test_for_cal,
                            )
                            | {"test_used_for_fit": False}
                        )
                if target_col == "result":
                    # One row is one canonical game pair. Serving scores that
                    # pair once and returns its exact complement if callers
                    # reverse the teams, so these are pairwise—not team-row—
                    # calibration metrics.
                    report["pairwise_test_metrics"] = {
                        "raw": raw_metrics,
                        "calibrated": calibrated_metrics,
                        "max_probability_sum_error": 0.0,
                    }
                self.store_calibration_report(report)
            if target_col == "result":
                self._store_pickle(
                    f"{self.model_name}_outcome_matchup_schema.pkl",
                    {
                        "version": 1,
                        "canonical_key": "teamid_then_teamname",
                        "excluded_features": ["first_pick", "side_win_likelihood"],
                    },
                )
        elif self.problem_type == "regression" and is_prop_target(target_col):
            self.prop_calibrator = self.fit_prop_calibrator(
                y_cal_fit,
                model.predict(X_cal_fit),
                meta_cal_fit_for_cal,
                y_cal_select,
                model.predict(X_cal_select),
                meta_cal_select_for_cal,
                y_cal_full,
                model.predict(X_cal_full),
                meta_cal_full_for_cal,
            )

        # ─────────────────── Post-training Feature Selection Report ─────────────────── #
        # For importance-based methods, report which features would be selected
        if feature_selection in ("importance", "cumulative", "report"):
            try:
                if feature_selection == "report":
                    selected = FeatureSelector.write_temporal_recommendation_report(
                        model=model,
                        X_train=X_train,
                        X_validation=X_val,
                        y_validation=y_val,
                        model_name=self.model_name,
                        max_features=self.max_features,
                        feature_counts=SELECTED_FEATURE_COUNTS,
                        problem_type=self.problem_type,
                    ).get("recommended_features", [])
                elif feature_selection == "importance":
                    selected = FeatureSelector.select_by_importance(
                        model, X_train, threshold=feature_selection_threshold
                    )
                else:
                    selected = FeatureSelector.select_by_cumulative_importance(
                        model, X_train, cumulative_threshold=0.95
                    )
                logger.info(
                    "Feature selection would keep %d/%d features",
                    len(selected),
                    len(train_cols),
                )
                # Store selected features for future use
                self._store_pickle(
                    f"{self.model_name}_selected_features.pkl",
                    selected,
                )
            except Exception as e:
                logger.warning("Feature selection report failed: %s", e)

        # ─────────────────────────── Validate ──────────────────────────── #
        if validate:
            eval_meta_for_test = meta_df.loc[
                X_test.index,
                [
                    c
                    for c in [
                        "league",
                        "league_region",
                        "league_tier",
                        "strength_pool",
                        "patch",
                        "side",
                        "gameid",
                        "series_id",
                        "source_gameid",
                        "target_gameid",
                        "game",
                        "next_map_number",
                        "date",
                    ]
                    if c in meta_df.columns
                ],
            ]
            if "league" in eval_meta_for_test:
                from oracle_bets_core.league_selection import actionable_leagues

                eval_meta_for_test["actionable"] = eval_meta_for_test["league"].isin(
                    actionable_leagues()
                )
            self.store_sealed_evaluation(
                sealed_test_features,
                y_test,
                eval_meta_for_test,
            )
            self.validate_model(
                model,
                X_test,
                y_test,
                eval_gameids,
                eval_sides,
                eval_meta_for_test,
            )

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
                    # classification: also persist cohort metrics with probabilities if available
                    y_proba = None
                    if self.problem_type == "classification":
                        y_proba = self._predict_positive_probability(
                            model,
                            X_test,
                            eval_meta_for_test,
                        )
                    self.store_cohort_metrics(
                        eval_meta_for_test, y_test, model.predict(X_test), y_proba
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
                hyperparams=self.model_hyperparameters(model),
                code_version=self.git_code_version(),
                data_hash=self.stable_dataframe_hash(self.training_data),
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
