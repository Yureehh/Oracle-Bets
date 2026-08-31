"""Candidate training triggers, promotion evidence, and immutable model bundles."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from functools import cache
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import numpy as np
from oracle_bets_core.evidence.performance import prediction_quality
from oracle_bets_core.io_utils import load_model
from oracle_bets_core.paths import (
    CONFIG_DIR,
    MODELS_DIR,
    PRODUCT_CONFIG,
    SUITE_ROOT,
    TRAINING_PLAYER_DATA,
    TRAINING_TEAM_DATA,
)
from oracle_bets_core.pd import pd

if TYPE_CHECKING:
    from collections.abc import Collection

_MAX_ROUTINE_RELATIVE_DEGRADATION = 0.01
_MIN_RETUNE_RELATIVE_IMPROVEMENT = 0.01
_MAX_BRIER_DEGRADATION = 0.002
_MAX_ECE_DEGRADATION = 0.01
_MAX_REGRESSION_MAE_DEGRADATION = 0.02
_MAX_COHORT_RELATIVE_REGRESSION = 0.02
_MIN_COHORT_SIZE = 30
_MIN_BOOTSTRAP_SAMPLES = 100
_MIN_PAIRED_LOSSES = 2
_MAX_BOOTSTRAP_INDEX_CELLS = 1_000_000
_DRIFT_MISSINGNESS_WARNING = 0.25
_DRIFT_TOP_FEATURE_MINIMUM_OVERLAP = 5
_DRIFT_PREDICTION_MEAN_WARNING = 0.05
_COMPLETE_LOL_BUNDLE_REQUIRED_ARTIFACTS = frozenset(
    {
        "team_league_mapping.parquet",
        "league_elo.parquet",
        "OutcomePrediction_LightGBM/OutcomePrediction_LightGBM.pkl",
        "OutcomePrediction_LightGBM/OutcomePrediction_LightGBM_feature_pipeline.pkl",
        "OutcomePrediction_LightGBM/OutcomePrediction_LightGBM_outcome_matchup_schema.pkl",
        "OutcomePrediction_LightGBM/OutcomePrediction_LightGBM_probability_calibrator.pkl",
        "OutcomePrediction_LightGBM/OutcomePrediction_LightGBM_probability_uncertainty.pkl",
        "SeriesWinnerPrediction_LightGBM/SeriesWinnerPrediction_LightGBM.pkl",
        "SeriesWinnerPrediction_LightGBM/SeriesWinnerPrediction_LightGBM_feature_pipeline.pkl",
        "SeriesWinnerPrediction_LightGBM/SeriesWinnerPrediction_LightGBM_outcome_matchup_schema.pkl",
        "SeriesWinnerPrediction_LightGBM/SeriesWinnerPrediction_LightGBM_probability_calibrator.pkl",
        "SeriesWinnerPrediction_LightGBM/SeriesWinnerPrediction_LightGBM_probability_uncertainty.pkl",
        "GamelengthPrediction_LightGBM/GamelengthPrediction_LightGBM.pkl",
        "GamelengthPrediction_LightGBM/GamelengthPrediction_LightGBM_feature_pipeline.pkl",
        "GamelengthPrediction_LightGBM/GamelengthPrediction_LightGBM_residual_summary.pkl",
        "GamelengthPrediction_LightGBM/GamelengthPrediction_LightGBM_prop_calibrator.pkl",
        "TotalKillsPrediction_LightGBM/TotalKillsPrediction_LightGBM.pkl",
        "TotalKillsPrediction_LightGBM/TotalKillsPrediction_LightGBM_feature_pipeline.pkl",
        "TotalKillsPrediction_LightGBM/TotalKillsPrediction_LightGBM_residual_summary.pkl",
        "TotalKillsPrediction_LightGBM/TotalKillsPrediction_LightGBM_prop_calibrator.pkl",
        "TotalTowersPrediction_LightGBM/TotalTowersPrediction_LightGBM.pkl",
        "TotalTowersPrediction_LightGBM/TotalTowersPrediction_LightGBM_feature_pipeline.pkl",
        "TotalTowersPrediction_LightGBM/TotalTowersPrediction_LightGBM_residual_summary.pkl",
        "TotalTowersPrediction_LightGBM/TotalTowersPrediction_LightGBM_prop_calibrator.pkl",
        "_evaluation/GamelengthPrediction_LightGBM/prop_evaluation_report.json",
        "_evaluation/TotalKillsPrediction_LightGBM/prop_evaluation_report.json",
        "_evaluation/TotalTowersPrediction_LightGBM/prop_evaluation_report.json",
    }
)


def _missing_required_bundle_artifacts(
    target: object,
    artifact_names: Collection[str],
) -> set[str]:
    if target != "complete_lol_bundle":
        return set()
    return set(_COMPLETE_LOL_BUNDLE_REQUIRED_ARTIFACTS) - set(artifact_names)


class ModelRegistryError(ValueError):
    """Raised when a model registry operation would violate its audit rules."""


class PromotionPolicy(StrEnum):
    """Distinct evidence standards for routine refreshes and Optuna research."""

    ROUTINE = "routine"
    OPTUNA = "optuna"


@dataclass(frozen=True)
class TrainingTriggerState:
    """New evidence accumulated since the last candidate training run."""

    new_valid_maps: int
    new_major_maps: int
    last_candidate_at: datetime
    evaluated_at: datetime

    def __post_init__(self) -> None:
        if self.new_valid_maps < 0 or self.new_major_maps < 0:
            raise ValueError("new map counts cannot be negative")
        _require_utc(self.last_candidate_at, field="last_candidate_at")
        _require_utc(self.evaluated_at, field="evaluated_at")
        if self.evaluated_at < self.last_candidate_at:
            raise ValueError("evaluated_at cannot precede last_candidate_at")


@dataclass(frozen=True)
class TrainingTriggerEvaluation:
    state: TrainingTriggerState
    triggered: bool
    reasons: tuple[str, ...]
    candidate_id: str


def evaluate_training_triggers_from_history(
    history: pd.DataFrame,
    *,
    registry: ModelRegistry,
    evaluated_at: datetime,
    major_leagues: Collection[str],
    valid_map_threshold: int = 50,
    major_map_threshold: int = 20,
    maximum_age: timedelta = timedelta(days=30),
) -> TrainingTriggerEvaluation:
    """Count new complete maps since the latest immutable candidate."""
    _require_utc(evaluated_at, field="evaluated_at")
    required = {"gameid", "date", "league", "datacompleteness"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"training trigger history missing columns: {sorted(missing)}")
    last_candidate_at = _latest_candidate_at(registry)
    if last_candidate_at is None:
        last_candidate_at = _history_start(history, evaluated_at=evaluated_at)
    dates = pd.to_datetime(history["date"], errors="coerce", utc=True)
    complete = history["datacompleteness"].astype(str).str.casefold().eq("complete")
    new = history.loc[complete & dates.gt(last_candidate_at)].copy()
    new["__date"] = dates.loc[new.index]
    maps = (
        new.sort_values("__date", kind="mergesort")
        .drop_duplicates(subset=["gameid"], keep="last")
        .reset_index(drop=True)
    )
    major = {league.casefold() for league in major_leagues}
    new_major_maps = int(maps["league"].astype(str).str.casefold().isin(major).sum())
    state = TrainingTriggerState(
        new_valid_maps=int(maps["gameid"].nunique()),
        new_major_maps=new_major_maps,
        last_candidate_at=last_candidate_at,
        evaluated_at=evaluated_at,
    )
    reasons: list[str] = []
    if state.new_valid_maps >= valid_map_threshold:
        reasons.append("new_valid_maps")
    if state.new_major_maps >= major_map_threshold:
        reasons.append("new_major_maps")
    if state.evaluated_at - state.last_candidate_at >= maximum_age:
        reasons.append("monthly")
    candidate_id = f"candidate-{evaluated_at.strftime('%Y%m%dT%H%M%SZ')}"
    return TrainingTriggerEvaluation(
        state=state,
        triggered=bool(reasons),
        reasons=tuple(reasons),
        candidate_id=candidate_id,
    )


def _latest_candidate_at(registry: ModelRegistry) -> datetime | None:
    created: list[datetime] = []
    for manifest_path in registry.candidates.glob("*/manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            parsed = datetime.fromisoformat(str(payload["created_at"]))
            _require_utc(parsed, field="candidate created_at")
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        created.append(parsed)
    return max(created) if created else None


def _history_start(history: pd.DataFrame, *, evaluated_at: datetime) -> datetime:
    dates = pd.to_datetime(history["date"], errors="coerce", utc=True).dropna()
    if dates.empty:
        return evaluated_at
    first = dates.min().to_pydatetime()
    return min(first, evaluated_at)


@dataclass(frozen=True)
class BootstrapComparison:
    """Paired estimate of candidate relative log-loss improvement."""

    relative_improvement: float
    lower_bound: float
    upper_bound: float
    confidence: float
    samples: int


def paired_bootstrap_improvement(
    champion_losses: tuple[float, ...] | list[float],
    candidate_losses: tuple[float, ...] | list[float],
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 7,
) -> BootstrapComparison:
    """Compare paired per-row losses using a deterministic bootstrap interval."""
    champion = _loss_array(champion_losses, field="champion_losses")
    candidate = _loss_array(candidate_losses, field="candidate_losses")
    if champion.shape != candidate.shape:
        raise ValueError("champion and candidate losses must have equal lengths")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if samples < _MIN_BOOTSTRAP_SAMPLES:
        raise ValueError(f"samples must be at least {_MIN_BOOTSTRAP_SAMPLES}")

    champion_mean = float(np.mean(champion))
    if champion_mean <= 0:
        raise ValueError("champion mean loss must be positive")

    paired_deltas = champion - candidate
    relative_improvement = float(np.mean(paired_deltas) / champion_mean)
    generator = np.random.default_rng(seed)
    # The fixed original champion mean keeps the interval paired while avoiding
    # a degenerate ratio when every candidate loss is an exact scalar multiple.
    bootstrapped = np.empty(samples, dtype=float)
    chunk_size = max(1, _MAX_BOOTSTRAP_INDEX_CELLS // champion.size)
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        sample_indices = generator.integers(
            0,
            champion.size,
            size=(stop - start, champion.size),
        )
        bootstrapped[start:stop] = (
            np.mean(paired_deltas[sample_indices], axis=1) / champion_mean
        )
    tail = (1 - confidence) / 2
    lower_bound, upper_bound = np.quantile(bootstrapped, [tail, 1 - tail])

    return BootstrapComparison(
        relative_improvement=relative_improvement,
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        confidence=confidence,
        samples=samples,
    )


@dataclass(frozen=True)
class PromotionEvidence:
    """Probability-quality evidence used by the automatic promotion gate."""

    champion_log_losses: tuple[float, ...]
    candidate_log_losses: tuple[float, ...]
    champion_brier: float
    candidate_brier: float
    cohort_log_loss: dict[str, tuple[float, float, int]]
    champion_ece: float = 0.0
    candidate_ece: float = 0.0
    regression_target_mae: dict[str, tuple[float, float]] | None = None
    champion_evidence_status: dict[str, str] | None = None
    candidate_evidence_status: dict[str, str] | None = None
    operational_failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _loss_array(self.champion_log_losses, field="champion_log_losses")
        _loss_array(self.candidate_log_losses, field="candidate_log_losses")
        _finite_nonnegative(self.champion_brier, field="champion_brier")
        _finite_nonnegative(self.candidate_brier, field="candidate_brier")
        _finite_nonnegative(self.champion_ece, field="champion_ece")
        _finite_nonnegative(self.candidate_ece, field="candidate_ece")
        for name, (champion, candidate, count) in self.cohort_log_loss.items():
            if not name.strip():
                raise ValueError("cohort names cannot be empty")
            _finite_nonnegative(champion, field=f"cohort {name} champion loss")
            _finite_nonnegative(candidate, field=f"cohort {name} candidate loss")
            if count < 0:
                raise ValueError(f"cohort {name} count cannot be negative")
        for name, (champion, candidate) in (self.regression_target_mae or {}).items():
            if not name.strip():
                raise ValueError("regression target names cannot be empty")
            _finite_nonnegative(champion, field=f"{name} champion MAE")
            _finite_nonnegative(candidate, field=f"{name} candidate MAE")


@dataclass(frozen=True)
class PromotionDecision:
    """Machine-readable result of the candidate promotion gate."""

    promote: bool
    reasons: tuple[str, ...]
    relative_improvement: float
    confidence_lower_bound: float
    confidence_degradation_upper_bound: float
    safety_failures: tuple[str, ...]
    policy: PromotionPolicy


@dataclass(frozen=True)
class CandidateReview:
    """Persisted result of replaying two bundles on the candidate's sealed rows."""

    candidate_id: str
    champion_id: str | None
    status: str
    policy: PromotionPolicy
    promoted: bool
    reasons: tuple[str, ...]
    row_fingerprints: dict[str, str]
    evidence: dict[str, Any]


def evaluate_promotion(
    evidence: PromotionEvidence,
    *,
    policy: PromotionPolicy = PromotionPolicy.ROUTINE,
    minimum_relative_improvement: float = _MIN_RETUNE_RELATIVE_IMPROVEMENT,
    maximum_relative_degradation: float = _MAX_ROUTINE_RELATIVE_DEGRADATION,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    seed: int = 7,
    maximum_cohort_relative_regression: float = (_MAX_COHORT_RELATIVE_REGRESSION),
    minimum_cohort_size: int = _MIN_COHORT_SIZE,
) -> PromotionDecision:
    """Apply the confirmed probability-quality and cohort safety gates."""
    policy = PromotionPolicy(policy)
    if minimum_relative_improvement < 0:
        raise ValueError("minimum_relative_improvement cannot be negative")
    if maximum_relative_degradation < 0:
        raise ValueError("maximum_relative_degradation cannot be negative")
    if maximum_cohort_relative_regression < 0:
        raise ValueError("maximum_cohort_relative_regression cannot be negative")
    if minimum_cohort_size <= 0:
        raise ValueError("minimum_cohort_size must be positive")

    comparison = paired_bootstrap_improvement(
        evidence.champion_log_losses,
        evidence.candidate_log_losses,
        confidence=confidence,
        samples=bootstrap_samples,
        seed=seed,
    )
    reasons: list[str] = []
    safety_failures: list[str] = []

    degradation_upper_bound = -comparison.lower_bound
    if policy is PromotionPolicy.ROUTINE:
        if degradation_upper_bound > maximum_relative_degradation:
            reasons.append("log_loss_noninferiority_failed")
    elif (
        comparison.relative_improvement < minimum_relative_improvement
        or comparison.lower_bound <= 0
    ):
        reasons.append("meaningful_improvement_not_proven")
    if evidence.candidate_brier - evidence.champion_brier > _MAX_BRIER_DEGRADATION:
        safety_failures.append("brier_regression")
    if evidence.candidate_ece - evidence.champion_ece > _MAX_ECE_DEGRADATION:
        safety_failures.append("ece_regression")

    for cohort in sorted(evidence.cohort_log_loss):
        champion_loss, candidate_loss, count = evidence.cohort_log_loss[cohort]
        if (
            not cohort.startswith("actionable")
            or count < minimum_cohort_size
            or champion_loss == 0
        ):
            continue
        relative_regression = (candidate_loss - champion_loss) / champion_loss
        if relative_regression > maximum_cohort_relative_regression:
            safety_failures.append(f"cohort_regression:{cohort}")

    safety_failures.extend(
        f"operational_failure:{reason}" for reason in evidence.operational_failures
    )

    reasons.extend(safety_failures)
    return PromotionDecision(
        promote=not reasons,
        reasons=tuple(reasons),
        relative_improvement=comparison.relative_improvement,
        confidence_lower_bound=comparison.lower_bound,
        confidence_degradation_upper_bound=degradation_upper_bound,
        safety_failures=tuple(safety_failures),
        policy=policy,
    )


def review_candidate_on_sealed_rows(
    registry: ModelRegistry,
    candidate_id: str,
    *,
    policy: PromotionPolicy = PromotionPolicy.ROUTINE,
    automatic: bool = True,
    reviewed_at: datetime,
    bootstrap_samples: int = 10_000,
) -> CandidateReview:
    """Replay champion and candidate on one sealed holdout and apply promotion gates."""
    _require_utc(reviewed_at, field="reviewed_at")
    policy = PromotionPolicy(policy)
    champion_id = registry.champion_id()
    if not registry.verify_bundle(candidate_id):
        raise ModelRegistryError(f"candidate bundle is unhealthy: {candidate_id}")
    if champion_id is None:
        candidate_series = (
            registry.candidates
            / candidate_id
            / _EVALUATION_MODELS["series_winner"]
            / f"{_EVALUATION_MODELS['series_winner']}.pkl"
        )
        if candidate_series.is_file():
            return _review_first_winner_v2_candidate(
                registry,
                candidate_id=candidate_id,
                champion_id=None,
                policy=policy,
                reviewed_at=reviewed_at,
                bootstrap_samples=bootstrap_samples,
            )
        return _write_candidate_review(
            registry,
            CandidateReview(
                candidate_id=candidate_id,
                champion_id=None,
                status="bootstrap_manual_review_required",
                policy=policy,
                promoted=False,
                reasons=("no_champion_comparator",),
                row_fingerprints={},
                evidence={
                    "drift_review": _bundle_drift_baseline(
                        registry.candidates / candidate_id
                    )
                },
            ),
            reviewed_at=reviewed_at,
        )
    direct_series_name = _EVALUATION_MODELS["series_winner"]
    champion_series = (
        registry.candidates
        / champion_id
        / direct_series_name
        / f"{direct_series_name}.pkl"
    )
    if not champion_series.is_file():
        return _review_first_winner_v2_candidate(
            registry,
            candidate_id=candidate_id,
            champion_id=champion_id,
            policy=policy,
            reviewed_at=reviewed_at,
            bootstrap_samples=bootstrap_samples,
        )
    try:
        winner_evidence, fingerprints, report = _replay_promotion_evidence(
            registry,
            candidate_id=candidate_id,
            champion_id=champion_id,
        )
        decisions = {
            target: evaluate_promotion(
                evidence,
                policy=policy,
                bootstrap_samples=bootstrap_samples,
            )
            for target, evidence in winner_evidence.items()
        }
    except Exception as error:
        return _write_candidate_review(
            registry,
            CandidateReview(
                candidate_id=candidate_id,
                champion_id=champion_id,
                status="blocked",
                policy=policy,
                promoted=False,
                reasons=(f"sealed_replay_failed:{type(error).__name__}:{error}",),
                row_fingerprints={},
                evidence={},
            ),
            reviewed_at=reviewed_at,
        )

    reasons = tuple(
        f"{target}:{reason}"
        for target, decision in decisions.items()
        for reason in decision.reasons
    )
    report["strategy_readiness"] = _build_review_readiness(
        registry.candidates / candidate_id,
        candidate_id=candidate_id,
        reviewed_at=reviewed_at,
        series_report=report["sealed_rows"]["series_winner"],
        blocked=bool(reasons),
    )
    decision_report = {
        target: {
            "promote": decision.promote,
            "reasons": list(decision.reasons),
            "relative_improvement": decision.relative_improvement,
            "confidence_lower_bound": decision.confidence_lower_bound,
            "confidence_degradation_upper_bound": (
                decision.confidence_degradation_upper_bound
            ),
            "safety_failures": list(decision.safety_failures),
        }
        for target, decision in decisions.items()
    }
    all_winners_pass = all(decision.promote for decision in decisions.values())
    should_promote = (
        automatic and policy is PromotionPolicy.ROUTINE and all_winners_pass
    )
    if should_promote:
        _write_candidate_review(
            registry,
            CandidateReview(
                candidate_id=candidate_id,
                champion_id=champion_id,
                status="promotion_approved",
                policy=policy,
                promoted=False,
                reasons=reasons,
                row_fingerprints=fingerprints,
                evidence=report | {"winner_decisions": decision_report},
            ),
            reviewed_at=reviewed_at,
        )
        registry.promote(
            candidate_id,
            promoted_at=reviewed_at,
            reason="routine candidate passed sealed-row non-inferiority gates",
        )
    if should_promote:
        status = "auto_promoted"
    elif all_winners_pass and policy is PromotionPolicy.OPTUNA:
        status = "manual_review_required"
    else:
        status = "blocked"
    return _write_candidate_review(
        registry,
        CandidateReview(
            candidate_id=candidate_id,
            champion_id=champion_id,
            status=status,
            policy=policy,
            promoted=should_promote,
            reasons=reasons,
            row_fingerprints=fingerprints,
            evidence=report | {"winner_decisions": decision_report},
        ),
        reviewed_at=reviewed_at,
    )


def _review_first_winner_v2_candidate(
    registry: ModelRegistry,
    *,
    candidate_id: str,
    champion_id: str | None,
    policy: PromotionPolicy,
    reviewed_at: datetime,
    bootstrap_samples: int,
) -> CandidateReview:
    """Compare both first-generation Winner V2 targets with rating baselines."""
    root = registry.candidates / candidate_id
    reasons: list[str] = []
    operational_failures = _candidate_operational_failures(root)
    winner_reports: dict[str, Any] = {}
    fingerprints: dict[str, str] = {}
    for target in sorted(_WINNER_EVALUATION_TARGETS):
        try:
            target_reasons, fingerprint, report = (
                _review_winner_target_against_rating_baseline(
                    root,
                    target=target,
                    operational_failures=(
                        operational_failures if target == "series_winner" else []
                    ),
                    bootstrap_samples=bootstrap_samples,
                )
            )
            reasons.extend(f"{target}:{reason}" for reason in target_reasons)
            fingerprints[target] = fingerprint
            winner_reports[target] = report
        except Exception as error:
            reasons.append(
                f"{target}:baseline_review_failed:{type(error).__name__}:{error}"
            )

    evidence_report: dict[str, Any] = {
        "first_winner_v2": True,
        "comparator": "predeclared_rating_logistic_baseline",
        "winner_targets": winner_reports,
        "manual_promotion_required": True,
    }
    evidence_report["strategy_readiness"] = _build_review_readiness(
        root,
        candidate_id=candidate_id,
        reviewed_at=reviewed_at,
        series_report=winner_reports.get("series_winner", {}),
        blocked=bool(reasons),
    )

    review = CandidateReview(
        candidate_id=candidate_id,
        champion_id=champion_id,
        status="manual_review_required" if not reasons else "blocked",
        policy=policy,
        promoted=False,
        reasons=tuple(dict.fromkeys(reasons)),
        row_fingerprints=fingerprints,
        evidence=evidence_report,
    )
    return _write_candidate_review(registry, review, reviewed_at=reviewed_at)


def _review_winner_target_against_rating_baseline(
    root: Path,
    *,
    target: str,
    operational_failures: list[str],
    bootstrap_samples: int,
) -> tuple[list[str], str, dict[str, Any]]:
    """Review one Winner V2 target without borrowing evidence from another."""
    from lol_bets.prediction_models.gbdt_model import (
        MAX_ABSOLUTE_CALIBRATION_INTERCEPT,
        MAX_CALIBRATION_SLOPE,
        MIN_CALIBRATION_SLOPE,
        GradientBoostingModel,
        conservative_probability_report,
    )

    model_name = _EVALUATION_MODELS[target]
    evaluation = root / "_evaluation" / model_name
    raw = pd.read_parquet(evaluation / "features.parquet")
    labels = pd.read_parquet(evaluation / "labels.parquet")
    actual = pd.to_numeric(labels["actual"], errors="raise").to_numpy(dtype=float)
    metadata = labels.drop(columns=["actual"])
    pipeline = load_model(root / model_name / f"{model_name}_feature_pipeline.pkl")
    model = load_model(root / model_name / f"{model_name}.pkl")
    calibrator = load_model(
        root / model_name / f"{model_name}_probability_calibrator.pkl"
    )
    uncertainty = load_model(
        root / model_name / f"{model_name}_probability_uncertainty.pkl"
    )
    transformed = pipeline.transform(raw.copy())
    baseline = np.asarray(model.rating_baseline_probability(transformed), dtype=float)
    candidate = np.asarray(
        calibrator.predict(model.predict_proba(transformed)[:, 1], metadata=metadata),
        dtype=float,
    )
    baseline_quality = prediction_quality(
        actual.astype(int).tolist(), baseline.tolist()
    )
    candidate_quality = prediction_quality(
        actual.astype(int).tolist(), candidate.tolist()
    )
    cluster_col = "series_id" if target == "next_map_winner" else None
    if cluster_col is None:
        baseline_losses = _binary_log_losses(actual, baseline)
        candidate_losses = _binary_log_losses(actual, candidate)
    else:
        baseline_losses = _clustered_binary_log_losses(
            labels, actual, baseline, cluster_col=cluster_col
        )
        candidate_losses = _clustered_binary_log_losses(
            labels, actual, candidate, cluster_col=cluster_col
        )
    evidence = PromotionEvidence(
        champion_log_losses=tuple(baseline_losses),
        candidate_log_losses=tuple(candidate_losses),
        champion_brier=baseline_quality.brier,
        candidate_brier=candidate_quality.brier,
        champion_ece=baseline_quality.calibration_error,
        candidate_ece=candidate_quality.calibration_error,
        cohort_log_loss=_cohort_replay_losses(
            labels,
            actual,
            baseline,
            candidate,
            cluster_col=cluster_col,
        ),
        operational_failures=tuple(operational_failures),
    )
    decision = evaluate_promotion(
        evidence,
        policy=PromotionPolicy.ROUTINE,
        bootstrap_samples=bootstrap_samples,
    )
    calibration = GradientBoostingModel.compute_probability_calibration_metrics(
        pd.Series(actual), candidate
    )
    reasons = list(decision.reasons)
    warnings: list[str] = []
    readiness_failures: list[str] = []
    slope = calibration.get("calibration_slope")
    intercept = calibration.get("calibration_intercept")
    if slope is None or not (
        MIN_CALIBRATION_SLOPE <= float(slope) <= MAX_CALIBRATION_SLOPE
    ):
        warnings.append("calibration_slope_outside_0.8_1.2")
    if intercept is None or abs(float(intercept)) > MAX_ABSOLUTE_CALIBRATION_INTERCEPT:
        warnings.append("calibration_intercept_above_0.10")
    if target == "series_winner":
        readiness_failures.extend(warnings)
    conservative_report: dict[str, Any] | None = None
    if target == "series_winner":
        conservative_report = conservative_probability_report(
            model,
            transformed,
            pd.Series(actual),
            calibrator,
            uncertainty,
            metadata,
        )
        if conservative_report["coverage"]["passed"] is not True:
            readiness_failures.append("conservative_probability_coverage_failed")
    report = {
        "sealed_rows": len(labels),
        "bootstrap_unit": cluster_col or "series",
        "bootstrap_units": len(baseline_losses),
        "rating_baseline": asdict(baseline_quality),
        "candidate": asdict(candidate_quality),
        "calibration": calibration,
        "relative_improvement": decision.relative_improvement,
        "confidence_lower_bound": decision.confidence_lower_bound,
        "confidence_degradation_upper_bound": (
            decision.confidence_degradation_upper_bound
        ),
        "cohorts": {
            name: {
                "rating_baseline_log_loss": values[0],
                "candidate_log_loss": values[1],
                "count": values[2],
            }
            for name, values in evidence.cohort_log_loss.items()
        },
        "operational_failures": list(operational_failures),
        "reasons": list(dict.fromkeys(reasons)),
        "warnings": warnings,
        "readiness_failures": list(dict.fromkeys(readiness_failures)),
        "conservative_probability": conservative_report,
        "readiness_cohorts": _series_readiness_cohorts(
            labels,
            actual,
            baseline,
            candidate,
        ),
    }
    return list(dict.fromkeys(reasons)), _frame_fingerprint(labels), report


def load_candidate_review(
    registry: ModelRegistry,
    candidate_id: str,
) -> dict[str, Any] | None:
    path = registry.root / "reviews" / f"{candidate_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class CandidateManifest:
    """Reproducibility facts that identify one trained candidate."""

    model_id: str
    sport: str
    target: str
    created_at: datetime
    code_version: str
    data_manifest: str
    feature_fingerprint: str
    config_hash: str
    dependency_lock_hash: str
    random_seed: int
    metrics: dict[str, float]

    def __post_init__(self) -> None:
        for field_name in (
            "model_id",
            "sport",
            "target",
            "code_version",
            "data_manifest",
            "feature_fingerprint",
            "config_hash",
            "dependency_lock_hash",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} cannot be empty")
        if "/" in self.model_id or "\\" in self.model_id:
            raise ValueError("model_id cannot contain path separators")
        _require_utc(self.created_at, field="created_at")
        if self.random_seed < 0:
            raise ValueError("random_seed cannot be negative")
        if not self.metrics:
            raise ValueError("metrics cannot be empty")
        for name, value in self.metrics.items():
            if not name.strip() or not math.isfinite(value):
                raise ValueError("metrics require non-empty names and finite values")


class ModelRegistry:
    """Filesystem registry with immutable candidates and an auditable pointer."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.candidates = self.root / "candidates"
        self.champion_pointer = self.root / "champion.json"
        self.history_path = self.root / "champion_history.jsonl"
        self.transition_path = self.root / "champion_transition.json"
        self.actionability_path = self.root / "model_actionability.json"
        self.actionability_history_path = (
            self.root / "model_actionability_history.jsonl"
        )
        self.strategy_readiness_path = self.root / "strategy_readiness.json"
        self.strategy_readiness_history_path = (
            self.root / "strategy_readiness_history.jsonl"
        )
        self.candidates.mkdir(parents=True, exist_ok=True)
        self._recover_champion_transition()

    def register_candidate(
        self,
        manifest: CandidateManifest,
        artifacts: dict[str, Path],
    ) -> Path:
        """Copy a complete candidate bundle into the registry exactly once."""
        destination = self.candidates / manifest.model_id
        if destination.exists():
            raise ModelRegistryError(
                f"candidate bundle already exists: {manifest.model_id}"
            )
        if not artifacts:
            raise ModelRegistryError("candidate bundle requires at least one artifact")

        normalized = {
            self._validate_artifact_name(name): Path(source)
            for name, source in artifacts.items()
        }
        missing = [str(path) for path in normalized.values() if not path.is_file()]
        if missing:
            raise ModelRegistryError(f"candidate artifacts do not exist: {missing}")
        incomplete = _missing_required_bundle_artifacts(manifest.target, normalized)
        if incomplete:
            raise ModelRegistryError(
                "complete LoL bundle is missing required artifacts: "
                f"{sorted(incomplete)}"
            )

        with tempfile.TemporaryDirectory(
            prefix=f".{manifest.model_id}-",
            dir=self.candidates,
        ) as temporary_directory:
            temporary = Path(temporary_directory)
            file_manifest: dict[str, dict[str, Any]] = {}
            for name, source in sorted(normalized.items()):
                stored_path = temporary / name
                stored_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, stored_path)
                file_manifest[name] = {
                    "sha256": _sha256(stored_path),
                    "size_bytes": stored_path.stat().st_size,
                }

            payload = _manifest_payload(manifest)
            payload["files"] = file_manifest
            _write_json(temporary / "manifest.json", payload)
            try:
                temporary.rename(destination)
            except FileExistsError as error:
                raise ModelRegistryError(
                    f"candidate bundle already exists: {manifest.model_id}"
                ) from error

        return destination

    def verify_bundle(self, model_id: str) -> bool:
        """Return whether every declared artifact is present and unchanged."""
        bundle = self.candidates / model_id
        manifest_path = bundle / "manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = manifest["files"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return False
        if (
            not isinstance(files, dict)
            or not files
            or _missing_required_bundle_artifacts(manifest.get("target"), files)
        ):
            return False

        for name, facts in files.items():
            try:
                safe_name = self._validate_artifact_name(name)
                artifact = bundle / safe_name
                expected_hash = facts["sha256"]
                expected_size = facts["size_bytes"]
            except (KeyError, TypeError, ModelRegistryError):
                return False
            if (
                not artifact.is_file()
                or artifact.stat().st_size != expected_size
                or _sha256(artifact) != expected_hash
            ):
                return False
        return True

    def champion_id(self) -> str | None:
        """Read the current champion model identifier, if one is selected."""
        if not self.champion_pointer.is_file():
            return None
        try:
            payload = json.loads(self.champion_pointer.read_text(encoding="utf-8"))
            model_id = payload["model_id"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ModelRegistryError("champion pointer is malformed") from error
        if not isinstance(model_id, str) or not model_id:
            raise ModelRegistryError("champion pointer model_id is invalid")
        return model_id

    def actionability(self, model_id: str) -> dict[str, Any]:
        """Return the serving actionability state for one registered model."""
        if not self.actionability_path.is_file():
            return {"status": "actionable", "reason": None, "changed_at": None}
        try:
            states = json.loads(self.actionability_path.read_text(encoding="utf-8"))
            state = states.get(model_id)
        except (OSError, json.JSONDecodeError, AttributeError) as error:
            raise ModelRegistryError(
                "model actionability registry is malformed"
            ) from error
        if not isinstance(state, dict):
            return {"status": "actionable", "reason": None, "changed_at": None}
        return state

    def is_actionable(self, model_id: str) -> bool:
        """Return whether a healthy model may support owner-facing reviews."""
        return (
            self.verify_bundle(model_id)
            and self.actionability(model_id).get("status") == "actionable"
        )

    def strategy_readiness(self) -> dict[str, Any] | None:
        """Return the active champion's separately reviewed strategy scope."""
        if not self.strategy_readiness_path.is_file():
            return None
        try:
            payload = json.loads(
                self.strategy_readiness_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError, TypeError) as error:
            raise ModelRegistryError(
                "strategy readiness pointer is malformed"
            ) from error
        if (
            not isinstance(payload, dict)
            or payload.get("candidate_id") != self.champion_id()
        ):
            raise ModelRegistryError("strategy readiness does not match the champion")
        return payload

    def quarantine(
        self,
        model_id: str,
        *,
        quarantined_at: datetime,
        reason: str,
    ) -> None:
        """Disable owner-facing use while preserving diagnostic serving."""
        _require_utc(quarantined_at, field="quarantined_at")
        reason = reason.strip()
        if not reason:
            raise ModelRegistryError("model quarantine reason cannot be empty")
        if not self.verify_bundle(model_id):
            raise ModelRegistryError(
                f"candidate bundle failed checksum verification: {model_id}"
            )
        states: dict[str, Any] = {}
        if self.actionability_path.is_file():
            try:
                loaded = json.loads(self.actionability_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    states = loaded
            except (OSError, json.JSONDecodeError) as error:
                raise ModelRegistryError(
                    "model actionability registry is malformed"
                ) from error
        state = {
            "status": "research_only",
            "reason": reason,
            "changed_at": quarantined_at.isoformat(),
        }
        if states.get(model_id) == state:
            return
        states[model_id] = state
        _atomic_write_json(self.actionability_path, states)
        history = {"model_id": model_id, **state}
        with self.actionability_history_path.open("a", encoding="utf-8") as handle:
            handle.write(_stable_json(history) + "\n")

    def artifact_path(
        self,
        name: str,
        *,
        model_id: str | None = None,
    ) -> Path:
        """Return one checksum-verified artifact from a candidate bundle."""
        safe_name = self._validate_artifact_name(name)
        artifacts = self.verified_artifact_paths(model_id=model_id)
        if safe_name not in artifacts:
            selected = model_id or self.champion_id()
            raise ModelRegistryError(
                f"candidate {selected} does not contain artifact: {safe_name}"
            )
        return artifacts[safe_name]

    def verified_artifact_paths(
        self,
        *,
        model_id: str | None = None,
    ) -> dict[str, Path]:
        """Verify one immutable bundle once and return all declared paths."""
        selected = model_id or self.champion_id()
        if selected is None:
            raise ModelRegistryError("no champion is selected")
        if not self.verify_bundle(selected):
            raise ModelRegistryError(
                f"candidate bundle failed checksum verification: {selected}"
            )
        manifest_path = self.candidates / selected / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = manifest["files"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ModelRegistryError("candidate manifest is malformed") from error
        return {
            self._validate_artifact_name(name): self.candidates / selected / name
            for name in files
        }

    def promote(
        self,
        model_id: str,
        *,
        promoted_at: datetime,
        reason: str,
    ) -> None:
        """Set a verified candidate as champion and append the decision."""
        self._change_champion(
            model_id,
            changed_at=promoted_at,
            reason=reason,
            action="promote",
        )

    def rollback(
        self,
        model_id: str,
        *,
        rolled_back_at: datetime,
        reason: str,
    ) -> None:
        """Restore a verified earlier candidate and append the decision."""
        self._change_champion(
            model_id,
            changed_at=rolled_back_at,
            reason=reason,
            action="rollback",
        )

    def _change_champion(
        self,
        model_id: str,
        *,
        changed_at: datetime,
        reason: str,
        action: str,
    ) -> None:
        _require_utc(changed_at, field="changed_at")
        if not reason.strip():
            raise ModelRegistryError("champion change reason cannot be empty")
        if not self.verify_bundle(model_id):
            raise ModelRegistryError(
                f"candidate bundle failed checksum verification: {model_id}"
            )
        if self.actionability(model_id).get("status") != "actionable":
            raise ModelRegistryError(
                f"quarantined model cannot be promoted: {model_id}"
            )
        direct_model = (
            self.candidates
            / model_id
            / "SeriesWinnerPrediction_LightGBM"
            / "SeriesWinnerPrediction_LightGBM.pkl"
        )
        manifest = json.loads(
            (self.candidates / model_id / "manifest.json").read_text(encoding="utf-8")
        )
        if direct_model.is_file() and manifest.get("target") == "complete_lol_bundle":
            review = load_candidate_review(self, model_id)
            accepted_statuses = {"manual_review_required", "promotion_approved"}
            if (
                not review
                or review.get("status") not in accepted_statuses
                or review.get("reasons")
            ):
                raise ModelRegistryError(
                    "Winner V2 promotion requires a passing sealed-row review"
                )
            readiness = review.get("evidence", {}).get("strategy_readiness")
            if (
                not isinstance(readiness, dict)
                or readiness.get("candidate_id") != model_id
            ):
                raise ModelRegistryError(
                    "Winner V2 promotion requires reviewed strategy readiness"
                )
        else:
            readiness = None

        previous = self.champion_id()
        pointer = {
            "model_id": model_id,
            "selected_at": changed_at.isoformat(),
        }
        history = {
            "action": action,
            "at": changed_at.isoformat(),
            "from_model_id": previous,
            "to_model_id": model_id,
            "reason": reason.strip(),
        }
        history["transition_id"] = hashlib.sha256(
            _stable_json(history).encode()
        ).hexdigest()[:24]
        transition = {"pointer": pointer, "history": history}
        if readiness is not None:
            transition["strategy_readiness"] = readiness
        _atomic_write_json(self.transition_path, transition)
        _atomic_write_json(self.champion_pointer, pointer)
        if readiness is not None:
            self._activate_strategy_readiness(
                readiness,
                transition_id=history["transition_id"],
                activated_at=changed_at,
            )
        self._append_champion_history(history)
        self.transition_path.unlink(missing_ok=True)

    def _recover_champion_transition(self) -> None:
        if not self.transition_path.is_file():
            return
        try:
            transition = json.loads(self.transition_path.read_text(encoding="utf-8"))
            pointer = transition["pointer"]
            history = transition["history"]
            model_id = str(pointer["model_id"])
            transition_id = str(history["transition_id"])
            readiness = transition.get("strategy_readiness")
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ModelRegistryError("champion transition is malformed") from error
        if not self.verify_bundle(model_id):
            raise ModelRegistryError(
                f"pending champion transition bundle is unhealthy: {model_id}"
            )
        _atomic_write_json(self.champion_pointer, pointer)
        if isinstance(readiness, dict):
            self._activate_strategy_readiness(
                readiness,
                transition_id=transition_id,
                activated_at=datetime.fromisoformat(str(pointer["selected_at"])),
            )
        self._append_champion_history(history, transition_id=transition_id)
        self.transition_path.unlink(missing_ok=True)

    def _activate_strategy_readiness(
        self,
        readiness: dict[str, Any],
        *,
        transition_id: str,
        activated_at: datetime,
    ) -> None:
        payload = dict(readiness)
        payload["activated_at"] = activated_at.isoformat()
        payload["transition_id"] = transition_id
        _atomic_write_json(self.strategy_readiness_path, payload)
        if self.strategy_readiness_history_path.is_file():
            for line in self.strategy_readiness_history_path.read_text(
                encoding="utf-8"
            ).splitlines():
                try:
                    if json.loads(line).get("transition_id") == transition_id:
                        return
                except json.JSONDecodeError:
                    continue
        with self.strategy_readiness_history_path.open("a", encoding="utf-8") as stream:
            stream.write(_stable_json(payload) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _append_champion_history(
        self,
        history: dict[str, Any],
        *,
        transition_id: str | None = None,
    ) -> None:
        identity = transition_id or str(history["transition_id"])
        if self.history_path.is_file():
            for line in self.history_path.read_text(encoding="utf-8").splitlines():
                try:
                    if json.loads(line).get("transition_id") == identity:
                        return
                except json.JSONDecodeError:
                    continue
        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(_stable_json(history) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _validate_artifact_name(name: str) -> str:
        if not isinstance(name, str) or not name:
            raise ModelRegistryError("artifact names must be non-empty strings")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "." in path.parts:
            raise ModelRegistryError(f"unsafe artifact name: {name}")
        return path.as_posix()


def resolve_serving_artifact(
    legacy_path: Path,
    *,
    registry_root: Path,
    legacy_root: Path,
    model_id: str | None = None,
) -> Path:
    """Resolve a verified champion artifact or retain bootstrap legacy paths."""
    registry = ModelRegistry(registry_root)
    champion = model_id or registry.champion_id()
    if champion is None:
        return Path(legacy_path)
    try:
        name = Path(legacy_path).resolve().relative_to(Path(legacy_root).resolve())
    except ValueError as error:
        raise ModelRegistryError(
            f"serving artifact is outside the model root: {legacy_path}"
        ) from error
    return registry.artifact_path(name.as_posix(), model_id=champion)


def register_current_candidate(
    *,
    registry: ModelRegistry,
    model_id: str,
    target: str,
    code_version: str,
    metrics: dict[str, float],
    created_at: datetime,
    random_seed: int = 7,
    model_root: Path = MODELS_DIR,
    config_root: Path = CONFIG_DIR,
    product_config: Path = PRODUCT_CONFIG,
    dependency_lock: Path = SUITE_ROOT / "uv.lock",
    training_paths: tuple[Path, ...] = (
        TRAINING_TEAM_DATA,
        TRAINING_PLAYER_DATA,
    ),
) -> Path:
    """Freeze the complete current inference tree as an immutable candidate."""
    artifacts = {
        path.relative_to(model_root).as_posix(): path
        for path in sorted(model_root.rglob("*"))
        if path.is_file() and path.name != "README.md" and not path.name.startswith(".")
    }
    if not artifacts:
        raise ModelRegistryError(f"no inference artifacts found under {model_root}")
    manifest = CandidateManifest(
        model_id=model_id,
        sport="lol",
        target=target,
        created_at=created_at,
        code_version=code_version,
        data_manifest=_paths_fingerprint(training_paths),
        feature_fingerprint=_paths_fingerprint(
            tuple(sorted(config_root.rglob("*.json")))
        ),
        config_hash=_paths_fingerprint((product_config,)),
        dependency_lock_hash=_paths_fingerprint((dependency_lock,)),
        random_seed=random_seed,
        metrics=metrics,
    )
    return registry.register_candidate(manifest, artifacts)


_EVALUATION_MODELS = {
    "series_winner": "SeriesWinnerPrediction_LightGBM",
    "gamelength": "GamelengthPrediction_LightGBM",
    "total_kills": "TotalKillsPrediction_LightGBM",
    "total_towers": "TotalTowersPrediction_LightGBM",
}
_WINNER_EVALUATION_TARGETS = frozenset({"series_winner"})


def _replay_promotion_evidence(
    registry: ModelRegistry,
    *,
    candidate_id: str,
    champion_id: str,
) -> tuple[dict[str, PromotionEvidence], dict[str, str], dict[str, Any]]:
    candidate_root = registry.candidates / candidate_id
    champion_root = registry.candidates / champion_id
    operational_failures = _candidate_operational_failures(candidate_root)
    fingerprints: dict[str, str] = {}
    results: dict[str, dict[str, Any]] = {}
    regression_mae: dict[str, tuple[float, float]] = {}
    drift_targets: dict[str, dict[str, Any]] = {}
    winner_evidence: dict[str, PromotionEvidence] = {}

    for target, model_name in _EVALUATION_MODELS.items():
        evaluation = candidate_root / "_evaluation" / model_name
        raw = pd.read_parquet(evaluation / "features.parquet")
        labels = pd.read_parquet(evaluation / "labels.parquet")
        if len(raw) != len(labels) or not len(raw):
            raise ValueError(f"sealed {target} rows are empty or misaligned")
        fingerprints[target] = _frame_fingerprint(labels)
        actual = pd.to_numeric(labels["actual"], errors="raise").to_numpy(dtype=float)
        champion_prediction = _replay_bundle_target(
            champion_root,
            model_name=model_name,
            raw_features=raw,
            metadata=labels.drop(columns=["actual"]),
            classification=target in _WINNER_EVALUATION_TARGETS,
        )
        candidate_prediction = _replay_bundle_target(
            candidate_root,
            model_name=model_name,
            raw_features=raw,
            metadata=labels.drop(columns=["actual"]),
            classification=target in _WINNER_EVALUATION_TARGETS,
        )
        try:
            drift_targets[target] = _target_drift_review(
                raw,
                labels,
                champion_prediction=champion_prediction,
                candidate_prediction=candidate_prediction,
                candidate_root=candidate_root,
                champion_root=champion_root,
                model_name=model_name,
            )
        except Exception as error:
            drift_targets[target] = {
                "status": "unavailable",
                "warnings": [f"drift_review_unavailable:{type(error).__name__}"],
            }
        if target in _WINNER_EVALUATION_TARGETS:
            champion_quality = prediction_quality(
                actual.astype(int).tolist(), champion_prediction.tolist()
            )
            candidate_quality = prediction_quality(
                actual.astype(int).tolist(), candidate_prediction.tolist()
            )
            cluster_col = "series_id" if target == "next_map_winner" else None
            if cluster_col is None:
                champion_losses = _binary_log_losses(actual, champion_prediction)
                candidate_losses = _binary_log_losses(actual, candidate_prediction)
            else:
                champion_losses = _clustered_binary_log_losses(
                    labels,
                    actual,
                    champion_prediction,
                    cluster_col=cluster_col,
                )
                candidate_losses = _clustered_binary_log_losses(
                    labels,
                    actual,
                    candidate_prediction,
                    cluster_col=cluster_col,
                )
            target_cohorts = _cohort_replay_losses(
                labels,
                actual,
                champion_prediction,
                candidate_prediction,
                cluster_col=cluster_col,
            )
            winner_evidence[target] = PromotionEvidence(
                champion_log_losses=tuple(champion_losses),
                candidate_log_losses=tuple(candidate_losses),
                champion_brier=champion_quality.brier,
                candidate_brier=candidate_quality.brier,
                champion_ece=champion_quality.calibration_error,
                candidate_ece=candidate_quality.calibration_error,
                cohort_log_loss=target_cohorts,
            )
            results[target] = {
                "rows": len(labels),
                "bootstrap_unit": cluster_col or "series",
                "bootstrap_units": len(champion_losses),
                "champion": asdict(champion_quality),
                "candidate": asdict(candidate_quality),
                "cohorts": {
                    name: {
                        "champion_log_loss": values[0],
                        "candidate_log_loss": values[1],
                        "count": values[2],
                    }
                    for name, values in target_cohorts.items()
                },
                "readiness_cohorts": _series_readiness_cohorts(
                    labels,
                    actual,
                    champion_prediction,
                    candidate_prediction,
                ),
            }
        else:
            champion_error = float(np.mean(np.abs(actual - champion_prediction)))
            candidate_error = float(np.mean(np.abs(actual - candidate_prediction)))
            regression_mae[target] = (champion_error, candidate_error)
            results[target] = {
                "rows": len(labels),
                "champion_mae": champion_error,
                "candidate_mae": candidate_error,
            }

    missing_winner_evidence = _WINNER_EVALUATION_TARGETS - set(winner_evidence)
    if missing_winner_evidence:
        raise ValueError(
            f"sealed winner evaluation is missing: {sorted(missing_winner_evidence)}"
        )
    champion_status = _bundle_evidence_status(champion_root)
    candidate_status = _bundle_evidence_status(candidate_root)
    series_evidence = winner_evidence["series_winner"]
    winner_evidence["series_winner"] = PromotionEvidence(
        champion_log_losses=series_evidence.champion_log_losses,
        candidate_log_losses=series_evidence.candidate_log_losses,
        champion_brier=series_evidence.champion_brier,
        candidate_brier=series_evidence.candidate_brier,
        champion_ece=series_evidence.champion_ece,
        candidate_ece=series_evidence.candidate_ece,
        cohort_log_loss=series_evidence.cohort_log_loss,
        operational_failures=tuple(operational_failures),
    )
    return (
        winner_evidence,
        fingerprints,
        {
            "sealed_rows": results,
            "champion_evidence_status": champion_status,
            "candidate_evidence_status": candidate_status,
            "operational_failures": operational_failures,
            "drift_review": {
                "status": "warning_only",
                "promotion_gate_effect": "none",
                "targets": drift_targets,
            },
        },
    )


def _target_drift_review(
    raw: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    champion_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    candidate_root: Path,
    champion_root: Path,
    model_name: str,
) -> dict[str, Any]:
    candidate_split = _split_profile(candidate_root, model_name)
    champion_split = _split_profile(champion_root, model_name)
    missingness = raw.isna().mean().sort_values(ascending=False)
    candidate_attribution = _attribution_profile(candidate_root, model_name)
    champion_attribution = _attribution_profile(champion_root, model_name)
    candidate_top = set(candidate_attribution["top_features"][:10])
    champion_top = set(champion_attribution["top_features"][:10])
    warnings: list[str] = []
    if missingness.size and float(missingness.iloc[0]) > _DRIFT_MISSINGNESS_WARNING:
        warnings.append("high_feature_missingness")
    if candidate_split["leagues"] != champion_split["leagues"]:
        warnings.append("league_coverage_changed")
    if (
        candidate_top
        and champion_top
        and len(candidate_top & champion_top) < _DRIFT_TOP_FEATURE_MINIMUM_OVERLAP
    ):
        warnings.append("top_feature_attribution_changed")
    candidate_distribution = _prediction_distribution(candidate_prediction)
    champion_distribution = _prediction_distribution(champion_prediction)
    if (
        abs(candidate_distribution["mean"] - champion_distribution["mean"])
        > _DRIFT_PREDICTION_MEAN_WARNING
    ):
        warnings.append("prediction_mean_shift")
    return {
        "candidate_data_window": candidate_split["date_window"],
        "champion_data_window": champion_split["date_window"],
        "candidate_leagues": candidate_split["leagues"],
        "champion_leagues": champion_split["leagues"],
        "sealed_league_counts": {
            str(key): int(value)
            for key, value in labels.get("league", pd.Series(dtype=str))
            .value_counts()
            .items()
        },
        "feature_availability": {
            "columns": int(raw.shape[1]),
            "fully_unavailable": int(missingness.eq(1.0).sum()),
            "worst_missingness": {
                str(feature): float(fraction)
                for feature, fraction in missingness.head(20).items()
            },
        },
        "prediction_distribution": {
            "candidate": candidate_distribution,
            "champion": champion_distribution,
        },
        "attribution_stability": {
            "candidate": candidate_attribution,
            "champion": champion_attribution,
            "top_10_overlap": len(candidate_top & champion_top),
        },
        "warnings": warnings,
    }


def _bundle_drift_baseline(root: Path) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    for target, model_name in _EVALUATION_MODELS.items():
        split = _split_profile(root, model_name)
        targets[target] = {
            "data_window": split["date_window"],
            "leagues": split["leagues"],
            "attribution": _attribution_profile(root, model_name),
        }
    return {
        "status": "baseline_only",
        "promotion_gate_effect": "none",
        "targets": targets,
    }


@cache
def _split_profile(root: Path, model_name: str) -> dict[str, Any]:
    path = root / "_evaluation" / model_name / "split_report.json"
    try:
        test = json.loads(path.read_text(encoding="utf-8")).get("test", {})
    except (OSError, json.JSONDecodeError, TypeError):
        test = {}
    return {
        "date_window": {"min": test.get("date_min"), "max": test.get("date_max")},
        "leagues": sorted(str(value) for value in (test.get("league_counts") or {})),
    }


@cache
def _attribution_profile(root: Path, model_name: str) -> dict[str, Any]:
    path = root / "_evaluation" / "summary.json"
    try:
        models = json.loads(path.read_text(encoding="utf-8"))["models"]
        model = next(item for item in models if item.get("model_name") == model_name)
        features = [str(item["feature"]) for item in model.get("top_features", [])]
    except (OSError, json.JSONDecodeError, KeyError, StopIteration, TypeError):
        features = []
    families: dict[str, int] = {}
    for feature in features[:20]:
        tokens = feature.removeprefix("delta_").split("_", 1)
        family = tokens[0]
        families[family] = families.get(family, 0) + 1
    return {"top_features": features[:20], "top_families": families}


def _prediction_distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _replay_bundle_target(
    root: Path,
    *,
    model_name: str,
    raw_features: pd.DataFrame,
    metadata: pd.DataFrame,
    classification: bool,
) -> np.ndarray:
    model_root = root / model_name
    pipeline = load_model(model_root / f"{model_name}_feature_pipeline.pkl")
    model = load_model(model_root / f"{model_name}.pkl")
    transformed = pipeline.transform(raw_features.copy())
    if not classification:
        return np.asarray(model.predict(transformed), dtype=float)
    calibrator_path = model_root / f"{model_name}_probability_calibrator.pkl"
    if not calibrator_path.is_file():
        raise ValueError(f"probability calibrator missing for {root.name}")
    calibrator = load_model(calibrator_path)
    raw_probability = np.asarray(model.predict_proba(transformed)[:, 1], dtype=float)
    return np.asarray(
        calibrator.predict(raw_probability, metadata=metadata), dtype=float
    )


def _binary_log_losses(actual: np.ndarray, probability: np.ndarray) -> list[float]:
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    return (-(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped))).tolist()


def _clustered_binary_log_losses(
    labels: pd.DataFrame,
    actual: np.ndarray,
    probability: np.ndarray,
    *,
    cluster_col: str,
) -> list[float]:
    """Average correlated row losses within a cluster before bootstrapping."""
    if cluster_col not in labels or labels[cluster_col].isna().any():
        raise ValueError(f"sealed labels require complete {cluster_col} clusters")
    frame = pd.DataFrame(
        {
            "cluster": labels[cluster_col].astype(str).to_numpy(),
            "loss": _binary_log_losses(actual, probability),
        }
    )
    return frame.groupby("cluster", sort=False)["loss"].mean().tolist()


def _cohort_replay_losses(
    labels: pd.DataFrame,
    actual: np.ndarray,
    champion: np.ndarray,
    candidate: np.ndarray,
    *,
    cluster_col: str | None = None,
) -> dict[str, tuple[float, float, int]]:
    cohorts: dict[str, np.ndarray] = {
        "all_research_all_supported": np.ones(len(labels), dtype=bool)
    }
    if "actionable" in labels:
        cohorts["actionable_tier1_plus_erls"] = (
            labels["actionable"].fillna(False).astype(bool).to_numpy()
        )
    for column, prefix in (
        ("league", "league"),
        ("league_region", "region"),
        ("league_tier", "tier"),
    ):
        if column not in labels:
            continue
        values = labels[column].fillna("unknown").astype(str)
        for value in sorted(values.unique()):
            cohorts[f"{prefix}:{value}"] = values.eq(value).to_numpy()
    champion_losses = np.asarray(_binary_log_losses(actual, champion))
    candidate_losses = np.asarray(_binary_log_losses(actual, candidate))
    results: dict[str, tuple[float, float, int]] = {}
    for name, mask in cohorts.items():
        if not np.any(mask):
            continue
        if cluster_col is None:
            results[name] = (
                float(np.mean(champion_losses[mask])),
                float(np.mean(candidate_losses[mask])),
                int(np.sum(mask)),
            )
            continue
        if cluster_col not in labels or labels[cluster_col].isna().any():
            raise ValueError(f"sealed labels require complete {cluster_col} clusters")
        clustered = (
            pd.DataFrame(
                {
                    "cluster": labels.loc[mask, cluster_col].astype(str).to_numpy(),
                    "champion": champion_losses[mask],
                    "candidate": candidate_losses[mask],
                }
            )
            .groupby("cluster", sort=False)[["champion", "candidate"]]
            .mean()
        )
        results[name] = (
            float(clustered["champion"].mean()),
            float(clustered["candidate"].mean()),
            int(len(clustered)),
        )
    return results


def _series_readiness_cohorts(
    labels: pd.DataFrame,
    actual: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Build only the finite owner-actionable series cohort lattice."""
    from oracle_bets_core.league_selection import actionable_leagues

    from lol_bets.operations.readiness import classification_cohort_evidence

    if "gameid" not in labels or labels["gameid"].isna().any():
        raise ValueError("series readiness requires unique fixture identities")
    actionable = (
        labels["actionable"].fillna(False).astype(bool).to_numpy()
        if "actionable" in labels
        else np.zeros(len(labels), dtype=bool)
    )
    masks = {"actionable_tier1_plus_erls": actionable}
    if "league" in labels:
        leagues = labels["league"].fillna("unknown").astype(str)
        for league in actionable_leagues():
            masks[f"league:{league}"] = actionable & leagues.eq(league).to_numpy()
    evidence = classification_cohort_evidence(
        actual,
        baseline,
        candidate,
        labels["gameid"].astype(str).to_numpy(),
        masks,
    )
    return {name: asdict(value) for name, value in evidence.items()}


def _build_review_readiness(
    root: Path,
    *,
    candidate_id: str,
    reviewed_at: datetime,
    series_report: dict[str, Any],
    blocked: bool,
) -> dict[str, Any]:
    """Attach model readiness to a review without changing promotion health."""
    from oracle_bets_core.league_selection import actionable_leagues

    from lol_bets.operations.readiness import (
        CohortEvidence,
        build_readiness_artifact,
    )

    statuses = _bundle_evidence_status(root)
    if blocked:
        statuses["series_winner"] = "review_required"
    raw_cohorts = series_report.get("readiness_cohorts")
    if not isinstance(raw_cohorts, dict):
        raw_cohorts = {}
    cohorts = {
        str(name): CohortEvidence(**payload)
        for name, payload in raw_cohorts.items()
        if isinstance(payload, dict)
    }
    preregistered = (
        "actionable_tier1_plus_erls",
        *(f"league:{league}" for league in actionable_leagues()),
    )
    return build_readiness_artifact(
        candidate_id=candidate_id,
        reviewed_at=reviewed_at,
        target_statuses=statuses,
        series_cohorts=cohorts,
        preregistered_cohorts=preregistered,
        series_recommendation_failures=tuple(
            str(reason) for reason in series_report.get("readiness_failures", ())
        ),
    )


def _bundle_evidence_status(root: Path) -> dict[str, str]:
    summary_path = root / "_evaluation" / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    statuses = {
        str(model["target"]): str(model["evidence_status"])
        for model in payload["models"]
    }
    for target in ("gamelength", "total_kills", "total_towers"):
        model_name = _EVALUATION_MODELS[target]
        report_path = root / "_evaluation" / model_name / "prop_evaluation_report.json"
        calibrator_path = root / model_name / f"{model_name}_prop_calibrator.pkl"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            baseline = report["constant_baseline"]
            semantics = report["line_semantics"]
            dimensions = set(report["cohort_dimensions"])
            contract_ready = bool(
                baseline.get("fit_split") == "train"
                and isinstance(baseline.get("metrics"), dict)
                and semantics.get("version") == 1
                and semantics.get("supported")
                in {"continuous_threshold", "half_lines_only"}
                and semantics.get("push_model") is False
                and {"league", "map_number"}.issubset(dimensions)
                and calibrator_path.is_file()
            )
        except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
            contract_ready = False
        if not contract_ready:
            statuses[target] = "review_required"
    return statuses


def _candidate_operational_failures(root: Path) -> list[str]:  # noqa: PLR0912
    failures: list[str] = []
    for model_name in _EVALUATION_MODELS.values():
        report_path = root / "_evaluation" / model_name / "split_report.json"
        try:
            checks = json.loads(report_path.read_text(encoding="utf-8"))["checks"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
            failures.append(f"missing_split_integrity:{model_name}")
            continue
        if checks.get("gameids_disjoint") is not True:
            failures.append(f"split_overlap:{model_name}")
        if checks.get("temporal_ordered") is not True:
            failures.append(f"temporal_order_failed:{model_name}")
    for target in sorted(_WINNER_EVALUATION_TARGETS):
        outcome = _EVALUATION_MODELS[target]
        prefix = "" if target == "series_winner" else "next_map_"
        schema_path = root / outcome / f"{outcome}_outcome_matchup_schema.pkl"
        try:
            schema = load_model(schema_path)
        except Exception:
            failures.append(f"{prefix}missing_symmetric_matchup_schema")
        else:
            excluded = set(schema.get("excluded_features") or [])
            if not {"first_pick", "side_win_likelihood"}.issubset(excluded):
                failures.append(f"{prefix}prematch_side_feature_contract_failed")
        try:
            pipeline = load_model(root / outcome / f"{outcome}_feature_pipeline.pkl")
            model = load_model(root / outcome / f"{outcome}.pkl")
            columns = tuple(str(value) for value in pipeline.train_columns)
            rating_columns = tuple(str(value) for value in model.rating_columns)
            lineage = json.loads(
                (root / outcome / f"{outcome}_feature_lineage.json").read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            failures.append(f"{prefix}winner_train_serve_contract_unreadable")
            continue

        from lol_bets.operations.winner_validation import (
            FORBIDDEN_WINNER_FEATURE_FRAGMENTS,
        )
        from lol_bets.prediction_models.winner_model import (
            has_complete_direct_rating_contract,
        )

        if not set(rating_columns).issubset(columns):
            failures.append(f"{prefix}rating_train_serve_parity_failed")
        if not callable(getattr(model, "rating_baseline_probability", None)):
            failures.append(f"{prefix}calibrated_rating_baseline_missing")
        if not has_complete_direct_rating_contract(rating_columns):
            failures.append(f"{prefix}direct_rating_family_missing")
        forbidden = FORBIDDEN_WINNER_FEATURE_FRAGMENTS
        if target == "next_map_winner":
            allowed_state = {
                "current_series",
                "game_in_series",
                "is_deciding_game",
                "maps_completed",
                "next_map_number",
                "series_wins_before",
                "series_losses_before",
                "series_score",
            }
            forbidden = tuple(
                fragment for fragment in forbidden if fragment not in allowed_state
            )
        if any(
            fragment in column.casefold()
            for column in columns
            for fragment in forbidden
        ):
            failures.append(f"{prefix}forbidden_winner_feature")
        lineage_columns = {
            str(item.get("feature")) for item in lineage if isinstance(item, dict)
        }
        valid_availability = {"strictly_before_fixture_start"} | (
            {"after_previous_map_before_target_map"}
            if target == "next_map_winner"
            else set()
        )
        if lineage_columns != set(columns) or any(
            item.get("availability_timestamp") not in valid_availability
            or item.get("model_eligible") is not True
            for item in lineage
            if isinstance(item, dict)
        ):
            failures.append(f"{prefix}winner_feature_lineage_failed")
    return failures


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    payload = json.dumps(
        frame.astype(str).to_dict(orient="records"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_candidate_review(
    registry: ModelRegistry,
    review: CandidateReview,
    *,
    reviewed_at: datetime,
) -> CandidateReview:
    destination = registry.root / "reviews" / f"{review.candidate_id}.json"
    payload = asdict(review)
    payload["policy"] = review.policy.value
    payload["reviewed_at"] = reviewed_at.isoformat()
    _atomic_write_json(destination, payload)
    return review


def _manifest_payload(manifest: CandidateManifest) -> dict[str, Any]:
    payload = asdict(manifest)
    payload["created_at"] = manifest.created_at.isoformat()
    payload["metrics"] = dict(sorted(manifest.metrics.items()))
    return payload


def _loss_array(values: tuple[float, ...] | list[float], *, field: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < _MIN_PAIRED_LOSSES:
        raise ValueError(f"{field} must contain at least {_MIN_PAIRED_LOSSES} values")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{field} must contain finite non-negative values")
    return array


def _finite_nonnegative(value: float, *, field: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be finite and non-negative")


def _require_utc(value: datetime, *, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must be timezone-aware UTC")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _paths_fingerprint(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    if not paths:
        digest.update(b"empty")
    for path in sorted((Path(value) for value in paths), key=str):
        digest.update(str(path).encode())
        if not path.is_file():
            digest.update(b":missing")
            continue
        digest.update(b":")
        digest.update(_sha256(path).encode())
    return digest.hexdigest()


def _stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(_stable_json(payload) + "\n", encoding="utf-8")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_stable_json(payload) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
