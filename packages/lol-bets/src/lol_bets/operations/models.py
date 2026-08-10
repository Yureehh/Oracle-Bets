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
    from collections.abc import Callable, Collection

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


def should_train_candidate(
    state: TrainingTriggerState,
    *,
    valid_map_threshold: int = 50,
    major_map_threshold: int = 20,
    maximum_age: timedelta = timedelta(days=30),
) -> bool:
    """Return whether any confirmed candidate-training trigger has fired."""
    if valid_map_threshold <= 0 or major_map_threshold <= 0:
        raise ValueError("map thresholds must be positive")
    if maximum_age <= timedelta(0):
        raise ValueError("maximum_age must be positive")
    return (
        state.new_valid_maps >= valid_map_threshold
        or state.new_major_maps >= major_map_threshold
        or state.evaluated_at - state.last_candidate_at >= maximum_age
    )


@dataclass(frozen=True)
class TrainingTriggerEvaluation:
    state: TrainingTriggerState
    triggered: bool
    reasons: tuple[str, ...]
    candidate_id: str


@dataclass(frozen=True)
class CandidateTrainingResult:
    evaluation: TrainingTriggerEvaluation
    trained: bool
    registered_model_id: str | None


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


def orchestrate_candidate_training(
    evaluation: TrainingTriggerEvaluation,
    *,
    train_candidate: Callable[[], None],
    register_candidate: Callable[[str], None],
) -> CandidateTrainingResult:
    """Train and immutably register only when an evidence trigger fired."""
    if not evaluation.triggered:
        return CandidateTrainingResult(evaluation, False, None)
    train_candidate()
    register_candidate(evaluation.candidate_id)
    return CandidateTrainingResult(
        evaluation,
        True,
        evaluation.candidate_id,
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


def evaluate_promotion(  # noqa: PLR0912
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
        if count < minimum_cohort_size or champion_loss == 0:
            continue
        relative_regression = (candidate_loss - champion_loss) / champion_loss
        if relative_regression > maximum_cohort_relative_regression:
            safety_failures.append(f"cohort_regression:{cohort}")

    for target, (champion_mae, candidate_mae) in sorted(
        (evidence.regression_target_mae or {}).items()
    ):
        if champion_mae == 0:
            if candidate_mae > 0:
                safety_failures.append(f"mae_regression:{target}")
            continue
        if (
            candidate_mae - champion_mae
        ) / champion_mae >= _MAX_REGRESSION_MAE_DEGRADATION:
            safety_failures.append(f"mae_regression:{target}")

    status_rank = {
        "below_constant_baseline": 0,
        "review_required": 0,
        "weak_signal": 1,
        "meets_basic_sanity": 2,
    }
    champion_status = evidence.champion_evidence_status or {}
    candidate_status = evidence.candidate_evidence_status or {}
    for target, previous in sorted(champion_status.items()):
        current = candidate_status.get(target)
        if current is None or status_rank.get(current, -1) < status_rank.get(
            previous, -1
        ):
            safety_failures.append(f"evidence_status_downgrade:{target}")

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
    try:
        evidence, fingerprints, report = _replay_promotion_evidence(
            registry,
            candidate_id=candidate_id,
            champion_id=champion_id,
        )
        decision = evaluate_promotion(
            evidence,
            policy=policy,
            bootstrap_samples=bootstrap_samples,
        )
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

    should_promote = (
        automatic and policy is PromotionPolicy.ROUTINE and decision.promote
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
                reasons=decision.reasons,
                row_fingerprints=fingerprints,
                evidence=report
                | {
                    "relative_improvement": decision.relative_improvement,
                    "confidence_lower_bound": decision.confidence_lower_bound,
                    "confidence_degradation_upper_bound": (
                        decision.confidence_degradation_upper_bound
                    ),
                    "safety_failures": list(decision.safety_failures),
                },
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
    elif decision.promote and policy is PromotionPolicy.OPTUNA:
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
            reasons=decision.reasons,
            row_fingerprints=fingerprints,
            evidence=report
            | {
                "relative_improvement": decision.relative_improvement,
                "confidence_lower_bound": decision.confidence_lower_bound,
                "confidence_degradation_upper_bound": (
                    decision.confidence_degradation_upper_bound
                ),
                "safety_failures": list(decision.safety_failures),
            },
        ),
        reviewed_at=reviewed_at,
    )


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
        if not isinstance(files, dict) or not files:
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

    def artifact_path(
        self,
        name: str,
        *,
        model_id: str | None = None,
    ) -> Path:
        """Return one checksum-verified artifact from a candidate bundle."""
        safe_name = self._validate_artifact_name(name)
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
        if safe_name not in files:
            raise ModelRegistryError(
                f"candidate {selected} does not contain artifact: {safe_name}"
            )
        return self.candidates / selected / safe_name

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
        _atomic_write_json(
            self.transition_path,
            {"pointer": pointer, "history": history},
        )
        _atomic_write_json(self.champion_pointer, pointer)
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
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ModelRegistryError("champion transition is malformed") from error
        if not self.verify_bundle(model_id):
            raise ModelRegistryError(
                f"pending champion transition bundle is unhealthy: {model_id}"
            )
        _atomic_write_json(self.champion_pointer, pointer)
        self._append_champion_history(history, transition_id=transition_id)
        self.transition_path.unlink(missing_ok=True)

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
) -> Path:
    """Resolve a verified champion artifact or retain bootstrap legacy paths."""
    registry = ModelRegistry(registry_root)
    champion = registry.champion_id()
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
    "result": "OutcomePrediction_LightGBM",
    "gamelength": "GamelengthPrediction_LightGBM",
    "total_kills": "TotalKillsPrediction_LightGBM",
    "total_towers": "TotalTowersPrediction_LightGBM",
}


def _replay_promotion_evidence(
    registry: ModelRegistry,
    *,
    candidate_id: str,
    champion_id: str,
) -> tuple[PromotionEvidence, dict[str, str], dict[str, Any]]:
    candidate_root = registry.candidates / candidate_id
    champion_root = registry.candidates / champion_id
    operational_failures = _candidate_operational_failures(candidate_root)
    fingerprints: dict[str, str] = {}
    results: dict[str, dict[str, Any]] = {}
    cohort_log_loss: dict[str, tuple[float, float, int]] = {}
    regression_mae: dict[str, tuple[float, float]] = {}
    drift_targets: dict[str, dict[str, Any]] = {}
    outcome_losses: tuple[list[float], list[float]] | None = None
    champion_brier = candidate_brier = 0.0
    champion_ece = candidate_ece = 0.0

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
            classification=target == "result",
        )
        candidate_prediction = _replay_bundle_target(
            candidate_root,
            model_name=model_name,
            raw_features=raw,
            metadata=labels.drop(columns=["actual"]),
            classification=target == "result",
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
        if target == "result":
            champion_quality = prediction_quality(
                actual.astype(int).tolist(), champion_prediction.tolist()
            )
            candidate_quality = prediction_quality(
                actual.astype(int).tolist(), candidate_prediction.tolist()
            )
            champion_losses = _binary_log_losses(actual, champion_prediction)
            candidate_losses = _binary_log_losses(actual, candidate_prediction)
            outcome_losses = (champion_losses, candidate_losses)
            champion_brier = champion_quality.brier
            candidate_brier = candidate_quality.brier
            champion_ece = champion_quality.calibration_error
            candidate_ece = candidate_quality.calibration_error
            cohort_log_loss = _cohort_replay_losses(
                labels,
                actual,
                champion_prediction,
                candidate_prediction,
            )
            results[target] = {
                "rows": len(labels),
                "champion": asdict(champion_quality),
                "candidate": asdict(candidate_quality),
                "cohorts": {
                    name: {
                        "champion_log_loss": values[0],
                        "candidate_log_loss": values[1],
                        "count": values[2],
                    }
                    for name, values in cohort_log_loss.items()
                },
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

    if outcome_losses is None:
        raise ValueError("sealed outcome evaluation is missing")
    champion_status = _bundle_evidence_status(champion_root)
    candidate_status = _bundle_evidence_status(candidate_root)
    evidence = PromotionEvidence(
        champion_log_losses=tuple(outcome_losses[0]),
        candidate_log_losses=tuple(outcome_losses[1]),
        champion_brier=champion_brier,
        candidate_brier=candidate_brier,
        champion_ece=champion_ece,
        candidate_ece=candidate_ece,
        cohort_log_loss=cohort_log_loss,
        regression_target_mae=regression_mae,
        champion_evidence_status=champion_status,
        candidate_evidence_status=candidate_status,
        operational_failures=tuple(operational_failures),
    )
    return (
        evidence,
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


def _cohort_replay_losses(
    labels: pd.DataFrame,
    actual: np.ndarray,
    champion: np.ndarray,
    candidate: np.ndarray,
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
    return {
        name: (
            float(np.mean(champion_losses[mask])),
            float(np.mean(candidate_losses[mask])),
            int(np.sum(mask)),
        )
        for name, mask in cohorts.items()
        if np.any(mask)
    }


def _bundle_evidence_status(root: Path) -> dict[str, str]:
    summary_path = root / "_evaluation" / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        str(model["target"]): str(model["evidence_status"])
        for model in payload["models"]
    }


def _candidate_operational_failures(root: Path) -> list[str]:
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
    outcome = _EVALUATION_MODELS["result"]
    schema_path = root / outcome / f"{outcome}_outcome_matchup_schema.pkl"
    try:
        schema = load_model(schema_path)
    except Exception:
        failures.append("missing_symmetric_matchup_schema")
    else:
        excluded = set(schema.get("excluded_features") or [])
        if not {"first_pick", "side_win_likelihood"}.issubset(excluded):
            failures.append("prematch_side_feature_contract_failed")
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
