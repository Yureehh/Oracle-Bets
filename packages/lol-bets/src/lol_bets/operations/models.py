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
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import numpy as np
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

_MIN_PROMOTION_IMPROVEMENT = 0.01
_MAX_COHORT_RELATIVE_REGRESSION = 0.02
_MIN_COHORT_SIZE = 30
_MIN_BOOTSTRAP_SAMPLES = 100
_MIN_PAIRED_LOSSES = 2


class ModelRegistryError(ValueError):
    """Raised when a model registry operation would violate its audit rules."""


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
    sample_indices = generator.integers(
        0,
        champion.size,
        size=(samples, champion.size),
    )
    # The fixed original champion mean keeps the interval paired while avoiding
    # a degenerate ratio when every candidate loss is an exact scalar multiple.
    bootstrapped = np.mean(paired_deltas[sample_indices], axis=1) / champion_mean
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

    def __post_init__(self) -> None:
        _loss_array(self.champion_log_losses, field="champion_log_losses")
        _loss_array(self.candidate_log_losses, field="candidate_log_losses")
        _finite_nonnegative(self.champion_brier, field="champion_brier")
        _finite_nonnegative(self.candidate_brier, field="candidate_brier")
        for name, (champion, candidate, count) in self.cohort_log_loss.items():
            if not name.strip():
                raise ValueError("cohort names cannot be empty")
            _finite_nonnegative(champion, field=f"cohort {name} champion loss")
            _finite_nonnegative(candidate, field=f"cohort {name} candidate loss")
            if count < 0:
                raise ValueError(f"cohort {name} count cannot be negative")


@dataclass(frozen=True)
class PromotionDecision:
    """Machine-readable result of the candidate promotion gate."""

    promote: bool
    reasons: tuple[str, ...]
    relative_improvement: float
    confidence_lower_bound: float
    safety_failures: tuple[str, ...]


def evaluate_promotion(
    evidence: PromotionEvidence,
    *,
    minimum_relative_improvement: float = _MIN_PROMOTION_IMPROVEMENT,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    seed: int = 7,
    maximum_cohort_relative_regression: float = (_MAX_COHORT_RELATIVE_REGRESSION),
    minimum_cohort_size: int = _MIN_COHORT_SIZE,
) -> PromotionDecision:
    """Apply the confirmed probability-quality and cohort safety gates."""
    if minimum_relative_improvement < 0:
        raise ValueError("minimum_relative_improvement cannot be negative")
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

    if comparison.relative_improvement < minimum_relative_improvement:
        reasons.append("relative_improvement_below_1_percent")
    if comparison.lower_bound <= 0:
        reasons.append("confidence_interval_not_positive")
    if evidence.candidate_brier > evidence.champion_brier:
        safety_failures.append("brier_regression")

    for cohort in sorted(evidence.cohort_log_loss):
        champion_loss, candidate_loss, count = evidence.cohort_log_loss[cohort]
        if count < minimum_cohort_size or champion_loss == 0:
            continue
        relative_regression = (candidate_loss - champion_loss) / champion_loss
        if relative_regression > maximum_cohort_relative_regression:
            safety_failures.append(f"cohort_regression:{cohort}")

    reasons.extend(safety_failures)
    return PromotionDecision(
        promote=not reasons,
        reasons=tuple(reasons),
        relative_improvement=comparison.relative_improvement,
        confidence_lower_bound=comparison.lower_bound,
        safety_failures=tuple(safety_failures),
    )


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
        self.candidates.mkdir(parents=True, exist_ok=True)

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

        _atomic_write_json(self.champion_pointer, pointer)
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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
