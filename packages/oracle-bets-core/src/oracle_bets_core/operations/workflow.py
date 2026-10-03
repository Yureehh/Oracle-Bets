"""Resumable named-step execution and evidence-backed run journaling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

from oracle_bets_core.evidence import EvidenceStore, EvidenceTable

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class WorkflowStep:
    """One ordered operation with an explicit mutation classification."""

    name: str
    action: Callable[[], str]
    writes: bool
    retryable: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("workflow step name cannot be empty")


@dataclass(frozen=True)
class StepOutcome:
    name: str
    status: str
    detail: str
    attempts: int

    @property
    def ok(self) -> bool:
        return self.status in {"completed", "skipped_completed", "skipped_dry_run"}


@dataclass(frozen=True)
class WorkflowOutcome:
    run_key: str
    steps: tuple[StepOutcome, ...]
    dry_run: bool

    @property
    def ok(self) -> bool:
        return all(step.ok for step in self.steps)


class WorkflowJournal(Protocol):
    """Minimal durable journal used for duplicate and resume protection."""

    def completed_steps(self, run_key: str) -> set[str]:
        """Return successful step names already recorded for this run."""

    def record(
        self,
        run_key: str,
        *,
        step: str,
        status: str,
        detail: str,
        attempt: int,
        occurred_at: datetime,
    ) -> None:
        """Append one step attempt."""


class EvidenceWorkflowJournal:
    """Append workflow step attempts into the canonical evidence database."""

    def __init__(
        self,
        store: EvidenceStore,
        *,
        run_type: str,
        started_at: datetime,
    ) -> None:
        _require_utc(started_at)
        self.store = store
        self.run_type = run_type
        self.started_at = started_at

    def ensure_run(self, run_key: str) -> str:
        run_id = _run_id(run_key)
        existing = self.store.get(EvidenceTable.RUNS, run_id)
        if existing is not None:
            if existing["run_type"] != self.run_type:
                raise ValueError(
                    f"run key already belongs to run type {existing['run_type']}"
                )
            return run_id
        self.store.append(
            EvidenceTable.RUNS,
            {
                "id": run_id,
                "run_type": self.run_type,
                "started_at": self.started_at,
                "status": "started",
                "idempotency_key": run_key,
                "payload_json": {"run_key": run_key},
            },
        )
        return run_id

    def completed_steps(self, run_key: str) -> set[str]:
        run_id = self.ensure_run(run_key)
        completed: set[str] = set()
        for row in self.store.list(EvidenceTable.RUN_EVENTS):
            if row["run_id"] != run_id or row["status"] != "completed":
                continue
            payload = json.loads(row["payload_json"])
            completed.add(str(payload["step"]))
        return completed

    def record(
        self,
        run_key: str,
        *,
        step: str,
        status: str,
        detail: str,
        attempt: int,
        occurred_at: datetime,
    ) -> None:
        _require_utc(occurred_at)
        run_id = self.ensure_run(run_key)
        identity = f"{run_key}|{step}|{status}|{attempt}|{occurred_at.isoformat()}"
        event_id = f"run-event-{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
        self.store.append(
            EvidenceTable.RUN_EVENTS,
            {
                "id": event_id,
                "run_id": run_id,
                "event_at": occurred_at,
                "event_type": "workflow_step",
                "status": status,
                "idempotency_key": identity,
                "payload_json": {
                    "step": step,
                    "detail": detail,
                    "attempt": attempt,
                },
            },
        )


def run_workflow(
    run_key: str,
    steps: tuple[WorkflowStep, ...] | list[WorkflowStep],
    *,
    journal: WorkflowJournal | None,
    dry_run: bool,
    retry_attempts: int = 3,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> WorkflowOutcome:
    """Execute, resume, retry, and stop an ordered named-step workflow."""
    if not run_key.strip():
        raise ValueError("run_key cannot be empty")
    if retry_attempts <= 0:
        raise ValueError("retry_attempts must be positive")
    names = [step.name for step in steps]
    if len(names) != len(set(names)):
        raise ValueError("workflow step names must be unique")
    completed = journal.completed_steps(run_key) if journal and not dry_run else set()
    outcomes: list[StepOutcome] = []
    blocked = False

    for step in steps:
        if blocked:
            outcomes.append(
                StepOutcome(step.name, "skipped_after_failure", "blocked", 0)
            )
            continue
        if step.name in completed and step.writes:
            outcomes.append(
                StepOutcome(
                    step.name,
                    "skipped_completed",
                    "already completed for run key",
                    0,
                )
            )
            continue
        if dry_run and step.writes:
            outcomes.append(
                StepOutcome(
                    step.name,
                    "skipped_dry_run",
                    "write suppressed by dry-run policy",
                    0,
                )
            )
            continue

        allowed_attempts = retry_attempts if step.retryable else 1
        for attempt in range(1, allowed_attempts + 1):
            try:
                detail = step.action()
            except Exception as error:
                detail = _sanitized_error(error)
                if journal and not dry_run:
                    journal.record(
                        run_key,
                        step=step.name,
                        status="failed",
                        detail=detail,
                        attempt=attempt,
                        occurred_at=clock(),
                    )
                if attempt == allowed_attempts:
                    outcomes.append(StepOutcome(step.name, "failed", detail, attempt))
                    blocked = True
                continue

            outcomes.append(StepOutcome(step.name, "completed", str(detail), attempt))
            if journal and not dry_run:
                journal.record(
                    run_key,
                    step=step.name,
                    status="completed",
                    detail=str(detail),
                    attempt=attempt,
                    occurred_at=clock(),
                )
            break

    return WorkflowOutcome(run_key=run_key, steps=tuple(outcomes), dry_run=dry_run)


def daily_run_key(
    sport: str,
    *,
    scheduled_for: datetime,
    config: dict[str, Any],
) -> str:
    """Build a stable key for one scheduled UTC day and effective config."""
    _require_utc(scheduled_for)
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"daily:{sport}:{scheduled_for.date().isoformat()}:{fingerprint}"


def _run_id(run_key: str) -> str:
    return f"run-{hashlib.sha256(run_key.encode()).hexdigest()[:24]}"


def _sanitized_error(error: Exception) -> str:
    text = " ".join(str(error).split())
    return f"{type(error).__name__}: {text[:300]}"


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("workflow timestamps must be timezone-aware UTC")
