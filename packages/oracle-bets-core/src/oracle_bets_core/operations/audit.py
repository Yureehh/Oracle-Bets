"""Light monthly owner audit built entirely from canonical evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.operations.health import (
    SystemHealthReport,
    evaluate_system_health,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from oracle_bets_core.config import ProductConfig

_LAST_MONTH = 12

_PERIOD_TIMESTAMP_COLUMNS: Mapping[EvidenceTable, str] = {
    EvidenceTable.RUNS: "started_at",
    EvidenceTable.RUN_EVENTS: "event_at",
    EvidenceTable.SOURCE_SNAPSHOTS: "observed_at",
    EvidenceTable.FIXTURES: "start_time",
    EvidenceTable.MODEL_VERSIONS: "created_at",
    EvidenceTable.PREDICTIONS: "created_at",
    EvidenceTable.MARKET_CANDIDATES: "discovered_at",
    EvidenceTable.MARKET_SNAPSHOTS: "observed_at",
    EvidenceTable.PROPOSALS: "created_at",
    EvidenceTable.APPROVALS: "created_at",
    EvidenceTable.PAPER_POSITIONS: "opened_at",
    EvidenceTable.SETTLEMENTS: "settled_at",
    EvidenceTable.CORRECTIONS: "created_at",
}


@dataclass(frozen=True)
class MonthlyAuditReport:
    period: str
    generated_at: datetime
    status: str
    product_config_hash: str
    product_config_changed: bool | None
    evidence_counts: Mapping[str, int]
    period_activity: Mapping[str, int]
    health: SystemHealthReport
    prediction_coverage: Mapping[str, int | float | None]
    accepted_risks: tuple[str, ...]
    review_checklist: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "period": self.period,
            "generated_at": _utc_text(self.generated_at),
            "status": self.status,
            "configuration": {
                "product_config_hash": self.product_config_hash,
                "changed_since_previous_monthly_audit": self.product_config_changed,
            },
            "evidence_counts": dict(self.evidence_counts),
            "period_activity": dict(self.period_activity),
            "health": self.health.to_dict(),
            "prediction_coverage": dict(self.prediction_coverage),
            "accepted_risks": list(self.accepted_risks),
            "review_checklist": list(self.review_checklist),
            "owner_signoff_required": True,
        }


def build_monthly_audit(
    store: EvidenceStore,
    *,
    period: str,
    config: ProductConfig,
    generated_at: datetime | None = None,
) -> MonthlyAuditReport:
    """Create one precise monthly review; never claim owner sign-off."""
    start, end = _period_bounds(period)
    now = generated_at or datetime.now(UTC)
    _require_utc(now)
    config_payload = config.to_dict()
    config_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    previous_hash = _latest_monthly_config_hash(store)
    health = evaluate_system_health(store, now=now)
    coverage = _prediction_coverage(store)
    status = "critical" if health.status == "critical" else "owner_review_required"
    return MonthlyAuditReport(
        period=period,
        generated_at=now,
        status=status,
        product_config_hash=config_hash,
        product_config_changed=(
            None if previous_hash is None else previous_hash != config_hash
        ),
        evidence_counts={table.value: store.count(table) for table in EvidenceTable},
        period_activity=_period_activity(store, start=start, end=end),
        health=health,
        prediction_coverage=coverage,
        accepted_risks=_accepted_risks(config),
        review_checklist=(
            "Review every warning and critical health check.",
            "Review source volume, missingness, schema, quarantine, and identity conflicts.",
            "Compare shadow prediction quality by league, market, strategy, model, mode, and edge band.",
            "Review paper ROI, CLV, calibration, drawdown, and uncertainty; do not use point ROI alone.",
            "Review all corrections, failed runs, stale proposals, and provider failures.",
            "Review new model candidates; promotion remains a separate owner-controlled decision.",
            "Create and verify a backup, then copy it to owner-controlled encrypted storage.",
            "Record owner comments and any approved rule change as a new versioned decision.",
        ),
    )


def record_monthly_audit(
    store: EvidenceStore,
    report: MonthlyAuditReport,
) -> str:
    """Append the monthly report as a run, source snapshot, and run event."""
    payload = report.to_dict()
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    run_id = f"run-monthly-audit-{digest}"
    store.append(
        EvidenceTable.RUNS,
        {
            "id": run_id,
            "run_type": "monthly_audit",
            "started_at": report.generated_at,
            "status": report.status,
            "idempotency_key": run_id,
            "payload_json": {
                "period": report.period,
                "product_config_hash": report.product_config_hash,
            },
        },
    )
    snapshot_id = f"source-monthly-audit-{digest}"
    store.append(
        EvidenceTable.SOURCE_SNAPSHOTS,
        {
            "id": snapshot_id,
            "run_id": run_id,
            "provider": "oracle_bets",
            "source_type": "monthly_audit",
            "observed_at": report.generated_at,
            "source_uri": f"oracle-bets://monthly-audit/{report.period}",
            "schema_fingerprint": "oracle-bets:monthly-audit:v1",
            "idempotency_key": snapshot_id,
            "payload_json": payload,
        },
    )
    event_id = f"run-event-monthly-audit-{digest}"
    store.append(
        EvidenceTable.RUN_EVENTS,
        {
            "id": event_id,
            "run_id": run_id,
            "event_at": report.generated_at,
            "event_type": "monthly_audit_generated",
            "status": report.status,
            "idempotency_key": event_id,
            "payload_json": {
                "period": report.period,
                "source_snapshot_id": snapshot_id,
                "owner_signoff_required": True,
            },
        },
    )
    return run_id


def write_monthly_audit(
    report: MonthlyAuditReport,
    *,
    json_path: Path,
    markdown_path: Path,
) -> tuple[Path, Path]:
    """Write stable owner-readable JSON and Markdown report artifacts."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        json_path,
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
    )
    health_lines = "\n".join(
        f"- `{check.level.value}` — **{check.name}**: {check.summary}"
        for check in report.health.checks
    )
    activity_lines = "\n".join(
        f"- {name}: {count}" for name, count in report.period_activity.items()
    )
    checklist = "\n".join(f"- [ ] {item}" for item in report.review_checklist)
    risks = "\n".join(f"- {item}" for item in report.accepted_risks)
    markdown = f"""# Oracle Bets monthly audit — {report.period}

Generated: {_utc_text(report.generated_at)}

Status: **{report.status}**. This report is not owner sign-off.

## Configuration

- Product config SHA-256: `{report.product_config_hash}`
- Changed since previous audit: `{report.product_config_changed}`

## Period activity

{activity_lines}

## System health

{health_lines}

## Prediction coverage

- Tracked predictions: {report.prediction_coverage["tracked_predictions"]}
- Resolved predictions: {report.prediction_coverage["resolved_predictions"]}
- Resolution coverage: {report.prediction_coverage["resolution_fraction"]}

## Accepted risks still in force

{risks}

## Required owner review

{checklist}
"""
    _atomic_write(markdown_path, markdown)
    return json_path, markdown_path


def _period_bounds(period: str) -> tuple[datetime, datetime]:
    try:
        start = datetime.strptime(period, "%Y-%m").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError("period must use YYYY-MM") from error
    if start.month == _LAST_MONTH:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start, end


def _period_activity(
    store: EvidenceStore,
    *,
    start: datetime,
    end: datetime,
) -> dict[str, int]:
    activity: dict[str, int] = {}
    with store.connection(read_only=True) as connection:
        for table, timestamp_column in _PERIOD_TIMESTAMP_COLUMNS.items():
            row = connection.execute(
                f"SELECT COUNT(*) AS count FROM {table.value} "  # noqa: S608
                f"WHERE {timestamp_column} >= ? AND {timestamp_column} < ?",
                (_utc_text(start), _utc_text(end)),
            ).fetchone()
            activity[table.value] = int(row["count"])
    return activity


def _prediction_coverage(store: EvidenceStore) -> dict[str, int | float | None]:
    with store.connection(read_only=True) as connection:
        tracked = int(
            connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM predictions prediction
                WHERE NOT EXISTS (
                    SELECT 1 FROM corrections correction
                    WHERE correction.target_table = 'predictions'
                      AND correction.target_id = prediction.id
                )
                """
            ).fetchone()["count"]
        )
        resolved = int(
            connection.execute(
                """
                SELECT COUNT(DISTINCT prediction.id) AS count
                FROM predictions prediction
                JOIN source_snapshots outcome
                  ON outcome.source_type = 'prediction_outcome'
                 AND json_extract(
                     outcome.payload_json,
                     '$.prediction_id'
                 ) = prediction.id
                WHERE NOT EXISTS (
                    SELECT 1 FROM corrections correction
                    WHERE correction.target_table = 'predictions'
                      AND correction.target_id = prediction.id
                )
                  AND NOT EXISTS (
                    SELECT 1 FROM corrections correction
                    WHERE correction.target_table = 'source_snapshots'
                      AND correction.target_id = outcome.id
                )
                """
            ).fetchone()["count"]
        )
    return {
        "tracked_predictions": tracked,
        "resolved_predictions": resolved,
        "resolution_fraction": resolved / tracked if tracked else None,
    }


def _latest_monthly_config_hash(store: EvidenceStore) -> str | None:
    for row in reversed(store.list(EvidenceTable.SOURCE_SNAPSHOTS)):
        if row["source_type"] != "monthly_audit":
            continue
        try:
            payload = json.loads(str(row["payload_json"]))
            value = payload["configuration"]["product_config_hash"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if isinstance(value, str) and value:
            return value
    return None


def _accepted_risks(config: ProductConfig) -> tuple[str, ...]:
    risks: list[str] = []
    if config.market.read_only:
        risks.append(
            "Market comparison is read-only; no wallet, signer, order, or fund-movement path exists."
        )
    if not config.promotion.automatic:
        risks.append(
            "Model promotion requires an explicit owner review after temporal calibration and profit evidence."
        )
    return tuple(risks)


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("monthly audit generated_at must be timezone-aware UTC")
