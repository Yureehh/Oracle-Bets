"""Unified, evidence-driven operational health evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.evidence.schema import SCHEMA_VERSION

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_MINIMUM_COMPARISON_REPORTS = 2
_EXTREME_PROBABILITY_LOW = 0.001
_EXTREME_PROBABILITY_HIGH = 0.999


class HealthLevel(StrEnum):
    OK = "pass"
    NOT_ENOUGH_DATA = "not_enough_data"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class HealthThresholds:
    fixture_source_warning_hours: int = 26
    fixture_source_critical_hours: int = 48
    quality_source_warning_hours: int = 48
    volume_warning_drop: float = 0.20
    volume_critical_drop: float = 0.50
    missing_warning_fraction: float = 0.05
    missing_critical_fraction: float = 0.20
    extreme_probability_fraction: float = 0.05
    missing_uncertainty_fraction: float = 0.20
    cohort_probability_drift: float = 0.15
    repeated_failure_warning: int = 2
    repeated_failure_critical: int = 3

    def __post_init__(self) -> None:
        if not (
            0 < self.fixture_source_warning_hours < self.fixture_source_critical_hours
        ):
            raise ValueError("fixture freshness thresholds must be increasing")
        for field_name in (
            "volume_warning_drop",
            "volume_critical_drop",
            "missing_warning_fraction",
            "missing_critical_fraction",
            "extreme_probability_fraction",
            "missing_uncertainty_fraction",
            "cohort_probability_drift",
        ):
            value = float(getattr(self, field_name))
            if not 0 < value <= 1:
                raise ValueError(f"{field_name} must be in (0, 1]")


@dataclass(frozen=True)
class HealthCheck:
    name: str
    level: HealthLevel
    summary: str
    measured: Mapping[str, Any]
    next_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level.value,
            "summary": self.summary,
            "measured": dict(self.measured),
            "next_action": self.next_action,
        }


@dataclass(frozen=True)
class SystemHealthReport:
    generated_at: datetime
    checks: tuple[HealthCheck, ...]

    @property
    def status(self) -> str:
        levels = {check.level for check in self.checks}
        if HealthLevel.CRITICAL in levels:
            return "critical"
        if HealthLevel.WARNING in levels or HealthLevel.NOT_ENOUGH_DATA in levels:
            return "warning"
        return "ok"

    @property
    def exit_code(self) -> int:
        return 2 if self.status == "critical" else 1 if self.status == "warning" else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "generated_at": _utc_text(self.generated_at),
            "status": self.status,
            "checks": [check.to_dict() for check in self.checks],
        }


def evaluate_system_health(
    store: EvidenceStore,
    *,
    now: datetime | None = None,
    thresholds: HealthThresholds | None = None,
    quality_report_path: Path | None = None,
) -> SystemHealthReport:
    """Evaluate all confirmed operational signals from immutable evidence."""
    evaluated_at = now or datetime.now(UTC)
    _require_utc(evaluated_at)
    rules = thresholds or HealthThresholds()
    checks: list[HealthCheck] = []
    checks.extend(_database_checks(store))
    if any(check.level is HealthLevel.CRITICAL for check in checks):
        return SystemHealthReport(evaluated_at, tuple(checks))

    snapshots = list(store.list(EvidenceTable.SOURCE_SNAPSHOTS))
    checks.append(_fixture_freshness_check(snapshots, evaluated_at, rules))
    quality_reports = _quality_reports(snapshots, quality_report_path)
    checks.extend(_quality_checks(quality_reports, evaluated_at, rules))
    checks.append(_identity_check(store))
    checks.append(_probability_check(store, rules))
    checks.append(_cohort_drift_check(store, rules))
    checks.append(_market_check(store, evaluated_at))
    checks.append(_network_failure_check(store, evaluated_at, rules))
    return SystemHealthReport(evaluated_at, tuple(checks))


def record_system_health(
    store: EvidenceStore,
    report: SystemHealthReport,
) -> str:
    """Append one system-health report and one structured run event."""
    payload = report.to_dict()
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    run_id = f"run-system-health-{digest}"
    store.append(
        EvidenceTable.RUNS,
        {
            "id": run_id,
            "run_type": "system_health",
            "started_at": report.generated_at,
            "status": report.status,
            "idempotency_key": run_id,
            "payload_json": {"report_digest": digest},
        },
    )
    event_id = f"run-event-system-health-{digest}"
    store.append(
        EvidenceTable.RUN_EVENTS,
        {
            "id": event_id,
            "run_id": run_id,
            "event_at": report.generated_at,
            "event_type": "system_health_evaluation",
            "status": report.status,
            "idempotency_key": event_id,
            "payload_json": payload,
        },
    )
    return run_id


def write_system_health_report(
    report: SystemHealthReport,
    destination: Path,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _database_checks(store: EvidenceStore) -> list[HealthCheck]:
    checks: list[HealthCheck] = []
    try:
        integrity = store.integrity_check()
    except Exception as error:
        return [
            HealthCheck(
                "database_integrity",
                HealthLevel.CRITICAL,
                f"Evidence database cannot be read: {type(error).__name__}",
                {},
                "Restore the latest verified backup or repair database access.",
            )
        ]
    checks.append(
        HealthCheck(
            "database_integrity",
            HealthLevel.OK if integrity == "ok" else HealthLevel.CRITICAL,
            f"SQLite integrity is {integrity}.",
            {"integrity": integrity},
            None
            if integrity == "ok"
            else "Stop decisions and restore a verified backup.",
        )
    )
    try:
        version = store.schema_version()
    except Exception as error:
        version = None
        summary = f"Evidence schema cannot be read: {type(error).__name__}."
    else:
        summary = f"Evidence schema version is {version}."
    checks.append(
        HealthCheck(
            "evidence_schema",
            HealthLevel.OK if version == SCHEMA_VERSION else HealthLevel.CRITICAL,
            summary,
            {"found": version, "expected": SCHEMA_VERSION},
            (
                None
                if version == SCHEMA_VERSION
                else "Stop decisions and run only the reviewed schema migration."
            ),
        )
    )
    return checks


def _fixture_freshness_check(
    snapshots: list[dict[str, Any]],
    now: datetime,
    thresholds: HealthThresholds,
) -> HealthCheck:
    timestamps = [
        _parse_utc(row["observed_at"])
        for row in snapshots
        if row["source_type"] == "future_fixtures"
    ]
    if not timestamps:
        return HealthCheck(
            "fixture_freshness",
            HealthLevel.NOT_ENOUGH_DATA,
            "No canonical fixture source snapshot exists.",
            {"latest_observed_at": None},
            "Run the daily workflow before using predictions.",
        )
    latest = max(timestamps)
    age_hours = max(0.0, (now - latest).total_seconds() / 3600)
    if age_hours >= thresholds.fixture_source_critical_hours:
        level = HealthLevel.CRITICAL
    elif age_hours >= thresholds.fixture_source_warning_hours:
        level = HealthLevel.WARNING
    else:
        level = HealthLevel.OK
    return HealthCheck(
        "fixture_freshness",
        level,
        f"Latest fixture evidence is {age_hours:.1f} hours old.",
        {"latest_observed_at": _utc_text(latest), "age_hours": age_hours},
        (
            None
            if level is HealthLevel.OK
            else "Refresh PandaScore fixtures and lineups before reviewing markets."
        ),
    )


def _quality_reports(
    snapshots: list[dict[str, Any]],
    fallback_path: Path | None,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for row in snapshots:
        if row["source_type"] != "data_quality_report":
            continue
        payload = _json_object(row["payload_json"])
        if payload:
            reports.append(payload)
    if not reports and fallback_path is not None and fallback_path.is_file():
        try:
            payload = json.loads(fallback_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            reports.append(payload)
    return sorted(reports, key=lambda item: str(item.get("generated_at") or ""))


def _quality_checks(
    reports: list[dict[str, Any]],
    now: datetime,
    thresholds: HealthThresholds,
) -> list[HealthCheck]:
    if not reports:
        missing = HealthCheck(
            "source_quality_evidence",
            HealthLevel.NOT_ENOUGH_DATA,
            "No ingestion quality report exists.",
            {},
            "Run ingestion and review its quality report before using predictions.",
        )
        return [missing, missing_for("source_volume"), missing_for("source_schema")]

    latest = reports[-1]
    generated_at = _parse_utc(latest["generated_at"])
    age_hours = max(0.0, (now - generated_at).total_seconds() / 3600)
    freshness_level = (
        HealthLevel.WARNING
        if age_hours >= thresholds.quality_source_warning_hours
        else HealthLevel.OK
    )
    checks = [
        HealthCheck(
            "source_quality_evidence",
            freshness_level,
            f"Latest ingestion quality report is {age_hours:.1f} hours old.",
            {
                "generated_at": _utc_text(generated_at),
                "age_hours": age_hours,
            },
            (
                None
                if freshness_level is HealthLevel.OK
                else "Run incremental ingestion and inspect quarantined rows."
            ),
        )
    ]
    checks.append(_missingness_check(latest, thresholds))
    checks.append(_volume_check(reports, thresholds))
    checks.append(_source_schema_check(reports))
    return checks


def missing_for(name: str) -> HealthCheck:
    return HealthCheck(
        name,
        HealthLevel.NOT_ENOUGH_DATA,
        f"{name.replace('_', ' ').title()} cannot be evaluated.",
        {},
        "Generate at least one current ingestion quality report.",
    )


def _missingness_check(
    report: dict[str, Any],
    thresholds: HealthThresholds,
) -> HealthCheck:
    rows = max(int(report.get("input_rows") or 0), 1)
    missing = report.get("missing_values")
    values = missing if isinstance(missing, dict) else {}
    largest_field, largest_count = max(
        ((str(key), int(value)) for key, value in values.items()),
        key=lambda item: item[1],
        default=("none", 0),
    )
    fraction = largest_count / rows
    hard_identity_missing = max(
        (int(values.get(field, 0)) for field in ("gameid", "teamid", "teamname")),
        default=0,
    )
    if hard_identity_missing or fraction >= thresholds.missing_critical_fraction:
        level = HealthLevel.CRITICAL
    elif fraction >= thresholds.missing_warning_fraction:
        level = HealthLevel.WARNING
    else:
        level = HealthLevel.OK
    return HealthCheck(
        "source_missingness",
        level,
        f"Largest required-field missing fraction is {fraction:.1%} ({largest_field}).",
        {
            "largest_field": largest_field,
            "largest_missing_count": largest_count,
            "largest_missing_fraction": fraction,
            "identity_missing_count": hard_identity_missing,
        },
        (
            None
            if level is HealthLevel.OK
            else "Review source nulls and quarantine affected games before training."
        ),
    )


def _volume_check(
    reports: list[dict[str, Any]],
    thresholds: HealthThresholds,
) -> HealthCheck:
    latest = int(reports[-1].get("accepted_games") or 0)
    if len(reports) < _MINIMUM_COMPARISON_REPORTS:
        return HealthCheck(
            "source_volume",
            HealthLevel.NOT_ENOUGH_DATA,
            f"Accepted game volume is {latest}; no earlier report is available.",
            {"accepted_games": latest, "previous_accepted_games": None},
            "Keep the next report so volume change can be measured.",
        )
    previous = int(reports[-2].get("accepted_games") or 0)
    drop = max(0.0, (previous - latest) / previous) if previous else 0.0
    if drop >= thresholds.volume_critical_drop:
        level = HealthLevel.CRITICAL
    elif drop >= thresholds.volume_warning_drop:
        level = HealthLevel.WARNING
    else:
        level = HealthLevel.OK
    return HealthCheck(
        "source_volume",
        level,
        f"Accepted games changed from {previous} to {latest} ({drop:.1%} drop).",
        {
            "accepted_games": latest,
            "previous_accepted_games": previous,
            "drop_fraction": drop,
        },
        (
            None
            if level is HealthLevel.OK
            else "Inspect the history manifest, upstream coverage, and quarantine reasons."
        ),
    )


def _source_schema_check(reports: list[dict[str, Any]]) -> HealthCheck:
    latest = str(reports[-1].get("schema_fingerprint") or "")
    previous = (
        str(reports[-2].get("schema_fingerprint") or "")
        if len(reports) >= _MINIMUM_COMPARISON_REPORTS
        else None
    )
    changed = previous is not None and latest != previous
    return HealthCheck(
        "source_schema",
        HealthLevel.CRITICAL if changed or not latest else HealthLevel.OK,
        (
            "Upstream schema fingerprint changed."
            if changed
            else "Upstream schema fingerprint is stable."
        ),
        {"latest": latest or None, "previous": previous, "changed": changed},
        (
            "Stop ingestion consumers and review the provider schema change."
            if changed or not latest
            else None
        ),
    )


def _identity_check(store: EvidenceStore) -> HealthCheck:
    with store.connection(read_only=True) as connection:
        conflicts = connection.execute(
            """
            SELECT provider, provider_entity_id,
                   COUNT(DISTINCT identity_id) AS identity_count
            FROM provider_links
            WHERE valid_to IS NULL
            GROUP BY provider, provider_entity_id
            HAVING COUNT(DISTINCT identity_id) > 1
            """
        ).fetchall()
        missing_links = int(
            connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM fixtures fixture
                WHERE NOT EXISTS (
                    SELECT 1 FROM provider_links link
                    WHERE link.identity_id = fixture.team_a_identity_id
                )
                   OR NOT EXISTS (
                    SELECT 1 FROM provider_links link
                    WHERE link.identity_id = fixture.team_b_identity_id
                )
                """
            ).fetchone()["count"]
        )
    level = (
        HealthLevel.CRITICAL
        if conflicts
        else (HealthLevel.WARNING if missing_links else HealthLevel.OK)
    )
    return HealthCheck(
        "identity_resolution",
        level,
        (
            f"{len(conflicts)} active provider-ID conflicts; "
            f"{missing_links} fixtures have no provider-linked team."
        ),
        {
            "active_provider_conflicts": len(conflicts),
            "fixtures_without_provider_team_link": missing_links,
        },
        (
            None
            if level is HealthLevel.OK
            else "Resolve provider identity links before using affected fixtures."
        ),
    )


def _probability_check(
    store: EvidenceStore,
    thresholds: HealthThresholds,
) -> HealthCheck:
    predictions = list(store.list(EvidenceTable.PREDICTIONS))
    if not predictions:
        return missing_for("model_probability_health")
    extreme = 0
    missing_uncertainty = 0
    invalid = 0
    for row in predictions:
        try:
            point = float(row["probability_point"])
            lower = float(row["probability_lower"])
            upper = float(row["probability_upper"])
        except (TypeError, ValueError):
            invalid += 1
            continue
        if not all(math.isfinite(value) for value in (point, lower, upper)):
            invalid += 1
        if point <= _EXTREME_PROBABILITY_LOW or point >= _EXTREME_PROBABILITY_HIGH:
            extreme += 1
        warnings = _json_list(row["warnings_json"])
        if lower == point == upper or "uncertainty_interval_unavailable" in warnings:
            missing_uncertainty += 1
    total = len(predictions)
    extreme_fraction = extreme / total
    missing_fraction = missing_uncertainty / total
    if invalid:
        level = HealthLevel.CRITICAL
    elif (
        extreme_fraction >= thresholds.extreme_probability_fraction
        or missing_fraction >= thresholds.missing_uncertainty_fraction
    ):
        level = HealthLevel.WARNING
    else:
        level = HealthLevel.OK
    return HealthCheck(
        "model_probability_health",
        level,
        (
            f"{invalid} invalid, {extreme_fraction:.1%} extreme, "
            f"{missing_fraction:.1%} without useful uncertainty."
        ),
        {
            "prediction_count": total,
            "invalid_count": invalid,
            "extreme_fraction": extreme_fraction,
            "missing_uncertainty_fraction": missing_fraction,
        },
        (
            None
            if level is HealthLevel.OK
            else "Review calibration, uncertainty artifacts, and the affected model."
        ),
    )


def _cohort_drift_check(
    store: EvidenceStore,
    thresholds: HealthThresholds,
) -> HealthCheck:
    rows = _resolved_prediction_rows(store)
    by_league: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for row in rows:
        by_league[str(row["league"])].append(
            (_parse_utc(row["observed_at"]), float(row["probability_point"]))
        )
    drift: dict[str, float] = {}
    minimum_window = 5
    for league, observations in by_league.items():
        ordered = sorted(observations)
        if len(ordered) < minimum_window * 2:
            continue
        recent = [value for _, value in ordered[-minimum_window:]]
        prior = [value for _, value in ordered[-minimum_window * 2 : -minimum_window]]
        drift[league] = abs(sum(recent) / len(recent) - sum(prior) / len(prior))
    if not drift:
        return HealthCheck(
            "cohort_probability_drift",
            HealthLevel.NOT_ENOUGH_DATA,
            "No league has two resolved five-prediction windows.",
            {"eligible_leagues": 0},
            "Continue recording outcomes for every shadow prediction.",
        )
    maximum_league = max(drift, key=drift.__getitem__)
    maximum = drift[maximum_league]
    level = (
        HealthLevel.WARNING
        if maximum >= thresholds.cohort_probability_drift
        else HealthLevel.OK
    )
    return HealthCheck(
        "cohort_probability_drift",
        level,
        f"Largest recent probability shift is {maximum:.1%} in {maximum_league}.",
        {"by_league": dict(sorted(drift.items())), "maximum": maximum},
        (
            None
            if level is HealthLevel.OK
            else "Review league mix, roster changes, patches, and calibration."
        ),
    )


def _resolved_prediction_rows(store: EvidenceStore) -> list[dict[str, Any]]:
    with store.connection(read_only=True) as connection:
        rows = connection.execute(
            """
            SELECT prediction.probability_point,
                   fixture.competition_id AS league,
                   outcome.observed_at
            FROM predictions prediction
            JOIN fixtures fixture ON fixture.id = prediction.fixture_id
            JOIN source_snapshots outcome
              ON json_extract(outcome.payload_json, '$.prediction_id') = prediction.id
            WHERE outcome.source_type = 'prediction_outcome'
              AND NOT EXISTS (
                  SELECT 1 FROM corrections correction
                  WHERE correction.target_table = 'source_snapshots'
                    AND correction.target_id = outcome.id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM corrections correction
                  WHERE correction.target_table = 'predictions'
                    AND correction.target_id = prediction.id
              )
            """
        ).fetchall()
    return [dict(row) for row in rows]


def _market_check(store: EvidenceStore, now: datetime) -> HealthCheck:
    cutoff = now - timedelta(hours=24)
    predictions = [
        row
        for row in store.list(EvidenceTable.PREDICTIONS)
        if _parse_utc(row["created_at"]) >= cutoff
    ]
    candidates = [
        row
        for row in store.list(EvidenceTable.MARKET_CANDIDATES)
        if _parse_utc(row["discovered_at"]) >= cutoff
    ]
    snapshots = {
        str(row["market_candidate_id"])
        for row in store.list(EvidenceTable.MARKET_SNAPSHOTS)
        if _parse_utc(row["observed_at"]) >= cutoff
    }
    accepted = [
        row
        for row in candidates
        if str(row["match_status"]) == "typed_exact"
        or str(row["match_status"]).startswith("accepted")
    ]
    level = (
        HealthLevel.WARNING
        if predictions and (not accepted or not snapshots)
        else HealthLevel.OK
    )
    return HealthCheck(
        "market_observation",
        level,
        (
            f"Last 24h: {len(predictions)} predictions, {len(candidates)} candidates, "
            f"{len(accepted)} exact matches, {len(snapshots)} candidates with books."
        ),
        {
            "predictions_24h": len(predictions),
            "candidates_24h": len(candidates),
            "exact_matches_24h": len(accepted),
            "candidates_with_books_24h": len(snapshots),
        },
        (
            None
            if level is HealthLevel.OK
            else "Check Gamma/CLOB availability and exact market matching."
        ),
    )


def _network_failure_check(
    store: EvidenceStore,
    now: datetime,
    thresholds: HealthThresholds,
) -> HealthCheck:
    cutoff = now - timedelta(hours=24)
    failures: Counter[str] = Counter()
    for row in store.list(EvidenceTable.RUN_EVENTS):
        if _parse_utc(row["event_at"]) < cutoff or row["status"] != "failed":
            continue
        payload = _json_object(row["payload_json"])
        step = str(payload.get("step") or row["event_type"])
        detail = str(payload.get("detail") or "").casefold()
        if any(
            token in detail
            for token in ("network", "timeout", "connection", "provider", "http")
        ) or "provider" in str(row["event_type"]):
            failures[step] += 1
    maximum = max(failures.values(), default=0)
    if maximum >= thresholds.repeated_failure_critical:
        level = HealthLevel.CRITICAL
    elif maximum >= thresholds.repeated_failure_warning:
        level = HealthLevel.WARNING
    else:
        level = HealthLevel.OK
    return HealthCheck(
        "repeated_network_failures",
        level,
        f"Highest repeated provider/network failure count in 24h is {maximum}.",
        {"failures_by_step": dict(sorted(failures.items())), "maximum": maximum},
        (
            None
            if level is HealthLevel.OK
            else "Check provider status, credentials, network access, and retry logs."
        ),
    )


def _json_object(value: Any) -> dict[str, Any]:
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _json_list(value: Any) -> list[Any]:
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        return []
    return decoded if isinstance(decoded, list) else []


def _parse_utc(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    return parsed.astimezone(UTC)


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("health evaluation time must be timezone-aware UTC")
