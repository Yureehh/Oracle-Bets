from __future__ import annotations

from datetime import UTC, datetime, timedelta

from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.operations.health import (
    HealthLevel,
    evaluate_system_health,
    record_system_health,
    write_system_health_report,
)

NOW = datetime(2026, 7, 26, 12, tzinfo=UTC)
REPEATED_FAILURES = 3


def _store(tmp_path) -> EvidenceStore:
    store = EvidenceStore(tmp_path / "health.db")
    store.initialize_schema()
    return store


def _append_run(store: EvidenceStore, run_id: str = "run") -> None:
    store.append(
        EvidenceTable.RUNS,
        {
            "id": run_id,
            "run_type": "test",
            "started_at": NOW,
            "status": "completed",
            "idempotency_key": run_id,
            "payload_json": {},
        },
    )


def _quality_snapshot(
    store: EvidenceStore,
    *,
    snapshot_id: str,
    generated_at: datetime,
    schema: str,
    accepted_games: int,
) -> None:
    store.append(
        EvidenceTable.SOURCE_SNAPSHOTS,
        {
            "id": snapshot_id,
            "run_id": "run",
            "provider": "oracles_elixir",
            "source_type": "data_quality_report",
            "observed_at": generated_at,
            "source_uri": "fixture://quality",
            "schema_fingerprint": schema,
            "idempotency_key": snapshot_id,
            "payload_json": {
                "generated_at": generated_at,
                "schema_fingerprint": schema,
                "input_rows": 1200,
                "accepted_games": accepted_games,
                "missing_values": {
                    "gameid": 0,
                    "teamid": 0,
                    "teamname": 0,
                    "playername": 0,
                },
            },
        },
    )


def _check(report, name: str):
    return next(check for check in report.checks if check.name == name)


def test_missing_monitoring_evidence_is_visible_not_a_false_pass(tmp_path):
    store = _store(tmp_path)

    report = evaluate_system_health(
        store,
        now=NOW,
        quality_report_path=None,
    )

    assert report.status == "warning"
    assert _check(report, "database_integrity").level is HealthLevel.OK
    assert _check(report, "fixture_freshness").level is HealthLevel.NOT_ENOUGH_DATA
    assert (
        _check(report, "source_quality_evidence").level is HealthLevel.NOT_ENOUGH_DATA
    )


def test_schema_change_and_repeated_provider_failures_are_critical(tmp_path):
    store = _store(tmp_path)
    _append_run(store)
    store.append(
        EvidenceTable.SOURCE_SNAPSHOTS,
        {
            "id": "fixture-source",
            "run_id": "run",
            "provider": "pandascore",
            "source_type": "future_fixtures",
            "observed_at": NOW,
            "source_uri": "fixture://schedule",
            "schema_fingerprint": "schedule-v1",
            "idempotency_key": "fixture-source",
            "payload_json": {},
        },
    )
    _quality_snapshot(
        store,
        snapshot_id="quality-old",
        generated_at=NOW - timedelta(days=1),
        schema="schema-v1",
        accepted_games=100,
    )
    _quality_snapshot(
        store,
        snapshot_id="quality-new",
        generated_at=NOW,
        schema="schema-v2",
        accepted_games=40,
    )
    for index in range(REPEATED_FAILURES):
        store.append(
            EvidenceTable.RUN_EVENTS,
            {
                "id": f"failure-{index}",
                "run_id": "run",
                "event_at": NOW - timedelta(minutes=index),
                "event_type": "provider_read",
                "status": "failed",
                "idempotency_key": f"failure-{index}",
                "payload_json": {
                    "step": "pandascore",
                    "detail": "provider connection timeout",
                },
            },
        )

    report = evaluate_system_health(
        store,
        now=NOW,
        quality_report_path=None,
    )

    assert report.status == "critical"
    assert _check(report, "source_schema").level is HealthLevel.CRITICAL
    assert _check(report, "source_volume").level is HealthLevel.CRITICAL
    assert _check(report, "repeated_network_failures").level is HealthLevel.CRITICAL


def test_health_report_is_recorded_idempotently_and_written_atomically(tmp_path):
    store = _store(tmp_path)
    report = evaluate_system_health(
        store,
        now=NOW,
        quality_report_path=None,
    )

    first = record_system_health(store, report)
    second = record_system_health(store, report)
    destination = write_system_health_report(report, tmp_path / "health.json")

    assert first == second
    assert store.count(EvidenceTable.RUNS) == 1
    assert store.count(EvidenceTable.RUN_EVENTS) == 1
    assert destination.is_file()
    assert '"status": "warning"' in destination.read_text()
