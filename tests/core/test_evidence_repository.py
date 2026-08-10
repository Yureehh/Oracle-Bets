from __future__ import annotations

import csv
import json
import sqlite3
from datetime import UTC, datetime

import pytest
from oracle_bets_core.evidence import (
    EvidenceConflictError,
    EvidenceStore,
    EvidenceTable,
)

NOW = datetime(2026, 7, 26, 8, 15, tzinfo=UTC)
BATCH_SIZE = 3
EVIDENCE_SCHEMA_VERSION = 3


@pytest.fixture
def evidence_store(tmp_path):
    store = EvidenceStore(tmp_path / "oracle_bets.db")
    store.initialize_schema()
    return store


def _run_record(**overrides):
    record = {
        "id": "run-1",
        "run_type": "daily_lol",
        "started_at": NOW,
        "status": "started",
        "idempotency_key": "daily-lol:2026-07-26",
        "payload_json": {"dry_run": False},
    }
    record.update(overrides)
    return record


def test_schema_has_separate_append_only_record_tables(evidence_store):
    expected = {
        "runs",
        "run_events",
        "source_snapshots",
        "identities",
        "provider_links",
        "fixtures",
        "model_versions",
        "predictions",
        "forecasts",
        "market_candidates",
        "market_snapshots",
        "proposals",
        "approvals",
        "paper_positions",
        "settlements",
        "corrections",
    }

    assert expected <= evidence_store.table_names()
    assert evidence_store.schema_version() == EVIDENCE_SCHEMA_VERSION
    assert evidence_store.integrity_check() == "ok"

    with evidence_store.connection(read_only=True) as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1


def test_append_is_idempotent_only_for_identical_content(evidence_store):
    first = evidence_store.append(EvidenceTable.RUNS, _run_record())
    repeated = evidence_store.append(EvidenceTable.RUNS, _run_record())

    assert first == "run-1"
    assert repeated == first
    assert evidence_store.count(EvidenceTable.RUNS) == 1

    with pytest.raises(EvidenceConflictError, match="different content"):
        evidence_store.append(
            EvidenceTable.RUNS,
            _run_record(id="run-2", status="completed"),
        )


def test_prediction_chain_enforces_foreign_keys(evidence_store):
    evidence_store.append(EvidenceTable.RUNS, _run_record())
    evidence_store.append(
        EvidenceTable.IDENTITIES,
        {
            "id": "team-a",
            "entity_type": "team",
            "canonical_name": "Team A",
            "created_at": NOW,
            "idempotency_key": "team:team-a",
            "payload_json": {},
        },
    )
    evidence_store.append(
        EvidenceTable.IDENTITIES,
        {
            "id": "team-b",
            "entity_type": "team",
            "canonical_name": "Team B",
            "created_at": NOW,
            "idempotency_key": "team:team-b",
            "payload_json": {},
        },
    )
    evidence_store.append(
        EvidenceTable.FIXTURES,
        {
            "id": "fixture-1",
            "run_id": "run-1",
            "sport": "lol",
            "competition_id": "LCK",
            "team_a_identity_id": "team-a",
            "team_b_identity_id": "team-b",
            "start_time": NOW,
            "best_of": 3,
            "status": "scheduled",
            "idempotency_key": "fixture:panda:1",
            "payload_json": {},
        },
    )
    evidence_store.append(
        EvidenceTable.MODEL_VERSIONS,
        {
            "id": "model-1",
            "sport": "lol",
            "target": "map_win",
            "created_at": NOW,
            "artifact_uri": "models/lol/candidates/model-1",
            "artifact_checksum": "abc123",
            "idempotency_key": "model:model-1",
            "payload_json": {},
        },
    )
    evidence_store.append(
        EvidenceTable.PREDICTIONS,
        {
            "id": "prediction-1",
            "run_id": "run-1",
            "fixture_id": "fixture-1",
            "model_version_id": "model-1",
            "selection_id": "team-a",
            "mode": "prematch",
            "created_at": NOW,
            "probability_point": "0.61",
            "probability_lower": "0.54",
            "probability_upper": "0.68",
            "warnings_json": [],
            "idempotency_key": "prediction:fixture-1:model-1:team-a",
            "payload_json": {"drivers": ["rating difference"]},
        },
    )

    prediction = evidence_store.get(EvidenceTable.PREDICTIONS, "prediction-1")
    assert prediction["fixture_id"] == "fixture-1"
    assert json.loads(prediction["warnings_json"]) == []

    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        evidence_store.append(
            EvidenceTable.PREDICTIONS,
            {
                "id": "prediction-broken",
                "run_id": "run-1",
                "fixture_id": "missing",
                "model_version_id": "model-1",
                "selection_id": "team-a",
                "mode": "prematch",
                "created_at": NOW,
                "probability_point": "0.50",
                "probability_lower": "0.40",
                "probability_upper": "0.60",
                "warnings_json": [],
                "idempotency_key": "prediction:broken",
                "payload_json": {},
            },
        )


def test_evidence_rows_cannot_be_updated_or_deleted(evidence_store):
    evidence_store.append(EvidenceTable.RUNS, _run_record())

    with (
        pytest.raises(sqlite3.IntegrityError, match="append-only"),
        evidence_store.connection() as conn,
    ):
        conn.execute("UPDATE runs SET status = 'completed' WHERE id = 'run-1'")

    with (
        pytest.raises(sqlite3.IntegrityError, match="append-only"),
        evidence_store.connection() as conn,
    ):
        conn.execute("DELETE FROM runs WHERE id = 'run-1'")


def test_correction_appends_without_mutating_target(evidence_store):
    evidence_store.append(EvidenceTable.RUNS, _run_record())
    evidence_store.append(
        EvidenceTable.CORRECTIONS,
        {
            "id": "correction-1",
            "target_table": "runs",
            "target_id": "run-1",
            "created_at": NOW,
            "reason": "The final run status is recorded as a new fact.",
            "replacement_id": None,
            "idempotency_key": "correction:run-1:final",
            "payload_json": {"status": "completed"},
        },
    )

    assert evidence_store.get(EvidenceTable.RUNS, "run-1")["status"] == "started"
    assert (
        evidence_store.get(EvidenceTable.CORRECTIONS, "correction-1")["target_id"]
        == "run-1"
    )


def test_append_many_is_atomic_and_idempotent(evidence_store):
    records = [
        {
            "id": f"run-{index}",
            "run_type": "test",
            "started_at": NOW,
            "status": "started",
            "idempotency_key": f"run-{index}",
            "payload_json": {"index": index},
        }
        for index in range(BATCH_SIZE)
    ]

    assert evidence_store.append_many(EvidenceTable.RUNS, records) == [
        "run-0",
        "run-1",
        "run-2",
    ]
    assert evidence_store.append_many(EvidenceTable.RUNS, records) == [
        "run-0",
        "run-1",
        "run-2",
    ]
    assert evidence_store.count(EvidenceTable.RUNS) == BATCH_SIZE

    conflicting = [*records, {**records[0], "status": "completed"}]
    with pytest.raises(EvidenceConflictError):
        evidence_store.append_many(EvidenceTable.RUNS, conflicting)
    assert evidence_store.count(EvidenceTable.RUNS) == BATCH_SIZE


def test_cross_table_append_is_atomic(evidence_store):
    records = [
        (EvidenceTable.RUNS, _run_record()),
        (
            EvidenceTable.RUN_EVENTS,
            {
                "id": "event-broken",
                "run_id": "missing-run",
                "event_at": NOW,
                "event_type": "test",
                "status": "failed",
                "idempotency_key": "event-broken",
                "payload_json": {},
            },
        ),
    ]

    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        evidence_store.append_transaction(records)

    assert evidence_store.count(EvidenceTable.RUNS) == 0


def test_dry_run_rolls_back_all_writes(evidence_store):
    dry_store = EvidenceStore(evidence_store.path, dry_run=True)

    assert dry_store.append(EvidenceTable.RUNS, _run_record()) == "run-1"
    assert evidence_store.count(EvidenceTable.RUNS) == 0


def test_read_only_connection_rejects_writes(evidence_store):
    with (
        evidence_store.connection(read_only=True) as conn,
        pytest.raises(sqlite3.OperationalError),
    ):
        conn.execute(
            "INSERT INTO runs "
            "(id, run_type, started_at, status, idempotency_key, payload_json, content_hash) "
            "VALUES ('run-x', 'test', '2026-07-26T08:15:00Z', 'started', "
            "'run-x', '{}', 'hash')"
        )


def test_exports_are_stable_and_do_not_mutate_evidence(evidence_store, tmp_path):
    evidence_store.append(EvidenceTable.RUNS, _run_record())
    json_path = tmp_path / "runs.json"
    csv_path = tmp_path / "runs.csv"

    evidence_store.export_json(EvidenceTable.RUNS, json_path)
    evidence_store.export_csv(EvidenceTable.RUNS, csv_path)

    json_rows = json.loads(json_path.read_text())
    with csv_path.open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert json_rows[0]["id"] == "run-1"
    assert csv_rows[0]["id"] == "run-1"
    assert evidence_store.count(EvidenceTable.RUNS) == 1
