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
from oracle_bets_core.evidence.schema import SCHEMA_SQL

NOW = datetime(2026, 7, 26, 8, 15, tzinfo=UTC)
BATCH_SIZE = 3
EVIDENCE_SCHEMA_VERSION = 7
LEGACY_SCHEMA_VERSION = 6


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
        "bets",
        "bet_events",
        "corrections",
    }

    assert expected <= evidence_store.table_names()
    assert evidence_store.schema_version() == EVIDENCE_SCHEMA_VERSION
    assert evidence_store.integrity_check() == "ok"

    with evidence_store.connection(read_only=True) as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1


def test_schema_upgrade_records_auditable_transactional_migration(tmp_path):
    path = tmp_path / "migration.db"
    legacy_schema = SCHEMA_SQL.replace(
        "'settlements', 'bets', 'bet_events', 'corrections'",
        "'settlements', 'corrections'",
    )
    with sqlite3.connect(path) as conn:
        conn.executescript(legacy_schema)
        conn.execute("DROP TABLE bet_events")
        conn.execute("DROP TABLE bets")
        conn.execute(
            "INSERT INTO evidence_schema_version VALUES (4, '2026-01-01T00:00:00Z')"
        )

    store = EvidenceStore(path)
    store.initialize_schema()

    assert store.schema_version() == EVIDENCE_SCHEMA_VERSION
    with store.connection(read_only=True) as conn:
        rows = conn.execute(
            "SELECT from_version, to_version FROM evidence_schema_migrations "
            "ORDER BY from_version"
        ).fetchall()
    assert [tuple(row) for row in rows] == [(4, 5), (5, 6), (6, 7)]
    assert {"bets", "bet_events"} <= store.table_names()
    store.append(
        EvidenceTable.CORRECTIONS,
        {
            "id": "bet-correction",
            "target_table": "bets",
            "target_id": "bet-1",
            "created_at": NOW,
            "reason": "Migration supports unified-ledger corrections.",
            "replacement_id": None,
            "idempotency_key": "bet-correction",
            "payload_json": {},
        },
    )


def test_schema_upgrade_creates_tables_missing_from_older_additive_schema(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE evidence_schema_version "
            "(version INTEGER NOT NULL, installed_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO evidence_schema_version VALUES (1, '2026-01-01T00:00:00Z')"
        )

    store = EvidenceStore(path)
    store.initialize_schema()

    assert store.schema_version() == EVIDENCE_SCHEMA_VERSION
    assert "forecasts" in store.table_names()
    assert store.integrity_check() == "ok"


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


@pytest.mark.parametrize("target_table", ["bets", "bet_events"])
def test_unified_ledger_rows_can_receive_append_only_corrections(
    evidence_store, target_table
):
    correction_id = f"correction-{target_table}"
    evidence_store.append(
        EvidenceTable.CORRECTIONS,
        {
            "id": correction_id,
            "target_table": target_table,
            "target_id": f"{target_table}-1",
            "created_at": NOW,
            "reason": "Owner supplied corrected audit metadata.",
            "replacement_id": None,
            "idempotency_key": correction_id,
            "payload_json": {},
        },
    )

    assert evidence_store.get(EvidenceTable.CORRECTIONS, correction_id) is not None


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


def _legacy_quote_database(path):
    store = EvidenceStore(path, lock_writes=False)
    store.initialize_schema()
    store.append(EvidenceTable.RUNS, _run_record())
    for team in ("a", "b"):
        store.append(
            EvidenceTable.IDENTITIES,
            {
                "id": team,
                "entity_type": "team",
                "canonical_name": team,
                "created_at": NOW,
                "idempotency_key": team,
                "payload_json": {},
            },
        )
    store.append(
        EvidenceTable.FIXTURES,
        {
            "id": "fixture",
            "run_id": "run-1",
            "sport": "lol",
            "competition_id": "LPL",
            "team_a_identity_id": "a",
            "team_b_identity_id": "b",
            "start_time": NOW,
            "status": "not_started",
            "idempotency_key": "fixture",
            "payload_json": {},
        },
    )
    store.append(
        EvidenceTable.MARKET_CANDIDATES,
        {
            "id": "market",
            "run_id": "run-1",
            "fixture_id": "fixture",
            "provider": "thunderpick",
            "provider_market_id": "winner",
            "discovered_at": NOW,
            "match_status": "matched",
            "idempotency_key": "market",
            "payload_json": {"selection": "a"},
        },
    )
    store.append(
        EvidenceTable.BETS,
        {
            "id": "bet",
            "review_id": "run-1",
            "fixture_id": "fixture",
            "market_candidate_id": "market",
            "mode": "paper",
            "provider": "thunderpick",
            "target": "series_winner",
            "selection": "a",
            "opened_at": NOW,
            "currency": "EUR",
            "bankroll_before": "1000",
            "stake_percent": "1",
            "stake_amount": "10",
            "accepted_odds": "2",
            "actor_id": "owner",
            "evidence_classification": "model_positive_ev",
            "idempotency_key": "bet",
            "payload_json": {"reason": "legacy fact"},
        },
    )
    store.append(
        EvidenceTable.BET_EVENTS,
        {
            "id": "event",
            "bet_id": "bet",
            "event_at": NOW,
            "event_type": "settlement",
            "actor_id": "owner",
            "idempotency_key": "event",
            "payload_json": {"result": "win", "closing_odds": "1.8"},
        },
    )
    with sqlite3.connect(path) as conn:
        for table, column in (
            ("market_candidates", "prediction_id"),
            ("bets", "accepted_snapshot_id"),
            ("bet_events", "closing_snapshot_id"),
        ):
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
        conn.execute("UPDATE evidence_schema_version SET version = 6")
    return store


def test_v6_copy_migration_preserves_legacy_hashes(tmp_path):
    legacy = _legacy_quote_database(tmp_path / "original.db")
    before = {table: legacy.list(table) for table in EvidenceTable}
    path = tmp_path / "copy.db"
    with sqlite3.connect(legacy.path) as source, sqlite3.connect(path) as destination:
        source.backup(destination)
    migrated = EvidenceStore(path, lock_writes=False)
    migrated.initialize_schema()
    for table, rows in before.items():
        actual = migrated.list(table)
        assert len(actual) == len(rows)
        for old, new in zip(rows, actual, strict=True):
            assert {key: new[key] for key in old} == old
            assert all(new[key] is None for key in new.keys() - old.keys())
    assert legacy.schema_version() == LEGACY_SCHEMA_VERSION
    with migrated.connection(read_only=True) as conn:
        for table, column in (
            ("market_candidates", "prediction_id"),
            ("bets", "accepted_snapshot_id"),
            ("bet_events", "closing_snapshot_id"),
        ):
            assert column in {
                row[1] for row in conn.execute(f"PRAGMA table_info({table})")
            }
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_failed_migration_rolls_back_columns_and_version(tmp_path, monkeypatch):
    from oracle_bets_core.evidence import repository

    store = _legacy_quote_database(tmp_path / "legacy.db")
    before = store.list(EvidenceTable.RUNS)
    monkeypatch.setitem(
        repository.TABLE_COLUMNS,
        EvidenceTable.RUNS,
        (*repository.TABLE_COLUMNS[EvidenceTable.RUNS], "deliberately_missing"),
    )
    with pytest.raises(repository.EvidenceSchemaError, match="deliberately_missing"):
        store.initialize_schema()
    assert store.schema_version() == LEGACY_SCHEMA_VERSION
    assert store.list(EvidenceTable.RUNS) == before
    with store.connection(read_only=True) as conn:
        assert "prediction_id" not in {
            row[1] for row in conn.execute("PRAGMA table_info(market_candidates)")
        }


def test_migration_refuses_invalid_legacy_foreign_key_without_partial_columns(tmp_path):
    from oracle_bets_core.evidence.repository import EvidenceSchemaError

    store = _legacy_quote_database(tmp_path / "invalid-legacy.db")
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TRIGGER prevent_bets_update")
        connection.execute("UPDATE bets SET market_candidate_id = 'missing-market'")
    with pytest.raises(EvidenceSchemaError, match="foreign-key"):
        store.initialize_schema()
    assert store.schema_version() == LEGACY_SCHEMA_VERSION
    with store.connection(read_only=True) as connection:
        assert "accepted_snapshot_id" not in {
            row[1] for row in connection.execute("PRAGMA table_info(bets)")
        }
