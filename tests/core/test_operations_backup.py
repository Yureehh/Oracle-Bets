import hashlib
import json
import sqlite3
import stat
from datetime import UTC, datetime

import pytest
from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.operations import (
    EvidenceWorkflowJournal,
    WorkflowStep,
    daily_run_key,
    run_workflow,
)
from oracle_bets_core.operations.backup import (
    create_evidence_backup,
    export_all_evidence,
    verify_evidence_backup,
)

NOW = datetime(2026, 7, 27, 8, tzinfo=UTC)
EXPECTED_EXPORTED_FILES = len(EvidenceTable) + 1
EXPECTED_RETRY_ATTEMPTS = 2
LEGACY_SCHEMA_VERSION = 3
PRIVATE_FILE_MODE = 0o600
PRIVATE_DIRECTORY_MODE = 0o700


def _journal(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    store.initialize_schema()
    journal = EvidenceWorkflowJournal(
        store,
        run_type="daily_lol",
        started_at=NOW,
    )
    return store, journal


def test_retry_recovery_reruns_guards_and_skips_completed_writes(tmp_path):
    store, journal = _journal(tmp_path)
    attempts = {"read": 0, "write": 0}

    def flaky_read():
        attempts["read"] += 1
        if attempts["read"] < EXPECTED_RETRY_ATTEMPTS:
            raise ConnectionError("temporary provider outage")
        return "read ok"

    def write():
        attempts["write"] += 1
        return "write ok"

    steps = (
        WorkflowStep("read-provider", flaky_read, writes=False, retryable=True),
        WorkflowStep("write-evidence", write, writes=True),
    )
    first = run_workflow(
        "daily:lol:2026-07-27:test",
        steps,
        journal=journal,
        dry_run=False,
        clock=lambda: NOW,
    )
    resumed = run_workflow(
        "daily:lol:2026-07-27:test",
        steps,
        journal=journal,
        dry_run=False,
        clock=lambda: NOW,
    )

    assert first.ok
    assert first.steps[0].attempts == EXPECTED_RETRY_ATTEMPTS
    assert [step.status for step in resumed.steps] == [
        "completed",
        "skipped_completed",
    ]
    assert attempts == {"read": 3, "write": 1}
    events = store.list(EvidenceTable.RUN_EVENTS)
    assert [event["status"] for event in events] == [
        "failed",
        "completed",
        "completed",
        "completed",
    ]


def test_resume_reuses_original_run_when_process_start_time_changes(tmp_path):
    store, first_journal = _journal(tmp_path)
    run_key = "daily:lol:2026-07-27:restart"
    first_journal.ensure_run(run_key)
    restarted = EvidenceWorkflowJournal(
        store,
        run_type="daily_lol",
        started_at=datetime(2026, 7, 27, 9, tzinfo=UTC),
    )

    assert restarted.ensure_run(run_key) == first_journal.ensure_run(run_key)
    assert store.count(EvidenceTable.RUNS) == 1


def test_failure_stops_later_steps_and_returns_non_success(tmp_path):
    _store, journal = _journal(tmp_path)
    called = {"later": False}

    def fail():
        raise RuntimeError("database unavailable")

    def later():
        called["later"] = True
        return "must not run"

    outcome = run_workflow(
        "daily:lol:2026-07-27:failure",
        (
            WorkflowStep("core-database", fail, writes=True),
            WorkflowStep("publish", later, writes=True),
        ),
        journal=journal,
        dry_run=False,
        clock=lambda: NOW,
    )

    assert not outcome.ok
    assert [step.status for step in outcome.steps] == [
        "failed",
        "skipped_after_failure",
    ]
    assert not called["later"]


def test_dry_run_executes_reads_but_performs_no_writes_or_journal_changes(tmp_path):
    store, journal = _journal(tmp_path)
    before = store.count(EvidenceTable.RUNS)
    called = {"read": 0, "write": 0}

    outcome = run_workflow(
        "daily:lol:2026-07-27:dry",
        (
            WorkflowStep(
                "read",
                lambda: called.__setitem__("read", called["read"] + 1) or "ok",
                writes=False,
            ),
            WorkflowStep(
                "write",
                lambda: called.__setitem__("write", called["write"] + 1) or "ok",
                writes=True,
            ),
        ),
        journal=journal,
        dry_run=True,
        clock=lambda: NOW,
    )

    assert outcome.ok
    assert called == {"read": 1, "write": 0}
    assert outcome.steps[1].status == "skipped_dry_run"
    assert store.count(EvidenceTable.RUNS) == before


def test_daily_run_key_is_stable_for_config_and_changes_with_rules():
    first = daily_run_key(
        "lol",
        scheduled_for=NOW,
        config={"window": 2, "leagues": ["LCK"]},
    )
    second = daily_run_key(
        "lol",
        scheduled_for=NOW,
        config={"leagues": ["LCK"], "window": 2},
    )
    changed = daily_run_key(
        "lol",
        scheduled_for=NOW,
        config={"window": 3, "leagues": ["LCK"]},
    )

    assert first == second
    assert changed != first


def test_backup_restore_verification_and_complete_export(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    store.initialize_schema()
    store.append(
        EvidenceTable.RUNS,
        {
            "id": "run-1",
            "run_type": "test",
            "started_at": NOW,
            "status": "completed",
            "idempotency_key": "test-run",
            "payload_json": {"ok": True},
        },
    )

    backup = create_evidence_backup(
        store.path,
        tmp_path / "backups",
        created_at=NOW,
    )
    verified = verify_evidence_backup(backup.path)
    exports = export_all_evidence(
        store,
        tmp_path / "exports",
        file_format="json",
    )

    assert backup.sha256 == verified.sha256
    assert verified.integrity == "ok"
    assert len(exports) == EXPECTED_EXPORTED_FILES
    assert (tmp_path / "exports/runs.json").is_file()
    assert (tmp_path / "exports/manifest.json").is_file()
    assert stat.S_IMODE(backup.path.stat().st_mode) == PRIVATE_FILE_MODE
    assert stat.S_IMODE((tmp_path / "exports").stat().st_mode) == PRIVATE_DIRECTORY_MODE
    assert all(
        stat.S_IMODE(path.stat().st_mode) == PRIVATE_FILE_MODE for path in exports
    )
    restored = EvidenceStore(backup.path)
    assert {table.value: restored.count(table) for table in EvidenceTable} == {
        table.value: store.count(table) for table in EvidenceTable
    }
    manifest = json.loads((tmp_path / "exports/manifest.json").read_text())
    assert manifest["files"] == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in exports
        if path.name != "manifest.json"
    }


def test_backup_verification_accepts_migratable_older_schema(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    store.initialize_schema()
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE evidence_schema_version SET version = ?", (LEGACY_SCHEMA_VERSION,)
        )

    backup = create_evidence_backup(store.path, tmp_path / "backups", created_at=NOW)

    assert backup.integrity == "ok"
    assert backup.schema_version == LEGACY_SCHEMA_VERSION


def test_backup_verification_rejects_unknown_future_schema(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    store.initialize_schema()
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE evidence_schema_version SET version = 999")

    with pytest.raises(ValueError, match="outside supported range"):
        create_evidence_backup(store.path, tmp_path / "backups", created_at=NOW)
