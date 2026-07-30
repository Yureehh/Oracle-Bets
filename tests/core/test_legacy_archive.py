from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime

import pytest
from oracle_bets_core.evidence.archive import (
    ArchiveError,
    archive_legacy_database,
    verify_archive,
)

NOW = datetime(2026, 7, 26, 8, 15, tzinfo=UTC)


def _create_legacy_database(path):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE bets (id INTEGER PRIMARY KEY, selection TEXT)")
        conn.execute("INSERT INTO bets (selection) VALUES ('Team A')")


def test_legacy_archive_copies_hashes_and_verifies_without_deleting_source(tmp_path):
    source = tmp_path / "ledger.db"
    archive_dir = tmp_path / "archives"
    _create_legacy_database(source)

    result = archive_legacy_database(source, archive_dir, archived_at=NOW)

    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source.exists()
    assert result.source_path == source
    assert result.archive_path.exists()
    assert result.sha256 == expected_hash
    assert result.byte_count == source.stat().st_size
    assert verify_archive(result)


def test_archive_refuses_to_overwrite_an_existing_archive(tmp_path):
    source = tmp_path / "ledger.db"
    archive_dir = tmp_path / "archives"
    _create_legacy_database(source)
    archive_legacy_database(source, archive_dir, archived_at=NOW)

    with pytest.raises(ArchiveError, match="already exists"):
        archive_legacy_database(source, archive_dir, archived_at=NOW)


def test_archive_requires_a_real_sqlite_database(tmp_path):
    source = tmp_path / "ledger.db"
    source.write_text("not sqlite")

    with pytest.raises(ArchiveError, match="SQLite"):
        archive_legacy_database(source, tmp_path / "archives", archived_at=NOW)
