"""Recoverable archive support for the incompatible legacy ledger."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from oracle_bets_core.paths import LEGACY_LEDGER_ARCHIVE_DIR, LEGACY_LEDGER_DB


class ArchiveError(RuntimeError):
    """Raised when a legacy database cannot be archived safely."""


@dataclass(frozen=True)
class ArchiveResult:
    source_path: Path
    archive_path: Path
    sha256: str
    byte_count: int
    archived_at: datetime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sqlite(path: Path) -> None:
    if not path.is_file():
        msg = f"Legacy database does not exist: {path}"
        raise ArchiveError(msg)
    try:
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as conn:
            result = conn.execute("PRAGMA quick_check").fetchone()
    except sqlite3.DatabaseError as exc:
        msg = f"Legacy file is not a readable SQLite database: {path}"
        raise ArchiveError(msg) from exc
    if result is None or result[0] != "ok":
        msg = f"Legacy SQLite database failed quick_check: {path}"
        raise ArchiveError(msg)


def archive_legacy_database(
    source_path: str | Path = LEGACY_LEDGER_DB,
    archive_dir: str | Path = LEGACY_LEDGER_ARCHIVE_DIR,
    *,
    archived_at: datetime | None = None,
) -> ArchiveResult:
    """Copy and hash the legacy ledger without changing or deleting its source."""
    source = Path(source_path)
    destination_dir = Path(archive_dir)
    _validate_sqlite(source)
    timestamp = archived_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(timestamp):
        msg = "Archive timestamp must be timezone-aware UTC."
        raise ArchiveError(msg)
    source_hash = _sha256(source)
    stamp = timestamp.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = destination_dir / (
        f"{source.stem}-{stamp}-{source_hash[:12]}{source.suffix}"
    )
    if destination.exists():
        msg = f"Archive already exists and will not be overwritten: {destination}"
        raise ArchiveError(msg)
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    result = ArchiveResult(
        source_path=source,
        archive_path=destination,
        sha256=source_hash,
        byte_count=source.stat().st_size,
        archived_at=timestamp,
    )
    if not verify_archive(result):
        msg = f"Archive verification failed: {destination}"
        raise ArchiveError(msg)
    return result


def verify_archive(result: ArchiveResult) -> bool:
    """Verify archive bytes and SQLite integrity against its recorded source hash."""
    if not result.archive_path.is_file():
        return False
    if result.archive_path.stat().st_size != result.byte_count:
        return False
    if _sha256(result.archive_path) != result.sha256:
        return False
    try:
        _validate_sqlite(result.archive_path)
    except ArchiveError:
        return False
    return True
