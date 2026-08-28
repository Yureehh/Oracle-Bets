"""Verified SQLite evidence backup, restore checks, and complete exports."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.evidence.schema import SCHEMA_VERSION

if TYPE_CHECKING:
    from pathlib import Path

_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600


@dataclass(frozen=True)
class BackupResult:
    path: Path
    sha256: str
    schema_version: int
    integrity: str


def create_evidence_backup(
    source: Path,
    destination_directory: Path,
    *,
    created_at: datetime | None = None,
) -> BackupResult:
    """Create a consistent SQLite online backup and verify it immediately."""
    timestamp = created_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise ValueError("backup timestamp must be UTC")
    if not source.is_file():
        raise FileNotFoundError(f"evidence database does not exist: {source}")
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination_directory.chmod(_PRIVATE_DIRECTORY_MODE)
    destination = destination_directory / (
        f"oracle-bets-{timestamp.strftime('%Y%m%dT%H%M%SZ')}.db"
    )
    if destination.exists():
        raise FileExistsError(f"backup already exists: {destination}")

    with (
        sqlite3.connect(source) as source_connection,
        sqlite3.connect(destination) as destination_connection,
    ):
        source_connection.backup(destination_connection)
    destination.chmod(_PRIVATE_FILE_MODE)
    return verify_evidence_backup(destination)


def verify_evidence_backup(path: Path) -> BackupResult:
    """Check SQLite integrity and require a supported evidence schema version."""
    if not path.is_file():
        raise FileNotFoundError(f"backup does not exist: {path}")
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        try:
            version = int(
                connection.execute(
                    "SELECT version FROM evidence_schema_version"
                ).fetchone()[0]
            )
        except (sqlite3.DatabaseError, TypeError) as error:
            raise ValueError(
                "backup is not an Oracle Bets evidence database"
            ) from error
    if integrity != "ok":
        raise ValueError(f"backup integrity check failed: {integrity}")
    if not 1 <= version <= SCHEMA_VERSION:
        raise ValueError(
            f"backup schema version {version} is outside supported range "
            f"1..{SCHEMA_VERSION}"
        )
    return BackupResult(
        path=path,
        sha256=_sha256(path),
        schema_version=version,
        integrity=integrity,
    )


def export_all_evidence(
    store: EvidenceStore,
    destination_directory: Path,
    *,
    file_format: str,
) -> tuple[Path, ...]:
    """Export every evidence table plus a deterministic export manifest."""
    if file_format not in {"json", "csv"}:
        raise ValueError("evidence export format must be json or csv")
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination_directory.chmod(_PRIVATE_DIRECTORY_MODE)
    outputs: list[Path] = []
    for table in EvidenceTable:
        path = destination_directory / f"{table.value}.{file_format}"
        if file_format == "json":
            store.export_json(table, path)
        else:
            store.export_csv(table, path)
        outputs.append(path)
    manifest = destination_directory / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": store.schema_version(),
                "format": file_format,
                "files": {
                    path.name: _sha256(path)
                    for path in sorted(outputs, key=lambda item: item.name)
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.chmod(_PRIVATE_FILE_MODE)
    return (*outputs, manifest)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
