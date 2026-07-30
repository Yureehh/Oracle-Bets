"""SQLite repository with idempotent append-only writes."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from oracle_bets_core.evidence.schema import (
    EVIDENCE_TABLES,
    SCHEMA_SQL,
    SCHEMA_VERSION,
    append_only_triggers_sql,
)
from oracle_bets_core.paths import EVIDENCE_DB

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence


class EvidenceSchemaError(RuntimeError):
    """Raised when an evidence database has an incompatible schema."""


class EvidenceConflictError(RuntimeError):
    """Raised when an idempotency key is reused for different evidence."""


class EvidenceTable(StrEnum):
    RUNS = "runs"
    RUN_EVENTS = "run_events"
    SOURCE_SNAPSHOTS = "source_snapshots"
    IDENTITIES = "identities"
    PROVIDER_LINKS = "provider_links"
    FIXTURES = "fixtures"
    MODEL_VERSIONS = "model_versions"
    PREDICTIONS = "predictions"
    MARKET_CANDIDATES = "market_candidates"
    MARKET_SNAPSHOTS = "market_snapshots"
    PROPOSALS = "proposals"
    APPROVALS = "approvals"
    PAPER_POSITIONS = "paper_positions"
    SETTLEMENTS = "settlements"
    CORRECTIONS = "corrections"


TABLE_COLUMNS: dict[EvidenceTable, frozenset[str]] = {
    EvidenceTable.RUNS: frozenset(
        {
            "id",
            "run_type",
            "started_at",
            "status",
            "idempotency_key",
            "payload_json",
        }
    ),
    EvidenceTable.RUN_EVENTS: frozenset(
        {
            "id",
            "run_id",
            "event_at",
            "event_type",
            "status",
            "idempotency_key",
            "payload_json",
        }
    ),
    EvidenceTable.SOURCE_SNAPSHOTS: frozenset(
        {
            "id",
            "run_id",
            "provider",
            "source_type",
            "observed_at",
            "source_uri",
            "schema_fingerprint",
            "idempotency_key",
            "payload_json",
        }
    ),
    EvidenceTable.IDENTITIES: frozenset(
        {
            "id",
            "entity_type",
            "canonical_name",
            "created_at",
            "idempotency_key",
            "payload_json",
        }
    ),
    EvidenceTable.PROVIDER_LINKS: frozenset(
        {
            "id",
            "identity_id",
            "provider",
            "provider_entity_id",
            "valid_from",
            "valid_to",
            "idempotency_key",
            "payload_json",
        }
    ),
    EvidenceTable.FIXTURES: frozenset(
        {
            "id",
            "run_id",
            "sport",
            "competition_id",
            "team_a_identity_id",
            "team_b_identity_id",
            "start_time",
            "best_of",
            "status",
            "idempotency_key",
            "payload_json",
        }
    ),
    EvidenceTable.MODEL_VERSIONS: frozenset(
        {
            "id",
            "sport",
            "target",
            "created_at",
            "artifact_uri",
            "artifact_checksum",
            "idempotency_key",
            "payload_json",
        }
    ),
    EvidenceTable.PREDICTIONS: frozenset(
        {
            "id",
            "run_id",
            "fixture_id",
            "model_version_id",
            "selection_id",
            "mode",
            "created_at",
            "probability_point",
            "probability_lower",
            "probability_upper",
            "warnings_json",
            "idempotency_key",
            "payload_json",
        }
    ),
    EvidenceTable.MARKET_CANDIDATES: frozenset(
        {
            "id",
            "run_id",
            "fixture_id",
            "provider",
            "provider_market_id",
            "provider_selection_id",
            "discovered_at",
            "match_status",
            "rejection_reason",
            "idempotency_key",
            "payload_json",
        }
    ),
    EvidenceTable.MARKET_SNAPSHOTS: frozenset(
        {
            "id",
            "market_candidate_id",
            "observed_at",
            "sequence_number",
            "intended_stake_units",
            "expected_decimal_odds",
            "available_stake_units",
            "book_json",
            "idempotency_key",
            "payload_json",
        }
    ),
    EvidenceTable.PROPOSALS: frozenset(
        {
            "id",
            "run_id",
            "prediction_id",
            "market_snapshot_id",
            "created_at",
            "state",
            "rejection_reason",
            "ruleset_version",
            "strategy",
            "stake_units",
            "idempotency_key",
            "payload_json",
        }
    ),
    EvidenceTable.APPROVALS: frozenset(
        {
            "id",
            "proposal_id",
            "created_at",
            "actor_id",
            "decision",
            "idempotency_key",
            "payload_json",
        }
    ),
    EvidenceTable.PAPER_POSITIONS: frozenset(
        {
            "id",
            "proposal_id",
            "opened_at",
            "strategy",
            "stake_units",
            "decimal_odds",
            "state",
            "idempotency_key",
            "payload_json",
        }
    ),
    EvidenceTable.SETTLEMENTS: frozenset(
        {
            "id",
            "paper_position_id",
            "settled_at",
            "result",
            "result_source",
            "pnl_units",
            "idempotency_key",
            "payload_json",
        }
    ),
    EvidenceTable.CORRECTIONS: frozenset(
        {
            "id",
            "target_table",
            "target_id",
            "created_at",
            "reason",
            "replacement_id",
            "idempotency_key",
            "payload_json",
        }
    ),
}

JSON_COLUMNS = {"payload_json", "warnings_json", "book_json"}


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return _utc_text(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    msg = f"Unsupported evidence value: {type(value).__name__}"
    raise TypeError(msg)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        msg = "Evidence timestamps must be timezone-aware UTC."
        raise ValueError(msg)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _database_uri(path: Path) -> str:
    return f"file:{quote(str(path.resolve()), safe='/')}?mode=ro"


class EvidenceStore:
    """One SQLite evidence store with append-only and dry-run guarantees."""

    def __init__(
        self,
        path: str | Path = EVIDENCE_DB,
        *,
        dry_run: bool = False,
    ) -> None:
        self.path = Path(path)
        self.dry_run = dry_run

    @contextmanager
    def connection(
        self,
        *,
        read_only: bool = False,
    ) -> Iterator[sqlite3.Connection]:
        if read_only:
            conn = sqlite3.connect(_database_uri(self.path), uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        if read_only:
            conn.execute("PRAGMA query_only=ON")
        else:
            conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            if read_only or self.dry_run:
                conn.rollback()
            else:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        if self.dry_run and not self.path.exists():
            msg = "Dry-run evidence store must already be initialized."
            raise EvidenceSchemaError(msg)
        with self.connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.executescript(append_only_triggers_sql())
            rows = conn.execute(
                "SELECT version FROM evidence_schema_version"
            ).fetchall()
            if not rows:
                conn.execute(
                    "INSERT INTO evidence_schema_version (version, installed_at) "
                    "VALUES (?, ?)",
                    (SCHEMA_VERSION, _utc_text(datetime.now(UTC))),
                )
            elif len(rows) != 1 or int(rows[0]["version"]) != SCHEMA_VERSION:
                versions = [int(row["version"]) for row in rows]
                msg = (
                    f"Evidence schema is incompatible: found {versions}, "
                    f"expected {SCHEMA_VERSION}."
                )
                raise EvidenceSchemaError(msg)

    def append(
        self,
        table: EvidenceTable,
        values: Mapping[str, Any],
    ) -> str:
        table, normalized = self._prepare_record(table, values)
        with self.connection() as conn:
            return self._insert_record(conn, table, normalized)

    def append_many(
        self,
        table: EvidenceTable,
        records: Sequence[Mapping[str, Any]],
    ) -> Sequence[str]:
        """Append many records atomically using one SQLite transaction."""
        prepared = [self._prepare_record(table, record) for record in records]
        if not prepared:
            return []
        ids: list[str] = []
        with self.connection() as conn:
            for prepared_table, normalized in prepared:
                ids.append(self._insert_record(conn, prepared_table, normalized))
        return ids

    def _prepare_record(
        self,
        table: EvidenceTable,
        values: Mapping[str, Any],
    ) -> tuple[EvidenceTable, dict[str, Any]]:
        if not isinstance(table, EvidenceTable):
            try:
                table = EvidenceTable(table)
            except ValueError as exc:
                msg = f"Unsupported evidence table: {table}"
                raise ValueError(msg) from exc
        payload = dict(values)
        allowed = TABLE_COLUMNS[table]
        unknown = set(payload) - allowed
        if unknown:
            msg = f"Unknown {table.value} fields: {', '.join(sorted(unknown))}."
            raise ValueError(msg)
        if "id" not in payload or "idempotency_key" not in payload:
            msg = f"{table.value} evidence requires id and idempotency_key."
            raise ValueError(msg)

        normalized = {
            key: self._normalize_value(key, value) for key, value in payload.items()
        }
        normalized["content_hash"] = hashlib.sha256(
            _canonical_json(normalized).encode()
        ).hexdigest()
        return table, normalized

    @staticmethod
    def _insert_record(
        conn: sqlite3.Connection,
        table: EvidenceTable,
        normalized: dict[str, Any],
    ) -> str:
        columns = list(normalized)
        placeholders = ", ".join("?" for _ in columns)
        column_sql = ", ".join(columns)
        sql = (
            f"INSERT INTO {table.value} ({column_sql}) "  # noqa: S608
            f"VALUES ({placeholders})"
        )
        try:
            conn.execute(sql, [normalized[column] for column in columns])
        except sqlite3.IntegrityError as exc:
            existing = conn.execute(
                f"SELECT id, content_hash FROM {table.value} "  # noqa: S608
                "WHERE idempotency_key = ?",
                (normalized["idempotency_key"],),
            ).fetchone()
            if existing is None:
                raise
            if existing["content_hash"] != normalized["content_hash"]:
                msg = f"{table.value} idempotency key already stores different content."
                raise EvidenceConflictError(msg) from exc
            return str(existing["id"])
        return str(normalized["id"])

    @staticmethod
    def _normalize_value(column: str, value: Any) -> Any:
        if column in JSON_COLUMNS:
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError as exc:
                    msg = f"{column} must contain valid JSON."
                    raise ValueError(msg) from exc
                return _canonical_json(parsed)
            return _canonical_json(value)
        if isinstance(value, datetime):
            value = _utc_text(value)
        elif isinstance(value, Decimal):
            value = str(value)
        elif isinstance(value, StrEnum):
            value = value.value
        elif isinstance(value, bool):
            value = int(value)
        return value

    def get(self, table: EvidenceTable, record_id: str) -> dict[str, Any] | None:
        table = EvidenceTable(table)
        with self.connection(read_only=True) as conn:
            row = conn.execute(
                f"SELECT * FROM {table.value} WHERE id = ?",  # noqa: S608
                (record_id,),
            ).fetchone()
        return dict(row) if row else None

    def list(self, table: EvidenceTable) -> Sequence[dict[str, Any]]:
        table = EvidenceTable(table)
        with self.connection(read_only=True) as conn:
            rows = conn.execute(
                f"SELECT * FROM {table.value} ORDER BY rowid",  # noqa: S608
            ).fetchall()
        return [dict(row) for row in rows]

    def count(self, table: EvidenceTable) -> int:
        table = EvidenceTable(table)
        with self.connection(read_only=True) as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS count FROM {table.value}",  # noqa: S608
            ).fetchone()
        return int(row["count"])

    def table_names(self) -> set[str]:
        with self.connection(read_only=True) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        return {str(row["name"]) for row in rows}

    def schema_version(self) -> int:
        with self.connection(read_only=True) as conn:
            row = conn.execute("SELECT version FROM evidence_schema_version").fetchone()
        if row is None:
            msg = "Evidence schema version is missing."
            raise EvidenceSchemaError(msg)
        return int(row["version"])

    def integrity_check(self) -> str:
        with self.connection(read_only=True) as conn:
            row = conn.execute("PRAGMA integrity_check").fetchone()
        return str(row[0])

    def export_json(self, table: EvidenceTable, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.list(table), indent=2, ensure_ascii=False, sort_keys=True)
            + "\n"
        )
        return destination

    def export_csv(self, table: EvidenceTable, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = self.list(table)
        with destination.open("w", newline="") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return destination


assert set(EVIDENCE_TABLES) == {table.value for table in EvidenceTable}
