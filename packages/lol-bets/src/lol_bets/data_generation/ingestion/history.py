"""Incremental history refreshes and periodic full reconciliation manifests."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from oracle_bets_core.logger import logger
from oracle_bets_core.pd import pd

from lol_bets.data_generation.ingestion.quality import schema_fingerprint
from lol_bets.data_generation.ingestion.snapshot_io import (
    atomic_json,
    prune_generations,
)
from lol_bets.data_generation.ingestion.snapshot_io import (
    atomic_symlink as _atomic_symlink,
)
from lol_bets.data_generation.ingestion.snapshot_io import (
    remove_tree as _remove_tree,
)
from lol_bets.data_generation.ingestion.snapshot_io import (
    sha256_file as _sha256_file,
)

ROW_ID_COLUMNS = ("gameid", "side", "position")
ENTITY_ID_COLUMNS = ("playerid", "teamid")


class SourceHistoryError(RuntimeError):
    """Raised when a history refresh cannot be reconciled safely."""


class HistoryRefreshMode(StrEnum):
    INCREMENTAL = "incremental"
    FULL = "full"


@dataclass(frozen=True)
class HistoryManifest:
    mode: HistoryRefreshMode
    refreshed_at: str
    existing_schema_fingerprint: str | None
    incoming_schema_fingerprint: str
    rows_before: int
    incoming_rows: int
    rows_after: int
    added_rows: int
    updated_rows: int
    unchanged_rows: int
    removed_rows: int
    source_snapshot_id: str | None = None
    snapshot_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "refreshed_at": self.refreshed_at,
            "existing_schema_fingerprint": self.existing_schema_fingerprint,
            "incoming_schema_fingerprint": self.incoming_schema_fingerprint,
            "rows_before": self.rows_before,
            "incoming_rows": self.incoming_rows,
            "rows_after": self.rows_after,
            "added_rows": self.added_rows,
            "updated_rows": self.updated_rows,
            "unchanged_rows": self.unchanged_rows,
            "removed_rows": self.removed_rows,
            "source_snapshot_id": self.source_snapshot_id,
            "snapshot_id": self.snapshot_id,
        }


def refresh_years(
    current_year: int,
    mode: HistoryRefreshMode,
    *,
    years_back: int = 3,
) -> list[int]:
    if years_back <= 0:
        msg = "years_back must be positive."
        raise SourceHistoryError(msg)
    if mode == HistoryRefreshMode.INCREMENTAL:
        return [current_year]
    if mode == HistoryRefreshMode.FULL:
        return list(range(current_year, current_year - years_back, -1))
    msg = f"Unsupported history refresh mode: {mode}"
    raise SourceHistoryError(msg)


def _validate_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        msg = "History refresh timestamp must be timezone-aware UTC."
        raise SourceHistoryError(msg)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _validate_schema(existing: pd.DataFrame, incoming: pd.DataFrame) -> None:
    required = set(ROW_ID_COLUMNS + ENTITY_ID_COLUMNS)
    missing = required - set(incoming.columns)
    if missing:
        msg = f"Incoming history schema is missing identity columns: {sorted(missing)}"
        raise SourceHistoryError(msg)
    if existing.empty and not len(existing.columns):
        return
    if set(existing.columns) != set(incoming.columns):
        missing_incoming = set(existing.columns) - set(incoming.columns)
        added_incoming = set(incoming.columns) - set(existing.columns)
        msg = (
            "History schema drift detected; "
            f"missing={sorted(missing_incoming)}, added={sorted(added_incoming)}"
        )
        raise SourceHistoryError(msg)


def _with_row_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    entity_id = out["playerid"].where(out["playerid"].notna(), out["teamid"])
    is_team = out["position"].astype(str).str.casefold().eq("team")
    if "playername" in out.columns:
        entity_id = entity_id.where(
            entity_id.notna(),
            out["playername"].where(~is_team),
        )
    if "teamname" in out.columns:
        entity_id = entity_id.where(
            entity_id.notna(),
            out["teamname"].where(is_team),
        )
    if entity_id.isna().any():
        msg = "History rows require a provider ID or entity name for identity."
        raise SourceHistoryError(msg)
    out["__row_key"] = [
        "|".join(str(value) for value in row)
        for row in zip(
            out["gameid"],
            out["side"],
            out["position"],
            entity_id,
            strict=False,
        )
    ]
    duplicated = out["__row_key"].duplicated(keep=False)
    if duplicated.any():
        sample = sorted(out.loc[duplicated, "__row_key"].unique())[:5]
        msg = f"History contains duplicate row identities: {sample}"
        raise SourceHistoryError(msg)
    return out


def _row_payloads(df: pd.DataFrame, columns: list[str]) -> dict[str, str]:
    payloads: dict[str, str] = {}
    for _, row in df.iterrows():
        payload = {
            column: None if pd.isna(row[column]) else row[column] for column in columns
        }
        payloads[str(row["__row_key"])] = json.dumps(
            payload,
            default=str,
            separators=(",", ":"),
            sort_keys=True,
        )
    return payloads


def merge_history(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    mode: HistoryRefreshMode,
    refreshed_at: datetime,
    source_snapshot_id: str | None = None,
) -> tuple[pd.DataFrame, HistoryManifest]:
    """Reconcile one provider refresh using stable game-row identities."""
    if not isinstance(mode, HistoryRefreshMode):
        mode = HistoryRefreshMode(mode)
    _validate_schema(existing, incoming)
    incoming_ordered = incoming.copy()
    if len(existing.columns):
        incoming_ordered = incoming_ordered[list(existing.columns)]
    existing_keyed = (
        _with_row_key(existing)
        if len(existing)
        else existing.assign(__row_key=pd.Series(dtype="object"))
    )
    incoming_keyed = _with_row_key(incoming_ordered)
    columns = [column for column in incoming_ordered.columns if column != "__row_key"]
    existing_payloads = _row_payloads(existing_keyed, columns)
    incoming_payloads = _row_payloads(incoming_keyed, columns)
    existing_keys = set(existing_payloads)
    incoming_keys = set(incoming_payloads)
    shared = existing_keys & incoming_keys
    added = incoming_keys - existing_keys
    removed = existing_keys - incoming_keys
    updated = {
        key for key in shared if existing_payloads[key] != incoming_payloads[key]
    }
    unchanged = shared - updated

    if mode == HistoryRefreshMode.FULL:
        merged_keyed = incoming_keyed
        removed_rows = len(removed)
    else:
        incoming_gameids = set(incoming_keyed["gameid"])
        replace_mask = existing_keyed["gameid"].isin(incoming_gameids)
        replaced = existing_keyed[replace_mask]
        retained = existing_keyed[~replace_mask]
        merged_keyed = pd.concat([retained, incoming_keyed], ignore_index=True)
        removed_rows = len(set(replaced["__row_key"]) - incoming_keys)
    merged_keyed = merged_keyed.sort_values("__row_key", kind="mergesort")
    merged = merged_keyed.drop(columns="__row_key").reset_index(drop=True)
    manifest = HistoryManifest(
        mode=mode,
        refreshed_at=_validate_utc(refreshed_at),
        existing_schema_fingerprint=(
            schema_fingerprint(existing) if len(existing.columns) else None
        ),
        incoming_schema_fingerprint=schema_fingerprint(incoming_ordered),
        rows_before=len(existing),
        incoming_rows=len(incoming),
        rows_after=len(merged),
        added_rows=len(added),
        updated_rows=len(updated),
        unchanged_rows=len(unchanged),
        removed_rows=removed_rows,
        source_snapshot_id=source_snapshot_id,
    )
    return merged, manifest


def write_history_manifest(
    manifest: HistoryManifest,
    destination: str | Path,
) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)
    return path


def publish_history_snapshot(
    data: pd.DataFrame,
    manifest: HistoryManifest,
    *,
    raw_path: str | Path,
    manifest_path: str | Path,
    generations_dir: str | Path,
    pointer_path: str | Path,
) -> str:
    """
    Publish one immutable history generation and atomically advance its pointer.

    Readers must resolve ``pointer_path`` through :func:`read_history_snapshot`.
    Legacy ``raw_path`` and ``manifest_path`` are maintained as atomic symlinks
    for older integrations, but never participate in generation construction.
    """
    if data.empty:
        raise SourceHistoryError("Cannot publish an empty history snapshot")
    root = Path(generations_dir)
    root.mkdir(parents=True, exist_ok=True)
    previous_snapshot_id = _current_snapshot_id(Path(pointer_path))
    snapshot_id = _snapshot_id(data, manifest)
    generation = root / snapshot_id
    if generation.exists():
        raise SourceHistoryError(f"History generation already exists: {snapshot_id}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}.", dir=root))
    published = False
    try:
        data_file = temporary / "raw_data.parquet"
        data.to_parquet(data_file, index=False)
        data_sha256 = _sha256_file(data_file)
        generation_manifest = manifest.to_dict() | {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "data_file": "raw_data.parquet",
            "data_sha256": data_sha256,
        }
        _atomic_json(temporary / "manifest.json", generation_manifest)
        temporary.replace(generation)
        published = True

        pointer = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "generation": snapshot_id,
            "data_file": f"generations/{snapshot_id}/raw_data.parquet",
            "manifest_file": f"generations/{snapshot_id}/manifest.json",
        }
        _atomic_json(Path(pointer_path), pointer)
        _atomic_symlink(generation / "raw_data.parquet", Path(raw_path))
        _atomic_symlink(generation / "manifest.json", Path(manifest_path))
        prune_generations(
            root,
            {snapshot_id, previous_snapshot_id},
            label="history",
            logger=logger,
        )
    except Exception:
        if temporary.exists():
            _remove_tree(temporary)
        if published:
            try:
                current = json.loads(Path(pointer_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError):
                current = {}
            if current.get("snapshot_id") != snapshot_id:
                _remove_tree(generation)
        raise
    return snapshot_id


def read_history_snapshot(
    *,
    pointer_path: str | Path,
    root: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve and verify the one current history generation."""
    pointer_file = Path(pointer_path)
    try:
        pointer = json.loads(pointer_file.read_text(encoding="utf-8"))
        snapshot_id = str(pointer["snapshot_id"])
        data_rel = Path(str(pointer["data_file"]))
        manifest_rel = Path(str(pointer["manifest_file"]))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise SourceHistoryError("Current history pointer is malformed") from error
    base = Path(root) if root is not None else pointer_file.parent
    if (
        not re.fullmatch(r"history-[a-f0-9]{24}", snapshot_id)
        or data_rel.is_absolute()
        or manifest_rel.is_absolute()
        or ".." in data_rel.parts
        or ".." in manifest_rel.parts
    ):
        raise SourceHistoryError("Current history pointer contains unsafe paths")
    expected_prefix = Path("generations") / snapshot_id
    if (
        data_rel != expected_prefix / "raw_data.parquet"
        or manifest_rel != expected_prefix / "manifest.json"
    ):
        raise SourceHistoryError("Current history pointer contains mixed paths")
    data_path = base / data_rel
    manifest_path = base / manifest_rel
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise SourceHistoryError("Current history manifest is unreadable") from error
    if manifest.get("snapshot_id") != snapshot_id:
        raise SourceHistoryError("Current history pointer and manifest disagree")
    if not data_path.is_file():
        raise SourceHistoryError("Current history data file is missing")
    expected_hash = manifest.get("data_sha256")
    if not isinstance(expected_hash, str) or _sha256_file(data_path) != expected_hash:
        raise SourceHistoryError("Current history data checksum does not match")
    return data_path, manifest


def current_history_data_path(
    raw_path: str | Path,
    *,
    pointer_path: str | Path,
) -> Path:
    """Return the verified current data file, or the legacy path at bootstrap."""
    pointer = Path(pointer_path)
    if pointer.is_file():
        data_path, _ = read_history_snapshot(pointer_path=pointer, root=pointer.parent)
        return data_path
    return Path(raw_path)


def _snapshot_id(data: pd.DataFrame, manifest: HistoryManifest) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(manifest.to_dict(), sort_keys=True, default=str).encode())
    digest.update("|".join(map(str, data.columns)).encode())
    digest.update(str(data.shape).encode())
    return f"history-{digest.hexdigest()[:24]}"


def _current_snapshot_id(pointer: Path) -> str | None:
    try:
        value = json.loads(pointer.read_text(encoding="utf-8")).get("snapshot_id")
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    snapshot_id = str(value or "")
    return snapshot_id if re.fullmatch(r"history-[a-f0-9]{24}", snapshot_id) else None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_json(path, payload, default=str)
