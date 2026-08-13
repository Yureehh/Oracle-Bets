"""Incremental history refreshes and periodic full reconciliation manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from oracle_bets_core.pd import pd

from lol_bets.data_generation.ingestion.quality import schema_fingerprint

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
