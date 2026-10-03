"""Immutable paired feature histories, published only after both tables are complete."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from oracle_bets_core.pd import pd

from lol_bets.data_generation.ingestion.snapshot_io import atomic_json, sha256_file

_FEATURE_SNAPSHOT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class FeatureSnapshot:
    snapshot_id: str
    directory: Path
    manifest: dict[str, Any]

    def read(self, entity: Literal["teams", "players"]) -> pd.DataFrame:
        if entity not in {"teams", "players"}:
            raise ValueError("Unknown feature entity")
        path = self.directory / f"{entity}.parquet"
        if sha256_file(path) != self.manifest["files"][path.name]:
            raise ValueError(f"Feature snapshot checksum mismatch: {entity}")
        return pd.read_parquet(path)


def publish_feature_snapshot(
    team_history: pd.DataFrame,
    player_history: pd.DataFrame,
    *,
    team_columns: list[str],
    player_columns: list[str],
    source_manifest: dict[str, Any],
    code_sha256: str,
    training_generation_id: str,
    root: Path,
    observed_at: datetime | None = None,
) -> FeatureSnapshot:
    """
    Keep every after-match state; never backdate a generation's availability.

    Match completion is a lower bound on state availability. The entire generation
    is usable only after publication, enforced separately by the loader.
    """
    root = Path(root)
    if not code_sha256 or not training_generation_id:
        raise ValueError(
            "Feature snapshots require code and training generation provenance"
        )
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".features-", dir=root))
    try:
        files = {}
        for entity, history, columns in (
            ("teams", team_history, team_columns),
            ("players", player_history, player_columns),
        ):
            missing = {*columns, "date", "gamelength"} - set(history.columns)
            if missing:
                raise ValueError(f"Feature history missing columns: {sorted(missing)}")
            dates = pd.to_datetime(history["date"], utc=True, errors="coerce")
            durations = pd.to_numeric(history["gamelength"], errors="coerce")
            if history.empty or dates.isna().any() or not durations.gt(0).all():
                raise ValueError(
                    "Feature history requires dates and positive gamelength"
                )
            out = history.loc[:, list(dict.fromkeys([*columns, "date"]))].copy()
            out = out.rename(columns={c: c.removesuffix("_after") for c in columns})
            out["date"] = dates
            out["state_available_at"] = dates + pd.to_timedelta(durations, unit="m")
            path = temporary / f"{entity}.parquet"
            out.to_parquet(path, index=False)
            files[path.name] = sha256_file(path)
        # Timestamp after serializing both tables, not at the start of a long build.
        available = pd.to_datetime(observed_at or datetime.now(UTC), utc=True)
        manifest = {
            "schema_version": _FEATURE_SNAPSHOT_SCHEMA_VERSION,
            "observed_at": available.isoformat(),
            "source_manifest": source_manifest,
            "code_sha256": code_sha256,
            "training_generation_id": training_generation_id,
            "files": files,
        }
        digest = hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest()
        snapshot_id = f"features-{digest[:24]}"
        manifest["snapshot_id"] = snapshot_id
        atomic_json(temporary / "manifest.json", manifest)
        generation = root / snapshot_id
        if not generation.exists():
            temporary.rename(generation)
        snapshot = load_feature_snapshot(root, snapshot_id=snapshot_id)
        atomic_json(root / "current.json", {"snapshot_id": snapshot_id})
        return snapshot
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_feature_snapshot(
    root: Path,
    *,
    snapshot_id: str | None = None,
    decision_at: datetime | None = None,
) -> FeatureSnapshot:
    """Resolve once, validate both files, and retain the same generation for a pair."""
    root = Path(root)
    if snapshot_id is None:
        snapshot_id = json.loads((root / "current.json").read_text())["snapshot_id"]
    if not isinstance(snapshot_id, str) or not re.fullmatch(
        r"features-[a-f0-9]{24}", snapshot_id
    ):
        raise ValueError("Invalid feature snapshot ID")
    directory = root / snapshot_id
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("snapshot_id") != snapshot_id
        or manifest.get("schema_version") != _FEATURE_SNAPSHOT_SCHEMA_VERSION
        or not manifest.get("code_sha256")
        or not manifest.get("training_generation_id")
    ):
        raise ValueError("Feature snapshot manifest mismatch")
    payload = {k: v for k, v in manifest.items() if k != "snapshot_id"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if snapshot_id != f"features-{digest[:24]}":
        raise ValueError("Feature snapshot manifest checksum mismatch")
    for entity in ("teams", "players"):
        name = f"{entity}.parquet"
        if sha256_file(directory / name) != manifest["files"].get(name):
            raise ValueError(f"Feature snapshot checksum mismatch: {entity}")
    if decision_at is not None and pd.to_datetime(
        manifest["observed_at"], utc=True
    ) > pd.to_datetime(decision_at, utc=True):
        raise ValueError("Feature snapshot not available at the decision time")
    return FeatureSnapshot(snapshot_id, directory, manifest)
