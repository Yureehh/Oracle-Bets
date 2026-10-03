"""Bind loaded training tables to the generation which actually produced them."""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oracle_bets_core.io_utils import atomic_write_text
from oracle_bets_core.paths import PROCESSED_DIR, RAW_CURRENT_POINTER, SUITE_ROOT
from oracle_bets_core.pd import pd

from lol_bets.data_generation.ingestion.history import read_history_snapshot
from lol_bets.data_generation.ingestion.snapshot_io import sha256_file

MAP_INPUT_MANIFEST = PROCESSED_DIR / "training_inputs.json"
SERIES_INPUT_MANIFEST = PROCESSED_DIR / "series" / "training_inputs.json"


def code_fingerprint() -> str:
    """Include executable source and feature/rating configuration, including local edits."""
    paths = sorted(
        [
            path
            for path in (SUITE_ROOT / "packages").rglob("*.py")
            if "src" in path.parts
        ]
        + [
            path
            for path in (SUITE_ROOT / "config" / "lol").rglob("*.json")
            if "lightgbm" not in path.parts
        ]
    )
    digest = hashlib.sha256()
    digest.update(os.getenv("TRAINING_CONFIG_VARIANT", "").casefold().encode())
    for path in paths:
        digest.update(str(path.relative_to(SUITE_ROOT)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def current_source(pointer: Path) -> dict[str, Any]:
    _, manifest = read_history_snapshot(pointer_path=pointer)
    return {
        key: manifest[key]
        for key in ("snapshot_id", "data_sha256", "source_snapshot_id")
    }


def file_evidence(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    return {
        key: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for key, path in paths.items()
    }


def publish_generation(
    manifest_path: Path,
    paths: dict[str, Path],
    *,
    source: dict[str, Any],
    code: str,
    pointer: Path = RAW_CURRENT_POINTER,
    parent_generation: str | None = None,
    expected_files: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    files = file_evidence(paths)
    if expected_files is not None and files != expected_files:
        raise RuntimeError("Training tables changed during generation; rebuild them.")
    if source != current_source(pointer) or code != code_fingerprint():
        raise RuntimeError(
            "Training source or code changed during generation; rebuild tables."
        )
    manifest = {
        "schema_version": 1,
        "source": source,
        "code_sha256": code,
        "parent_generation": parent_generation,
        "files": files,
    }
    manifest["generation_id"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        manifest_path, json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    return manifest


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Training provenance missing or invalid at {path}; rebuild data and series tables."
        ) from exc
    unsigned = {key: value for key, value in manifest.items() if key != "generation_id"}
    if (
        manifest.get("schema_version") != 1
        or manifest.get("generation_id")
        != hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
    ):
        raise RuntimeError(
            f"Training generation manifest changed or invalid: {path}; rebuild tables."
        )
    return manifest


@dataclass
class TrainingInputs:
    manifests: dict[str, dict[str, Any]]
    manifest_paths: dict[str, Path]
    frames: dict[str, pd.DataFrame]
    pointer: Path

    def assert_unchanged(self) -> None:
        source = current_source(self.pointer)
        code = code_fingerprint()
        for name, manifest in self.manifests.items():
            if _read_manifest(self.manifest_paths[name]) != manifest:
                raise RuntimeError("Training generation changed after loading.")
            if source != manifest["source"] or code != manifest["code_sha256"]:
                raise RuntimeError(
                    "Training source or code changed; rebuild training tables."
                )
            for item in manifest["files"].values():
                if sha256_file(Path(item["path"])) != item["sha256"]:
                    raise RuntimeError(
                        f"Training table changed: {item['path']}; rebuild tables."
                    )


def load_training_inputs(
    map_manifest: Path = MAP_INPUT_MANIFEST,
    *,
    series_manifest: Path | None = None,
    pointer: Path = RAW_CURRENT_POINTER,
) -> TrainingInputs:
    paths = {"map": map_manifest}
    if series_manifest is not None:
        paths["series"] = series_manifest
    manifests = {key: _read_manifest(path) for key, path in paths.items()}
    if (
        "series" in manifests
        and manifests["series"]["parent_generation"]
        != manifests["map"]["generation_id"]
    ):
        raise RuntimeError(
            "Series tables reference a different map generation; rebuild series tables."
        )
    pinned = TrainingInputs(manifests, paths, {}, pointer)
    pinned.assert_unchanged()
    for manifest in manifests.values():
        for key, item in manifest["files"].items():
            data = Path(item["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise RuntimeError(
                    f"Training table changed while loading: {item['path']}"
                )
            pinned.frames[key] = pd.read_parquet(io.BytesIO(data))
    pinned.assert_unchanged()
    return pinned
