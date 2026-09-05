"""Small filesystem primitives shared by immutable ingestion snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import logging


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(
    path: Path,
    payload: dict[str, Any],
    *,
    default: Any = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(payload, indent=2, sort_keys=True, default=default) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_symlink(target: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target.resolve())
    temporary.replace(destination)


def remove_tree(path: Path) -> None:
    shutil.rmtree(path)


def prune_generations(
    root: Path,
    keep: set[str | None],
    *,
    label: str,
    logger: logging.Logger,
) -> None:
    """Retain current/previous immutable generations and best-effort prune older ones."""
    for generation in root.iterdir():
        if generation.name.startswith(".") or generation.name in keep:
            continue
        try:
            generation.unlink() if generation.is_symlink() else remove_tree(generation)
        except OSError as error:
            logger.warning(
                "Could not remove old %s generation %s: %s",
                label,
                generation,
                error,
            )
