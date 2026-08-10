"""Fail-closed Git provenance for immutable registered training runs."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from oracle_bets_core.paths import SUITE_ROOT

_CHANGE_PREVIEW_LIMIT = 20


class DirtyWorktreeError(RuntimeError):
    """Raised when a model would be registered from unreproducible source."""


@dataclass(frozen=True)
class RepositoryProvenance:
    revision: str
    clean: bool
    changes: tuple[str, ...]


def require_clean_repository(
    root: str | Path = SUITE_ROOT,
) -> RepositoryProvenance:
    """Return exact HEAD only when tracked and untracked source state is clean."""
    repository = Path(root)
    revision = _git(repository, "rev-parse", "HEAD")
    status = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    changes = tuple(line for line in status.splitlines() if line.strip())
    if changes:
        preview = "\n".join(changes[:_CHANGE_PREVIEW_LIMIT])
        if len(changes) > _CHANGE_PREVIEW_LIMIT:
            preview += f"\n... and {len(changes) - _CHANGE_PREVIEW_LIMIT} more"
        raise DirtyWorktreeError(
            "Registered training requires a clean Git worktree. Commit or remove "
            f"all source changes before training:\n{preview}"
        )
    return RepositoryProvenance(revision=revision, clean=True, changes=())


def _git(root: Path, *arguments: str) -> str:
    executable = shutil.which("git")
    if executable is None:
        raise DirtyWorktreeError("Git is required for registered training provenance")
    try:
        result = subprocess.run(  # noqa: S603
            [executable, *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise DirtyWorktreeError(
            f"Could not establish Git provenance for registered training: {error}"
        ) from error
    return result.stdout.strip()
