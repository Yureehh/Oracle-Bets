"""Cross-process maintenance lock shared by every state-writing entrypoint."""

from __future__ import annotations

import fcntl
import hashlib
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from oracle_bets_core.paths import SUITE_ROOT

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import TextIO


class MaintenanceBusyError(RuntimeError):
    """Raised when an incompatible Oracle Bets process still owns the lock."""


def maintenance_lock_path(root: Path = SUITE_ROOT) -> Path:
    fingerprint = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"oracle-bets-{fingerprint}.lock"


@contextmanager
def maintenance_lock(
    *,
    exclusive: bool,
    blocking: bool = True,
    root: Path = SUITE_ROOT,
) -> Iterator[TextIO]:
    """Hold the repository maintenance lock for the complete operation."""
    path = maintenance_lock_path(root)
    handle = path.open("a+", encoding="utf-8")
    path.chmod(0o600)
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    if not blocking:
        operation |= fcntl.LOCK_NB
    try:
        fcntl.flock(handle.fileno(), operation)
    except BlockingIOError as error:
        handle.close()
        kind = "writer" if exclusive else "maintenance"
        raise MaintenanceBusyError(
            f"Oracle Bets {kind} lock is busy; stop the Gateway bot and other commands."
        ) from error
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()} {'exclusive' if exclusive else 'shared'}\n")
        handle.flush()
        yield handle
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
