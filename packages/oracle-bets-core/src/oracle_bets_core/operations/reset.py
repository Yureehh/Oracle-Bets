"""Guarded archive-and-reset workflow for a new LoL paper epoch."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.io_utils import atomic_write_text
from oracle_bets_core.maintenance import maintenance_lock
from oracle_bets_core.operations.backup import (
    create_evidence_backup,
    verify_evidence_backup,
)
from oracle_bets_core.paths import (
    DATA_DIR,
    EVIDENCE_DB,
    LOGS_DIR,
    MODELS_DIR,
    PRODUCT_STATE_DIR,
    REPORTS_DIR,
    SUITE_ROOT,
)

_TOKEN_TTL = timedelta(minutes=20)
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600


class ResetSafetyError(RuntimeError):
    """Raised before any deletion when a reset guard is not satisfied."""


@dataclass(frozen=True)
class ResetPlanResult:
    plan_path: Path
    archive_path: Path
    manifest_path: Path
    confirmation_token: str
    expires_at: datetime
    file_count: int
    total_bytes: int


def default_reset_paths(root: Path = SUITE_ROOT) -> tuple[Path, ...]:
    """Return the finite generated-state allowlist for this repository."""
    return (
        DATA_DIR,
        PRODUCT_STATE_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        LOGS_DIR,
        root / "site",
        root / ".hypothesis",
        root / ".pytest_cache",
        root / ".ruff_cache",
    )


def create_reset_plan(
    *,
    archive_directory: Path,
    epoch: str,
    root: Path = SUITE_ROOT,
    paths: tuple[Path, ...] | None = None,
    evidence_database: Path = EVIDENCE_DB,
    now: datetime | None = None,
) -> ResetPlanResult:
    """Archive and restore-verify generated state, then issue one short-lived token."""
    created_at = now or datetime.now(UTC)
    _require_utc(created_at)
    if not epoch.strip():
        raise ResetSafetyError("paper epoch cannot be empty")
    root = root.resolve()
    commit = _clean_commit(root)
    selected = tuple(paths or default_reset_paths(root))
    archive_directory = archive_directory.expanduser().resolve()
    if _is_relative_to(archive_directory, root):
        raise ResetSafetyError("reset archive must live outside the repository")
    archive_directory.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIRECTORY_MODE)
    archive_directory.chmod(_PRIVATE_DIRECTORY_MODE)

    with maintenance_lock(exclusive=True, blocking=False, root=root):
        _assert_gateway_stopped(evidence_database.parent / "discord-bot.lock")
        _assert_database_quiescent(evidence_database)
        manifest = _build_manifest(
            root,
            selected,
            epoch=epoch,
            commit=commit,
            evidence_database=evidence_database,
        )
        stamp = created_at.strftime("%Y%m%dT%H%M%SZ")
        base = f"oracle-bets-{epoch}-{stamp}"
        archive_path = archive_directory / f"{base}.tar.gz"
        manifest_path = archive_directory / f"{base}.manifest.json"
        plan_path = archive_directory / f"{base}.plan.json"
        for destination in (archive_path, manifest_path, plan_path):
            if destination.exists():
                raise ResetSafetyError(f"reset artifact already exists: {destination}")
        atomic_write_text(
            manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            mode=_PRIVATE_FILE_MODE,
        )
        _write_archive(
            archive_path,
            root=root,
            paths=selected,
            manifest_path=manifest_path,
            evidence_database=evidence_database,
            created_at=created_at,
        )
        archive_path.chmod(_PRIVATE_FILE_MODE)
        archive_sha = _sha256(archive_path)
        manifest_sha = _sha256(manifest_path)
        _verify_archive(
            archive_path,
            manifest,
            expected_archive_sha=archive_sha,
        )
        token = secrets.token_urlsafe(24)
        expires_at = created_at + _TOKEN_TTL
        plan = {
            "schema_version": 1,
            "epoch": epoch,
            "created_at": created_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "repository": str(root),
            "commit": commit,
            "archive_path": str(archive_path),
            "archive_sha256": archive_sha,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "token_sha256": _hash_token(token),
            "used": False,
        }
        atomic_write_text(
            plan_path,
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            mode=_PRIVATE_FILE_MODE,
        )
    files = [entry for entry in manifest["entries"] if entry["type"] == "file"]
    return ResetPlanResult(
        plan_path=plan_path,
        archive_path=archive_path,
        manifest_path=manifest_path,
        confirmation_token=token,
        expires_at=expires_at,
        file_count=len(files),
        total_bytes=sum(int(entry["size"]) for entry in files),
    )


def apply_reset_plan(
    plan_path: Path,
    *,
    token: str,
    now: datetime | None = None,
) -> str:
    """Revalidate a reset plan and delete only its exact allowlisted roots."""
    applied_at = now or datetime.now(UTC)
    _require_utc(applied_at)
    plan_path = plan_path.expanduser().resolve()
    plan = _read_json(plan_path)
    root = Path(str(plan["repository"])).resolve()
    if plan.get("used") or plan_path.with_suffix(plan_path.suffix + ".used").exists():
        raise ResetSafetyError("reset confirmation has already been used")
    if _hash_token(token) != plan.get("token_sha256"):
        raise ResetSafetyError("reset confirmation token is invalid")
    if applied_at > datetime.fromisoformat(str(plan["expires_at"])):
        raise ResetSafetyError("reset confirmation token has expired")
    if _clean_commit(root) != plan.get("commit"):
        raise ResetSafetyError("repository commit changed after reset planning")
    archive_path = Path(str(plan["archive_path"]))
    manifest_path = Path(str(plan["manifest_path"]))
    if _sha256(archive_path) != plan.get("archive_sha256"):
        raise ResetSafetyError("reset archive hash changed")
    if _sha256(manifest_path) != plan.get("manifest_sha256"):
        raise ResetSafetyError("reset manifest hash changed")
    manifest = _read_json(manifest_path)
    selected = tuple(root / value for value in manifest["roots"])
    evidence_database = root / str(manifest["evidence_database"])

    with maintenance_lock(exclusive=True, blocking=False, root=root):
        _assert_gateway_stopped(evidence_database.parent / "discord-bot.lock")
        _assert_database_quiescent(evidence_database)
        current = _build_manifest(
            root,
            selected,
            epoch=str(manifest["epoch"]),
            commit=str(manifest["commit"]),
            evidence_database=evidence_database,
        )
        if _canonical_json(current) != _canonical_json(manifest):
            raise ResetSafetyError("generated state changed after reset planning")
        _verify_archive(
            archive_path,
            manifest,
            expected_archive_sha=str(plan["archive_sha256"]),
        )
        for path in selected:
            _assert_safe_root(root, path)
        for path in selected:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        for path in selected:
            path.mkdir(parents=True, exist_ok=True)
        store = EvidenceStore(evidence_database, lock_writes=False)
        store.initialize_schema()
        store.append(
            EvidenceTable.RUNS,
            {
                "id": f"paper-epoch-{manifest['epoch']}",
                "run_type": "paper_epoch",
                "started_at": applied_at,
                "status": "initialized",
                "idempotency_key": f"paper-epoch:{manifest['epoch']}",
                "payload_json": {
                    "epoch": manifest["epoch"],
                    "archive_sha256": plan["archive_sha256"],
                    "manifest_sha256": plan["manifest_sha256"],
                    "source_commit": plan["commit"],
                },
            },
        )
        marker = plan_path.with_suffix(plan_path.suffix + ".used")
        atomic_write_text(
            marker,
            json.dumps(
                {"applied_at": applied_at.isoformat(), "epoch": manifest["epoch"]},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            mode=_PRIVATE_FILE_MODE,
        )
        plan["used"] = True
        plan["applied_at"] = applied_at.isoformat()
        atomic_write_text(
            plan_path,
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            mode=_PRIVATE_FILE_MODE,
        )
    return str(manifest["epoch"])


def _build_manifest(
    root: Path,
    paths: tuple[Path, ...],
    *,
    epoch: str,
    commit: str,
    evidence_database: Path,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    roots: list[str] = []
    volatile_sqlite_paths = {
        Path(f"{evidence_database}-wal").resolve(strict=False),
        Path(f"{evidence_database}-shm").resolve(strict=False),
    }
    for selected in paths:
        path = selected.resolve(strict=False)
        _assert_safe_root(root, path)
        relative_root = path.relative_to(root).as_posix()
        roots.append(relative_root)
        if not path.exists():
            continue
        for item in (path, *sorted(path.rglob("*"))):
            if item.resolve(strict=False) in volatile_sqlite_paths:
                continue
            stat = item.lstat()
            if item.is_symlink():
                raise ResetSafetyError(f"reset path contains a symlink: {item}")
            if os.path.ismount(item):
                raise ResetSafetyError(f"reset path contains a mount point: {item}")
            relative = item.relative_to(root).as_posix()
            if item.is_dir():
                entries.append({"path": relative, "type": "directory", "size": 0})
            elif item.is_file():
                entries.append(
                    {
                        "path": relative,
                        "type": "file",
                        "size": stat.st_size,
                        "sha256": _sha256(item),
                    }
                )
            else:
                raise ResetSafetyError(f"unsupported reset filesystem entry: {item}")
    return {
        "schema_version": 1,
        "epoch": epoch,
        "repository": str(root),
        "commit": commit,
        "evidence_database": evidence_database.resolve(strict=False)
        .relative_to(root)
        .as_posix(),
        "roots": roots,
        "entries": entries,
    }


def _write_archive(
    destination: Path,
    *,
    root: Path,
    paths: tuple[Path, ...],
    manifest_path: Path,
    evidence_database: Path,
    created_at: datetime,
) -> None:
    with tempfile.TemporaryDirectory(prefix="oracle-bets-reset-") as temporary:
        metadata = Path(temporary)
        shutil.copy2(manifest_path, metadata / "manifest.json")
        if evidence_database.is_file():
            backup = create_evidence_backup(
                evidence_database,
                metadata / "evidence",
                created_at=created_at,
            )
            shutil.copy2(backup.path, metadata / "evidence.db")
        with tarfile.open(destination, "w:gz") as archive:
            for path in paths:
                if path.exists():
                    archive.add(path, arcname=path.resolve().relative_to(root))
            archive.add(metadata / "manifest.json", arcname="_reset/manifest.json")
            evidence_copy = metadata / "evidence.db"
            if evidence_copy.is_file():
                archive.add(evidence_copy, arcname="_reset/evidence.db")


def _verify_archive(
    archive_path: Path,
    manifest: dict[str, Any],
    *,
    expected_archive_sha: str,
) -> None:
    if _sha256(archive_path) != expected_archive_sha:
        raise ResetSafetyError("reset archive checksum verification failed")
    with tempfile.TemporaryDirectory(prefix="oracle-bets-restore-") as temporary:
        destination = Path(temporary)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(destination, filter="data")
        for entry in manifest["entries"]:
            restored = destination / entry["path"]
            if entry["type"] == "directory":
                if not restored.is_dir():
                    raise ResetSafetyError(f"archive lost directory: {entry['path']}")
            elif not restored.is_file() or _sha256(restored) != entry["sha256"]:
                raise ResetSafetyError(f"archive restore mismatch: {entry['path']}")
        backup = destination / "_reset/evidence.db"
        if backup.is_file():
            verify_evidence_backup(backup)


def _clean_commit(root: Path) -> str:
    git = shutil.which("git")
    if git is None:
        raise ResetSafetyError("git executable is required for reset provenance")
    status = subprocess.run(  # noqa: S603 - resolved local git executable, fixed argv
        [git, "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ResetSafetyError("repository tracked worktree must be clean")
    return subprocess.run(  # noqa: S603 - resolved local git executable, fixed argv
        [git, "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _assert_safe_root(root: Path, path: Path) -> None:
    if not _is_relative_to(path, root) or path == root:
        raise ResetSafetyError(f"reset path escapes repository: {path}")
    if path.exists() and (path.is_symlink() or os.path.ismount(path)):
        raise ResetSafetyError(f"reset root is a symlink or mount point: {path}")


def _assert_gateway_stopped(lock_path: Path) -> None:
    if not lock_path.exists():
        return
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise ResetSafetyError("Discord Gateway bot is still running") from error
    finally:
        handle.close()


def _assert_database_quiescent(path: Path) -> None:
    if not path.is_file():
        return
    try:
        with sqlite3.connect(path, timeout=0) as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.execute("BEGIN EXCLUSIVE")
            connection.rollback()
    except sqlite3.OperationalError as error:
        raise ResetSafetyError("evidence database has an active writer") from error


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResetSafetyError(f"invalid reset artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ResetSafetyError(f"invalid reset artifact: {path}")
    return payload


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("reset timestamps must be timezone-aware UTC")
