"""Refresh and validate the public Oracle's Elixir CSV source."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import requests
from oracle_bets_core.logger import logger
from oracle_bets_core.paths import RAW_DATA

from lol_bets.data_generation.ingestion.snapshot_io import (
    atomic_json as _atomic_json,
)
from lol_bets.data_generation.ingestion.snapshot_io import (
    atomic_symlink as _atomic_symlink,
)
from lol_bets.data_generation.ingestion.snapshot_io import (
    prune_generations,
)
from lol_bets.data_generation.ingestion.snapshot_io import (
    remove_tree as _remove_tree,
)
from lol_bets.data_generation.ingestion.snapshot_io import (
    sha256_file as _sha256_file,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

SOURCE_FILE_SUFFIX = "_LoL_esports_match_data_from_OraclesElixir.csv"
SOURCE_CACHE_DIRECTORY = RAW_DATA.parent / "oracles_elixir_cache"
SOURCE_MANIFEST = "source_manifest.json"
SOURCE_GENERATIONS_DIRECTORY = "generations"
SOURCE_CURRENT_POINTER = "current.json"
PUBLIC_DRIVE_FILE_IDS = {
    2024: "1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN",  # pragma: allowlist secret
    2025: "1v6LRphp2kYciU4SXp0PCjEMuev1bDejc",  # pragma: allowlist secret
    2026: "1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm",  # pragma: allowlist secret
}
PUBLIC_DRIVE_FOLDER_ID = (  # pragma: allowlist secret
    "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"  # pragma: allowlist secret
)
PUBLIC_DRIVE_FOLDER_URL = (
    f"https://drive.google.com/drive/folders/{PUBLIC_DRIVE_FOLDER_ID}"
)
TAKEOUT_EXPORTS_URL = "https://takeout-pa.clients6.google.com/v1/exports"
ARCHIVE_HOST = "storage.googleapis.com"
ARCHIVE_PATH_PREFIX = "/drive-bulk-export-anonymous/"
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
EXPORT_POLL_SECONDS = 5.0
EXPORT_MAX_POLLS = 60
DEFAULT_MAX_AGE = timedelta(hours=48)
DEFAULT_MINIMUM_BYTES = 1_024


class OracleSourceReadinessError(RuntimeError):
    """Raised when local source files cannot safely support a rebuild."""


class OracleSourceRefreshError(RuntimeError):
    """Raised when public source files cannot be refreshed safely."""


@dataclass(frozen=True)
class OracleSourceFile:
    year: int
    path: str
    size_bytes: int
    modified_at: datetime

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["modified_at"] = self.modified_at.isoformat()
        return payload


@dataclass(frozen=True)
class OracleSourceReadiness:
    source_directory: str
    source_is_symlink: bool
    required_years: tuple[int, ...]
    checked_at: datetime
    files: tuple[OracleSourceFile, ...]
    current_year_max_match_at: datetime | None
    issues: tuple[str, ...]
    snapshot_id: str | None = None

    @property
    def ready(self) -> bool:
        return not self.issues

    def raise_if_unready(self) -> None:
        if self.ready:
            return
        raise OracleSourceReadinessError(
            "Oracle's Elixir source is not ready: " + ", ".join(self.issues)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "source_directory": self.source_directory,
            "source_is_symlink": self.source_is_symlink,
            "required_years": list(self.required_years),
            "checked_at": self.checked_at.isoformat(),
            "files": [item.to_dict() for item in self.files],
            "current_year_max_match_at": (
                self.current_year_max_match_at.isoformat()
                if self.current_year_max_match_at
                else None
            ),
            "issues": list(self.issues),
            "snapshot_id": self.snapshot_id,
        }


@dataclass(frozen=True)
class OracleSourceRefresh:
    source_directory: str
    downloaded_at: datetime
    files: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_directory": self.source_directory,
            "downloaded_at": self.downloaded_at.isoformat(),
            "files": list(self.files),
        }


def oracle_source_directory() -> Path:
    """Return the managed cache, or an explicit owner-provided local source."""
    return _resolve_source_directory(managed_oracle_source_directory())


def managed_oracle_source_directory() -> Path:
    """Return the stable source root that owns generations and its pointer."""
    configured = os.getenv("ORACLES_ELIXIR_LOCAL_DIR")
    return Path(configured).expanduser() if configured else SOURCE_CACHE_DIRECTORY


def source_snapshot_id(source_directory: str | Path) -> str | None:
    """Return the immutable source generation used by one reader."""
    manifest = Path(source_directory) / SOURCE_MANIFEST
    if not manifest.is_file():
        return None
    try:
        value = json.loads(manifest.read_text(encoding="utf-8")).get("snapshot_id")
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise OracleSourceReadinessError(
            "source generation manifest is malformed"
        ) from error
    if value is None:
        return None
    snapshot = str(value)
    if not re.fullmatch(r"source-[a-f0-9]{24}", snapshot):
        raise OracleSourceReadinessError("source generation manifest has an invalid ID")
    return snapshot


def _resolve_source_directory(root: Path) -> Path:
    pointer = root / SOURCE_CURRENT_POINTER
    if not pointer.is_file():
        return root
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        generation = str(payload["generation"])
        snapshot_id = str(payload["snapshot_id"])
        if generation != snapshot_id or not re.fullmatch(
            r"source-[a-f0-9]{24}", generation
        ):
            raise ValueError("invalid source generation identifier")  # noqa: TRY301
        directory = root / SOURCE_GENERATIONS_DIRECTORY / generation
        manifest = json.loads((directory / SOURCE_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise OracleSourceReadinessError(
            "current Oracle's Elixir source pointer is malformed"
        ) from error
    if manifest.get("snapshot_id") != snapshot_id:
        raise OracleSourceReadinessError("source pointer and manifest disagree")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise OracleSourceReadinessError("source generation manifest has no files")
    for item in files:
        if not isinstance(item, dict):
            raise OracleSourceReadinessError("source generation manifest is malformed")
        filename = str(item.get("filename", ""))
        try:
            expected_filename = f"{int(item['year'])}{SOURCE_FILE_SUFFIX}"
        except (KeyError, TypeError, ValueError) as error:
            raise OracleSourceReadinessError(
                "source generation manifest has an invalid year"
            ) from error
        expected = item.get("sha256")
        path = directory / filename
        if (
            filename != expected_filename
            or PurePosixPath(filename).name != filename
            or not isinstance(expected, str)
            or not path.is_file()
        ):
            raise OracleSourceReadinessError("source generation is incomplete")
        if _sha256_file(path) != expected:
            raise OracleSourceReadinessError("source generation checksum mismatch")
    return directory


def refresh_oracle_source(  # noqa: PLR0912, PLR0915
    source_directory: str | Path | None = None,
    *,
    required_years: Sequence[int] | None = None,
    session: requests.Session | None = None,
    now: datetime | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> OracleSourceRefresh:
    """Refresh required files through Drive's anonymous bulk-export service."""
    root = (
        Path(source_directory)
        if source_directory
        else managed_oracle_source_directory()
    )
    cloud_storage = Path.home() / "Library" / "CloudStorage"
    if root.is_symlink() or root.resolve().is_relative_to(cloud_storage):
        raise OracleSourceRefreshError(
            "Refusing to update a symlinked/cloud-backed source directory; "
            "use the managed local Oracle's Elixir cache."
        )
    root.mkdir(parents=True, exist_ok=True)
    downloaded_at = (now or datetime.now(UTC)).astimezone(UTC)
    years = tuple(
        required_years or range(downloaded_at.year - 2, downloaded_at.year + 1)
    )
    missing_ids = [year for year in years if year not in PUBLIC_DRIVE_FILE_IDS]
    if missing_ids:
        raise OracleSourceRefreshError(
            f"No reviewed public Drive file ID configured for years: {missing_ids}"
        )

    client = session or requests.Session()
    owns_session = session is None
    staged: list[tuple[Path, Path, dict[str, Any], datetime]] = []
    archives: list[Path] = []
    generation: Path | None = None
    staging_root: Path | None = None
    published = False
    previous_snapshot_id = _current_source_snapshot_id(root)
    try:
        export_key = _public_export_key(client)
        export_job_id = _start_public_export(client, export_key, years)
        archive_urls = _wait_for_public_export(
            client,
            export_key,
            export_job_id,
            sleeper=sleeper,
        )
        archives = _download_export_archives(client, root, archive_urls)
        generations = root / SOURCE_GENERATIONS_DIRECTORY
        generations.mkdir(parents=True, exist_ok=True)
        staging_root = Path(tempfile.mkdtemp(prefix=".source-", dir=generations))
        staged = _stage_source_files(staging_root, archives, years)
        for temporary, destination, _metadata, remote_modified_at in staged:
            temporary.replace(destination)
            timestamp = remote_modified_at.timestamp()
            os.utime(destination, (timestamp, timestamp))
        files = tuple(metadata for _, _, metadata, _ in staged)
        digest = hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
        generation = generations / f"source-{digest}"
        _write_source_manifest(
            staging_root / SOURCE_MANIFEST,
            downloaded_at,
            export_job_id,
            files,
            snapshot_id=generation.name,
        )
        if generation.exists():
            _remove_tree(staging_root)
            staging_root = None
        else:
            staging_root.replace(generation)
        published = True
        _atomic_json(
            root / SOURCE_CURRENT_POINTER,
            {
                "schema_version": 1,
                "snapshot_id": generation.name,
                "generation": generation.name,
                "manifest_file": f"{SOURCE_GENERATIONS_DIRECTORY}/{generation.name}/{SOURCE_MANIFEST}",
            },
        )
        _atomic_symlink(generation / SOURCE_MANIFEST, root / SOURCE_MANIFEST)
        for metadata in files:
            filename = str(metadata["filename"])
            _atomic_symlink(generation / filename, root / filename)
        prune_generations(
            generations,
            {generation.name, previous_snapshot_id},
            label="source",
            logger=logger,
        )
    except (
        KeyError,
        OSError,
        requests.RequestException,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        raise OracleSourceRefreshError(
            f"Oracle's Elixir refresh failed: {exc}"
        ) from exc
    finally:
        for temporary, *_ in staged:
            temporary.unlink(missing_ok=True)
        for archive in archives:
            archive.unlink(missing_ok=True)
        if staging_root is not None and staging_root.exists():
            _remove_tree(staging_root)
        if published and generation is not None:
            try:
                current = json.loads(
                    (root / SOURCE_CURRENT_POINTER).read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError, TypeError):
                current = {}
            if current.get("snapshot_id") != generation.name:
                _remove_tree(generation)
        if owns_session:
            client.close()

    return OracleSourceRefresh(str(root), downloaded_at, files)


def _public_export_key(session: requests.Session) -> str:
    response = session.get(PUBLIC_DRIVE_FOLDER_URL, timeout=(10, 30))
    try:
        response.raise_for_status()
        match = re.search(r'"yLTeS":"([^"]+)"', response.text)
        if match is None:
            raise ValueError("public Drive export key is unavailable")
        return match.group(1)
    finally:
        response.close()


def _start_public_export(
    session: requests.Session,
    export_key: str,
    years: Sequence[int],
) -> str:
    response = session.post(
        TAKEOUT_EXPORTS_URL,
        params={"key": export_key},
        json={
            "archivePrefix": "OracleElixir",
            "items": [{"id": PUBLIC_DRIVE_FILE_IDS[year]} for year in years],
        },
        timeout=(10, 30),
    )
    try:
        response.raise_for_status()
        return str(response.json()["exportJob"]["id"])
    finally:
        response.close()


def _wait_for_public_export(
    session: requests.Session,
    export_key: str,
    export_job_id: str,
    *,
    sleeper: Callable[[float], None],
) -> tuple[str, ...]:
    for poll in range(EXPORT_MAX_POLLS):
        response = session.get(
            f"{TAKEOUT_EXPORTS_URL}/{export_job_id}",
            params={"key": export_key},
            headers={"PbiToken": str(uuid.uuid4())},
            timeout=(10, 30),
        )
        try:
            response.raise_for_status()
            job = response.json()["exportJob"]
        finally:
            response.close()
        status = str(job.get("status", ""))
        if status == "SUCCEEDED":
            paths = tuple(
                str(archive["storagePath"]) for archive in job.get("archives", ())
            )
            if not paths:
                raise ValueError("public Drive export completed without archives")
            return paths
        if status not in {"QUEUED", "RUNNING"}:
            raise ValueError(
                f"public Drive export ended with status {status or 'unknown'}"
            )
        if poll + 1 < EXPORT_MAX_POLLS:
            sleeper(EXPORT_POLL_SECONDS)
    raise ValueError("public Drive export timed out")


def _download_export_archives(
    session: requests.Session,
    root: Path,
    archive_urls: Sequence[str],
) -> list[Path]:
    archives: list[Path] = []
    try:
        for index, archive_url in enumerate(archive_urls, start=1):
            _validate_archive_url(archive_url)
            logger.info("Downloading Oracle's Elixir source archive %s.", index)
            response = session.get(archive_url, stream=True, timeout=(10, 180))
            try:
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".oracle-export-{index}-",
                    suffix=".zip.part",
                    dir=root,
                    delete=False,
                ) as handle:
                    archive = Path(handle.name)
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                        if chunk:
                            handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                response.close()
            _require_zip_archive(archive)
            archives.append(archive)
    except Exception:
        for archive in archives:
            archive.unlink(missing_ok=True)
        raise
    return archives


def _require_zip_archive(path: Path) -> None:
    if zipfile.is_zipfile(path):
        return
    path.unlink(missing_ok=True)
    raise ValueError("public Drive export is not a valid ZIP archive")


def _validate_archive_url(value: str) -> None:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != ARCHIVE_HOST
        or not parsed.path.startswith(ARCHIVE_PATH_PREFIX)
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("public Drive export returned an untrusted archive URL")


def _stage_source_files(
    root: Path,
    archives: Sequence[Path],
    years: Sequence[int],
) -> list[tuple[Path, Path, dict[str, Any], datetime]]:
    expected = {f"{year}{SOURCE_FILE_SUFFIX}": year for year in years}
    entries: dict[str, tuple[Path, zipfile.ZipInfo]] = {}
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                name = PurePosixPath(info.filename).name
                if name not in expected:
                    continue
                if name in entries:
                    raise ValueError(f"duplicate source file in public export: {name}")
                entries[name] = (archive_path, info)
    missing = sorted(set(expected) - set(entries))
    if missing:
        raise ValueError(f"public Drive export is missing required files: {missing}")

    staged: list[tuple[Path, Path, dict[str, Any], datetime]] = []
    try:
        for filename, year in expected.items():
            archive_path, info = entries[filename]
            temporary, size_bytes, digest = _stage_zip_entry(
                root,
                archive_path,
                info,
                year,
            )
            modified_at = datetime(*info.date_time, tzinfo=UTC)
            metadata = {
                "year": year,
                "drive_file_id": PUBLIC_DRIVE_FILE_IDS[year],
                "filename": filename,
                "size_bytes": size_bytes,
                "sha256": digest,
                "remote_modified_at": modified_at.isoformat(),
                "status": "downloaded",
            }
            staged.append((temporary, root / filename, metadata, modified_at))
    except Exception:
        for temporary, *_ in staged:
            temporary.unlink(missing_ok=True)
        raise
    return staged


def _stage_zip_entry(
    root: Path,
    archive_path: Path,
    info: zipfile.ZipInfo,
    year: int,
) -> tuple[Path, int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{year}-",
        suffix=".part",
        dir=root,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            with zipfile.ZipFile(archive_path) as archive, archive.open(info) as source:
                while chunk := source.read(DOWNLOAD_CHUNK_BYTES):
                    handle.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
            _validate_complete_download(
                temporary,
                year,
                byte_count,
                expected_bytes=info.file_size,
            )
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return temporary, byte_count, digest.hexdigest()


def _validate_complete_download(
    path: Path,
    year: int,
    byte_count: int,
    *,
    expected_bytes: int | None = None,
) -> None:
    if expected_bytes is not None and byte_count != expected_bytes:
        raise ValueError(f"public export size mismatch for {year}")
    if byte_count < DEFAULT_MINIMUM_BYTES:
        raise ValueError(f"downloaded file for {year} is incomplete")
    with path.open(encoding="utf-8-sig", newline="") as source:
        fields = set(next(csv.reader(source), ()))
    if not {"gameid", "date"}.issubset(fields):
        raise ValueError(f"downloaded file for {year} is not an Oracle's Elixir CSV")


def _write_source_manifest(
    path: Path,
    downloaded_at: datetime,
    export_job_id: str,
    files: tuple[dict[str, object], ...],
    *,
    snapshot_id: str | None = None,
) -> None:
    payload = {
        "schema_version": 2,
        "provider": "oracle_elixir_public_google_drive_bulk_export",
        "downloaded_at": downloaded_at.isoformat(),
        "export_job_id": export_job_id,
        "files": list(files),
    }
    if snapshot_id is not None:
        payload["snapshot_id"] = snapshot_id
    _atomic_json(path, payload)


def inspect_oracle_source(  # noqa: PLR0912
    source_directory: str | Path | None = None,
    *,
    required_years: Sequence[int] | None = None,
    now: datetime | None = None,
    maximum_current_year_age: timedelta = DEFAULT_MAX_AGE,
    minimum_bytes: int = DEFAULT_MINIMUM_BYTES,
    stability_seconds: float = 1.0,
    sleeper: Callable[[float], None] = time.sleep,
    require_symlink: bool = False,
) -> OracleSourceReadiness:
    """Inspect required local files without modifying or downloading source data."""
    checked_at = (now or datetime.now(UTC)).astimezone(UTC)
    years = tuple(required_years or range(checked_at.year - 2, checked_at.year + 1))
    root = (
        Path(source_directory)
        if source_directory
        else managed_oracle_source_directory()
    )
    requested_root = root
    snapshot_id: str | None = None
    try:
        pointer = root / SOURCE_CURRENT_POINTER
        if pointer.is_file():
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            snapshot_id = str(payload["snapshot_id"])
            root = _resolve_source_directory(root)
    except (
        OSError,
        OracleSourceReadinessError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ):
        return OracleSourceReadiness(
            str(requested_root),
            requested_root.is_symlink(),
            years,
            checked_at,
            (),
            None,
            ("current_source_pointer_invalid",),
            None,
        )
    if maximum_current_year_age <= timedelta(0):
        raise ValueError("maximum_current_year_age must be positive")
    if minimum_bytes <= 0 or stability_seconds < 0:
        raise ValueError("source size and stability thresholds are invalid")
    if not root.is_dir():
        return OracleSourceReadiness(
            str(requested_root),
            requested_root.is_symlink(),
            years,
            checked_at,
            (),
            None,
            ("source_directory_missing",),
            snapshot_id,
        )

    first_stats: dict[int, tuple[int, int]] = {}
    issues: list[str] = []
    if require_symlink and not root.is_symlink():
        issues.append("source_directory_not_symlink")
    for year in years:
        path = root / f"{year}{SOURCE_FILE_SUFFIX}"
        if not path.is_file():
            issues.append(f"missing_year:{year}")
            continue
        stat = path.stat()
        first_stats[year] = (stat.st_size, stat.st_mtime_ns)
    if stability_seconds:
        sleeper(stability_seconds)

    files: list[OracleSourceFile] = []
    for year in years:
        path = root / f"{year}{SOURCE_FILE_SUFFIX}"
        if year not in first_stats or not path.is_file():
            continue
        stat = path.stat()
        if first_stats[year] != (stat.st_size, stat.st_mtime_ns):
            issues.append(f"file_changed_during_check:{year}")
            continue
        if stat.st_size < minimum_bytes:
            issues.append(f"placeholder_or_incomplete:{year}")
        files.append(
            OracleSourceFile(
                year=year,
                path=str(path),
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            )
        )

    current = next((item for item in files if item.year == checked_at.year), None)
    if current and checked_at - current.modified_at > maximum_current_year_age:
        issues.append("current_year_file_stale")
    current_path = root / f"{checked_at.year}{SOURCE_FILE_SUFFIX}"
    maximum_match_at = _maximum_match_datetime(current_path) if current else None
    if current and maximum_match_at is None:
        issues.append("current_year_date_missing")
    elif (
        current
        and maximum_match_at is not None
        and maximum_match_at.year != checked_at.year
    ):
        issues.append("current_season_rows_missing")

    return OracleSourceReadiness(
        source_directory=str(requested_root),
        source_is_symlink=requested_root.is_symlink(),
        required_years=years,
        checked_at=checked_at,
        files=tuple(files),
        current_year_max_match_at=maximum_match_at,
        issues=tuple(dict.fromkeys(issues)),
        snapshot_id=snapshot_id,
    )


def require_oracle_source_ready(**kwargs) -> OracleSourceReadiness:
    report = inspect_oracle_source(**kwargs)
    report.raise_if_unready()
    return report


def _maximum_match_datetime(path: Path) -> datetime | None:
    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            rows = csv.DictReader(source)
            if "date" not in (rows.fieldnames or ()):
                return None
            maximum: datetime | None = None
            for row in rows:
                value = str(row.get("date") or "").strip()
                if not value:
                    continue
                try:
                    parsed = datetime.fromisoformat(value)
                except ValueError:
                    continue
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                parsed = parsed.astimezone(UTC)
                maximum = parsed if maximum is None else max(maximum, parsed)
            return maximum
    except OSError:
        return None


def _current_source_snapshot_id(root: Path) -> str | None:
    try:
        value = json.loads(
            (root / SOURCE_CURRENT_POINTER).read_text(encoding="utf-8")
        ).get("snapshot_id")
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    snapshot_id = str(value or "")
    return snapshot_id if re.fullmatch(r"source-[a-f0-9]{24}", snapshot_id) else None
