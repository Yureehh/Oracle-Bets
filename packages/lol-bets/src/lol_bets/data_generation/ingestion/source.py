"""Refresh and validate the public Oracle's Elixir CSV source."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from oracle_bets_core.paths import RAW_DATA

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

SOURCE_FILE_SUFFIX = "_LoL_esports_match_data_from_OraclesElixir.csv"
SOURCE_CACHE_DIRECTORY = RAW_DATA.parent / "oracles_elixir_cache"
SOURCE_MANIFEST = "source_manifest.json"
PUBLIC_DRIVE_FILE_IDS = {
    2024: "1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN",  # pragma: allowlist secret
    2025: "1v6LRphp2kYciU4SXp0PCjEMuev1bDejc",  # pragma: allowlist secret
    2026: "1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm",  # pragma: allowlist secret
}
PUBLIC_DOWNLOAD_URL = "https://drive.usercontent.google.com/download"
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
    configured = os.getenv("ORACLES_ELIXIR_LOCAL_DIR")
    return Path(configured).expanduser() if configured else SOURCE_CACHE_DIRECTORY


def refresh_oracle_source(
    source_directory: str | Path | None = None,
    *,
    required_years: Sequence[int] | None = None,
    session: requests.Session | None = None,
    now: datetime | None = None,
) -> OracleSourceRefresh:
    """Atomically refresh required public Drive files into the managed cache."""
    root = Path(source_directory) if source_directory else oracle_source_directory()
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
    staged: list[tuple[Path, Path, dict[str, object], datetime]] = []
    try:
        staged.extend(
            _download_source_file(
                client,
                root,
                year,
                PUBLIC_DRIVE_FILE_IDS[year],
                downloaded_at,
            )
            for year in years
        )
        for temporary, destination, _metadata, remote_modified_at in staged:
            temporary.replace(destination)
            timestamp = remote_modified_at.timestamp()
            os.utime(destination, (timestamp, timestamp))
        files = tuple(metadata for _, _, metadata, _ in staged)
        _write_source_manifest(root / SOURCE_MANIFEST, downloaded_at, files)
    except (OSError, requests.RequestException, ValueError) as exc:
        raise OracleSourceRefreshError(
            f"Oracle's Elixir refresh failed: {exc}"
        ) from exc
    finally:
        for temporary, *_ in staged:
            temporary.unlink(missing_ok=True)

    return OracleSourceRefresh(str(root), downloaded_at, files)


def _download_source_file(
    session: requests.Session,
    root: Path,
    year: int,
    file_id: str,
    downloaded_at: datetime,
) -> tuple[Path, Path, dict[str, object], datetime]:
    filename = f"{year}{SOURCE_FILE_SUFFIX}"
    response = session.get(
        PUBLIC_DOWNLOAD_URL,
        params={"id": file_id, "export": "download", "confirm": "t"},
        stream=True,
        timeout=(10, 180),
    )
    try:
        response.raise_for_status()
        disposition = response.headers.get("Content-Disposition", "")
        if filename not in disposition:
            raise ValueError(f"unexpected download response for {year}")
        remote_modified_at = _remote_modified_at(response, downloaded_at)
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
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
                _validate_complete_download(temporary, year, byte_count)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        metadata = {
            "year": year,
            "drive_file_id": file_id,
            "filename": filename,
            "size_bytes": byte_count,
            "sha256": digest.hexdigest(),
            "remote_modified_at": remote_modified_at.isoformat(),
        }
        return temporary, root / filename, metadata, remote_modified_at
    finally:
        response.close()


def _remote_modified_at(response: requests.Response, fallback: datetime) -> datetime:
    value = response.headers.get("Last-Modified")
    if not value:
        return fallback
    parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _validate_complete_download(path: Path, year: int, byte_count: int) -> None:
    if byte_count < DEFAULT_MINIMUM_BYTES:
        raise ValueError(f"downloaded file for {year} is incomplete")
    with path.open(encoding="utf-8-sig", newline="") as source:
        fields = set(next(csv.reader(source), ()))
    if not {"gameid", "date"}.issubset(fields):
        raise ValueError(f"downloaded file for {year} is not an Oracle's Elixir CSV")


def _write_source_manifest(
    path: Path,
    downloaded_at: datetime,
    files: tuple[dict[str, object], ...],
) -> None:
    payload = {
        "schema_version": 1,
        "provider": "oracle_elixir_public_google_drive",
        "downloaded_at": downloaded_at.isoformat(),
        "files": list(files),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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
    root = Path(source_directory) if source_directory else oracle_source_directory()
    if maximum_current_year_age <= timedelta(0):
        raise ValueError("maximum_current_year_age must be positive")
    if minimum_bytes <= 0 or stability_seconds < 0:
        raise ValueError("source size and stability thresholds are invalid")
    if not root.is_dir():
        return OracleSourceReadiness(
            str(root),
            root.is_symlink(),
            years,
            checked_at,
            (),
            None,
            ("source_directory_missing",),
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
    elif current and maximum_match_at.year != checked_at.year:
        issues.append("current_season_rows_missing")

    return OracleSourceReadiness(
        source_directory=str(root),
        source_is_symlink=root.is_symlink(),
        required_years=years,
        checked_at=checked_at,
        files=tuple(files),
        current_year_max_match_at=maximum_match_at,
        issues=tuple(dict.fromkeys(issues)),
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
