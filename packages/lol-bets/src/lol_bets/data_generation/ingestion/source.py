"""Read-only readiness checks for the locally synced Oracle's Elixir source."""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from oracle_bets_core.paths import RAW_DATA

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

SOURCE_FILE_SUFFIX = "_LoL_esports_match_data_from_OraclesElixir.csv"
DEFAULT_MAX_AGE = timedelta(hours=48)
DEFAULT_MINIMUM_BYTES = 1_024


class OracleSourceReadinessError(RuntimeError):
    """Raised when local source files cannot safely support a rebuild."""


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


def inspect_oracle_source(  # noqa: PLR0912
    source_directory: str | Path = RAW_DATA.parent / "oracles_elixir",
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
    root = Path(source_directory)
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
    kwargs.setdefault("require_symlink", True)
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
