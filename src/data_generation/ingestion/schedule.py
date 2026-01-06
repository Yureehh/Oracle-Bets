"""
Schedule Data Ingestion.

Fetches upcoming League of Legends matches from the PandaScore API
(https://pandascore.co).  Designed for testability (DI), observability
(structured logging) and robust error handling.
"""

from __future__ import annotations

import datetime as dt
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import requests
from dateutil import parser
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
from utils.io_utils import safe_store_df_as_parquet
from utils.logger import LOG_TOPIC, instantiate_logger  # your helper
from utils.paths import SCHEDULE
from utils.pd import pd

if TYPE_CHECKING:
    from collections.abc import Iterator

schedule_logger = instantiate_logger(LOG_TOPIC.data_generation.schedule_GENERATION)

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class ScheduleError(Exception):
    """Base class for schedule-related errors."""


class PandaScoreAPIError(ScheduleError):
    """Raised when the PandaScore API request ultimately fails."""


class DataValidationError(ScheduleError):
    """Raised when the API payload cannot be parsed into the expected schema."""


# --------------------------------------------------------------------------- #
# Dependency-injection hooks / Strategies
# --------------------------------------------------------------------------- #


class Sleeper(Protocol):
    """Callable sleep strategy – useful for test fakes."""

    def __call__(self, seconds: float, /) -> None: ...


def default_sleep(seconds: float) -> None:  # pragma: no cover
    time.sleep(seconds)


# --------------------------------------------------------------------------- #
# Constants & helpers
# --------------------------------------------------------------------------- #
PANDASCORE_BASE_URL = "https://api.pandascore.co/lol/matches/upcoming"
ACCEPT_JSON_HEADER: dict[str, str] = {"Accept": "application/json"}
DEFAULT_PER_PAGE = 100
START_DATETIME_COLUMN = "Start (UTC)"
DEFAULT_REFRESH_HOURS = 48.0

load_dotenv()


def _schedule_is_stale(path: str | os.PathLike, max_age_hours: float) -> bool:
    try:
        mtime = Path(path).stat().st_mtime
    except FileNotFoundError:
        return True
    age_hours = (time.time() - mtime) / 3600.0
    return age_hours >= max_age_hours


# --------------------------------------------------------------------------- #
# Main dataclass
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class PandaScoreSchedule:
    """
    Ingests schedule data from PandaScore and returns a pandas DataFrame.

    Parameters
    ----------
    api_key:
        PandaScore bearer token.  Falls back to the ``PANDASCORE_API_KEY`` env
        variable if not supplied.
    session:
        Injected HTTP client; defaults to a fresh ``requests.Session``.
    headers:
        Extra HTTP headers merged with ``ACCEPT_JSON_HEADER``.
    base_url:
        Endpoint for fetching upcoming matches.
    per_page:
        API pagination size.
    max_retries:
        Retry count before giving up and raising ``PandaScoreAPIError``.
    sleep_fn:
        Strategy for waiting between retries/pages (DI for unit tests).
    logger:
        Inject a custom logger if you don’t want to use ``schedule_logger``.

    """

    api_key: str | None = None
    session: requests.Session = field(default_factory=requests.Session)
    headers: dict[str, str] = field(default_factory=lambda: ACCEPT_JSON_HEADER.copy())
    base_url: str = PANDASCORE_BASE_URL
    per_page: int = DEFAULT_PER_PAGE
    max_retries: int = 3
    sleep_fn: Sleeper = default_sleep
    logger: Any = schedule_logger  # keep type flexible for custom wrappers

    # --------------------------------------------------------------------- #
    # Lifecycle
    # --------------------------------------------------------------------- #
    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.getenv("PANDASCORE_API_KEY")
        if not self.api_key:
            msg = (
                "PandaScore API key missing – supply via constructor or "
                "PANDASCORE_API_KEY environment variable."
            )
            raise ScheduleError(msg)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get_schedule(
        self,
        start_datetime: str | dt.datetime,
        end_datetime: str | dt.datetime | None = None,
        max_day_range: int = 14,
        time_format: str | None = "%Y-%m-%dT%H:%M:%S%z",
        leagues: str | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame of upcoming matches within the given time window."""
        start_dt = self._parse_datetime(start_datetime)
        if end_datetime is None:
            end_datetime = start_dt + dt.timedelta(days=max_day_range)
        start_dt, end_dt = self._validate_and_parse_dates(
            start_dt, end_datetime, max_day_range
        )

        schedule_df = pd.DataFrame()
        for matches in self._fetch_all_matches():
            parsed = self._parse_and_filter_matches(
                matches, start_dt, end_dt, time_format
            )
            if parsed.empty:
                continue
            schedule_df = self._append_to_schedule(schedule_df, parsed)
            if self._should_stop_fetching(parsed, end_dt):
                break

        if leagues:
            schedule_df = self.filter_by_league(schedule_df, leagues)

        self.logger.info("Fetched %s unique matches.", len(schedule_df))
        return schedule_df.reset_index(drop=True)

    # --------------------------------------------------------------------- #
    # I/O helpers
    # --------------------------------------------------------------------- #
    def _fetch_matches(self, page: int) -> list[dict[str, Any]]:
        """Fetch a single page from PandaScore with retry/back-off."""
        params = {
            "sort": "",
            "page": page,
            "per_page": self.per_page,
            "token": self.api_key,
        }
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(
                    self.base_url,
                    headers=self.headers,
                    params=params,
                    timeout=10,
                )
                resp.raise_for_status()
                self.logger.debug("Page %s OK – %s bytes", page, len(resp.content))
                return resp.json()
            except requests.RequestException as exc:
                self.logger.warning(
                    "API error (attempt %s/%s) – %s", attempt, self.max_retries, exc
                )
                if attempt == self.max_retries:
                    msg = "Max retries exceeded"
                    raise PandaScoreAPIError(msg) from exc
                self.sleep_fn(2**attempt)  # exponential back-off
        return []  # unreachable but satisfies type checker

    def _fetch_all_matches(self) -> Iterator[list[dict[str, Any]]]:
        """Generator yielding lists of match dicts page-by-page."""
        page = 1
        while True:
            matches = self._fetch_matches(page)
            if not matches:
                if page == 1:
                    self.logger.warning("No matches returned from PandaScore.")
                break
            yield matches
            self.logger.debug("Yielded %s matches from page %s", len(matches), page)
            page += 1
            self.sleep_fn(1.2)

    # --------------------------------------------------------------------- #
    # Parsing / validation
    # --------------------------------------------------------------------- #
    @staticmethod
    def _parse_matches_response(matches: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert raw JSON list into a normalized DataFrame."""
        data: list[dict[str, Any]] = []
        for m in matches:
            scheduled_at = m.get("scheduled_at")
            if not scheduled_at:
                continue
            opponents = m.get("opponents", [])
            blue = opponents[0].get("opponent", {}).get("name") if opponents else None
            red = (
                opponents[1].get("opponent", {}).get("name")
                if len(opponents) > 1
                else None
            )
            data.append(
                {
                    "match_id": m.get("id"),
                    "league": m.get("league", {}).get("name"),
                    "Blue": blue,
                    "Red": red,
                    START_DATETIME_COLUMN: scheduled_at,
                    "Best Of": m.get("number_of_games"),
                }
            )
        if not data:
            msg = "API payload contained no parsable matches."
            raise DataValidationError(msg)
        return pd.DataFrame(data)

    @staticmethod
    def filter_by_league(df: pd.DataFrame, leagues: str) -> pd.DataFrame:
        """Case-insensitive filter on the “league” column."""
        leagues_set = {lg.strip().lower() for lg in leagues.split(",")}
        out = df[df["league"].str.lower().isin(leagues_set)]
        schedule_logger.debug("Leagues filter → %s rows", len(out))
        return out

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _parse_datetime(value: str | dt.datetime) -> dt.datetime:
        dt_obj = parser.isoparse(value) if isinstance(value, str) else value
        return dt_obj.astimezone(dt.UTC)

    def _validate_and_parse_dates(
        self,
        start: str | dt.datetime,
        end: str | dt.datetime,
        max_range: int,
    ) -> tuple[dt.datetime, dt.datetime]:
        start_dt, end_dt = self._parse_datetime(start), self._parse_datetime(end)
        if start_dt > end_dt:
            msg = "start_datetime must be ≤ end_datetime."
            raise ScheduleError(msg)
        if (end_dt - start_dt).days > max_range:
            msg = f"Range exceeds {max_range} days."
            raise ScheduleError(msg)
        return start_dt, end_dt

    def _parse_and_filter_matches(
        self,
        matches: list[dict[str, Any]],
        start_dt: dt.datetime,
        end_dt: dt.datetime,
        time_format: str | None,
    ) -> pd.DataFrame:
        games_df = self._parse_matches_response(matches)
        raw_times = games_df[START_DATETIME_COLUMN]
        if time_format:
            parsed = pd.to_datetime(
                raw_times,
                format=time_format,
                errors="coerce",
                utc=True,
            )
            if parsed.isna().all():
                parsed = pd.to_datetime(raw_times, errors="coerce", utc=True)
        else:
            parsed = pd.to_datetime(raw_times, errors="coerce", utc=True)
        games_df[START_DATETIME_COLUMN] = parsed
        games_df = games_df.dropna(subset=[START_DATETIME_COLUMN])
        mask = (games_df[START_DATETIME_COLUMN] >= start_dt) & (
            games_df[START_DATETIME_COLUMN] <= end_dt
        )
        return games_df.loc[mask]

    @staticmethod
    def _append_to_schedule(curr: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        combined = pd.concat([curr, new], ignore_index=True)
        return combined.drop_duplicates(subset=["match_id"], keep="last")

    @staticmethod
    def _should_stop_fetching(df: pd.DataFrame, end_dt: dt.datetime) -> bool:
        return df[START_DATETIME_COLUMN].min() > end_dt

    @staticmethod
    def load_schedule(
        path: str | os.PathLike, leagues: str | None = None
    ) -> pd.DataFrame:
        """
        Load a previously saved schedule DataFrame from disk.

        Supports Parquet (.parquet/.pq/.parq) and CSV. Returns a DataFrame,
        optionally filtered by comma-separated league names (case-insensitive).
        """
        p = os.fspath(path)
        if not Path(p).exists():
            msg = f"Schedule file not found: {p}"
            raise FileNotFoundError(msg)

        df: pd.DataFrame | None = None

        # Try Parquet first (fastparquet -> pyarrow fallback)
        try:
            if p.lower().endswith((".parquet", ".pq", ".parq")):
                try:
                    df = pd.read_parquet(p, engine="fastparquet")
                except (ImportError, ValueError):
                    df = pd.read_parquet(p)
            else:
                # Try reading as Parquet anyway; if it fails, we'll try CSV
                try:
                    df = pd.read_parquet(p)
                except Exception:
                    df = None
            if df is None:
                df = pd.read_csv(p)
        except Exception as e:
            msg = f"Failed to load schedule file '{p}': {e}"
            raise ScheduleError(msg) from e

        if "league" not in df.columns:
            msg = "Schedule file missing required 'league' column."
            raise DataValidationError(msg)

        if leagues:
            df = PandaScoreSchedule.filter_by_league(df, leagues)

        schedule_logger.info("Loaded local schedule '%s' with %s rows.", p, len(df))
        return df.reset_index(drop=True)


def fetch_and_store_schedule(
    *,
    start_datetime: dt.datetime | None = None,
    window_days: int = 7,
    leagues: str | None = None,
    save_path: str | os.PathLike | None = SCHEDULE,
) -> pd.DataFrame:
    """
    Fetch upcoming matches, persist to disk, and return a DataFrame.
    """
    start_dt = start_datetime or dt.datetime.now(dt.UTC)
    schedule = PandaScoreSchedule()
    df = schedule.get_schedule(start_datetime=start_dt, max_day_range=window_days)
    if save_path is not None:
        safe_store_df_as_parquet(df, save_path, [schedule_logger])
        schedule_logger.info("Schedule stored to %s (%s rows).", save_path, len(df))
    else:
        schedule_logger.info("Fetched %s upcoming matches (no file output).", len(df))
    if leagues:
        return PandaScoreSchedule.filter_by_league(df, leagues).reset_index(drop=True)
    return df.reset_index(drop=True)


def get_or_update_schedule(
    *,
    leagues: str | None = None,
    window_days: int = 7,
    save_path: str | os.PathLike = SCHEDULE,
    max_age_hours: float | None = DEFAULT_REFRESH_HOURS,
    force_refresh: bool = True,
) -> pd.DataFrame:
    """
    Load the stored schedule; fetch and persist if forced, missing, invalid, or stale.
    """
    try:
        if force_refresh:
            schedule_logger.info("Force-refreshing schedule at %s.", save_path)
            return fetch_and_store_schedule(
                window_days=window_days, leagues=leagues, save_path=save_path
            )
        if max_age_hours is not None and _schedule_is_stale(save_path, max_age_hours):
            schedule_logger.info(
                "Schedule at %s is stale (>%s hours); refreshing.",
                save_path,
                max_age_hours,
            )
            return fetch_and_store_schedule(
                window_days=window_days, leagues=leagues, save_path=save_path
            )
        return PandaScoreSchedule.load_schedule(save_path, leagues)
    except (FileNotFoundError, ScheduleError, DataValidationError):
        return fetch_and_store_schedule(
            window_days=window_days, leagues=leagues, save_path=save_path
        )


if __name__ == "__main__":
    # Example usage
    schedule = PandaScoreSchedule()
    schedule_df = schedule.get_schedule(start_datetime=dt.datetime.now(dt.UTC))
    schedule_logger.info("Schedule DataFrame:\n%s", schedule_df.head())
