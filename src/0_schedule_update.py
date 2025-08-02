"""
schedule_fetcher.py – helper around `PandaScoreSchedule`

Used by the Discord bot (or any other caller) to keep an up-to-date Parquet
schedule _and/or_ an in-memory DataFrame.

Key API
-------
* **get_upcoming_matches** – pure function, returns `pd.DataFrame`.
* **main** – convenience wrapper you can import and call; not a CLI parser.

No command-line interface is provided/needed.
"""

from __future__ import annotations

import datetime as dt
import sys
from typing import TYPE_CHECKING, Final

from dateutil import parser
from dotenv import load_dotenv

from ingestion.schedule import PandaScoreSchedule
from utils.io_utils import safe_store_df_as_parquet
from utils.logger import LOG_TOPIC, instantiate_logger
from utils.paths import SCHEDULE

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

# --------------------------------------------------------------------------- #
# Config & logging
# --------------------------------------------------------------------------- #
load_dotenv()

LOG = instantiate_logger(LOG_TOPIC.SCHEDULE_GENERATION)

TIME_FMT: Final = "%Y-%m-%dT%H:%M:%SZ"
MAX_RANGE_DAYS: Final = 7


class ScheduleFetchError(Exception):
    """Custom error for schedule fetching issues."""


def _to_utc(dt_or_str: dt.datetime | str | None) -> dt.datetime:
    """Return a timezone-aware UTC datetime from str / datetime / None."""
    if dt_or_str is None:
        return dt.datetime.now(dt.UTC)
    if isinstance(dt_or_str, str):
        dt_or_str = parser.isoparse(dt_or_str)
    return dt_or_str.astimezone(dt.UTC)


# --------------------------------------------------------------------------- #
# Public helper
# --------------------------------------------------------------------------- #
def get_upcoming_matches(
    start_datetime: dt.datetime | None = None,
    window_days: int = MAX_RANGE_DAYS,
    leagues: str | None = None,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Fetch upcoming matches starting from *start_datetime* (UTC).

    Parameters
    ----------
    start_datetime
        Beginning of the window; defaults to current UTC time.
    window_days
        Horizon length in days.
    leagues
        Optional comma-separated league filter.
    save_path
        Optional Parquet output path (``None`` → no persistence).

    Returns
    -------
    pd.DataFrame
        Normalised schedule (may be empty).

    """
    start_dt_utc = _to_utc(start_datetime)
    end_dt_utc = start_dt_utc + dt.timedelta(days=window_days)

    schedule = PandaScoreSchedule()
    df = schedule.get_schedule(
        start_datetime=start_dt_utc.isoformat(),
        end_datetime=end_dt_utc.isoformat(),
        max_day_range=window_days,
        time_format=TIME_FMT,
        leagues=leagues,
    )

    if save_path:
        safe_store_df_as_parquet(df, save_path, [LOG])
        LOG.info("Schedule stored to %s (%s rows).", save_path, len(df))
    else:
        LOG.info("Fetched %s upcoming matches (no file output).", len(df))

    return df


# --------------------------------------------------------------------------- #
# Convenience wrapper for app code (non-CLI)
# --------------------------------------------------------------------------- #
def main(
    start_datetime: dt.datetime | None = None,
    window_days: int = MAX_RANGE_DAYS,
    leagues: str | None = None,
    persist: bool = True,
) -> pd.DataFrame:
    """
    Fetch the schedule and optionally persist it to ``utils.paths.SCHEDULE``.

    Parameters
    ----------
    start_datetime
        Beginning of the window; defaults to current UTC time.
    window_days
        Days ahead to look.
    leagues
        Comma-separated league list.
    persist
        If ``True`` stores the DataFrame to ``SCHEDULE``.

    """
    try:
        save_path: str | Path | None = SCHEDULE if persist else None
        return get_upcoming_matches(
            start_datetime=start_datetime,
            window_days=window_days,
            leagues=leagues,
            save_path=save_path,
        )
    except Exception:
        LOG.exception("Schedule fetch failed")
        raise


if __name__ == "__main__":  # pragma: no cover
    try:
        main(start_datetime="2025-08-01T00:00:00Z", window_days=MAX_RANGE_DAYS)
    except ScheduleFetchError:
        LOG.exception("Failed to fetch schedule. Exiting with error.")
        sys.exit(1)
