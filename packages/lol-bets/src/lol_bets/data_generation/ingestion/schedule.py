"""
Schedule Data Ingestion.

Fetches upcoming League of Legends matches from the PandaScore API
(https://pandascore.co).  Designed for testability (DI), observability
(structured logging) and robust error handling.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import requests
from dateutil import parser
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
from oracle_bets_core.io_utils import safe_store_df_as_parquet
from oracle_bets_core.logger import LOG_TOPIC, instantiate_logger  # your helper
from oracle_bets_core.paths import SCHEDULE
from oracle_bets_core.pd import pd

if TYPE_CHECKING:
    from collections.abc import Iterator

schedule_logger = instantiate_logger(LOG_TOPIC.SCHEDULE_GENERATION)

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
PANDASCORE_MATCH_URL = "https://api.pandascore.co/matches/{match_id}"
ACCEPT_JSON_HEADER: dict[str, str] = {"Accept": "application/json"}
DEFAULT_PER_PAGE = 100
SCHEDULE_COLUMNS: tuple[str, ...] = (
    "provider",
    "provider_match_id",
    "match_key",
    "fixture_version",
    "league",
    "serie",
    "serie_id",
    "tournament",
    "tournament_id",
    "team_a",
    "team_b",
    "team_a_id",
    "team_b_id",
    "team_a_lineup_json",
    "team_b_lineup_json",
    "lineup_source",
    "lineup_observed_at",
    "lineup_refresh_error",
    "start_utc",
    "best_of",
    "status",
    "market_query",
    "discord_label",
)
LEGACY_COLUMN_RENAMES = {
    "match_id": "provider_match_id",
    "Blue": "team_a",
    "Red": "team_b",
    "Start (UTC)": "start_utc",
    "Best Of": "best_of",
}

load_dotenv()


def _clean_text(value: Any) -> str:
    """Normalize API text into a compact parseable string."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def _match_key(match_id: Any, team_a: str, team_b: str, start_utc: Any) -> str:
    if match_id not in (None, ""):
        return f"pandascore:{match_id}"
    raw = "|".join([team_a.casefold(), team_b.casefold(), str(start_utc)])
    slug = re.sub(r"[^a-z0-9]+", "-", raw.casefold()).strip("-")
    return f"pandascore:{slug}"


def fixture_version(row: Any) -> str:
    """Hash the provider facts that can invalidate a prediction or approval."""
    payload = {
        key: _stable_fixture_value(row.get(key))
        for key in (
            "provider_match_id",
            "league",
            "serie_id",
            "tournament_id",
            "team_a_id",
            "team_b_id",
            "team_a",
            "team_b",
            "start_utc",
            "best_of",
            "status",
            "team_a_lineup_json",
            "team_b_lineup_json",
        )
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"fixture-{hashlib.sha256(encoded.encode()).hexdigest()[:20]}"


def _stable_fixture_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, dt.datetime | pd.Timestamp):
        parsed = pd.Timestamp(value)
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize(dt.UTC)
        return cast("pd.Timestamp", parsed.tz_convert(dt.UTC)).isoformat()
    return str(value).strip()


def _market_query(league: str, team_a: str, team_b: str) -> str:
    return " ".join(part for part in (league, team_a, team_b) if part)


def _discord_label(league: str, team_a: str, team_b: str, best_of: Any) -> str:
    series = "BO?" if pd.isna(best_of) else f"BO{best_of}"
    matchup = " vs ".join(part or "TBD" for part in (team_a, team_b))
    return f"{league or 'Unknown league'} | {matchup} | {series}"


def _lineup_json(team: dict[str, Any]) -> str:
    """Preserve provider player IDs, names, and roles without guessing."""
    players = team.get("players")
    if not isinstance(players, list):
        return "[]"
    lineup = []
    for player in players:
        if not isinstance(player, dict):
            continue
        name = _clean_text(player.get("name"))
        role = _clean_text(player.get("role") or player.get("position"))
        if not name or not role:
            continue
        lineup.append(
            {
                "provider_player_id": _clean_text(player.get("id")),
                "name": name,
                "role": role,
            }
        )
    return json.dumps(
        sorted(lineup, key=lambda item: (item["role"], item["name"])),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _normalize_lineup_json(value: Any) -> str:
    if value is None or (not isinstance(value, (list, dict)) and bool(pd.isna(value))):
        return "[]"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    text = _clean_text(value)
    return "[]" if text in {"", "<NA>", "nan"} else text


def normalize_schedule_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a stable snake_case schedule frame, including legacy file support."""
    if df.empty:
        return pd.DataFrame(columns=pd.Index(SCHEDULE_COLUMNS))

    out = df.rename(columns=LEGACY_COLUMN_RENAMES).copy()
    if "provider" not in out.columns:
        out["provider"] = "pandascore"
    for col in SCHEDULE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    text_cols = [
        "league",
        "serie",
        "serie_id",
        "tournament",
        "tournament_id",
        "team_a",
        "team_b",
        "status",
        "lineup_source",
        "lineup_refresh_error",
    ]
    for col in text_cols:
        out[col] = out[col].map(_clean_text)

    out["provider_match_id"] = out["provider_match_id"].map(_clean_text)
    out["team_a_id"] = out["team_a_id"].map(_clean_text)
    out["team_b_id"] = out["team_b_id"].map(_clean_text)
    for col in ("team_a_lineup_json", "team_b_lineup_json"):
        out[col] = out[col].map(_normalize_lineup_json)
    out["lineup_observed_at"] = pd.to_datetime(
        out["lineup_observed_at"], errors="coerce", utc=True
    )
    out["start_utc"] = pd.to_datetime(out["start_utc"], errors="coerce", utc=True)
    out["best_of"] = pd.to_numeric(out["best_of"], errors="coerce").astype("Int64")
    out["match_key"] = [
        _match_key(match_id, team_a, team_b, start_utc)
        for match_id, team_a, team_b, start_utc in zip(
            out["provider_match_id"],
            out["team_a"],
            out["team_b"],
            out["start_utc"],
            strict=False,
        )
    ]
    out["market_query"] = [
        _market_query(league, team_a, team_b)
        for league, team_a, team_b in zip(
            out["league"], out["team_a"], out["team_b"], strict=False
        )
    ]
    out["discord_label"] = [
        _discord_label(league, team_a, team_b, best_of)
        for league, team_a, team_b, best_of in zip(
            out["league"],
            out["team_a"],
            out["team_b"],
            out["best_of"],
            strict=False,
        )
    ]
    out["fixture_version"] = [fixture_version(row) for _, row in out.iterrows()]

    out = out[list(SCHEDULE_COLUMNS)]
    out = out.sort_values(["start_utc", "league", "team_a"], kind="mergesort")
    return out.reset_index(drop=True)


def validate_duplicate_fixture_agreement(df: pd.DataFrame) -> None:
    """Reject duplicate provider fixtures whose identity facts disagree."""
    if df.empty or "match_key" not in df.columns:
        return
    duplicate_rows = df[df.duplicated(subset=["match_key"], keep=False)]
    fact_columns = (
        "provider",
        "provider_match_id",
        "league",
        "team_a",
        "team_b",
        "team_a_id",
        "team_b_id",
        "start_utc",
        "best_of",
    )
    for match_key, group in duplicate_rows.groupby("match_key", observed=True):
        normalized = group[list(fact_columns)].astype(str).drop_duplicates()
        if len(normalized) > 1:
            msg = (
                f"Schedule contains conflicting duplicate fixture {match_key}; "
                "provider ID, teams, competition, time, and format must agree."
            )
            raise DataValidationError(msg)


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
            parsed = self._parse_and_filter_matches(matches, start_dt, end_dt)
            if parsed.empty:
                continue
            schedule_df = self._append_to_schedule(schedule_df, parsed)
            if self._should_stop_fetching(parsed, end_dt):
                break

        schedule_df = normalize_schedule_frame(schedule_df)

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
            team_a = opponents[0].get("opponent", {}) if opponents else {}
            team_b_payload = (
                opponents[1].get("opponent", {}) if len(opponents) > 1 else {}
            )
            league = _clean_text(m.get("league", {}).get("name"))
            team_a_name = _clean_text(team_a.get("name"))
            team_b_name = _clean_text(team_b_payload.get("name"))
            match_id = m.get("id")
            best_of = m.get("number_of_games")
            data.append(
                {
                    "provider": "pandascore",
                    "provider_match_id": match_id,
                    "league": league,
                    "serie": m.get("serie", {}).get("full_name")
                    or m.get("serie", {}).get("name"),
                    "serie_id": m.get("serie", {}).get("id"),
                    "tournament": m.get("tournament", {}).get("name"),
                    "tournament_id": m.get("tournament", {}).get("id"),
                    "team_a": team_a_name,
                    "team_b": team_b_name,
                    "team_a_id": team_a.get("id"),
                    "team_b_id": team_b_payload.get("id"),
                    "team_a_lineup_json": _lineup_json(team_a),
                    "team_b_lineup_json": _lineup_json(team_b_payload),
                    "lineup_source": "pandascore_upcoming_match",
                    "lineup_observed_at": dt.datetime.now(dt.UTC),
                    "lineup_refresh_error": "",
                    "start_utc": scheduled_at,
                    "best_of": best_of,
                    "status": m.get("status"),
                    "match_key": _match_key(
                        match_id, team_a_name, team_b_name, scheduled_at
                    ),
                    "market_query": _market_query(league, team_a_name, team_b_name),
                    "discord_label": _discord_label(
                        league, team_a_name, team_b_name, best_of
                    ),
                }
            )
        if not data:
            # A page of only TBD/unscheduled matches is normal near the end of
            # the upcoming feed; it must not abort the whole fetch.
            schedule_logger.warning(
                "API page contained no parsable matches (%d raw entries); skipping.",
                len(matches),
            )
            return pd.DataFrame()
        parsed = normalize_schedule_frame(pd.DataFrame(data))
        validate_duplicate_fixture_agreement(parsed)
        return parsed

    @staticmethod
    def filter_by_league(df: pd.DataFrame, leagues: str) -> pd.DataFrame:
        """Case-insensitive filter on the “league” column."""
        leagues_set = {lg.strip().lower() for lg in leagues.split(",")}
        out = df[df["league"].fillna("").str.lower().isin(leagues_set)]
        schedule_logger.debug("Leagues filter → %s rows", len(out))
        return out

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _parse_datetime(value: str | dt.datetime) -> dt.datetime:
        dt_obj = parser.isoparse(value) if isinstance(value, str) else value
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=dt.UTC)
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
    ) -> pd.DataFrame:
        games_df = self._parse_matches_response(matches)
        if games_df.empty:
            return games_df
        # normalize_schedule_frame already parsed start_utc to UTC datetimes;
        games_df["start_utc"] = pd.to_datetime(
            games_df["start_utc"], errors="coerce", utc=True
        )
        games_df = games_df.dropna(subset=["start_utc"])
        mask = (games_df["start_utc"] >= start_dt) & (games_df["start_utc"] <= end_dt)
        return games_df.loc[mask]

    @staticmethod
    def _append_to_schedule(curr: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        validate_duplicate_fixture_agreement(curr)
        validate_duplicate_fixture_agreement(new)
        combined = pd.concat([curr, new], ignore_index=True)
        combined = normalize_schedule_frame(combined)
        return combined.drop_duplicates(subset=["match_key"], keep="last")

    @staticmethod
    def _should_stop_fetching(df: pd.DataFrame, end_dt: dt.datetime) -> bool:
        return df["start_utc"].min() > end_dt


@dataclass(slots=True)
class PandaScoreLineupRefresher:
    """Refresh exact fixture and expected-lineup facts from match detail."""

    api_key: str | None = None
    session: requests.Session = field(default_factory=requests.Session)
    headers: dict[str, str] = field(default_factory=lambda: ACCEPT_JSON_HEADER.copy())
    match_url: str = PANDASCORE_MATCH_URL
    timeout_seconds: float = 10.0
    logger: Any = schedule_logger

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("PANDASCORE_API_KEY")
        if not self.api_key:
            raise ScheduleError(
                "PandaScore API key missing for expected-lineup refresh."
            )
        self.headers = {
            **self.headers,
            "Authorization": f"Bearer {self.api_key}",
        }

    def refresh(
        self,
        schedule: pd.DataFrame,
        *,
        observed_at: dt.datetime | None = None,
    ) -> pd.DataFrame:
        """Refresh each fixture independently; preserve explicit per-row errors."""
        refreshed = normalize_schedule_frame(schedule)
        observation = (observed_at or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
        authoritative_fields = (
            "league",
            "serie",
            "serie_id",
            "tournament",
            "tournament_id",
            "team_a",
            "team_b",
            "team_a_id",
            "team_b_id",
            "team_a_lineup_json",
            "team_b_lineup_json",
            "start_utc",
            "best_of",
            "status",
        )
        detail_access_error: str | None = None
        for index, row in refreshed.iterrows():
            match_id = _clean_text(row.get("provider_match_id"))
            if not match_id:
                refreshed.at[index, "lineup_refresh_error"] = (
                    "provider_match_id_missing"
                )
                continue
            if detail_access_error is not None:
                refreshed.at[index, "lineup_refresh_error"] = detail_access_error
                continue
            try:
                response = self.session.get(
                    self.match_url.format(match_id=match_id),
                    headers=self.headers,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                detail_row = _validated_match_detail(response.json(), match_id)
                for field_name in authoritative_fields:
                    refreshed.at[index, field_name] = detail_row[field_name]
                refreshed.at[index, "lineup_source"] = "pandascore_match_detail"
                refreshed.at[index, "lineup_observed_at"] = observation
                refreshed.at[index, "lineup_refresh_error"] = ""
            except requests.HTTPError as exc:
                status = getattr(exc.response, "status_code", None)
                if status in {401, 403}:
                    detail_access_error = f"match_detail_http_{status}"
                    refreshed.at[index, "lineup_refresh_error"] = detail_access_error
                    self.logger.warning(
                        "PandaScore match-detail lineup refresh disabled for this "
                        "run after HTTP %s; embedded schedule lineups retained. "
                        "Check API-key and plan permissions.",
                        status,
                    )
                    continue
                error = f"HTTPError: {' '.join(str(exc).split())[:200]}"
                refreshed.at[index, "lineup_refresh_error"] = error
                self.logger.warning(
                    "Expected-lineup refresh failed for match %s: %s",
                    match_id,
                    error,
                )
            except (
                DataValidationError,
                requests.RequestException,
                TypeError,
                ValueError,
            ) as exc:
                error = f"{type(exc).__name__}: {' '.join(str(exc).split())[:200]}"
                refreshed.at[index, "lineup_refresh_error"] = error
                self.logger.warning(
                    "Expected-lineup refresh failed for match %s: %s",
                    match_id,
                    error,
                )
        return normalize_schedule_frame(refreshed)


def _validated_match_detail(payload: Any, match_id: str) -> pd.Series:
    if not isinstance(payload, dict):
        raise DataValidationError("match detail response must be an object")
    detail = PandaScoreSchedule._parse_matches_response([payload])
    if detail.empty:
        raise DataValidationError("match detail contained no fixture")
    detail_row = detail.iloc[0]
    if _clean_text(detail_row["provider_match_id"]) != match_id:
        raise DataValidationError("match detail ID disagrees with request")
    return detail_row


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
        if df.empty:
            # Do not clobber a previously stored schedule with an empty frame:
            # the bot's !schedule command would go blank until the next fetch.
            schedule_logger.warning(
                "Fetched 0 upcoming matches; keeping existing schedule at %s.",
                save_path,
            )
        else:
            safe_store_df_as_parquet(df, Path(save_path), [schedule_logger])
            schedule_logger.info("Schedule stored to %s (%s rows).", save_path, len(df))
    else:
        schedule_logger.info("Fetched %s upcoming matches (no file output).", len(df))
    if leagues:
        return PandaScoreSchedule.filter_by_league(df, leagues).reset_index(drop=True)
    return df.reset_index(drop=True)
