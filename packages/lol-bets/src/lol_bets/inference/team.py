"""
Team model: fast, safe access to team & player snapshots.

- Caches parquet reads (single-process) to avoid repeated disk I/O.
- Case-insensitive matching with .casefold() (better than .lower() for i18n).
- Partial roster updates allowed; missing roles autocompleted from last roster.
- Strong validation + clear errors, but minimal noise in logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

from oracle_bets_core.logger import logger
from oracle_bets_core.paths import FLATTENED_PLAYERS, FLATTENED_TEAMS
from oracle_bets_core.pd import pd

from lol_bets.inference.team_resolver import TeamResolutionError, resolve_team_name

if TYPE_CHECKING:
    from collections.abc import Iterable

# ── small IO helper with engine fallback + caching ───────────────────────── #


@lru_cache(maxsize=4)
def _read_parquet_cached(path_str: str) -> pd.DataFrame:
    # sourcery skip: remove-unnecessary-cast
    path = str(path_str)
    try:
        return pd.read_parquet(path, engine="fastparquet")
    except (ImportError, ValueError):
        # fall back to pyarrow if available
        return pd.read_parquet(path)


def _require_columns(df: pd.DataFrame, required: Iterable[str], where: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = f"Missing columns in {where}: {missing}"
        raise ValueError(msg)


# ── Team ─────────────────────────────────────────────────────────────────── #

_EXPECTED_POS = ("top", "jng", "mid", "bot", "sup")
_VALID_SIDES = {"blue": "Blue", "red": "Red"}
_DEFAULT_GLICKO_PHI = 350.0
_DEFAULT_SKILL_SIGMA = 8.333


class InsufficientRosterHistoryError(ValueError):
    """Raised when a team lacks enough player history for safe inference."""


def _normalize_side(side: str) -> str:
    normalized = _VALID_SIDES.get(side.strip().casefold())
    if normalized is None:
        msg = "Side must be either 'Blue' or 'Red'."
        raise ValueError(msg)
    return normalized


def _days_ago(value, as_of: pd.Timestamp) -> float:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return float("nan")
    if getattr(timestamp, "tzinfo", None) is not None:
        timestamp = timestamp.tz_convert(None)
    return float(max(0, (as_of - timestamp.normalize()).days))


def _rating_uncertainty(row: pd.Series) -> float:
    values = []
    for column, baseline in (
        ("glicko2_phi", _DEFAULT_GLICKO_PHI),
        ("pl_sigma", _DEFAULT_SKILL_SIGMA),
        ("trueskill_sigma", _DEFAULT_SKILL_SIGMA),
    ):
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.notna(value):
            values.append(float(value) / baseline)
    return float(sum(values) / len(values)) if values else float("nan")


@dataclass
class Team:
    name: str
    side: str | None = None
    first_pick: bool | None = None
    as_of_date: object | None = None
    roster: dict[str, str | None] = field(
        default_factory=lambda: dict.fromkeys(_EXPECTED_POS)
    )

    # populated at init
    team_stats: pd.Series = field(init=False)
    player_stats: pd.DataFrame = field(init=False)
    established_roster: dict[str, str] = field(default_factory=dict, init=False)

    # cached tables
    _team_df: pd.DataFrame = field(init=False, repr=False)
    _player_df: pd.DataFrame = field(init=False, repr=False)
    _as_of: pd.Timestamp = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # load dataframes once (cached globally by lru_cache)
        self._team_df = _read_parquet_cached(str(FLATTENED_TEAMS))
        self._player_df = _read_parquet_cached(str(FLATTENED_PLAYERS))
        self._as_of = pd.to_datetime(cast("Any", self.as_of_date), errors="coerce")
        if pd.isna(self._as_of):
            self._as_of = pd.Timestamp.now()
        if getattr(self._as_of, "tzinfo", None) is not None:
            self._as_of = self._as_of.tz_convert(None)
        self._as_of = self._as_of.normalize()

        _require_columns(self._team_df, ["teamname"], "FLATTENED_TEAMS")
        _require_columns(
            self._player_df,
            ["teamname", "playername", "position", "date"],
            "FLATTENED_PLAYERS",
        )

        # set team_stats
        self.team_stats = self._lookup_team_row(self.name).copy()
        # Canonicalize the name so roster/player lookups use the Oracle's
        # Elixir teamname even when the caller passed an external alias.
        canonical_name = self.team_stats.get("teamname")
        if isinstance(canonical_name, str) and canonical_name:
            self.name = canonical_name
        if self.side:
            self.team_stats["side"] = _normalize_side(self.side)
            self.side = self.team_stats["side"]
        if self.first_pick is not None:
            self.team_stats["first_pick"] = int(bool(self.first_pick))

        # Complete roster from the latest known lineup, but retain how much of
        # the future lineup was supplied explicitly as an uncertainty signal.
        supplied_roles = {
            role
            for role, player in (self.roster or {}).items()
            if role in _EXPECTED_POS and isinstance(player, str) and player.strip()
        }
        try:
            last_roster = self._get_last_roster(self.name)
        except (ValueError, KeyError) as e:
            logger.debug("No last-roster fallback available: %s", e)
            last_roster = {}
        self.established_roster = dict(last_roster)
        filled = {**dict.fromkeys(_EXPECTED_POS), **(self.roster or {})}
        if any(v is None for v in filled.values()):
            for role in _EXPECTED_POS:
                if filled[role] is None and role in last_roster:
                    filled[role] = last_roster[role]

        missing_roles = [
            role
            for role, player in filled.items()
            if not isinstance(player, str) or not player.strip()
        ]
        if missing_roles:
            raise InsufficientRosterHistoryError(
                f"Insufficient historical roster coverage for '{self.name}': "
                f"missing {', '.join(missing_roles)}."
            )
        self.roster = filled
        self._validate_roster(strict=True)
        self.player_stats = self._lookup_players(self.roster)
        continuity = sum(
            str(self.roster[role]).casefold()
            == str(last_roster.get(role, "")).casefold()
            for role in _EXPECTED_POS
        ) / len(_EXPECTED_POS)
        known_fraction = len(supplied_roles) / len(_EXPECTED_POS)
        self.team_stats["roster_continuity"] = continuity
        self.team_stats["roster_uncertainty"] = max(
            1.0 - continuity, 1.0 - known_fraction
        )
        self.team_stats["days_since_last_game"] = _days_ago(
            self.team_stats.get("date"), self._as_of
        )
        self.team_stats["rating_uncertainty"] = _rating_uncertainty(self.team_stats)
        self.player_stats["days_since_last_game"] = self.player_stats["date"].map(
            lambda value: _days_ago(value, self._as_of)
        )
        self.player_stats["rating_uncertainty"] = self.player_stats.apply(
            _rating_uncertainty, axis=1
        )

    # ── lookups ──────────────────────────────────────────────────────────── #

    def _at_or_before_as_of(self, df: pd.DataFrame) -> pd.DataFrame:
        as_of = getattr(self, "_as_of", None)
        if "date" not in df or as_of is None:
            return df
        dates = pd.to_datetime(df["date"], errors="coerce")
        if getattr(dates.dt, "tz", None) is not None:
            dates = dates.dt.tz_convert(None)
        return df.loc[dates.dt.normalize() <= as_of]

    def _lookup_team_row(self, team_name: str) -> pd.Series:
        known_names = self._team_df["teamname"].dropna().astype(str).unique().tolist()
        resolved = resolve_team_name(team_name, known_names)
        if not resolved.ok:
            raise TeamResolutionError(resolved)
        if resolved.method != "exact":
            logger.info(
                "Resolved team '%s' -> '%s' via %s match.",
                team_name,
                resolved.resolved_name,
                resolved.method,
            )
        resolved_name = resolved.resolved_name
        if resolved_name is None:
            raise TeamResolutionError(resolved)
        key = resolved_name.casefold()
        rows = self._team_df[self._team_df["teamname"].str.casefold() == key]
        rows = self._at_or_before_as_of(rows)
        if rows.empty:  # defensive: resolver guarantees membership
            msg = f"Team '{team_name}' not found in teams table."
            raise ValueError(msg)
        if "date" in rows.columns:
            rows = rows.sort_values("date", ascending=False)
        return rows.iloc[0]

    def _get_last_roster(self, team_name: str) -> dict[str, str]:
        key = team_name.casefold()
        df = self._player_df[self._player_df["teamname"].str.casefold() == key]
        df = self._at_or_before_as_of(df)
        player_names = df["playername"].fillna("").astype(str).str.strip()
        df = df[player_names.ne("") & player_names.str.casefold().ne("nan")]
        if df.empty:
            msg = f"No player rows for team '{team_name}'."
            raise ValueError(msg)

        # latest per position
        latest = (
            df.sort_values(["position", "date"], ascending=[True, False])
            .groupby("position", observed=True)
            .first()
            .reset_index()
        )
        # ensure we have at least the expected roles
        missing = set(_EXPECTED_POS) - set(latest["position"].str.casefold())
        if missing:
            msg = f"Could not determine last roster for roles: {sorted(missing)}"
            raise ValueError(msg)

        # build canonical mapping role -> playername
        out: dict[str, str] = {}
        for _, row in latest.iterrows():
            role = str(row["position"]).casefold()
            name = str(row["playername"])
            if role in _EXPECTED_POS:
                out[role] = name
        return out

    def _lookup_players(self, roster: dict[str, str | None]) -> pd.DataFrame:
        # expect fully-populated roster already validated
        wanted = {role: (name or "") for role, name in roster.items()}
        wanted_lower = {role: name.casefold() for role, name in wanted.items()}
        requested_names = set(wanted_lower.values())
        team_key = self.name.casefold()

        df = self._player_df[
            self._player_df["playername"].str.casefold().isin(requested_names)
        ].copy()
        df = self._at_or_before_as_of(df)

        if df.empty:
            msg = f"No player stats found for roster of team '{self.name}'."
            raise ValueError(msg)

        df["_player_key"] = df["playername"].str.casefold()
        df["_team_match"] = df["teamname"].str.casefold().eq(team_key)

        # Prefer the latest row with the requested team. If an explicit future roster
        # contains a transferred player with no row for this team yet, fall back to
        # that player's latest global row.
        team_rows = (
            df[df["_team_match"]]
            .sort_values(["_player_key", "date"], ascending=[True, False])
            .drop_duplicates(subset=["_player_key"], keep="first")
        )
        found_team_rows = set(team_rows["_player_key"])
        fallback_rows = (
            df[~df["_player_key"].isin(found_team_rows)]
            .sort_values(["_player_key", "date"], ascending=[True, False])
            .drop_duplicates(subset=["_player_key"], keep="first")
        )
        df = pd.concat([team_rows, fallback_rows], ignore_index=True).drop(
            columns=["_player_key", "_team_match"]
        )

        # ensure we have one row per requested player
        have_lower = set(df["playername"].str.casefold())
        missing = [nm for nm in requested_names if nm not in have_lower]
        if missing:
            msg = f"Missing statistics for players: {sorted(set(missing))}"
            raise ValueError(msg)

        # attach canonical roles (by matching case-insensitive names back to roster)
        name_to_role = {wanted_lower[r]: r for r in _EXPECTED_POS}
        df["role"] = df["playername"].str.casefold().map(name_to_role).fillna("unknown")

        # re-order helpful columns if present
        cols = ["role", "playername", "teamname", "position", "date"]
        ordered = [c for c in cols if c in df.columns] + [
            c for c in df.columns if c not in cols
        ]
        return df[ordered]

    def _validate_roster(self, *, strict: bool = True) -> None:
        roles = set(self.roster.keys())
        missing_roles = [r for r in _EXPECTED_POS if r not in roles]
        if missing_roles:
            msg = f"Roster missing required roles: {missing_roles}"
            raise ValueError(msg)

        values = list(self.roster.values())
        if strict and any(
            v is None or not isinstance(v, str) or not v.strip() for v in values
        ):
            msg = "Roster must provide a non-empty player name for every role."
            raise ValueError(msg)

        # warn on duplicates (same name in multiple roles)
        names_norm = [str(v).casefold() for v in values if isinstance(v, str)]
        if len(names_norm) != len(set(names_norm)):
            logger.warning(
                "Roster for '%s' contains duplicate player names.", self.name
            )
