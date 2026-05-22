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
from typing import TYPE_CHECKING

from oracle_bets_core.logger import logger
from oracle_bets_core.paths import FLATTENED_PLAYERS, FLATTENED_TEAMS
from oracle_bets_core.pd import pd

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


def _normalize_side(side: str) -> str:
    normalized = _VALID_SIDES.get(side.strip().casefold())
    if normalized is None:
        msg = "Side must be either 'Blue' or 'Red'."
        raise ValueError(msg)
    return normalized


@dataclass
class Team:
    name: str
    side: str | None = None
    first_pick: bool | None = None
    roster: dict[str, str | None] = field(
        default_factory=lambda: dict.fromkeys(_EXPECTED_POS)
    )

    # populated at init
    team_stats: pd.Series | None = field(default=None, init=False)
    player_stats: pd.DataFrame | None = field(default=None, init=False)

    # cached tables
    _team_df: pd.DataFrame = field(default=None, init=False, repr=False)
    _player_df: pd.DataFrame = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # load dataframes once (cached globally by lru_cache)
        self._team_df = _read_parquet_cached(str(FLATTENED_TEAMS))
        self._player_df = _read_parquet_cached(str(FLATTENED_PLAYERS))

        _require_columns(self._team_df, ["teamname"], "FLATTENED_TEAMS")
        _require_columns(
            self._player_df,
            ["teamname", "playername", "position", "date"],
            "FLATTENED_PLAYERS",
        )

        # set team_stats
        self.team_stats = self._lookup_team_row(self.name).copy()
        if self.side:
            self.team_stats["side"] = _normalize_side(self.side)
            self.side = self.team_stats["side"]
        if self.first_pick is not None:
            self.team_stats["first_pick"] = int(bool(self.first_pick))

        # complete roster: if some roles missing/None, backfill from last roster
        filled = {**dict.fromkeys(_EXPECTED_POS), **(self.roster or {})}
        if any(v is None for v in filled.values()):
            try:
                last = self._get_last_roster(self.name)
                for role in _EXPECTED_POS:
                    if filled[role] is None and role in last:
                        filled[role] = last[role]
            except (ValueError, KeyError) as e:
                logger.debug("No last-roster fallback available: %s", e)

        self.roster = filled
        self._validate_roster(strict=True)
        self.player_stats = self._lookup_players(self.roster)

    # ── lookups ──────────────────────────────────────────────────────────── #

    def _lookup_team_row(self, team_name: str) -> pd.Series:
        key = team_name.casefold()
        rows = self._team_df[self._team_df["teamname"].str.casefold() == key]
        if rows.empty:
            msg = f"Team '{team_name}' not found in teams table."
            raise ValueError(msg)
        return rows.iloc[0]

    def _get_last_roster(self, team_name: str) -> dict[str, str]:
        key = team_name.casefold()
        df = self._player_df[self._player_df["teamname"].str.casefold() == key]
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

        df = self._player_df[
            self._player_df["playername"].str.casefold().isin(wanted_lower.values())
        ].copy()

        if df.empty:
            msg = f"No player stats found for roster of team '{self.name}'."
            raise ValueError(msg)

        # keep the latest row per player
        df = df.sort_values(["playername", "date"], ascending=[True, False])
        df = df.drop_duplicates(subset=["playername"], keep="first").reset_index(
            drop=True
        )

        # ensure we have one row per requested player
        have_lower = set(df["playername"].str.casefold())
        missing = [nm for nm in wanted_lower.values() if nm not in have_lower]
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

    # ── roster ops ───────────────────────────────────────────────────────── #

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

    def update_roster(self, players: dict[str, str]) -> None:
        """
        Partially update roster with {role: player}. Missing roles remain unchanged.
        After update we reload player_stats with latest snapshots.
        """
        for role, player in players.items():
            if role not in _EXPECTED_POS:
                logger.warning("Ignoring unexpected role '%s'.", role)
                continue
            self.roster[role] = player

        self._validate_roster(strict=True)
        self.player_stats = self._lookup_players(self.roster)

    # ── user-facing helpers ──────────────────────────────────────────────── #

    def get_team_info(self) -> pd.DataFrame:
        """Compact table with team + roster (good for Discord embeds)."""
        rows = [{"ROLE": "Team", "NAME": self.name}]
        rows += [{"ROLE": r.upper(), "NAME": self.roster[r]} for r in _EXPECTED_POS]
        return pd.DataFrame(rows)
