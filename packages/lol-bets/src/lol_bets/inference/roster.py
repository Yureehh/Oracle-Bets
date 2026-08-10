"""Conservative expected-lineup stability gate for pre-match inference."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import Any

from oracle_bets_core.pd import pd

EXPECTED_STARTERS = 5
SERIES_COMPLETION_BUFFER = timedelta(hours=6)
EXPECTED_ROLES = ("top", "jng", "mid", "bot", "sup")


class RosterGateState(StrEnum):
    UNKNOWN = "unknown"
    EMERGENCY_SUBSTITUTE = "emergency_substitute"
    STABILIZING = "stabilizing"
    STABLE = "stable"


@dataclass(frozen=True)
class RosterGateEvidence:
    """Facts known before a fixture; completed-series count comes from results."""

    team_id: str
    expected_player_ids: tuple[str, ...]
    established_player_ids: tuple[str, ...] | None
    completed_series_with_expected_roster: int
    emergency_substitute: bool = False

    def __post_init__(self) -> None:
        if not self.team_id.strip():
            raise ValueError("team_id cannot be empty")
        if self.completed_series_with_expected_roster < 0:
            raise ValueError("completed-series count cannot be negative")


@dataclass(frozen=True)
class HistoricalRosterEvidence:
    """Exact pre-fixture lineup repeated across the latest completed series."""

    roster: dict[str, str]
    series_ids: tuple[str, ...]
    series_dates: tuple[str, ...]
    evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": "historical_three_series",
            "roster": dict(self.roster),
            "series_ids": list(self.series_ids),
            "series_dates": list(self.series_dates),
            "evidence_hash": self.evidence_hash,
        }


@dataclass(frozen=True)
class RosterGateDecision:
    state: RosterGateState
    actionable: bool
    roster_version: str
    reasons: tuple[str, ...]
    completed_series: int
    required_completed_series: int


def evaluate_roster_gate(
    evidence: RosterGateEvidence,
    *,
    required_completed_series: int = 3,
) -> RosterGateDecision:
    """Allow an established roster or a changed roster proven across 3 series."""
    if required_completed_series <= 0:
        raise ValueError("required completed series must be positive")
    expected = _normalized_players(evidence.expected_player_ids)
    established = (
        _normalized_players(evidence.established_player_ids)
        if evidence.established_player_ids is not None
        else ()
    )
    version = _roster_version(evidence.team_id, expected)
    if len(expected) != EXPECTED_STARTERS:
        return _decision(
            RosterGateState.UNKNOWN,
            version,
            "expected_lineup_incomplete_or_ambiguous",
            evidence,
            required_completed_series,
        )
    if evidence.emergency_substitute:
        return _decision(
            RosterGateState.EMERGENCY_SUBSTITUTE,
            version,
            "emergency_substitute_shadow_only",
            evidence,
            required_completed_series,
        )
    if established and expected == established:
        return RosterGateDecision(
            state=RosterGateState.STABLE,
            actionable=True,
            roster_version=version,
            reasons=(),
            completed_series=evidence.completed_series_with_expected_roster,
            required_completed_series=required_completed_series,
        )
    if evidence.completed_series_with_expected_roster < required_completed_series:
        return _decision(
            RosterGateState.STABILIZING,
            version,
            "changed_roster_needs_three_completed_series",
            evidence,
            required_completed_series,
        )
    return RosterGateDecision(
        state=RosterGateState.STABLE,
        actionable=True,
        roster_version=version,
        reasons=(),
        completed_series=evidence.completed_series_with_expected_roster,
        required_completed_series=required_completed_series,
    )


def completed_series_with_roster(
    history: pd.DataFrame,
    *,
    team_id: str,
    team_name: str,
    expected_player_names: tuple[str, ...],
    before: Any,
) -> int:
    """Count consecutive completed series played with the exact expected five."""
    expected = _normalized_players(expected_player_names)
    if len(expected) != EXPECTED_STARTERS:
        return 0
    required = {
        "date",
        "gameid",
        "game",
        "teamid",
        "teamname",
        "playername",
        "position",
    }
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"roster history missing columns: {sorted(missing)}")

    cutoff = pd.Timestamp(before)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert(None)
    team = history.copy()
    dates = pd.to_datetime(team["date"], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    id_match = team["teamid"].astype(str).eq(str(team_id))
    name_match = team["teamname"].astype(str).str.casefold().eq(team_name.casefold())
    identity_match = id_match if team_id else name_match
    team = team.loc[identity_match & dates.lt(cutoff)].copy()
    team["date"] = dates.loc[team.index]
    team = team[
        team["playername"].notna()
        & team["position"].astype(str).str.casefold().isin(EXPECTED_ROLES)
    ]
    if team.empty:
        return 0

    maps: list[dict[str, Any]] = []
    for gameid, game in team.groupby("gameid", sort=False):
        players = _normalized_players(tuple(game["playername"].dropna().astype(str)))
        maps.append(
            {
                "gameid": str(gameid),
                "date": game["date"].min(),
                "game": int(pd.to_numeric(game["game"], errors="coerce").min()),
                "players": players,
            }
        )
    if not maps:
        return 0
    map_frame = pd.DataFrame(maps).sort_values(["date", "game"], kind="mergesort")
    previous_game = map_frame["game"].shift()
    gap = map_frame["date"].diff()
    map_frame["new_series"] = (
        map_frame["game"].eq(1)
        | map_frame["game"].le(previous_game)
        | gap.gt(pd.Timedelta(SERIES_COMPLETION_BUFFER))
    )
    map_frame["series_number"] = map_frame["new_series"].cumsum()

    completed_before = cutoff - pd.Timedelta(SERIES_COMPLETION_BUFFER)
    consecutive = 0
    for _, series in reversed(list(map_frame.groupby("series_number", sort=True))):
        if series["date"].max() > completed_before:
            continue
        exact = all(
            len(players) == EXPECTED_STARTERS and players == expected
            for players in series["players"]
        )
        if not exact:
            break
        consecutive += 1
    return consecutive


def infer_historical_roster(  # noqa: PLR0911, PLR0915
    history: pd.DataFrame,
    *,
    team_id: str,
    team_name: str,
    before: Any,
    required_completed_series: int = 3,
) -> HistoricalRosterEvidence | None:
    """Infer an exact role-mapped five from the latest consecutive series."""
    if required_completed_series <= 0:
        raise ValueError("required_completed_series must be positive")
    required = {
        "date",
        "gameid",
        "game",
        "teamid",
        "teamname",
        "playername",
        "position",
    }
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"roster history missing columns: {sorted(missing)}")

    cutoff = pd.Timestamp(before)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert(None)
    frame = history.copy()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    id_match = frame["teamid"].astype(str).eq(str(team_id)) if team_id else False
    name_match = frame["teamname"].astype(str).str.casefold().eq(team_name.casefold())
    identity_match = id_match if team_id else name_match
    frame = frame.loc[identity_match & dates.lt(cutoff)].copy()
    frame["date"] = dates.loc[frame.index]
    frame["position"] = frame["position"].astype(str).str.casefold()
    frame = frame[frame["playername"].notna() & frame["position"].isin(EXPECTED_ROLES)]
    if frame.empty:
        return None

    maps: list[dict[str, Any]] = []
    for gameid, game_rows in frame.groupby("gameid", sort=False):
        role_rows = game_rows[["position", "playername"]].drop_duplicates()
        role_counts = role_rows["position"].value_counts()
        valid_roles = set(role_rows["position"]) == set(EXPECTED_ROLES)
        unique_roles = bool((role_counts == 1).all())
        players = role_rows["playername"].astype(str).str.strip()
        unique_players = (
            len(players) == EXPECTED_STARTERS and players.nunique() == EXPECTED_STARTERS
        )
        roster = (
            {
                role: str(
                    role_rows.loc[role_rows["position"].eq(role), "playername"].iloc[0]
                ).strip()
                for role in EXPECTED_ROLES
            }
            if valid_roles and unique_roles and unique_players
            else None
        )
        game_number = pd.to_numeric(game_rows["game"], errors="coerce").min()
        maps.append(
            {
                "gameid": str(gameid),
                "date": game_rows["date"].min(),
                "game": int(game_number) if pd.notna(game_number) else 0,
                "roster": roster,
            }
        )
    map_frame = pd.DataFrame(maps).sort_values(["date", "game"], kind="mergesort")
    previous_game = map_frame["game"].shift()
    gap = map_frame["date"].diff()
    map_frame["new_series"] = (
        map_frame["game"].eq(1)
        | map_frame["game"].le(previous_game)
        | gap.gt(pd.Timedelta(SERIES_COMPLETION_BUFFER))
    )
    map_frame["series_number"] = map_frame["new_series"].cumsum()
    completed_before = cutoff - pd.Timedelta(SERIES_COMPLETION_BUFFER)
    completed = [
        series
        for _, series in map_frame.groupby("series_number", sort=True)
        if series["date"].max() <= completed_before
    ]
    selected = completed[-required_completed_series:]
    if len(selected) != required_completed_series:
        return None

    expected_key: tuple[tuple[str, str], ...] | None = None
    series_ids: list[str] = []
    series_dates: list[str] = []
    roster: dict[str, str] | None = None
    for series in selected:
        rosters = list(series["roster"])
        if any(item is None for item in rosters):
            return None
        keys = {
            tuple((role, str(item[role]).casefold()) for role in EXPECTED_ROLES)
            for item in rosters
        }
        if len(keys) != 1:
            return None
        key = next(iter(keys))
        if expected_key is not None and key != expected_key:
            return None
        expected_key = key
        roster = dict(rosters[0])
        gameids = tuple(sorted(series["gameid"].astype(str)))
        identity = hashlib.sha256("|".join(gameids).encode()).hexdigest()[:16]
        series_ids.append(f"series-{identity}")
        series_dates.append(pd.Timestamp(series["date"].max()).isoformat())
    if roster is None:
        return None
    payload = "|".join(
        (
            team_id or team_name.casefold(),
            *(f"{role}:{roster[role].casefold()}" for role in EXPECTED_ROLES),
            *series_ids,
        )
    )
    return HistoricalRosterEvidence(
        roster=roster,
        series_ids=tuple(series_ids),
        series_dates=tuple(series_dates),
        evidence_hash=(
            "historical-roster-" + hashlib.sha256(payload.encode()).hexdigest()[:20]
        ),
    )


def _decision(
    state: RosterGateState,
    version: str,
    reason: str,
    evidence: RosterGateEvidence,
    required: int,
) -> RosterGateDecision:
    return RosterGateDecision(
        state=state,
        actionable=False,
        roster_version=version,
        reasons=(reason,),
        completed_series=evidence.completed_series_with_expected_roster,
        required_completed_series=required,
    )


def _normalized_players(players: tuple[str, ...] | None) -> tuple[str, ...]:
    if players is None:
        return ()
    normalized = tuple(
        sorted(player.strip().casefold() for player in players if player.strip())
    )
    return normalized if len(normalized) == len(set(normalized)) else ()


def _roster_version(team_id: str, players: tuple[str, ...]) -> str:
    payload = "|".join((team_id.strip().casefold(), *players))
    return f"roster-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"
