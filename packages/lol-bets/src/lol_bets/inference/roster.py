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
    team = team.loc[(id_match | name_match) & dates.lt(cutoff)].copy()
    team["date"] = dates.loc[team.index]
    team = team[
        team["playername"].notna()
        & team["position"]
        .astype(str)
        .str.casefold()
        .isin({"top", "jng", "mid", "bot", "sup"})
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
