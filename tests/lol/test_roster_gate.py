from datetime import UTC, datetime, timedelta

from lol_bets.inference.roster import (
    RosterGateEvidence,
    RosterGateState,
    completed_series_with_roster,
    evaluate_roster_gate,
)
from oracle_bets_core.pd import pd

OLD = ("old-top", "old-jng", "old-mid", "old-bot", "old-sup")
NEW = ("new-top", "old-jng", "old-mid", "old-bot", "old-sup")
REQUIRED_STABLE_SERIES = 3


def _evidence(
    expected=NEW,
    *,
    completed=0,
    emergency=False,
    established=OLD,
):
    return RosterGateEvidence(
        team_id="team-1",
        expected_player_ids=expected,
        established_player_ids=established,
        completed_series_with_expected_roster=completed,
        emergency_substitute=emergency,
    )


def test_established_roster_is_immediately_actionable():
    decision = evaluate_roster_gate(_evidence(expected=OLD))

    assert decision.state is RosterGateState.STABLE
    assert decision.actionable


def test_changed_roster_requires_three_completed_series():
    before = evaluate_roster_gate(_evidence(completed=2))
    after = evaluate_roster_gate(_evidence(completed=3))

    assert before.state is RosterGateState.STABILIZING
    assert not before.actionable
    assert before.reasons == ("changed_roster_needs_three_completed_series",)
    assert after.actionable


def test_unknown_or_emergency_lineup_is_shadow_only():
    incomplete = evaluate_roster_gate(_evidence(expected=NEW[:4]))
    emergency = evaluate_roster_gate(_evidence(completed=20, emergency=True))

    assert incomplete.state is RosterGateState.UNKNOWN
    assert not incomplete.actionable
    assert emergency.state is RosterGateState.EMERGENCY_SUBSTITUTE
    assert not emergency.actionable


def _roster_history(series_count: int = 3) -> pd.DataFrame:
    rows = []
    start = datetime(2026, 7, 1, tzinfo=UTC)
    roles = ("top", "jng", "mid", "bot", "sup")
    for series_index in range(series_count):
        for game in (1, 2):
            played_at = start + timedelta(days=series_index, hours=game)
            for role, player in zip(roles, NEW, strict=True):
                rows.append(
                    {
                        "date": played_at,
                        "gameid": f"s{series_index}-g{game}",
                        "game": game,
                        "teamid": "team-1",
                        "teamname": "T1",
                        "playername": player,
                        "position": role,
                    }
                )
    return pd.DataFrame(rows)


def test_completed_series_counter_requires_consecutive_exact_lineups():
    history = _roster_history()

    completed = completed_series_with_roster(
        history,
        team_id="team-1",
        team_name="T1",
        expected_player_names=NEW,
        before=datetime(2026, 7, 5, tzinfo=UTC),
    )

    assert completed == REQUIRED_STABLE_SERIES

    changed = history.copy()
    changed.loc[changed["gameid"] == "s2-g2", "playername"] = "emergency"
    assert (
        completed_series_with_roster(
            changed,
            team_id="team-1",
            team_name="T1",
            expected_player_names=NEW,
            before=datetime(2026, 7, 5, tzinfo=UTC),
        )
        == 0
    )
