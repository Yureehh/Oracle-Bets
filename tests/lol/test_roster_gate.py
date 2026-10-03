from datetime import UTC, datetime, timedelta

import pytest
from lol_bets.inference.roster import (
    EXPECTED_ROLES,
    HistoricalRosterEvidence,
    RosterGateEvidence,
    RosterGateState,
    completed_series_with_roster,
    evaluate_roster_gate,
    infer_historical_roster,
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
    for series_index in range(series_count):
        for game in (1, 2):
            played_at = start + timedelta(days=series_index, hours=game)
            for role, player in zip(EXPECTED_ROLES, NEW, strict=True):
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


def test_historical_fallback_selects_most_frequent_players_from_last_ten_maps():
    inferred = infer_historical_roster(
        _roster_history(),
        team_id="team-1",
        team_name="T1",
        before=datetime(2026, 7, 5, tzinfo=UTC),
    )

    assert isinstance(inferred, HistoricalRosterEvidence)
    assert inferred.roster == dict(zip(EXPECTED_ROLES, NEW, strict=True))
    expected_maps = 6
    assert len(inferred.source_map_ids) == expected_maps
    assert all(share == 1.0 for share in inferred.appearance_shares.values())
    assert inferred.confidence == "high"
    assert inferred.evidence_hash.startswith("historical-roster-")


def test_historical_fallback_handles_substitutions_and_role_conflicts():
    history = _roster_history()
    changed = history.copy()
    changed.loc[
        (changed["gameid"] == "s2-g2") & changed["position"].eq("top"),
        "playername",
    ] = "emergency"
    duplicate_role = history.copy()
    duplicate_role.loc[
        (duplicate_role["gameid"] == "s2-g2") & duplicate_role["position"].eq("sup"),
        "position",
    ] = "top"

    kwargs = {
        "team_id": "team-1",
        "team_name": "T1",
        "before": datetime(2026, 7, 5, tzinfo=UTC),
    }
    changed_result = infer_historical_roster(changed, **kwargs)
    short_result = infer_historical_roster(_roster_history(2), **kwargs)
    conflict_result = infer_historical_roster(duplicate_role, **kwargs)
    assert changed_result is not None
    assert changed_result.roster["top"] == "new-top"
    assert changed_result.confidence in {"high", "medium"}
    assert short_result is not None
    assert short_result.confidence == "medium"
    assert conflict_result is not None
    assert conflict_result.confidence in {"high", "medium", "low"}


def test_historical_fallback_breaks_frequency_ties_by_most_recent_appearance():
    history = _roster_history(series_count=5)
    top_rows = history["position"].eq("top")
    top_indices = history.loc[top_rows].sort_values("date").index
    history.loc[top_indices[:5], "playername"] = "older-top"
    history.loc[top_indices[5:], "playername"] = "newer-top"

    inferred = infer_historical_roster(
        history,
        team_id="team-1",
        team_name="T1",
        before=datetime(2026, 7, 8, tzinfo=UTC),
    )

    assert inferred is not None
    assert inferred.roster["top"] == "newer-top"
    assert inferred.appearance_shares["top"] == pytest.approx(0.5)
    assert inferred.alternates["top"] == ("older-top",)
    assert len(inferred.substitutions["top"]) == len(inferred.source_map_ids) // 2


def test_historical_fallback_does_not_mix_a_reused_team_name():
    history = _roster_history()
    collision = history.copy()
    collision["teamid"] = "other-team"
    collision["playername"] = "wrong-" + collision["playername"]
    mixed = pd.concat([history, collision], ignore_index=True)

    inferred = infer_historical_roster(
        mixed,
        team_id="team-1",
        team_name="T1",
        before=datetime(2026, 7, 5, tzinfo=UTC),
    )

    assert inferred is not None
    assert inferred.roster == dict(zip(EXPECTED_ROLES, NEW, strict=True))
