from __future__ import annotations

from datetime import UTC, datetime

from lol_bets.inference.snapshots import publish_feature_snapshot
from lol_bets.inference.team import Team
from oracle_bets_core.pd import pd

TEAM_A_SHARED_ELO = 1500
LATEST_TEAM_ELO = 1700
EXPECTED_BREAK_THRESHOLD_DAYS = 45.0


def test_default_as_of_uses_the_utc_decision_clock(tmp_path) -> None:
    teams = pd.DataFrame(
        {
            "teamname": ["Team A"],
            "date": ["2026-01-01T12:00Z"],
            "gamelength": [30.0],
            "elo_after": [1500],
        }
    )
    players = pd.concat(
        [
            teams.assign(playername=role, position=role)
            for role in ("top", "jng", "mid", "bot", "sup")
        ],
        ignore_index=True,
    )
    columns = ["teamname", "date", "elo_after"]
    snapshot = publish_feature_snapshot(
        teams,
        players,
        team_columns=columns,
        player_columns=[*columns, "playername", "position"],
        source_manifest={},
        code_sha256="test-code",
        training_generation_id="test-generation",
        root=tmp_path,
        observed_at=datetime(2026, 1, 1, 13, tzinfo=UTC),
    )
    team = Team(
        "Team A",
        feature_snapshot=snapshot,
        decision_at="2026-01-02T12:00:00+02:00",
    )
    # Only 22 hours elapsed; midnight rounding would incorrectly add a day.
    assert team.team_stats["days_since_last_game"] == 0.0
    assert team.team_stats["is_after_break"] == 0
    assert team.player_stats["days_since_last_game"].tolist() == [0.0] * 5

    at_threshold = Team(
        "Team A",
        feature_snapshot=snapshot,
        as_of_date="2026-02-16T11:00:00Z",
        decision_at="2026-02-16T11:00:00Z",
    )
    assert (
        at_threshold.team_stats["days_since_last_game"] == EXPECTED_BREAK_THRESHOLD_DAYS
    )
    assert at_threshold.team_stats["is_after_break"] == 0

    after_break = Team(
        "Team A",
        feature_snapshot=snapshot,
        as_of_date="2026-02-17T12:00:00Z",
        decision_at="2026-02-17T12:00:00Z",
    )
    assert after_break.team_stats["is_after_break"] == 1


def test_lookup_players_prefers_requested_team_before_global_latest() -> None:
    team = Team.__new__(Team)
    team.name = "Team A"
    team._player_df = pd.DataFrame(
        {
            "teamname": [
                "Team A",
                "Team B",
                "Team A",
                "Team A",
                "Team A",
                "Team A",
            ],
            "playername": ["Shared", "Shared", "Jng", "Mid", "Bot", "Sup"],
            "position": ["top", "top", "jng", "mid", "bot", "sup"],
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-02-01",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-02",
                ]
            ),
            "elo": [1500, 1800, 1510, 1520, 1530, 1540],
        }
    )
    roster = {
        "top": "Shared",
        "jng": "Jng",
        "mid": "Mid",
        "bot": "Bot",
        "sup": "Sup",
    }

    out = team._lookup_players(roster)
    top = out[out["role"] == "top"].iloc[0]

    assert top["teamname"] == "Team A"
    assert top["elo"] == TEAM_A_SHARED_ELO


def test_lookup_team_row_prefers_latest_snapshot() -> None:
    team = Team.__new__(Team)
    team._team_df = pd.DataFrame(
        {
            "teamname": ["Team A", "Team A"],
            "date": pd.to_datetime(["2026-01-01", "2026-03-01"]),
            "elo": [1500, LATEST_TEAM_ELO],
        }
    )

    out = team._lookup_team_row("Team A")

    assert out["elo"] == LATEST_TEAM_ELO


def test_lookup_team_row_respects_as_of_date() -> None:
    team = Team.__new__(Team)
    team._as_of = pd.Timestamp("2026-02-01")
    team._team_df = pd.DataFrame(
        {
            "teamname": ["Team A", "Team A"],
            "date": pd.to_datetime(["2026-01-01", "2026-03-01"]),
            "elo": [1500, LATEST_TEAM_ELO],
        }
    )

    out = team._lookup_team_row("Team A")

    assert out["elo"] == TEAM_A_SHARED_ELO


def test_lookup_rejects_same_day_future_and_unfinished_match_state() -> None:
    team = Team.__new__(Team)
    team._as_of = pd.Timestamp("2026-02-01T12:00Z")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-02-01T10:00Z", "2026-02-01T11:50Z", "2026-02-01T18:00Z"]
            ),
            "state_available_at": pd.to_datetime(
                ["2026-02-01T10:30Z", "2026-02-01T12:20Z", "2026-02-01T18:30Z"]
            ),
            "gameid": ["completed", "unfinished", "future"],
        }
    )
    assert team._at_or_before_as_of(frame)["gameid"].tolist() == ["completed"]


def test_last_roster_ignores_blank_latest_player_names() -> None:
    team = Team.__new__(Team)
    team.name = "Team A"
    team._as_of = None
    team._player_df = pd.DataFrame(
        {
            "teamname": ["Team A"] * 6,
            "playername": ["Top", "", "Jng", "Mid", "Bot", "Sup"],
            "position": ["top", "top", "jng", "mid", "bot", "sup"],
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-02-01",
                    "2026-02-01",
                    "2026-02-01",
                    "2026-02-01",
                    "2026-02-01",
                ]
            ),
        }
    )

    roster = team._get_last_roster("Team A")

    assert roster["top"] == "Top"
