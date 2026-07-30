from __future__ import annotations

from lol_bets.inference.team import Team
from oracle_bets_core.pd import pd

TEAM_A_SHARED_ELO = 1500
LATEST_TEAM_ELO = 1700


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
