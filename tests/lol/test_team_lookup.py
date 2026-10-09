from __future__ import annotations

from datetime import UTC, datetime

import pytest
from lol_bets.data_generation.feature_engineering.ratings_features.elo import (
    INACTIVITY_GRACE_DAYS,
    INACTIVITY_HALF_LIFE_DAYS,
)
from lol_bets.data_generation.feature_engineering.ratings_features.plackett_luce import (
    DEFAULT_SIGMA as PL_SIGMA,
)
from lol_bets.data_generation.feature_engineering.ratings_features.trueskill import (
    DEFAULT_SIGMA as TS_SIGMA,
)
from lol_bets.inference import team as team_module
from lol_bets.inference.snapshots import publish_feature_snapshot
from lol_bets.inference.team import (
    InsufficientRosterHistoryError,
    Team,
    _advance_inactive_ratings,
    _latest_rating_leagues,
)
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


def test_same_day_activity_does_not_advance_frozen_feature_state(tmp_path) -> None:
    teams = pd.DataFrame(
        {
            "teamname": ["Team A", "Team A"],
            "date": ["2026-01-01T10:00Z", "2026-01-02T10:00Z"],
            "gamelength": [30.0, 30.0],
            "elo_after": [1500.0, 1700.0],
            "glicko2_mu_after": [1500.0, 1600.0],
            "glicko2_phi_after": [100.0, 80.0],
        }
    )
    players = pd.concat(
        [
            teams.assign(playername=role, position=role)
            for role in ("top", "jng", "mid", "bot", "sup")
        ],
        ignore_index=True,
    )
    snapshot = publish_feature_snapshot(
        teams,
        players,
        team_columns=[
            "teamname",
            "date",
            "elo_after",
            "glicko2_mu_after",
            "glicko2_phi_after",
        ],
        player_columns=[
            "teamname",
            "date",
            "elo_after",
            "glicko2_mu_after",
            "glicko2_phi_after",
            "playername",
            "position",
        ],
        source_manifest={},
        code_sha256="test-code",
        training_generation_id="test-generation",
        root=tmp_path,
        observed_at=datetime(2026, 1, 2, 13, tzinfo=UTC),
    )
    team = Team(
        "Team A",
        as_of_date="2026-01-02T12:00Z",
        decision_at="2026-01-02T14:00Z",
        feature_snapshot=snapshot,
    )

    assert team.team_stats["elo"] == TEAM_A_SHARED_ELO
    assert team.player_stats["elo"].tolist() == [TEAM_A_SHARED_ELO] * 5
    assert team.team_stats["days_since_last_game"] == 0.0
    assert team.player_stats["days_since_last_game"].tolist() == [0.0] * 5

    tomorrow = Team(
        "Team A",
        as_of_date="2026-01-03T12:00Z",
        decision_at="2026-01-02T14:00Z",
        feature_snapshot=snapshot,
    )
    assert tomorrow.team_stats["elo"] == LATEST_TEAM_ELO
    assert tomorrow.player_stats["elo"].tolist() == [LATEST_TEAM_ELO] * 5

    later_at = pd.Timestamp("2026-04-01T12:00Z")
    later = Team(
        "Team A",
        league="LEC",
        as_of_date=later_at,
        decision_at=later_at,
        feature_snapshot=snapshot,
    )
    inactive_days = max(
        0,
        (later_at - pd.Timestamp("2026-01-02T10:00Z")).days - INACTIVITY_GRACE_DAYS,
    )
    retention = 0.5 ** (inactive_days / INACTIVITY_HALF_LIFE_DAYS)
    assert later.team_stats["elo"] == pytest.approx(1500 + 200 * retention)
    assert later.team_stats["glicko2_mu"] == pytest.approx(1500 + 100 * retention)
    assert later.team_stats["glicko2_phi"] == pytest.approx(
        350 - (350 - 80) * retention
    )
    assert later.player_stats["elo"].tolist() == pytest.approx(
        [1500 + 200 * retention] * 5
    )
    assert later.team_stats["league"] == "LEC"
    assert later.team_stats["strength_pool"] == "major"


def test_player_league_transfer_uses_last_non_cross_league(
    tmp_path, monkeypatch
) -> None:
    player_id = "oe:player:example"
    history = pd.DataFrame(
        {
            "playerid": [player_id, player_id],
            "league": ["LFL", "EWC"],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        }
    )
    assert _latest_rating_leagues(history, "playerid")[player_id] == "LFL"

    league_path = tmp_path / "league_elo.parquet"
    pd.DataFrame({"league": ["LFL", "LEC"], "elo": [1700.0, 1600.0]}).to_parquet(
        league_path
    )
    monkeypatch.setattr(team_module, "RATING_LEAGUE_ELO", league_path)
    team_module._league_elo_values.cache_clear()
    row = pd.Series(
        {
            "playerid": player_id,
            "date": "2026-01-02T10:00Z",
            "elo": 1600.0,
            "glicko2_mu": 1600.0,
            "glicko2_phi": 100.0,
            "pl_mu": 30.0,
            "pl_sigma": 2.0,
            "trueskill_mu": 30.0,
            "trueskill_sigma": 2.0,
        }
    )
    advanced = _advance_inactive_ratings(row, pd.Timestamp("2026-01-03"), "LEC", "LFL")

    assert advanced["elo"] == pytest.approx(1640.0)
    assert advanced["glicko2_mu"] == pytest.approx(1640.0)
    assert advanced["pl_mu"] == pytest.approx(25.0)
    assert advanced["pl_sigma"] == PL_SIGMA
    assert advanced["trueskill_mu"] == pytest.approx(25.0)
    assert advanced["trueskill_sigma"] == TS_SIGMA
    team_module._league_elo_values.cache_clear()


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
            "playerid": ["shared-a", "shared-b", "jng", "mid", "bot", "sup"],
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
    assert team._player_activity_dates["shared-a"] == pd.Timestamp("2026-01-01")


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


def test_frozen_state_includes_prior_day_game_completed_after_midnight() -> None:
    team = Team.__new__(Team)
    team._as_of = pd.Timestamp("2026-10-07T00:34:50Z")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-10-06T23:46:59Z", "2026-10-06T23:55:00Z", "2026-10-07T00:05:00Z"]
            ),
            "state_available_at": pd.to_datetime(
                ["2026-10-07T00:16:40Z", "2026-10-07T00:40:00Z", "2026-10-07T00:25:00Z"]
            ),
            "gameid": ["completed_prior_day", "unfinished_prior_day", "same_day"],
        }
    )
    assert team._at_or_before_as_of(frame, frozen_state=True)["gameid"].tolist() == [
        "completed_prior_day"
    ]


@pytest.mark.parametrize("prior_count", [0, 4])
def test_lookup_players_without_prior_state_reports_unavailable_history(
    prior_count,
) -> None:
    team = Team.__new__(Team)
    team.name = "Team A"
    team._as_of = pd.Timestamp("2026-10-07T12:00Z")
    roles = ("top", "jng", "mid", "bot", "sup")
    team._player_df = pd.DataFrame(
        {
            "teamname": ["Team A"] * len(roles),
            "playername": list(roles),
            "playerid": list(roles),
            "position": list(roles),
            "date": pd.to_datetime(
                ["2026-10-06T10:00Z"] * prior_count
                + ["2026-10-07T10:00Z"] * (len(roles) - prior_count)
            ),
            "state_available_at": pd.to_datetime(
                ["2026-10-06T10:30Z"] * prior_count
                + ["2026-10-07T10:30Z"] * (len(roles) - prior_count)
            ),
        }
    )
    with pytest.raises(
        InsufficientRosterHistoryError, match=r"No player stats|Missing statistics"
    ):
        team._lookup_players(dict(zip(roles, roles, strict=True)))


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
