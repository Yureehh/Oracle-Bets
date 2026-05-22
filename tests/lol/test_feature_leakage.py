import numpy as np
from lol_bets.data_generation.feature_engineering.features_generator import (
    FeatureGenerator,
    _add_expanding_mean,
)
from oracle_bets_core.pd import pd
from pandas.testing import assert_series_equal

PLAYER_A_FIRST_KILLS = 10
PLAYER_B_FIRST_KILLS = 100
EXPECTED_EPIC_MONSTERS = 6
EXPECTED_STRUCTURE_CONTROL = 8


def test_player_win_loss_metrics_shift_within_player_season_patch():
    df = pd.DataFrame(
        {
            "playerid": ["a", "a", "b", "b"],
            "season": ["2025", "2025", "2025", "2025"],
            "patch": ["15.1", "15.1", "15.1", "15.1"],
            "result": [1, 1, 1, 1],
            "kills": [10, 20, 100, 200],
            "deaths": [1, 2, 10, 20],
            "date": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"]
            ),
        }
    )

    out = FeatureGenerator.compute_win_loss_metrics(df)

    assert np.isnan(out.loc[0, "kills_prev_avg_season_win"])
    assert out.loc[1, "kills_prev_avg_season_win"] == PLAYER_A_FIRST_KILLS
    assert np.isnan(out.loc[2, "kills_prev_avg_season_win"])
    assert out.loc[3, "kills_prev_avg_season_win"] == PLAYER_B_FIRST_KILLS


def test_expanding_mean_shift_resets_at_group_boundary():
    df = pd.DataFrame(
        {
            "teamid": ["a", "a", "b", "b"],
            "season": ["2025", "2025", "2025", "2025"],
            "date": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"]
            ),
            "gamelength": [30.0, 40.0, 70.0, 90.0],
        }
    )

    out = _add_expanding_mean(
        df,
        group_cols=["teamid", "season"],
        value_cols=["gamelength"],
        prefix="team_season_avg_",
    )

    expected = pd.Series(
        [np.nan, 30.0, np.nan, 70.0], name="team_season_avg_gamelength"
    )
    assert_series_equal(
        out["team_season_avg_gamelength"].reset_index(drop=True), expected
    )


def test_head_to_head_wins_shift_within_matchup():
    df = pd.DataFrame(
        {
            "teamid": ["a", "b", "a", "b", "c", "d"],
            "gameid": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-03",
                ]
            ),
            "result": [1, 0, 0, 1, 1, 0],
            "side": ["Blue", "Red", "Blue", "Red", "Blue", "Red"],
        }
    )

    out = FeatureGenerator.add_head_to_head_history(df)
    rows = out.set_index(["gameid", "teamid"])

    assert rows.loc[("g1", "a"), "h2h_wins_before"] == 0
    assert rows.loc[("g2", "a"), "h2h_wins_before"] == 1
    assert rows.loc[("g3", "c"), "h2h_wins_before"] == 0


def test_team_control_features_are_bounded_and_objective_based():
    df = pd.DataFrame(
        {
            "kills": [12, 8],
            "total_kills": [20, 20],
            "towers": [7, 3],
            "total_towers": [10, 10],
            "dragons": [2, 1],
            "barons": [1, 0],
            "heralds": [0, 1],
            "elders": [0, 0],
            "void_grubs": [3, 0],
            "inhibitors": [1, 0],
            "goldat15": [25000, 23000],
            "golddiffat15": [2000, -2000],
            "xpat15": [18000, 17500],
            "xpdiffat15": [500, -500],
            "csat15": [520, 500],
            "csdiffat15": [20, -20],
            "goldat25": [43000, 40000],
            "golddiffat25": [3000, -3000],
            "xpat25": [33000, 32000],
            "xpdiffat25": [1000, -1000],
            "csat25": [830, 800],
            "csdiffat25": [30, -30],
        }
    )

    out = FeatureGenerator.add_team_control_features(df)

    assert out["kill_share"].between(0, 1).all()
    assert out["tower_share"].between(0, 1).all()
    assert out.loc[0, "epic_monsters"] == EXPECTED_EPIC_MONSTERS
    assert out.loc[0, "structure_control"] == EXPECTED_STRUCTURE_CONTROL
    assert out.loc[0, "golddiff_shareat15"] > 0
    assert out.loc[1, "golddiff_shareat15"] < 0
