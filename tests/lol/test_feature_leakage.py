import numpy as np
from lol_bets.data_generation.feature_engineering.features_generator import (
    FeatureGenerator,
    _add_expanding_mean,
)
from oracle_bets_core.pd import pd
from pandas.testing import assert_series_equal

PLAYER_A_FIRST_KILLS = 10
PLAYER_B_FIRST_KILLS = 100


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
