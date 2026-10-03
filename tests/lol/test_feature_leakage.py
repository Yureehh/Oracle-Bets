import numpy as np
import pytest
from lol_bets.data_generation.feature_engineering.features_generator import (
    FeatureGenerator,
    _add_expanding_mean,
)
from lol_bets.data_generation.feature_engineering.performance_features.entity_stats import (
    apply_ema,
    apply_opponent_stats,
)
from lol_bets.data_generation.feature_engineering.performance_features.patch_win_rate import (
    compute_ema_patch,
)
from lol_bets.prediction_models.gbdt_model import GradientBoostingModel
from oracle_bets_core.pd import pd
from pandas.testing import assert_series_equal

PLAYER_A_FIRST_KILLS = 10
PLAYER_B_FIRST_KILLS = 100
EXPECTED_EPIC_MONSTERS = 6
EXPECTED_STRUCTURE_CONTROL = 8
EXPECTED_GOLD_GROWTH = 1000
EXPECTED_WIN_GAMELENGTH = 1800
EXPECTED_TEAM_WPM = 3.2
EXPECTED_TEAM_VSPM = 6.2
EXPECTED_TEAM_CONTROL_WARDS = 9
EXPECTED_SAME_DATE_EXCLUDED_MEAN = 35.0
EXPECTED_PRIOR_H2H_SAME_DATE_EXCLUDED = 2
BLUE_GOLD_EMA = 110.0
RED_GOLD_EMA = 90.0
GOLD_EMA_DIFF = 20.0
TOP_OPP_KDA = 1.5
TOP_KDA_DIFF = 2.5
MID_OPP_KDA = 9.0
MID_KDA_DIFF = -1.0
MODEL_GOLD_EMA = 120.0
MODEL_OPP_GOLD_EMA = 100.0
MODEL_TOP_KDA = 5.0
MODEL_OPP_TOP_KDA = 3.0
MODEL_TOP_KDA_DIFF = 2.0
DEATHLESS_KDA = 11
EXPECTED_TWO_DAY_GAP = 2
EXPECTED_ROSTER_CONTINUITY = 0.8
EXPECTED_SAME_DAY_PLAYER_MEAN = 15.0
EXPECTED_PRIOR_LOSS_MEAN = 10.0
NEUTRAL_WIN_RATE = 0.5


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


def test_inactivity_and_roster_features_use_prior_distinct_dates():
    team_df = pd.DataFrame(
        {
            "teamid": ["a", "a", "a"],
            "date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-03"]),
            "season": ["2026", "2026", "2026"],
        }
    )
    player_rows = []
    for match_date, roster in (
        ("2026-01-01", ["top", "jng", "mid", "bot", "sup"]),
        ("2026-01-03", ["top", "jng", "new_mid", "bot", "sup"]),
    ):
        player_rows.extend(
            {"teamid": "a", "date": match_date, "playerid": player} for player in roster
        )
    player_df = pd.DataFrame(player_rows)

    gaps = FeatureGenerator.add_break_indicator(team_df)
    out = FeatureGenerator.add_roster_features(gaps, player_df)

    assert np.isnan(out.loc[0, "days_since_last_game"])
    assert np.isnan(out.loc[1, "days_since_last_game"])
    assert out.loc[2, "days_since_last_game"] == EXPECTED_TWO_DAY_GAP
    assert out.loc[2, "roster_continuity"] == EXPECTED_ROSTER_CONTINUITY
    assert out.loc[2, "roster_uncertainty"] == pytest.approx(
        1 - EXPECTED_ROSTER_CONTINUITY
    )


def test_player_win_loss_metrics_exclude_same_date_maps():
    # Same-date maps (e.g. games of one series) must not feed each other,
    # because Oracle's Elixir dates do not guarantee intra-day ordering.
    df = pd.DataFrame(
        {
            "playerid": ["a", "a", "a"],
            "season": ["2025", "2025", "2025"],
            "patch": ["15.1", "15.1", "15.1"],
            "result": [1, 1, 1],
            "kills": [10, 20, 30],
            "deaths": [1, 2, 3],
            "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02"]),
        }
    )

    out = FeatureGenerator.compute_win_loss_metrics(df)

    # Both 01-01 maps have no strictly-prior date -> NaN.
    assert np.isnan(out.loc[0, "kills_prev_avg_season_win"])
    assert np.isnan(out.loc[1, "kills_prev_avg_season_win"])
    # The 01-02 map sees the average of both prior-day maps.
    assert out.loc[2, "kills_prev_avg_season_win"] == EXPECTED_SAME_DAY_PLAYER_MEAN


def test_ema_features_exclude_other_games_on_the_same_date():
    frame = pd.DataFrame(
        {
            "teamid": ["a", "a", "a"],
            "date": pd.to_datetime(
                ["2026-01-01 10:00", "2026-01-01 18:00", "2026-01-02 10:00"]
            ),
            "kills": [10.0, 30.0, 50.0],
        }
    )

    out = apply_ema(frame, "teamid", ["kills"])

    assert np.isnan(out.loc[0, "ema_kills_before"])
    assert np.isnan(out.loc[1, "ema_kills_before"])
    assert out.loc[2, "ema_kills_before"] > EXPECTED_PRIOR_LOSS_MEAN


def test_patch_ema_features_exclude_other_games_on_the_same_date():
    frame = pd.DataFrame(
        {
            "teamid": ["a", "a", "a"],
            "gameid": ["g1", "g2", "g3"],
            "side": ["Blue", "Red", "Blue"],
            "patch": ["16.1"] * 3,
            "date": pd.to_datetime(
                ["2026-01-01 10:00", "2026-01-01 18:00", "2026-01-02 10:00"]
            ),
            "result": [1, 0, 1],
        }
    )

    out = compute_ema_patch(frame, "teamid")

    assert out.loc[0, "ema_patch_win_rate_before"] == NEUTRAL_WIN_RATE
    assert out.loc[1, "ema_patch_win_rate_before"] == NEUTRAL_WIN_RATE
    assert out.loc[0, "ema_patch_games_before"] == 0.0
    assert out.loc[1, "ema_patch_games_before"] == 0.0


def test_player_win_loss_metrics_ignore_missing_results():
    # A missing result must not be silently counted as a loss.
    df = pd.DataFrame(
        {
            "playerid": ["a", "a", "a"],
            "season": ["2025", "2025", "2025"],
            "patch": ["15.1", "15.1", "15.1"],
            "result": [0, None, 0],
            "kills": [10, 20, 30],
            "deaths": [1, 2, 3],
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        }
    )

    out = FeatureGenerator.compute_win_loss_metrics(df)

    # Loss history on 01-03 must only include the 01-01 loss (kills=10),
    # not the unknown-result 01-02 game.
    assert out.loc[2, "kills_prev_avg_season_loss"] == EXPECTED_PRIOR_LOSS_MEAN


def test_player_kda_uses_kills_plus_assists_for_deathless_games():
    df = pd.DataFrame(
        {
            "teamid": ["blue", "blue", "red", "red"],
            "gameid": ["g1", "g1", "g1", "g1"],
            "side": ["Blue", "Blue", "Red", "Red"],
            "position": ["mid", "sup", "mid", "sup"],
            "league": ["LEC", "LEC", "LEC", "LEC"],
            "kills": [4, 1, 2, 0],
            "assists": [7, 9, 3, 4],
            "deaths": [0, 2, 1, 4],
            "gamelength": [1800, 1800, 1800, 1800],
            "total_cs": [260, 30, 240, 35],
            "patch": ["16.1", "16.1", "16.1", "16.1"],
            "result": [1, 1, 0, 0],
            "playerid": ["caps", "support", "opp_mid", "opp_sup"],
            "date": pd.to_datetime(["2026-01-01"] * 4),
            "damagetakenperminute": [400, 600, 390, 620],
            "damagemitigatedperminute": [300, 500, 280, 520],
            "wpm": [0.5, 1.2, 0.4, 1.1],
            "wcpm": [0.2, 0.4, 0.2, 0.3],
        }
    )

    out = FeatureGenerator.generate_new_player_features(df)
    caps = out.loc[out["playerid"] == "caps"].iloc[0]

    assert caps["kda"] == DEATHLESS_KDA


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


def test_expanding_mean_can_exclude_same_date_rows():
    df = pd.DataFrame(
        {
            "teamid": ["a", "a", "a"],
            "season": ["2025", "2025", "2025"],
            "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02"]),
            "gamelength": [30.0, 40.0, 50.0],
        }
    )

    out = _add_expanding_mean(
        df,
        group_cols=["teamid", "season"],
        value_cols=["gamelength"],
        prefix="team_season_avg_",
        sort_also_by=["date"],
        exclude_same_date=True,
    )

    same_day = out[out["date"].eq(pd.Timestamp("2025-01-01"))]
    next_day = out[out["date"].eq(pd.Timestamp("2025-01-02"))].iloc[0]
    assert same_day["team_season_avg_gamelength"].isna().all()
    assert next_day["team_season_avg_gamelength"] == EXPECTED_SAME_DATE_EXCLUDED_MEAN


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


def test_head_to_head_excludes_same_date_maps():
    df = pd.DataFrame(
        {
            "teamid": ["a", "b", "a", "b", "a", "b"],
            "gameid": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-01",
                    "2025-01-01",
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-02",
                ]
            ),
            "result": [1, 0, 0, 1, 1, 0],
            "side": ["Blue", "Red", "Blue", "Red", "Blue", "Red"],
        }
    )

    out = FeatureGenerator.add_head_to_head_history(df).set_index(["gameid", "teamid"])

    assert out.loc[("g1", "a"), "h2h_games_before"] == 0
    assert out.loc[("g2", "a"), "h2h_games_before"] == 0
    assert (
        out.loc[("g3", "a"), "h2h_games_before"]
        == EXPECTED_PRIOR_H2H_SAME_DATE_EXCLUDED
    )
    assert out.loc[("g3", "a"), "h2h_wins_before"] == 1


def test_series_context_uses_explicit_best_of_not_completed_length():
    df = pd.DataFrame(
        {
            "gameid": ["match_game1", "match_game2", "match_game3"],
            "game": [1, 2, 3],
            "best_of": [5, 5, 5],
        }
    )

    out = FeatureGenerator.add_series_context(df)

    assert out["is_bo5"].tolist() == [1, 1, 1]
    assert out["is_bo3"].tolist() == [0, 0, 0]
    assert out["is_deciding_game"].tolist() == [0, 0, 0]


def test_series_context_does_not_infer_best_of_from_observed_game_count():
    df = pd.DataFrame(
        {
            "gameid": ["match_game1", "match_game2"],
            "game": [1, 2],
        }
    )

    out = FeatureGenerator.add_series_context(df)

    assert out["is_bo1"].tolist() == [0, 0]
    assert out["is_bo3"].tolist() == [0, 0]
    assert out["is_bo5"].tolist() == [0, 0]
    assert out["is_deciding_game"].tolist() == [0, 0]


def test_team_control_features_are_bounded_and_objective_based():
    df = pd.DataFrame(
        {
            "result": [1, 0],
            "gamelength": [1800, 2100],
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
            "goldat10": [24000, 24000],
            "golddiffat10": [1000, -1000],
            "xpat10": [12000, 12100],
            "xpdiffat10": [300, -300],
            "csat10": [350, 345],
            "csdiffat10": [10, -10],
            "goldat15": [25000, 23000],
            "golddiffat15": [2000, -2000],
            "xpat15": [18000, 17500],
            "xpdiffat15": [500, -500],
            "csat15": [520, 500],
            "csdiffat15": [20, -20],
            "goldat20": [33000, 31500],
            "golddiffat20": [2500, -2500],
            "xpat20": [26000, 25200],
            "xpdiffat20": [800, -800],
            "csat20": [680, 650],
            "csdiffat20": [30, -30],
            "goldat25": [43000, 40000],
            "golddiffat25": [3000, -3000],
            "xpat25": [33000, 32000],
            "xpdiffat25": [1000, -1000],
            "csat25": [830, 800],
            "csdiffat25": [30, -30],
            "firsttower": [1, 0],
            "firstdragon": [1, 0],
        }
    )

    out = FeatureGenerator.add_team_control_features(df)

    assert out["kill_share"].between(0, 1).all()
    assert out["tower_share"].between(0, 1).all()
    assert out.loc[0, "epic_monsters"] == EXPECTED_EPIC_MONSTERS
    assert out.loc[0, "structure_control"] == EXPECTED_STRUCTURE_CONTROL
    assert out.loc[0, "golddiff_shareat15"] > 0
    assert out.loc[1, "golddiff_shareat15"] < 0
    assert out.loc[0, "golddiff_growth_10_15"] == EXPECTED_GOLD_GROWTH
    assert out.loc[0, "ahead_goldat15"] == 1
    assert out.loc[0, "won_when_ahead_goldat15"] == 1
    assert out.loc[1, "lost_when_behind_goldat15"] == 1
    assert out.loc[0, "win_gamelength"] == EXPECTED_WIN_GAMELENGTH
    assert np.isnan(out.loc[0, "loss_gamelength"])
    assert out.loc[0, "towers_per_epic_monster"] > 0
    assert out.loc[0, "first_tower_to_win"] == 1


def test_team_vision_features_aggregate_player_rows():
    team_df = pd.DataFrame(
        {"gameid": ["g1", "g1"], "teamid": ["blue", "red"], "side": ["Blue", "Red"]}
    )
    player_df = pd.DataFrame(
        {
            "gameid": ["g1", "g1", "g1", "g1"],
            "teamid": ["blue", "blue", "red", "red"],
            "wpm": [1.2, 2.0, 0.5, 0.7],
            "wcpm": [0.4, 0.8, 0.3, 0.2],
            "vspm": [3.0, 3.2, 1.0, 1.4],
            "controlwardsbought": [4, 5, 2, 1],
        }
    )

    out = FeatureGenerator.add_team_vision_features(team_df, player_df)
    blue = out.loc[out["teamid"] == "blue"].iloc[0]

    assert blue["team_wpm"] == EXPECTED_TEAM_WPM
    assert blue["team_vspm"] == EXPECTED_TEAM_VSPM
    assert blue["team_controlwardsbought"] == EXPECTED_TEAM_CONTROL_WARDS


def test_opponent_ema_diff_does_not_overwrite_base_team_feature():
    df = pd.DataFrame(
        {
            "gameid": ["g1", "g1"],
            "side": ["Red", "Blue"],
            "ema_goldat15_before": [RED_GOLD_EMA, BLUE_GOLD_EMA],
        }
    )

    out = apply_opponent_stats(df, "team", ["goldat15"])
    blue = out.loc[out["side"] == "Blue"].iloc[0]

    assert blue["ema_goldat15_before"] == BLUE_GOLD_EMA
    assert blue["opp_ema_goldat15_before"] == RED_GOLD_EMA
    assert blue["diff_ema_goldat15_before"] == GOLD_EMA_DIFF


def test_player_opponent_ema_diff_is_position_aware():
    df = pd.DataFrame(
        {
            "gameid": ["g1", "g1", "g1", "g1"],
            "side": ["Blue", "Blue", "Red", "Red"],
            "position": ["top", "mid", "top", "mid"],
            "ema_kda_before": [
                TOP_OPP_KDA + TOP_KDA_DIFF,
                MID_OPP_KDA + MID_KDA_DIFF,
                TOP_OPP_KDA,
                MID_OPP_KDA,
            ],
        }
    )

    out = apply_opponent_stats(df, "player", ["kda"])
    top = out.loc[(out["side"] == "Blue") & (out["position"] == "top")].iloc[0]
    mid = out.loc[(out["side"] == "Blue") & (out["position"] == "mid")].iloc[0]

    assert top["opp_ema_kda_before"] == TOP_OPP_KDA
    assert top["diff_ema_kda_before"] == TOP_KDA_DIFF
    assert mid["opp_ema_kda_before"] == MID_OPP_KDA
    assert mid["diff_ema_kda_before"] == MID_KDA_DIFF


def test_model_preprocessing_keeps_base_ema_when_creating_diff():
    df = pd.DataFrame(
        {
            "ema_goldat15": [MODEL_GOLD_EMA],
            "opp_ema_goldat15": [MODEL_OPP_GOLD_EMA],
            "top_ema_kda": [MODEL_TOP_KDA],
            "top_opp_ema_kda": [MODEL_OPP_TOP_KDA],
        }
    )

    out = GradientBoostingModel.add_explicit_ema_diffs(df, drop_opponents=True)

    assert out.loc[0, "ema_goldat15"] == MODEL_GOLD_EMA
    assert out.loc[0, "diff_ema_goldat15"] == GOLD_EMA_DIFF
    assert out.loc[0, "top_ema_kda"] == MODEL_TOP_KDA
    assert out.loc[0, "top_diff_ema_kda"] == MODEL_TOP_KDA_DIFF
    assert "opp_ema_goldat15" not in out
    assert "top_opp_ema_kda" not in out


def test_model_preprocessing_overwrites_stale_matchup_diffs():
    df = pd.DataFrame(
        {
            "ema_goldat15": [MODEL_GOLD_EMA],
            "opp_ema_goldat15": [MODEL_OPP_GOLD_EMA],
            "diff_ema_goldat15": [999.0],
            "top_ema_kda": [MODEL_TOP_KDA],
            "top_opp_ema_kda": [MODEL_OPP_TOP_KDA],
            "top_diff_ema_kda": [999.0],
        }
    )

    out = GradientBoostingModel.add_explicit_ema_diffs(df, drop_opponents=True)

    assert out.loc[0, "diff_ema_goldat15"] == GOLD_EMA_DIFF
    assert out.loc[0, "top_diff_ema_kda"] == MODEL_TOP_KDA_DIFF


def test_production_pruning_keeps_signed_features_and_drops_only_constants():
    frame = pd.DataFrame(
        {
            "negative_mean_delta": [-5.0, -3.0, -7.0, -1.0],
            "positive_mean_delta": [1.0, 3.0, 2.0, 4.0],
            "constant": [1.0, 1.0, 1.0, 1.0],
        }
    )

    retained, dropped = GradientBoostingModel.drop_low_std_columns(None, frame)

    assert "negative_mean_delta" in retained
    assert "positive_mean_delta" in retained
    assert dropped == ["constant"]
