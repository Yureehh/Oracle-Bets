import json
import warnings

import numpy as np
import pytest
from lol_bets.prediction_models import gbdt_model as gbdt_module
from lol_bets.prediction_models.gbdt_model import GradientBoostingModel
from lol_bets.prediction_models.prop_features import build_game_level_prop_features
from oracle_bets_core.pd import pd

BLUE_KPM = 1.5
RED_KPM = -1.5
KPM_ABS_DIFF = 3.0
SELECTED_MAX_FEATURES = 3


class DummyModel(GradientBoostingModel):
    def train_model(self, **kwargs):
        assert kwargs is not None

    def _optimize_hyperparameters(self, *args):
        assert args is not None
        return {}


def test_prop_feature_builder_collapses_two_side_rows_to_one_game():
    X = pd.DataFrame(
        {
            "diff_ema_team_kpm": [BLUE_KPM, RED_KPM],
            "league_region": ["asia", "asia"],
        }
    )
    meta = pd.DataFrame(
        {
            "gameid": ["g1", "g1"],
            "side": ["Blue", "Red"],
            "date": ["2026-01-01", "2026-01-01"],
            "league": ["LPL", "LPL"],
        }
    )
    y = pd.Series([27.0, 27.0], name="total_kills")

    X_game, y_game, meta_game = build_game_level_prop_features(
        X, meta, y, target_col="total_kills"
    )

    assert len(X_game) == 1
    assert y_game.tolist() == [27.0]
    assert meta_game.loc[0, "gameid"] == "g1"
    assert X_game.loc[0, "blue_diff_ema_team_kpm"] == BLUE_KPM
    assert X_game.loc[0, "red_diff_ema_team_kpm"] == RED_KPM
    assert X_game.loc[0, "mean_diff_ema_team_kpm"] == 0.0
    assert X_game.loc[0, "absdiff_diff_ema_team_kpm"] == KPM_ABS_DIFF
    assert "result" not in X_game.columns


def test_prop_builder_drops_metadata_duplicated_in_feature_rows():
    X_side = pd.DataFrame(
        {
            "gameid": ["g1", "g1"],
            "side": ["Blue", "Red"],
            "league": ["LCK", "LCK"],
            "feature": [1.0, 2.0],
        }
    )
    meta = X_side[["gameid", "side", "league"]].copy()

    out, _, meta_out = build_game_level_prop_features(X_side, meta)

    assert out.columns.is_unique
    assert len(out) == 1
    assert meta_out.loc[0, "league"] == "LCK"


def test_prop_feature_builder_rejects_target_disagreement():
    X = pd.DataFrame({"diff_ema_team_kpm": [1.0, -1.0]})
    meta = pd.DataFrame({"gameid": ["g1", "g1"], "side": ["Blue", "Red"]})
    y = pd.Series([30.0, 31.0], name="total_kills")

    with pytest.raises(ValueError, match="side-level disagreement"):
        build_game_level_prop_features(X, meta, y, target_col="total_kills")


def test_prop_feature_builder_all_nan_mean_does_not_warn():
    X = pd.DataFrame({"diff_ema_team_kpm": [np.nan, np.nan]})
    meta = pd.DataFrame({"gameid": ["g1", "g1"], "side": ["Blue", "Red"]})
    y = pd.Series([27.0, 27.0], name="total_kills")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X_game, _, _ = build_game_level_prop_features(
            X, meta, y, target_col="total_kills"
        )

    assert caught == []
    assert pd.isna(X_game.loc[0, "mean_diff_ema_team_kpm"])
    assert pd.isna(X_game.loc[0, "absdiff_diff_ema_team_kpm"])
    assert pd.isna(X_game.loc[0, "sum_diff_ema_team_kpm"])


def test_compact_player_config_expands_to_role_features():
    candidates = GradientBoostingModel.compact_feature_candidates()

    assert "top_diff_ema_kda" in candidates
    assert "jng_diff_ema_kda" in candidates
    assert "players_trueskill_win_likelihood" in candidates
    assert "players_rating_consensus" in candidates
    assert "rating_disagreement" in candidates


def test_raw_season_is_training_metadata_not_model_feature():
    assert "season" in GradientBoostingModel._meta_columns()
    assert "season" not in GradientBoostingModel.compact_feature_candidates()


def test_rating_consensus_features_combine_team_and_player_strength_models():
    X = pd.DataFrame(
        {
            "elo_win_likelihood": [0.60],
            "trueskill_win_likelihood": [0.70],
            "players_elo_win_likelihood": [0.55],
            "players_glicko2_win_likelihood": [0.65],
            "players_pl_win_likelihood": [0.60],
            "players_trueskill_win_likelihood": [0.70],
        }
    )

    out = GradientBoostingModel.add_rating_consensus_features(X)

    assert out.loc[0, "team_rating_consensus"] == pytest.approx(0.65)
    assert out.loc[0, "players_rating_consensus"] == pytest.approx(0.625)
    assert out.loc[0, "rating_consensus"] == pytest.approx(0.6375)
    assert out.loc[0, "rating_disagreement"] > 0


def test_selected_feature_set_respects_max_features_and_anchors(tmp_path, monkeypatch):
    monkeypatch.setattr(gbdt_module, "FEATURE_REPORTS_DIR", tmp_path)
    report = {
        "recommended_features": ["ordinary_a", "ordinary_b", "ordinary_c"],
        "recommendations_by_count": {
            str(SELECTED_MAX_FEATURES): ["ordinary_a", "ordinary_b", "ordinary_c"]
        },
    }
    (tmp_path / "Dummy_recommended_compact_features.json").write_text(
        json.dumps(report)
    )
    model = DummyModel(
        model_name="Dummy",
        problem_type="classification",
        team_data=pd.DataFrame(),
        player_data=pd.DataFrame(),
        feature_set="selected",
        max_features=SELECTED_MAX_FEATURES,
    )
    X = pd.DataFrame(
        {
            "league_elo_win_likelihood": [0.6],
            "diff_ema_golddiffat15": [100.0],
            "ordinary_a": [1.0],
            "ordinary_b": [2.0],
            "ordinary_c": [3.0],
        }
    )

    out = model._apply_feature_set_filter(X)

    assert len(out.columns) == SELECTED_MAX_FEATURES
    assert "league_elo_win_likelihood" in out.columns
    assert "diff_ema_golddiffat15" in out.columns


def test_selected_feature_set_requires_a_recommendation_report(tmp_path, monkeypatch):
    monkeypatch.setattr(gbdt_module, "FEATURE_REPORTS_DIR", tmp_path)
    model = DummyModel(
        model_name="Dummy",
        problem_type="classification",
        team_data=pd.DataFrame(),
        player_data=pd.DataFrame(),
        feature_set="selected",
        max_features=SELECTED_MAX_FEATURES,
    )

    with pytest.raises(FileNotFoundError, match="recommendation report"):
        model._apply_feature_set_filter(pd.DataFrame({"feature": [1.0]}))
