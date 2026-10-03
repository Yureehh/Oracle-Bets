from __future__ import annotations

import pytest
from lol_bets.prediction_models.data_preprocessor import DataPreprocessor
from lol_bets.prediction_models.prop_features import (
    build_game_level_outcome_features,
)
from oracle_bets_core.pd import pd

EXPECTED_RATING_DELTA = -8.0
EXPECTED_NEXT_MAP_NUMBER = 2


def test_regression_preprocessing_does_not_filter_by_full_dataset_quantiles() -> None:
    team_data = pd.DataFrame(
        {
            "gameid": ["g1", "g2", "g3"],
            "side": ["Blue", "Blue", "Blue"],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-06-01"]),
            "result": [1, 1, 1],
            "gamelength": [31.0, 32.0, 99.0],
            "total_kills": [20.0, 21.0, 22.0],
            "total_towers": [11.0, 12.0, 13.0],
            "team_feature": [1.0, 2.0, 3.0],
        }
    )
    player_data = pd.DataFrame(
        {
            "gameid": ["g1", "g2", "g3"],
            "side": ["Blue", "Blue", "Blue"],
            "position": ["top", "top", "top"],
            "result": [1, 1, 1],
            "player_feature": [10.0, 20.0, 30.0],
        }
    )

    out = DataPreprocessor(team_data, player_data).preprocess(
        "gamelength",
        "regression",
    )

    assert out["gameid"].tolist() == ["g1", "g2", "g3"]
    assert out["gamelength"].tolist() == [31.0, 32.0, 99.0]


def test_player_targets_are_not_pivoted_into_features() -> None:
    team_data = pd.DataFrame(
        {
            "gameid": ["g1"],
            "side": ["Blue"],
            "result": [1],
            "gamelength": [31.0],
            "total_kills": [20.0],
            "total_towers": [11.0],
            "team_feature": [1.0],
        }
    )
    player_data = pd.DataFrame(
        {
            "gameid": ["g1"],
            "side": ["Blue"],
            "position": ["top"],
            "result": [1],
            "gamelength": [31.0],
            "total_kills": [20.0],
            "total_towers": [11.0],
            "player_feature": [10.0],
        }
    )

    out = DataPreprocessor(team_data, player_data).preprocess("result")

    assert "top_player_feature" in out.columns
    assert "top_result" not in out.columns
    assert "top_gamelength" not in out.columns
    assert "top_total_kills" not in out.columns
    assert "top_total_towers" not in out.columns


def test_outcome_matchup_features_are_canonical_and_side_free() -> None:
    features = pd.DataFrame(
        {
            "rating": [20.0, 12.0],
            "roster_uncertainty": [0.2, 0.6],
            "league_tier": ["major", "major"],
            "first_pick": [1.0, 0.0],
            "side_win_likelihood": [0.6, 0.4],
        }
    )
    metadata = pd.DataFrame(
        {
            "gameid": ["g1", "g1"],
            "teamid": ["z", "a"],
            "teamname": ["Zulu", "Alpha"],
            "side": ["Blue", "Red"],
            "date": ["2026-01-01", "2026-01-01"],
            "league": ["LCK", "LCK"],
        }
    )

    X_game, y_game, meta_game = build_game_level_outcome_features(
        features, metadata, pd.Series([1, 0])
    )

    assert X_game.loc[0, "delta_rating"] == EXPECTED_RATING_DELTA
    assert X_game.loc[0, "delta_roster_uncertainty"] == pytest.approx(0.4)
    assert X_game.loc[0, "pair_league_tier"] == "major|major"
    assert "delta_first_pick" not in X_game
    assert "delta_side_win_likelihood" not in X_game
    assert y_game.tolist() == [0.0]
    assert meta_game.loc[0, "canonical_teamname"] == "Alpha"


def test_outcome_matchup_preserves_series_cluster_identity() -> None:
    features = pd.DataFrame({"rating": [20.0, 12.0], "next_map_number": [2, 2]})
    metadata = pd.DataFrame(
        {
            "gameid": ["series-a:map-2", "series-a:map-2"],
            "series_id": ["series-a", "series-a"],
            "source_gameid": ["map-1", "map-1"],
            "target_gameid": ["map-2", "map-2"],
            "teamid": ["z", "a"],
            "teamname": ["Zulu", "Alpha"],
        }
    )

    _, _, meta_game = build_game_level_outcome_features(
        features, metadata, pd.Series([1, 0])
    )

    assert meta_game.loc[0, "series_id"] == "series-a"
    assert meta_game.loc[0, "source_gameid"] == "map-1"
    assert meta_game.loc[0, "target_gameid"] == "map-2"
    assert meta_game.loc[0, "next_map_number"] == EXPECTED_NEXT_MAP_NUMBER


def test_outcome_matchup_rejects_conflicting_series_cluster_identity() -> None:
    metadata = pd.DataFrame(
        {
            "gameid": ["g1", "g1"],
            "series_id": ["series-a", "series-b"],
            "teamid": ["z", "a"],
            "teamname": ["Zulu", "Alpha"],
        }
    )

    with pytest.raises(ValueError, match="conflicting identity metadata 'series_id'"):
        build_game_level_outcome_features(
            pd.DataFrame({"rating": [20.0, 12.0]}),
            metadata,
            pd.Series([1, 0]),
        )
