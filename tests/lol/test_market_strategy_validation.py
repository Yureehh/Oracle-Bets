import json

import pytest
from lol_bets.operations.market_validation import (
    _backtest_derived_markets,
    _map_cohort_metrics,
)
from oracle_bets_core.pd import pd

EXPECTED_TWO_ROWS = 2


def test_derived_market_backtest_uses_only_held_out_first_map_probability():
    manifest = pd.DataFrame(
        [
            {
                "series_id": "bo3",
                "best_of": 3,
                "map_count": 2,
                "team_a_id": "a",
                "team_b_id": "b",
                "source_map_ids": json.dumps(["bo3-map1", "bo3-map2"]),
                "map_winner_ids": json.dumps(["a", "a"]),
            },
            {
                "series_id": "bo5",
                "best_of": 5,
                "map_count": 5,
                "team_a_id": "a",
                "team_b_id": "b",
                "source_map_ids": json.dumps(
                    ["bo5-map1", "bo5-map2", "bo5-map3", "bo5-map4", "bo5-map5"]
                ),
                "map_winner_ids": json.dumps(["a", "b", "a", "b", "a"]),
            },
        ]
    )
    predictions = pd.DataFrame(
        [
            {"gameid": "bo3-map1", "proba": 0.7},
            {"gameid": "bo5-map1", "proba": 0.6},
            # Later-map prices must never be read for a prematch series backtest.
            {"gameid": "bo5-map2", "proba": 0.01},
        ]
    )

    metrics, counts = _backtest_derived_markets(manifest, predictions)

    assert counts == {"bo3": 1, "bo5": 1}
    assert set(metrics) == {
        "series_totals_map_path_v1",
        "series_handicap_map_path_v1",
    }
    assert all(
        value >= 0
        for strategy in metrics.values()
        for name, value in strategy.items()
        if name != "accuracy"
    )
    assert all(0 <= strategy["accuracy"] <= 1 for strategy in metrics.values())


def test_derived_market_backtest_rejects_invalid_probabilities():
    manifest = pd.DataFrame(
        [
            {
                "best_of": 3,
                "map_count": 2,
                "team_a_id": "a",
                "team_b_id": "b",
                "source_map_ids": '["map1", "map2"]',
                "map_winner_ids": '["a", "a"]',
            }
        ]
    )

    with pytest.raises(ValueError, match="outside"):
        _backtest_derived_markets(
            manifest,
            pd.DataFrame([{"gameid": "map1", "proba": 1.1}]),
        )


def test_map_cohorts_separate_map_one_later_actionable_and_probability_bands():
    manifest = pd.DataFrame([{"source_map_ids": '["map1", "map2", "map3"]'}])
    predictions = pd.DataFrame(
        [
            {"gameid": "map1", "actual": 1, "proba": 0.7, "actionable": True},
            {"gameid": "map2", "actual": 0, "proba": 0.5, "actionable": True},
            {"gameid": "map3", "actual": 0, "proba": 0.3, "actionable": False},
        ]
    )

    metrics = _map_cohort_metrics(manifest, predictions)

    assert metrics["map_1"]["count"] == 1
    assert metrics["later_maps"]["count"] == EXPECTED_TWO_ROWS
    assert metrics["actionable"]["count"] == EXPECTED_TWO_ROWS
    assert metrics["probability_0_00_0_40"]["count"] == 1
    assert metrics["probability_0_40_0_60"]["count"] == 1
    assert metrics["probability_0_60_1_00"]["count"] == 1
