from __future__ import annotations

import ast
from pathlib import Path

import pytest
from lol_bets.operations.market_strategies import (
    DecisionClass,
    decide_market,
    rank_market_decisions,
)

ONE_UNIT = 0.01
EXPECTED_EXPLORATION_SAMPLES = 2
PREDICTION_MODELS = (
    Path(__file__).resolve().parents[2]
    / "packages/lol-bets/src/lol_bets/prediction_models"
)


def test_direct_series_recommendation_is_deterministic_and_records_all_sizes():
    inputs = {
        "semantic_fingerprint": "series-team-a",
        "target": "series_winner",
        "readiness": "recommendation_active",
        "probability": 0.58,
        "conservative_probability": 0.56,
        "decimal_odds": 1.9,
        "is_model_favorite": True,
    }

    first = decide_market(**inputs)
    second = decide_market(**inputs)
    sizing = first.to_dict()["sizing"]

    assert first == second
    assert first.classification is DecisionClass.RECOMMENDED
    assert first.reason_codes == ("all_recommendation_gates_passed",)
    assert sizing["selected_path"] == "full_kelly"
    assert sizing["bankroll_fractions"]["flat_1u"] == ONE_UNIT
    assert sizing["bankroll_fractions"]["half_kelly"] == pytest.approx(
        sizing["bankroll_fractions"]["full_kelly"] / 2
    )
    assert sizing["bankroll_fractions"]["quarter_kelly"] == pytest.approx(
        sizing["bankroll_fractions"]["full_kelly"] / 4
    )


def test_negative_ev_exploration_uses_flat_unit_and_zero_kelly():
    decision = decide_market(
        semantic_fingerprint="map-team-a",
        target="map_winner",
        readiness="exploration",
        probability=0.52,
        conservative_probability=0.50,
        decimal_odds=1.8,
        is_model_favorite=True,
    )
    sizing = decision.to_dict()["sizing"]

    assert decision.classification is DecisionClass.EXPLORATION
    assert "target_exploration_only" in decision.reason_codes
    assert "point_ev_non_positive" in decision.reason_codes
    assert sizing["selected_path"] == "flat_1u"
    assert sizing["bankroll_fractions"] == {
        "flat_1u": ONE_UNIT,
        "full_kelly": 0.0,
        "half_kelly": 0.0,
        "quarter_kelly": 0.0,
    }


def test_display_only_and_existing_blocks_retain_every_reason():
    decision = decide_market(
        semantic_fingerprint="prop-over",
        target="total_kills_mean",
        readiness="display_only",
        probability=0.60,
        conservative_probability=0.58,
        decimal_odds=2.0,
        is_model_favorite=True,
        hard_blocks=("semantic_contract_invalid", "book_crossed"),
    )

    assert decision.classification is DecisionClass.NOT_COMPARABLE
    assert decision.reason_codes == (
        "semantic_contract_invalid",
        "book_crossed",
        "strategy_display_only",
    )
    assert decision.to_dict()["sizing"]["selected_path"] is None


def test_exploration_sampler_selects_once_per_target_and_period():
    actions = [
        {
            "fixture_key": "fixture-1",
            "classification": "exploration",
            "point_ev": value,
            "semantic_fingerprint": fingerprint,
            "semantic_key": {"target": "map_winner", "period": period},
            "reason_codes": [],
        }
        for value, fingerprint, period in (
            (0.0, "map1-a", "map:1"),
            (-0.1, "map1-b", "map:1"),
            (-0.2, "map2-a", "map:2"),
            (-0.3, "map2-b", "map:2"),
        )
    ]

    ranked = rank_market_decisions(actions)

    assert [
        row["semantic_fingerprint"] for row in ranked if row["ticket_eligible"]
    ] == [
        "map1-a",
        "map2-a",
    ]
    assert (
        sum(row["exploration_sampled"] for row in ranked)
        == EXPECTED_EXPLORATION_SAMPLES
    )
    assert "exploration_sample_already_filled" in ranked[1]["reason_codes"]


def test_exploration_sampler_keeps_fixtures_independent():
    semantic = {"target": "map_winner", "period": "map:1"}
    ranked = rank_market_decisions(
        [
            {
                "fixture_key": fixture,
                "classification": "exploration",
                "point_ev": 0.01,
                "semantic_fingerprint": f"{fixture}-map1",
                "semantic_key": semantic,
                "reason_codes": [],
            }
            for fixture in ("fixture-1", "fixture-2")
        ]
    )

    assert all(row["ticket_eligible"] for row in ranked)


def test_prediction_modules_do_not_import_market_or_betting_decision_helpers():
    forbidden = {"oracle_bets_core.betting", "oracle_bets_core.markets"}
    imports: set[str] = set()
    for path in PREDICTION_MODELS.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        imports.update(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )
        imports.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )

    assert forbidden.isdisjoint(imports)
