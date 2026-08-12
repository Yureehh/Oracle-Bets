from __future__ import annotations

import json
from pathlib import Path

import pytest
from lol_bets.prediction_models.feature_contract import (
    Availability,
    FeatureContractError,
    FeatureRegistry,
    FeatureRole,
    default_feature_registry,
)

ROOT = Path(__file__).resolve().parents[2]
TRAINING_CONFIGS = (
    ROOT / "config/lol/training/training_team_config.json",
    ROOT / "config/lol/training/training_compact_team_config.json",
    ROOT / "config/lol/training/training_player_config.json",
    ROOT / "config/lol/training/training_compact_player_config.json",
)


def _features(path):
    payload = json.loads(path.read_text())
    return next(iter(payload.values()))


def test_default_contract_separates_inputs_identifiers_targets_and_prohibited():
    registry = default_feature_registry()

    assert registry.resolve("diff_ema_goldat15_before").role == FeatureRole.INPUT
    assert registry.resolve("side").role == FeatureRole.IDENTIFIER
    assert registry.resolve("result").role == FeatureRole.TARGET
    assert registry.resolve("first_pick").role == FeatureRole.PROHIBITED
    assert registry.resolve("side_win_likelihood").role == FeatureRole.PROHIBITED


def test_prematch_model_inputs_reject_future_or_prohibited_facts():
    registry = default_feature_registry()

    registry.assert_model_inputs_available(
        ["diff_ema_goldat15_before", "roster_continuity"],
        at=Availability.PREMATCH,
    )
    with pytest.raises(FeatureContractError, match="first_pick"):
        registry.assert_model_inputs_available(
            ["diff_ema_goldat15_before", "first_pick"],
            at=Availability.PREMATCH,
        )
    with pytest.raises(FeatureContractError, match="result"):
        registry.assert_model_inputs_available(
            ["result"],
            at=Availability.PREMATCH,
        )


def test_unknown_features_fail_instead_of_becoming_implicit_inputs():
    with pytest.raises(FeatureContractError, match="unregistered"):
        default_feature_registry().resolve("mystery_future_stat")


def test_feature_lineage_declares_availability_swap_and_eligibility():
    registry = default_feature_registry()

    rating = registry.resolve("elo_win_likelihood")
    context = registry.resolve("best_of")
    target = registry.resolve("result")

    assert rating.availability_timestamp == "strictly_before_fixture_start"
    assert rating.swap_behavior == "negate_as_team_delta"
    assert rating.model_eligible
    assert context.swap_behavior == "invariant_context"
    assert not target.model_eligible


def test_feature_manifest_fingerprint_is_registration_order_independent():
    first = FeatureRegistry()
    first.register_exact(
        "feature_a",
        family="rating",
        role=FeatureRole.INPUT,
        availability=Availability.PREMATCH,
        missing_policy="unknown",
    )
    first.register_exact(
        "feature_b",
        family="form",
        role=FeatureRole.INPUT,
        availability=Availability.PREMATCH,
        missing_policy="unknown",
    )
    second = FeatureRegistry()
    second.register_exact(
        "feature_b",
        family="form",
        role=FeatureRole.INPUT,
        availability=Availability.PREMATCH,
        missing_policy="unknown",
    )
    second.register_exact(
        "feature_a",
        family="rating",
        role=FeatureRole.INPUT,
        availability=Availability.PREMATCH,
        missing_policy="unknown",
    )

    assert first.fingerprint(["feature_a", "feature_b"]) == second.fingerprint(
        ["feature_b", "feature_a"]
    )


def test_all_training_configs_have_no_prohibited_model_signal():
    registry = default_feature_registry()

    for path in TRAINING_CONFIGS:
        specs = [registry.resolve(name) for name in _features(path)]
        prohibited = [
            spec.name for spec in specs if spec.role == FeatureRole.PROHIBITED
        ]
        assert prohibited == [], f"{path.name}: {prohibited}"
