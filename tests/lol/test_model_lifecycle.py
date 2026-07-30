from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from lol_bets.operations.models import (
    CandidateManifest,
    ModelRegistry,
    ModelRegistryError,
    PromotionEvidence,
    TrainingTriggerState,
    evaluate_promotion,
    evaluate_training_triggers_from_history,
    orchestrate_candidate_training,
    paired_bootstrap_improvement,
    register_current_candidate,
    resolve_serving_artifact,
    should_train_candidate,
)
from oracle_bets_core.pd import pd

NOW = datetime(2026, 7, 26, 8, 15, tzinfo=UTC)
MIN_PROMOTION_IMPROVEMENT = 0.01
MAJOR_MAPS = 25


def test_paired_bootstrap_detects_repeatable_log_loss_improvement():
    champion_losses = [0.72, 0.68, 0.75, 0.64, 0.70, 0.73] * 20
    candidate_losses = [value * 0.96 for value in champion_losses]

    comparison = paired_bootstrap_improvement(
        champion_losses,
        candidate_losses,
        confidence=0.95,
        samples=2000,
        seed=7,
    )

    assert comparison.relative_improvement == pytest.approx(0.04)
    assert comparison.lower_bound > 0
    assert comparison.upper_bound > comparison.lower_bound


def test_promotion_requires_point_improvement_confidence_and_safety():
    evidence = PromotionEvidence(
        champion_log_losses=tuple([0.70, 0.72, 0.68, 0.71] * 40),
        candidate_log_losses=tuple([0.67, 0.69, 0.65, 0.68] * 40),
        champion_brier=0.22,
        candidate_brier=0.21,
        cohort_log_loss={
            "LCK": (0.69, 0.66, 50),
            "LEC": (0.71, 0.69, 50),
        },
    )

    decision = evaluate_promotion(evidence, bootstrap_samples=2000, seed=3)

    assert decision.promote
    assert decision.relative_improvement >= MIN_PROMOTION_IMPROVEMENT
    assert decision.confidence_lower_bound > 0
    assert decision.safety_failures == ()


def test_accuracy_gain_cannot_override_worse_probability_quality():
    evidence = PromotionEvidence(
        champion_log_losses=(0.60, 0.62, 0.61, 0.63) * 40,
        candidate_log_losses=(0.64, 0.65, 0.63, 0.66) * 40,
        champion_brier=0.21,
        candidate_brier=0.20,
        cohort_log_loss={},
    )

    decision = evaluate_promotion(evidence, bootstrap_samples=1000, seed=5)

    assert not decision.promote
    assert "relative_improvement_below_1_percent" in decision.reasons


def test_major_cohort_regression_blocks_aggregate_promotion():
    evidence = PromotionEvidence(
        champion_log_losses=(0.70, 0.72, 0.68, 0.71) * 40,
        candidate_log_losses=(0.67, 0.69, 0.65, 0.68) * 40,
        champion_brier=0.22,
        candidate_brier=0.21,
        cohort_log_loss={"LCK": (0.60, 0.66, 50)},
    )

    decision = evaluate_promotion(evidence, bootstrap_samples=1000, seed=3)

    assert not decision.promote
    assert decision.safety_failures == ("cohort_regression:LCK",)


@pytest.mark.parametrize(
    ("valid_maps", "major_maps", "last_days", "expected"),
    [
        (50, 0, 1, True),
        (0, 20, 1, True),
        (0, 0, 31, True),
        (49, 19, 29, False),
    ],
)
def test_candidate_training_triggers_are_explicit(
    valid_maps, major_maps, last_days, expected
):
    state = TrainingTriggerState(
        new_valid_maps=valid_maps,
        new_major_maps=major_maps,
        last_candidate_at=NOW - timedelta(days=last_days),
        evaluated_at=NOW,
    )

    assert should_train_candidate(state) is expected


def test_history_evidence_drives_training_and_immutable_registration(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    history = pd.DataFrame(
        [
            {
                "gameid": f"game-{index}",
                "date": NOW - timedelta(days=10) + timedelta(hours=index),
                "league": "LCK" if index < MAJOR_MAPS else "ERL",
                "datacompleteness": "complete",
            }
            for index in range(60)
        ]
    )
    evaluation = evaluate_training_triggers_from_history(
        history,
        registry=registry,
        evaluated_at=NOW,
        major_leagues={"LCK"},
    )
    calls = []

    result = orchestrate_candidate_training(
        evaluation,
        train_candidate=lambda: calls.append("trained"),
        register_candidate=lambda model_id: calls.append(model_id),
    )

    assert evaluation.triggered
    assert "new_valid_maps" in evaluation.reasons
    assert "new_major_maps" in evaluation.reasons
    assert result.trained
    assert calls == ["trained", evaluation.candidate_id]


def test_untriggered_evidence_neither_trains_nor_registers(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    candidate = registry.candidates / "candidate-existing"
    candidate.mkdir()
    (candidate / "manifest.json").write_text(
        json.dumps({"created_at": (NOW - timedelta(days=1)).isoformat()})
    )
    history = pd.DataFrame(
        [
            {
                "gameid": "old-game",
                "date": NOW - timedelta(days=2),
                "league": "LCK",
                "datacompleteness": "complete",
            }
        ]
    )
    evaluation = evaluate_training_triggers_from_history(
        history,
        registry=registry,
        evaluated_at=NOW,
        major_leagues={"LCK"},
    )

    def _must_not_register(model_id):
        pytest.fail(f"must not register {model_id}")

    result = orchestrate_candidate_training(
        evaluation,
        train_candidate=lambda: pytest.fail("must not train"),
        register_candidate=_must_not_register,
    )

    assert not evaluation.triggered
    assert not result.trained
    assert result.registered_model_id is None


def _manifest(model_id="candidate-1"):
    return CandidateManifest(
        model_id=model_id,
        sport="lol",
        target="map_win",
        created_at=NOW,
        code_version="tree-abc",
        data_manifest="data-abc",
        feature_fingerprint="features-abc",
        config_hash="config-abc",
        dependency_lock_hash="lock-abc",
        random_seed=7,
        metrics={"log_loss": 0.65, "brier": 0.21},
    )


def test_candidate_bundle_is_immutable_and_checksum_verified(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"model bytes")

    bundle = registry.register_candidate(_manifest(), {"model.pkl": artifact})

    assert registry.verify_bundle("candidate-1")
    stored = json.loads((bundle / "manifest.json").read_text())
    assert stored["files"]["model.pkl"]["sha256"]
    with pytest.raises(ModelRegistryError, match="already exists"):
        registry.register_candidate(_manifest(), {"model.pkl": artifact})

    (bundle / "model.pkl").write_bytes(b"tampered")
    assert not registry.verify_bundle("candidate-1")


def test_promotion_and_rollback_change_pointer_and_append_history(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    first_artifact = tmp_path / "first.pkl"
    second_artifact = tmp_path / "second.pkl"
    first_artifact.write_bytes(b"first")
    second_artifact.write_bytes(b"second")
    registry.register_candidate(_manifest("candidate-1"), {"model.pkl": first_artifact})
    registry.register_candidate(
        _manifest("candidate-2"), {"model.pkl": second_artifact}
    )

    registry.promote("candidate-1", promoted_at=NOW, reason="initial champion")
    registry.promote(
        "candidate-2",
        promoted_at=NOW + timedelta(minutes=1),
        reason="passed promotion gate",
    )
    registry.rollback(
        "candidate-1",
        rolled_back_at=NOW + timedelta(minutes=2),
        reason="owner rollback after artifact review",
    )

    assert registry.champion_id() == "candidate-1"
    history = [
        json.loads(line)
        for line in (tmp_path / "registry/champion_history.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [entry["to_model_id"] for entry in history] == [
        "candidate-1",
        "candidate-2",
        "candidate-1",
    ]
    assert history[-1]["action"] == "rollback"


def test_unhealthy_bundle_cannot_be_promoted(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"model")
    bundle = registry.register_candidate(_manifest(), {"model.pkl": artifact})
    (bundle / "model.pkl").write_bytes(b"tampered")

    with pytest.raises(ModelRegistryError, match="checksum"):
        registry.promote("candidate-1", promoted_at=NOW, reason="should fail")


def test_serving_resolves_only_verified_champion_artifacts(tmp_path):
    model_root = tmp_path / "models"
    registry_root = tmp_path / "registry"
    legacy = model_root / "OutcomePrediction" / "OutcomePrediction.pkl"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy")
    registry = ModelRegistry(registry_root)
    source = tmp_path / "candidate.pkl"
    source.write_bytes(b"candidate")
    name = "OutcomePrediction/OutcomePrediction.pkl"
    registry.register_candidate(_manifest(), {name: source})

    assert (
        resolve_serving_artifact(
            legacy,
            registry_root=registry_root,
            legacy_root=model_root,
        )
        == legacy
    )

    registry.promote("candidate-1", promoted_at=NOW, reason="gate passed")
    resolved = resolve_serving_artifact(
        legacy,
        registry_root=registry_root,
        legacy_root=model_root,
    )
    assert resolved.read_bytes() == b"candidate"


def test_serving_refuses_mixed_or_tampered_champion_bundle(tmp_path):
    model_root = tmp_path / "models"
    registry_root = tmp_path / "registry"
    registry = ModelRegistry(registry_root)
    source = tmp_path / "candidate.pkl"
    source.write_bytes(b"candidate")
    registry.register_candidate(_manifest(), {"other.pkl": source})
    registry.promote("candidate-1", promoted_at=NOW, reason="gate passed")

    with pytest.raises(ModelRegistryError, match="does not contain"):
        resolve_serving_artifact(
            model_root / "missing.pkl",
            registry_root=registry_root,
            legacy_root=model_root,
        )

    (registry_root / "candidates/candidate-1/other.pkl").write_bytes(b"tampered")
    with pytest.raises(ModelRegistryError, match="checksum"):
        registry.artifact_path("other.pkl")


def test_current_inference_tree_can_be_frozen_with_reproducibility_hashes(
    tmp_path,
):
    model_root = tmp_path / "models"
    config_root = tmp_path / "config"
    model = model_root / "OutcomePrediction" / "OutcomePrediction.pkl"
    feature_config = config_root / "training.json"
    product_config = tmp_path / "product.json"
    lock = tmp_path / "uv.lock"
    training = tmp_path / "training.parquet"
    model.parent.mkdir(parents=True)
    config_root.mkdir()
    model.write_bytes(b"model")
    feature_config.write_text('{"features":["elo"]}')
    product_config.write_text('{"schema_version":1}')
    lock.write_text("lock")
    training.write_bytes(b"training")
    registry = ModelRegistry(tmp_path / "registry")

    bundle = register_current_candidate(
        registry=registry,
        model_id="candidate-1",
        target="map_win",
        code_version="abc123",
        metrics={"log_loss": 0.64},
        created_at=NOW,
        model_root=model_root,
        config_root=config_root,
        product_config=product_config,
        dependency_lock=lock,
        training_paths=(training,),
    )

    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["data_manifest"]
    assert manifest["feature_fingerprint"]
    assert manifest["config_hash"]
    assert manifest["dependency_lock_hash"]
    assert registry.verify_bundle("candidate-1")
