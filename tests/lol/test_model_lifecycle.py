from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from lol_bets.operations.models import (
    CandidateManifest,
    ModelRegistry,
    ModelRegistryError,
    PromotionEvidence,
    PromotionPolicy,
    TrainingTriggerState,
    _target_drift_review,
    evaluate_promotion,
    evaluate_training_triggers_from_history,
    orchestrate_candidate_training,
    paired_bootstrap_improvement,
    register_current_candidate,
    resolve_serving_artifact,
    review_candidate_on_sealed_rows,
    should_train_candidate,
)
from oracle_bets_core.pd import pd

NOW = datetime(2026, 7, 26, 8, 15, tzinfo=UTC)
MIN_PROMOTION_IMPROVEMENT = 0.01
MAX_ROUTINE_DEGRADATION = 0.01
MAJOR_MAPS = 25
SEALED_ROWS = 80


class _IdentityPipeline:
    def transform(self, frame):
        return frame


class _IdentityCalibrator:
    def predict(self, values, metadata=None):
        del metadata
        return values


class _ProbabilityModel:
    def __init__(self, confidence):
        self.confidence = confidence

    def predict_proba(self, frame):
        import numpy as np

        point = np.where(
            frame["signal"].to_numpy() == 1, self.confidence, 1 - self.confidence
        )
        return np.column_stack([1 - point, point])


class _RegressionModel:
    def __init__(self, error):
        self.error = error

    def predict(self, frame):
        return frame["signal"].to_numpy() + self.error


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
        champion_ece=0.03,
        candidate_ece=0.025,
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
        champion_ece=0.03,
        candidate_ece=0.03,
        cohort_log_loss={},
    )

    decision = evaluate_promotion(evidence, bootstrap_samples=1000, seed=5)

    assert not decision.promote
    assert "log_loss_noninferiority_failed" in decision.reasons


def test_major_cohort_regression_blocks_aggregate_promotion():
    evidence = PromotionEvidence(
        champion_log_losses=(0.70, 0.72, 0.68, 0.71) * 40,
        candidate_log_losses=(0.67, 0.69, 0.65, 0.68) * 40,
        champion_brier=0.22,
        candidate_brier=0.21,
        champion_ece=0.03,
        candidate_ece=0.03,
        cohort_log_loss={"LCK": (0.60, 0.66, 50)},
    )

    decision = evaluate_promotion(evidence, bootstrap_samples=1000, seed=3)

    assert not decision.promote
    assert decision.safety_failures == ("cohort_regression:LCK",)


def test_routine_candidate_can_promote_when_safely_non_inferior():
    champion = tuple([0.60, 0.62, 0.61, 0.63] * 50)
    candidate = tuple(value * 1.002 for value in champion)
    evidence = PromotionEvidence(
        champion_log_losses=champion,
        candidate_log_losses=candidate,
        champion_brier=0.210,
        candidate_brier=0.211,
        champion_ece=0.030,
        candidate_ece=0.035,
        cohort_log_loss={"actionable": (0.61, 0.611, 200)},
        regression_target_mae={"gamelength": (4.0, 4.05)},
        champion_evidence_status={"result": "meets_basic_sanity"},
        candidate_evidence_status={"result": "meets_basic_sanity"},
    )

    decision = evaluate_promotion(evidence, bootstrap_samples=1000)

    assert decision.policy is PromotionPolicy.ROUTINE
    assert decision.promote
    assert decision.confidence_degradation_upper_bound <= MAX_ROUTINE_DEGRADATION


def test_optuna_candidate_requires_meaningful_improvement():
    losses = tuple([0.60, 0.62, 0.61, 0.63] * 50)
    evidence = PromotionEvidence(
        champion_log_losses=losses,
        candidate_log_losses=tuple(value * 0.998 for value in losses),
        champion_brier=0.21,
        candidate_brier=0.209,
        champion_ece=0.03,
        candidate_ece=0.03,
        cohort_log_loss={"actionable": (0.61, 0.609, 200)},
    )

    decision = evaluate_promotion(
        evidence,
        policy=PromotionPolicy.OPTUNA,
        bootstrap_samples=1000,
    )

    assert not decision.promote
    assert "meaningful_improvement_not_proven" in decision.reasons


def test_prop_or_evidence_status_regression_blocks_routine_promotion():
    losses = tuple([0.60, 0.62, 0.61, 0.63] * 50)
    evidence = PromotionEvidence(
        champion_log_losses=losses,
        candidate_log_losses=losses,
        champion_brier=0.21,
        candidate_brier=0.21,
        champion_ece=0.03,
        candidate_ece=0.03,
        cohort_log_loss={"actionable": (0.61, 0.61, 200)},
        regression_target_mae={"total_kills": (6.0, 6.2)},
        champion_evidence_status={"total_towers": "weak_signal"},
        candidate_evidence_status={"total_towers": "below_constant_baseline"},
    )

    decision = evaluate_promotion(evidence, bootstrap_samples=1000)

    assert not decision.promote
    assert "mae_regression:total_kills" in decision.safety_failures
    assert "evidence_status_downgrade:total_towers" in decision.safety_failures


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


def test_incomplete_champion_transition_recovers_history_without_duplication(
    tmp_path,
    monkeypatch,
):
    registry_path = tmp_path / "registry"
    registry = ModelRegistry(registry_path)
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"model")
    registry.register_candidate(_manifest("candidate-1"), {"model.pkl": artifact})
    original_append = registry._append_champion_history

    def interrupt(_history, **_kwargs):
        raise OSError("simulated history failure")

    monkeypatch.setattr(registry, "_append_champion_history", interrupt)
    with pytest.raises(OSError, match="history failure"):
        registry.promote("candidate-1", promoted_at=NOW, reason="gate passed")

    assert registry.transition_path.is_file()
    recovered = ModelRegistry(registry_path)
    assert recovered.champion_id() == "candidate-1"
    assert not recovered.transition_path.exists()
    history = recovered.history_path.read_text(encoding="utf-8").splitlines()
    assert len(history) == 1

    original_append(json.loads(history[0]))
    assert len(recovered.history_path.read_text(encoding="utf-8").splitlines()) == 1


def test_routine_review_replays_both_bundles_on_candidate_sealed_rows(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    _register_replay_bundle(
        registry,
        tmp_path / "champion",
        "champion",
        confidence=0.65,
        regression_error=1.0,
        include_evaluation=False,
    )
    _register_replay_bundle(
        registry,
        tmp_path / "candidate",
        "candidate",
        confidence=0.70,
        regression_error=0.5,
        include_evaluation=True,
    )
    registry.promote("champion", promoted_at=NOW, reason="bootstrap")

    review = review_candidate_on_sealed_rows(
        registry,
        "candidate",
        reviewed_at=NOW + timedelta(minutes=1),
        bootstrap_samples=500,
    )

    assert review.status == "auto_promoted"
    assert review.promoted
    assert registry.champion_id() == "candidate"
    assert set(review.row_fingerprints) == {
        "result",
        "gamelength",
        "total_kills",
        "total_towers",
    }
    assert review.evidence["sealed_rows"]["result"]["rows"] == SEALED_ROWS
    drift = review.evidence["drift_review"]
    assert drift["status"] == "warning_only"
    assert drift["promotion_gate_effect"] == "none"
    assert drift["targets"]["result"]["feature_availability"]["columns"] == 1


def test_routine_promotion_writes_recoverable_approval_before_pointer_change(
    tmp_path,
    monkeypatch,
):
    registry = ModelRegistry(tmp_path / "registry")
    _register_replay_bundle(
        registry,
        tmp_path / "champion",
        "champion",
        confidence=0.65,
        regression_error=1.0,
        include_evaluation=False,
    )
    _register_replay_bundle(
        registry,
        tmp_path / "candidate",
        "candidate",
        confidence=0.70,
        regression_error=0.5,
        include_evaluation=True,
    )
    registry.promote("champion", promoted_at=NOW, reason="bootstrap")

    def fail_promotion(*_args, **_kwargs):
        raise OSError("simulated pointer write failure")

    monkeypatch.setattr(registry, "promote", fail_promotion)
    with pytest.raises(OSError, match="pointer write failure"):
        review_candidate_on_sealed_rows(
            registry,
            "candidate",
            reviewed_at=NOW + timedelta(minutes=1),
            bootstrap_samples=500,
        )

    persisted = json.loads(
        (registry.root / "reviews" / "candidate.json").read_text(encoding="utf-8")
    )
    assert persisted["status"] == "promotion_approved"


def test_drift_diagnostic_failure_never_blocks_predictive_promotion(
    tmp_path,
    monkeypatch,
):
    registry = ModelRegistry(tmp_path / "registry")
    _register_replay_bundle(
        registry,
        tmp_path / "champion",
        "champion",
        confidence=0.65,
        regression_error=1.0,
        include_evaluation=False,
    )
    _register_replay_bundle(
        registry,
        tmp_path / "candidate",
        "candidate",
        confidence=0.70,
        regression_error=0.5,
        include_evaluation=True,
    )
    registry.promote("champion", promoted_at=NOW, reason="bootstrap")

    def fail_drift(*_args, **_kwargs):
        raise ValueError("bad drift")

    monkeypatch.setattr(
        "lol_bets.operations.models._target_drift_review",
        fail_drift,
    )

    review = review_candidate_on_sealed_rows(
        registry,
        "candidate",
        reviewed_at=NOW + timedelta(minutes=1),
        bootstrap_samples=500,
    )

    assert review.status == "auto_promoted"
    assert review.evidence["drift_review"]["targets"]["result"] == {
        "status": "unavailable",
        "warnings": ["drift_review_unavailable:ValueError"],
    }


def test_drift_review_reports_each_warning_without_becoming_a_gate(
    tmp_path,
    monkeypatch,
):
    candidate = tmp_path / "candidate"
    champion = tmp_path / "champion"
    monkeypatch.setattr(
        "lol_bets.operations.models._split_profile",
        lambda root, _model: {
            "date_window": {"min": "2026-01-01", "max": "2026-08-01"},
            "leagues": ["LCK", "LEC"] if root == candidate else ["LCK"],
        },
    )
    monkeypatch.setattr(
        "lol_bets.operations.models._attribution_profile",
        lambda root, _model: {
            "top_features": (
                [f"candidate_{index}" for index in range(10)]
                if root == candidate
                else [f"champion_{index}" for index in range(10)]
            ),
            "top_families": {},
        },
    )

    review = _target_drift_review(
        pd.DataFrame({"missing": [None, None, 1.0], "present": [1.0, 2.0, 3.0]}),
        pd.DataFrame({"league": ["LCK", "LCK", "LEC"]}),
        champion_prediction=np.asarray([0.4, 0.4, 0.4]),
        candidate_prediction=np.asarray([0.6, 0.6, 0.6]),
        candidate_root=candidate,
        champion_root=champion,
        model_name="OutcomePrediction_LightGBM",
    )

    assert set(review["warnings"]) == {
        "high_feature_missingness",
        "league_coverage_changed",
        "top_feature_attribution_changed",
        "prediction_mean_shift",
    }


def test_optuna_derived_review_never_auto_promotes(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    _register_replay_bundle(
        registry,
        tmp_path / "champion",
        "champion",
        confidence=0.65,
        regression_error=1.0,
        include_evaluation=False,
    )
    _register_replay_bundle(
        registry,
        tmp_path / "candidate",
        "candidate",
        confidence=0.72,
        regression_error=0.5,
        include_evaluation=True,
    )
    registry.promote("champion", promoted_at=NOW, reason="bootstrap")

    review = review_candidate_on_sealed_rows(
        registry,
        "candidate",
        policy=PromotionPolicy.OPTUNA,
        reviewed_at=NOW + timedelta(minutes=1),
        bootstrap_samples=500,
    )

    assert review.status == "manual_review_required"
    assert not review.promoted
    assert registry.champion_id() == "champion"


def _register_replay_bundle(
    registry,
    root,
    model_id,
    *,
    confidence,
    regression_error,
    include_evaluation,
):
    import numpy as np

    model_names = {
        "result": "OutcomePrediction_LightGBM",
        "gamelength": "GamelengthPrediction_LightGBM",
        "total_kills": "TotalKillsPrediction_LightGBM",
        "total_towers": "TotalTowersPrediction_LightGBM",
    }
    statuses = []
    for target, name in model_names.items():
        model_root = root / name
        model_root.mkdir(parents=True, exist_ok=True)
        _pickle(model_root / f"{name}_feature_pipeline.pkl", _IdentityPipeline())
        if target == "result":
            _pickle(model_root / f"{name}.pkl", _ProbabilityModel(confidence))
            _pickle(
                model_root / f"{name}_probability_calibrator.pkl", _IdentityCalibrator()
            )
            _pickle(
                model_root / f"{name}_outcome_matchup_schema.pkl",
                {
                    "version": 1,
                    "excluded_features": ["first_pick", "side_win_likelihood"],
                },
            )
        else:
            _pickle(model_root / f"{name}.pkl", _RegressionModel(regression_error))
        statuses.append(
            {
                "target": target,
                "evidence_status": (
                    "meets_basic_sanity" if target == "result" else "weak_signal"
                ),
            }
        )
        if include_evaluation:
            evaluation = root / "_evaluation" / name
            evaluation.mkdir(parents=True, exist_ok=True)
            if target == "result":
                actual = np.tile([1, 0], 40)
            else:
                actual = np.linspace(10, 20, 80)
            pd.DataFrame({"signal": actual}).to_parquet(
                evaluation / "features.parquet", index=False
            )
            pd.DataFrame(
                {
                    "actual": actual,
                    "gameid": [f"game-{index}" for index in range(80)],
                    "league": ["LCK"] * 80,
                    "league_region": ["Korea"] * 80,
                    "league_tier": ["tier1"] * 80,
                    "actionable": [True] * 80,
                }
            ).to_parquet(evaluation / "labels.parquet", index=False)
        evaluation = root / "_evaluation" / name
        evaluation.mkdir(parents=True, exist_ok=True)
        (evaluation / "split_report.json").write_text(
            json.dumps(
                {
                    "checks": {
                        "gameids_disjoint": True,
                        "temporal_ordered": True,
                    }
                }
            )
        )
    evaluation_root = root / "_evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    (evaluation_root / "summary.json").write_text(json.dumps({"models": statuses}))
    artifact_map = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    }
    registry.register_candidate(_manifest(model_id), artifact_map)


def _pickle(path, value):
    with path.open("wb") as stream:
        pickle.dump(value, stream)


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
