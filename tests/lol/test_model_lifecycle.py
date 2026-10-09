from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime, timedelta
from typing import ClassVar

import numpy as np
import pytest
from lol_bets.operations.models import (
    CandidateManifest,
    ModelRegistry,
    ModelRegistryError,
    PromotionEvidence,
    PromotionPolicy,
    _clustered_binary_log_losses,
    _cohort_replay_losses,
    _replay_bundle_target,
    _target_drift_review,
    evaluate_promotion,
    evaluate_training_triggers_from_history,
    paired_bootstrap_improvement,
    register_current_candidate,
    resolve_serving_artifact,
    review_candidate_on_sealed_rows,
)
from oracle_bets_core.pd import pd

NOW = datetime(2026, 7, 26, 8, 15, tzinfo=UTC)
MIN_PROMOTION_IMPROVEMENT = 0.01
MAX_ROUTINE_DEGRADATION = 0.01
MAJOR_MAPS = 25
SEALED_ROWS = 80


def test_next_map_evidence_clusters_correlated_maps_by_series() -> None:
    expected_clusters = 2
    labels = pd.DataFrame(
        {
            "series_id": ["s1", "s1", "s2"],
            "league": ["LCK", "LCK", "LCK"],
            "actionable": [True, True, True],
        }
    )
    actual = np.array([1.0, 0.0, 1.0])
    baseline = np.array([0.6, 0.6, 0.6])
    candidate = np.array([0.7, 0.4, 0.7])

    clustered = _clustered_binary_log_losses(
        labels,
        actual,
        candidate,
        cluster_col="series_id",
    )
    cohorts = _cohort_replay_losses(
        labels,
        actual,
        baseline,
        candidate,
        cluster_col="series_id",
    )

    assert len(clustered) == expected_clusters
    assert cohorts["all_research_all_supported"][2] == expected_clusters
    assert cohorts["actionable_tier1_plus_erls"][2] == expected_clusters


def test_clustered_next_map_evidence_requires_series_identity() -> None:
    with pytest.raises(ValueError, match="complete series_id clusters"):
        _clustered_binary_log_losses(
            pd.DataFrame({"series_id": ["s1", None]}),
            np.array([1.0, 0.0]),
            np.array([0.6, 0.4]),
            cluster_col="series_id",
        )


class _IdentityPipeline:
    train_columns: ClassVar[list[str]] = [
        "signal",
        "delta_elo",
        "delta_glicko2_mu",
        "delta_glicko2_phi",
        "delta_pl_mu",
        "delta_pl_sigma",
        "delta_trueskill_mu",
        "delta_trueskill_sigma",
        "delta_elo_win_likelihood",
        "delta_glicko2_win_likelihood",
        "delta_pl_win_likelihood",
        "delta_trueskill_win_likelihood",
    ]

    def transform(self, frame):
        output = frame.copy()
        for column in self.train_columns:
            if column not in output:
                output[column] = 0.5
        return output.loc[:, self.train_columns]


class _IdentityCalibrator:
    version = 3

    def predict(self, values, metadata=None):
        del metadata
        return values


class _IdentityUncertainty:
    version = 2
    fit_split = "uncertainty_fit"
    sample_count = SEALED_ROWS
    calibration_units = 10
    method = "week_block_q10_plus_one_sided_calibration_bias"
    unit = "iso_week"

    @staticmethod
    def interval(values):
        return values, values


class _ProbabilityModel:
    rating_columns = tuple(_IdentityPipeline.train_columns[1:])

    def __init__(self, confidence):
        self.confidence = confidence

    def predict_proba(self, frame):
        import numpy as np

        point = np.where(
            frame["signal"].to_numpy() == 1, self.confidence, 1 - self.confidence
        )
        return np.column_stack([1 - point, point])

    def rating_baseline_probability(self, frame):
        return self.predict_proba(frame)[:, 1]

    @staticmethod
    def conservative_probability(frame, calibrator=None, metadata=None):
        del calibrator, metadata
        return np.full(len(frame), 0.01)


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


def test_actionable_cohort_regression_blocks_aggregate_promotion():
    evidence = PromotionEvidence(
        champion_log_losses=(0.70, 0.72, 0.68, 0.71) * 40,
        candidate_log_losses=(0.67, 0.69, 0.65, 0.68) * 40,
        champion_brier=0.22,
        candidate_brier=0.21,
        champion_ece=0.03,
        candidate_ece=0.03,
        cohort_log_loss={"actionable_tier1_plus_erls": (0.60, 0.66, 50)},
    )

    decision = evaluate_promotion(evidence, bootstrap_samples=1000, seed=3)

    assert not decision.promote
    assert decision.safety_failures == ("cohort_regression:actionable_tier1_plus_erls",)


def test_small_cohort_regression_is_diagnostic_not_a_promotion_blocker():
    evidence = PromotionEvidence(
        champion_log_losses=(0.70, 0.72, 0.68, 0.71) * 40,
        candidate_log_losses=(0.67, 0.69, 0.65, 0.68) * 40,
        champion_brier=0.22,
        candidate_brier=0.21,
        champion_ece=0.03,
        candidate_ece=0.03,
        cohort_log_loss={"league:CBLOL": (0.60, 0.66, 50)},
    )

    decision = evaluate_promotion(
        evidence,
        policy=PromotionPolicy.OPTUNA,
        bootstrap_samples=1000,
        seed=3,
    )

    assert decision.promote
    assert decision.safety_failures == ()


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


def test_shadow_prop_regression_does_not_block_winner_promotion():
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

    assert decision.promote
    assert decision.safety_failures == ()


def test_history_evidence_drives_training_trigger(tmp_path):
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
    assert evaluation.triggered
    assert "new_valid_maps" in evaluation.reasons
    assert "new_major_maps" in evaluation.reasons


def test_recent_history_does_not_trigger_training(tmp_path):
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

    assert not evaluation.triggered


def _manifest(model_id="candidate-1", *, target="map_win"):
    return CandidateManifest(
        model_id=model_id,
        sport="lol",
        target=target,
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


def test_complete_lol_bundle_rejects_missing_serving_artifacts(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"model bytes")

    with pytest.raises(ModelRegistryError, match="missing required artifacts"):
        registry.register_candidate(
            _manifest(target="complete_lol_bundle"),
            {"model.pkl": artifact},
        )


def test_complete_lol_bundle_verification_checks_semantic_completeness(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"model bytes")
    bundle = registry.register_candidate(_manifest(), {"model.pkl": artifact})
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["target"] = "complete_lol_bundle"
    manifest_path.write_text(json.dumps(manifest))

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


def test_quarantined_model_is_research_only_and_cannot_be_promoted(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"model")
    registry.register_candidate(_manifest(), {"model.pkl": artifact})

    registry.quarantine(
        "candidate-1",
        quarantined_at=NOW,
        reason="winner v2 rebuild required",
    )

    assert registry.verify_bundle("candidate-1")
    assert not registry.is_actionable("candidate-1")
    assert registry.actionability("candidate-1")["status"] == "research_only"
    with pytest.raises(ModelRegistryError, match="quarantined model"):
        registry.promote("candidate-1", promoted_at=NOW, reason="unsafe")


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
        "series_winner",
        "gamelength",
        "total_kills",
        "total_towers",
    }
    assert review.evidence["sealed_rows"]["series_winner"]["rows"] == SEALED_ROWS
    drift = review.evidence["drift_review"]
    assert drift["status"] == "warning_only"
    assert drift["promotion_gate_effect"] == "none"
    assert drift["targets"]["series_winner"]["feature_availability"]["columns"] == len(
        _IdentityPipeline.train_columns
    )


def test_promotion_replay_rejects_absent_model_features_before_imputation(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    _register_replay_bundle(
        registry,
        tmp_path / "candidate",
        "candidate",
        confidence=0.7,
        regression_error=0.5,
        include_evaluation=True,
    )
    with pytest.raises(ValueError, match="missing trained features"):
        _replay_bundle_target(
            registry.candidates / "candidate",
            model_name="SeriesWinnerPrediction_LightGBM",
            raw_features=pd.DataFrame({"signal": [1.0]}),
            metadata=pd.DataFrame({"league": ["LCK"]}),
            classification=True,
        )


def test_experimental_next_map_regression_does_not_block_prematch_candidate(tmp_path):
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
        next_map_confidence=0.50,
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
    assert not any(reason.startswith("next_map_winner:") for reason in review.reasons)


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
    assert review.evidence["drift_review"]["targets"]["series_winner"] == {
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


def test_first_v2_without_legacy_champion_runs_internal_baseline_review(
    tmp_path, monkeypatch
):
    registry = ModelRegistry(tmp_path / "registry")
    _register_replay_bundle(
        registry,
        tmp_path / "candidate",
        "candidate",
        confidence=0.70,
        regression_error=0.5,
        include_evaluation=True,
    )
    monkeypatch.setattr(
        "lol_bets.prediction_models.gbdt_model.GradientBoostingModel."
        "compute_probability_calibration_metrics",
        lambda *_args, **_kwargs: {
            "calibration_slope": 1.0,
            "calibration_intercept": -0.15,
        },
    )
    monkeypatch.setattr(
        "lol_bets.prediction_models.gbdt_model.conservative_probability_report",
        lambda *_args, **_kwargs: {"coverage": {"passed": False}},
    )

    review = review_candidate_on_sealed_rows(
        registry,
        "candidate",
        reviewed_at=NOW,
        bootstrap_samples=500,
    )

    assert review.evidence["first_winner_v2"] is True
    assert review.evidence["comparator"] == "predeclared_rating_logistic_baseline"
    assert set(review.evidence["winner_targets"]) == {
        "series_winner",
    }
    assert "no_champion_comparator" not in review.reasons
    assert review.status == "manual_review_required"
    assert review.evidence["winner_targets"]["series_winner"]["warnings"] == [
        "calibration_intercept_above_0.10"
    ]
    readiness = review.evidence["strategy_readiness"]
    assert readiness["model_health_is_separate"] is True
    actionable = next(
        cell
        for cell in readiness["cells"]
        if cell["target"] == "series_winner"
        and cell["cohort"] == "actionable_tier1_plus_erls"
    )
    assert actionable["state"] == "exploration_only"
    assert actionable["reasons"] == [
        "calibration_intercept_above_0.10",
        "conservative_probability_coverage_failed",
    ]


def test_champion_transition_activates_reviewed_readiness_once(tmp_path, monkeypatch):
    registry = ModelRegistry(tmp_path / "registry")
    candidate = registry.candidates / "candidate"
    model = candidate / "SeriesWinnerPrediction_LightGBM"
    model.mkdir(parents=True)
    (model / "SeriesWinnerPrediction_LightGBM.pkl").write_bytes(b"model")
    (candidate / "manifest.json").write_text(
        json.dumps({"target": "complete_lol_bundle"}), encoding="utf-8"
    )
    readiness = {
        "candidate_id": "candidate",
        "schema_version": 1,
        "policy_version": "lol-readiness-v1",
        "cells": [],
    }
    reviews = registry.root / "reviews"
    reviews.mkdir()
    (reviews / "candidate.json").write_text(
        json.dumps(
            {
                "status": "manual_review_required",
                "reasons": [],
                "evidence": {"strategy_readiness": readiness},
            }
        ),
        encoding="utf-8",
    )

    def bundle_is_valid(model_id):
        return model_id == "candidate"

    monkeypatch.setattr(registry, "verify_bundle", bundle_is_valid)

    registry.promote("candidate", promoted_at=NOW, reason="reviewed")

    assert registry.strategy_readiness()["candidate_id"] == "candidate"
    history = registry.strategy_readiness_history_path.read_text().splitlines()
    assert len(history) == 1


def _register_replay_bundle(
    registry,
    root,
    model_id,
    *,
    confidence,
    next_map_confidence=None,
    regression_error,
    include_evaluation,
):
    import numpy as np

    model_names = {
        "series_winner": "SeriesWinnerPrediction_LightGBM",
        "next_map_winner": "NextMapWinnerPrediction_LightGBM",
        "gamelength": "GamelengthPrediction_LightGBM",
        "total_kills": "TotalKillsPrediction_LightGBM",
        "total_towers": "TotalTowersPrediction_LightGBM",
    }
    statuses = []
    for target, name in model_names.items():
        model_root = root / name
        model_root.mkdir(parents=True, exist_ok=True)
        _pickle(model_root / f"{name}_feature_pipeline.pkl", _IdentityPipeline())
        if target in {"series_winner", "next_map_winner"}:
            target_confidence = (
                next_map_confidence
                if target == "next_map_winner" and next_map_confidence is not None
                else confidence
            )
            _pickle(model_root / f"{name}.pkl", _ProbabilityModel(target_confidence))
            (model_root / f"{name}_feature_lineage.json").write_text(
                json.dumps(
                    [
                        {
                            "feature": column,
                            "source": column,
                            "availability_timestamp": ("strictly_before_fixture_start"),
                            "family": "prematch",
                            "swap_behavior": "canonical_team_a_minus_team_b",
                            "model_eligible": True,
                        }
                        for column in _IdentityPipeline.train_columns
                    ]
                )
            )
            _pickle(
                model_root / f"{name}_probability_calibrator.pkl", _IdentityCalibrator()
            )
            _pickle(
                model_root / f"{name}_probability_uncertainty.pkl",
                _IdentityUncertainty(),
            )
            _pickle(
                model_root / f"{name}_outcome_matchup_schema.pkl",
                {
                    "version": 1,
                    "canonical_key": "teamid_then_teamname",
                    "excluded_features": ["first_pick", "side_win_likelihood"],
                },
            )
        else:
            _pickle(model_root / f"{name}.pkl", _RegressionModel(regression_error))
        statuses.append(
            {
                "target": target,
                "evidence_status": (
                    "meets_basic_sanity"
                    if target in {"series_winner", "next_map_winner"}
                    else "weak_signal"
                ),
            }
        )
        if include_evaluation:
            evaluation = root / "_evaluation" / name
            evaluation.mkdir(parents=True, exist_ok=True)
            if target in {"series_winner", "next_map_winner"}:
                actual = np.tile([1, 0], 40)
            else:
                actual = np.linspace(10, 20, 80)
            pd.DataFrame(
                {"signal": actual}
                | dict.fromkeys(_IdentityPipeline.train_columns[1:], 0.5)
            ).to_parquet(evaluation / "features.parquet", index=False)
            pd.DataFrame(
                {
                    "actual": actual,
                    "gameid": [f"game-{index}" for index in range(80)],
                    "league": ["LCK"] * 80,
                    "league_region": ["Korea"] * 80,
                    "league_tier": ["tier1"] * 80,
                    "actionable": [True] * 80,
                    "series_id": [f"series-{index // 2}" for index in range(80)],
                    "date": pd.date_range("2026-01-01", periods=80, freq="D"),
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
