from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from lol_bets.operations.readiness import (
    CohortEvidence,
    build_readiness_artifact,
    classification_cohort_evidence,
)

NOW = datetime(2026, 8, 27, 12, tzinfo=UTC)
SHA256_LENGTH = 64
ALL_FIXTURE_CLUSTERS = 3
LCK_FIXTURE_CLUSTERS = 2


def _cell(artifact, target, cohort):
    return next(
        cell
        for cell in artifact["cells"]
        if cell["target"] == target and cell["cohort"] == cohort
    )


def test_underpowered_or_imbalanced_cohort_cannot_activate_recommendations():
    evidence = {
        "actionable": CohortEvidence(29, 0.5, 0.6, 0.59, 0.01),
        "league:LCK": CohortEvidence(50, 0.96, 0.6, 0.59, 0.01),
    }

    artifact = build_readiness_artifact(
        candidate_id="lol-test",
        reviewed_at=NOW,
        target_statuses={"series_winner": "meets_basic_sanity"},
        series_cohorts=evidence,
        preregistered_cohorts=("actionable", "league:LCK"),
    )

    assert _cell(artifact, "series_winner", "actionable")["state"] == "exploration_only"
    assert _cell(artifact, "series_winner", "league:LCK")["state"] == "exploration_only"


def test_healthy_model_and_powered_safe_cohort_can_activate_series_only():
    artifact = build_readiness_artifact(
        candidate_id="lol-test",
        reviewed_at=NOW,
        target_statuses={
            "series_winner": "meets_basic_sanity",
            "map_winner": "meets_basic_sanity",
            "gamelength": "weak_signal",
            "total_kills": "weak_signal",
            "total_towers": "below_constant_baseline",
        },
        series_cohorts={
            "actionable": CohortEvidence(80, 0.5, 0.6, 0.58, 0.01),
        },
        preregistered_cohorts=("actionable",),
    )

    assert (
        _cell(artifact, "series_winner", "actionable")["state"]
        == "recommendation_active"
    )
    assert (
        _cell(artifact, "map_winner:map_1", "all_actionable")["state"]
        == "exploration_only"
    )
    assert (
        _cell(artifact, "gamelength", "all_actionable")["state"] == "exploration_only"
    )
    assert _cell(artifact, "total_towers", "all_actionable")["state"] == "display_only"
    assert artifact["model_health_is_separate"] is True
    assert len(artifact["content_sha256"]) == SHA256_LENGTH


def test_recommendation_gate_failure_keeps_healthy_series_in_exploration() -> None:
    artifact = build_readiness_artifact(
        candidate_id="lol-test",
        reviewed_at=NOW,
        target_statuses={"series_winner": "meets_basic_sanity"},
        series_cohorts={
            "actionable": CohortEvidence(80, 0.5, 0.6, 0.58, 0.01),
        },
        preregistered_cohorts=("actionable",),
        series_recommendation_failures=("conservative_probability_coverage_failed",),
    )

    cell = _cell(artifact, "series_winner", "actionable")
    assert cell["state"] == "exploration_only"
    assert cell["reasons"] == ["conservative_probability_coverage_failed"]


def test_cohort_evidence_counts_unique_fixture_clusters_and_applies_family_bound():
    actual = np.array([1, 1, 0, 0, 1, 1])
    baseline = np.array([0.6, 0.6, 0.4, 0.4, 0.55, 0.55])
    candidate = np.array([0.62, 0.62, 0.38, 0.38, 0.57, 0.57])
    clusters = np.array(["a", "a", "b", "b", "c", "c"])

    evidence = classification_cohort_evidence(
        actual,
        baseline,
        candidate,
        clusters,
        {
            "actionable": np.ones(6, dtype=bool),
            "league:LCK": np.array([True, True, True, True, False, False]),
        },
        bootstrap_samples=200,
    )

    assert evidence["actionable"].fixture_clusters == ALL_FIXTURE_CLUSTERS
    assert evidence["league:LCK"].fixture_clusters == LCK_FIXTURE_CLUSTERS
    assert (
        evidence["actionable"].candidate_log_loss
        < evidence["actionable"].baseline_log_loss
    )
