"""Versioned, model-only strategy readiness decisions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np

READINESS_SCHEMA_VERSION = 1
READINESS_POLICY_VERSION = "lol-readiness-v1"
MIN_FIXTURE_CLUSTERS = 30
MIN_OUTCOME_FRACTION = 0.15
MAX_RELATIVE_LOG_LOSS_REGRESSION = 0.02
MIN_BOOTSTRAP_SAMPLES = 100
RECOMMENDATION_TARGET = "series_winner"
EXPLORATION_TARGETS = (
    "map_winner:map_1",
    "map_winner:later_maps",
    "next_map_winner",
    "series_total_maps",
    "series_handicap",
    "gamelength",
    "total_kills",
    "total_towers",
)


class ReadinessState(StrEnum):
    RECOMMENDATION_ACTIVE = "recommendation_active"
    EXPLORATION_ONLY = "exploration_only"
    DISPLAY_ONLY = "display_only"


@dataclass(frozen=True)
class CohortEvidence:
    """Fixture-clustered evidence for one preregistered model cohort."""

    fixture_clusters: int
    positive_fraction: float
    baseline_log_loss: float
    candidate_log_loss: float
    regression_upper_bound: float

    def __post_init__(self) -> None:
        if self.fixture_clusters < 0:
            raise ValueError("fixture_clusters cannot be negative")
        if not 0 <= self.positive_fraction <= 1:
            raise ValueError("positive_fraction must be in [0, 1]")
        for name in (
            "baseline_log_loss",
            "candidate_log_loss",
            "regression_upper_bound",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")


def classification_cohort_evidence(
    actual: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    clusters: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 2_000,
    seed: int = 41,
) -> dict[str, CohortEvidence]:
    """Build multiplicity-aware, fixture-clustered cohort evidence."""
    actual = np.asarray(actual, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    clusters = np.asarray(clusters).astype(str)
    if not (
        actual.shape == baseline.shape == candidate.shape == clusters.shape
        and actual.ndim == 1
    ):
        raise ValueError("cohort evidence arrays must be aligned one-dimensional rows")
    if not masks:
        return {}
    if not 0 < confidence < 1 or bootstrap_samples < MIN_BOOTSTRAP_SAMPLES:
        raise ValueError("invalid cohort bootstrap configuration")
    baseline_losses = _binary_losses(actual, baseline)
    candidate_losses = _binary_losses(actual, candidate)
    family_size = max(1, len(masks))
    quantile = 1.0 - (1.0 - confidence) / family_size
    result: dict[str, CohortEvidence] = {}
    for offset, (name, raw_mask) in enumerate(sorted(masks.items())):
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != actual.shape:
            raise ValueError(f"cohort mask is misaligned: {name}")
        if not np.any(mask):
            continue
        rows = _cluster_rows(
            actual[mask],
            baseline_losses[mask],
            candidate_losses[mask],
            clusters[mask],
        )
        baseline_mean = float(np.mean(rows[:, 1]))
        candidate_mean = float(np.mean(rows[:, 2]))
        if baseline_mean <= 0:
            upper = float("inf")
        else:
            generator = np.random.default_rng(seed + offset)
            indices = generator.integers(
                0,
                len(rows),
                size=(bootstrap_samples, len(rows)),
            )
            regressions = np.mean(rows[indices, 2] - rows[indices, 1], axis=1)
            upper = float(np.quantile(regressions / baseline_mean, quantile))
        result[name] = CohortEvidence(
            fixture_clusters=len(rows),
            positive_fraction=float(np.mean(rows[:, 0])),
            baseline_log_loss=baseline_mean,
            candidate_log_loss=candidate_mean,
            regression_upper_bound=upper,
        )
    return result


def build_readiness_artifact(
    *,
    candidate_id: str,
    reviewed_at: datetime,
    target_statuses: dict[str, str],
    series_cohorts: dict[str, CohortEvidence],
    preregistered_cohorts: tuple[str, ...],
    series_recommendation_failures: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build every target/cohort state without consulting market or final policy data."""
    if reviewed_at.tzinfo is None:
        raise ValueError("reviewed_at must be timezone aware")
    reviewed_at = reviewed_at.astimezone(UTC)
    cells: list[dict[str, Any]] = []
    series_structural = (
        target_statuses.get(RECOMMENDATION_TARGET) == "meets_basic_sanity"
    )
    for cohort in preregistered_cohorts:
        evidence = series_cohorts.get(cohort)
        reasons: list[str] = []
        if not series_structural:
            state = ReadinessState.DISPLAY_ONLY
            reasons.append("series_model_not_structurally_ready")
        elif series_recommendation_failures:
            state = ReadinessState.EXPLORATION_ONLY
            reasons.extend(series_recommendation_failures)
        elif evidence is None:
            state = ReadinessState.EXPLORATION_ONLY
            reasons.append("cohort_evidence_missing")
        else:
            if evidence.fixture_clusters < MIN_FIXTURE_CLUSTERS:
                reasons.append("underpowered_fixture_clusters")
            if not (
                MIN_OUTCOME_FRACTION
                <= evidence.positive_fraction
                <= 1.0 - MIN_OUTCOME_FRACTION
            ):
                reasons.append("outcome_balance_insufficient")
            if evidence.regression_upper_bound > MAX_RELATIVE_LOG_LOSS_REGRESSION:
                reasons.append("multiplicity_adjusted_regression_not_excluded")
            state = (
                ReadinessState.RECOMMENDATION_ACTIVE
                if not reasons
                else ReadinessState.EXPLORATION_ONLY
            )
        cells.append(
            _cell(RECOMMENDATION_TARGET, cohort, state, reasons, evidence=evidence)
        )

    for target in EXPLORATION_TARGETS:
        base_target = target.split(":", 1)[0]
        source_target = (
            "map_winner"
            if base_target in {"series_total_maps", "series_handicap"}
            else base_target
        )
        status = target_statuses.get(source_target)
        structurally_available = status in {"meets_basic_sanity", "weak_signal"}
        state = (
            ReadinessState.EXPLORATION_ONLY
            if structurally_available
            else ReadinessState.DISPLAY_ONLY
        )
        reason = (
            "experimental_target_requires_own_sealed_and_paper_evidence"
            if structurally_available
            else "target_model_not_structurally_ready"
        )
        cells.append(_cell(target, "all_actionable", state, [reason]))

    payload: dict[str, Any] = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "policy_version": READINESS_POLICY_VERSION,
        "candidate_id": candidate_id,
        "reviewed_at": reviewed_at.isoformat(),
        "decision_schedule": "fixed_monthly_owner_review",
        "model_health_is_separate": True,
        "cells": cells,
    }
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _cell(
    target: str,
    cohort: str,
    state: ReadinessState,
    reasons: list[str],
    *,
    evidence: CohortEvidence | None = None,
) -> dict[str, Any]:
    return {
        "target": target,
        "cohort": cohort,
        "state": state.value,
        "reasons": reasons,
        "evidence": asdict(evidence) if evidence is not None else None,
    }


def _binary_losses(actual: np.ndarray, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return -(actual * np.log(clipped) + (1.0 - actual) * np.log(1.0 - clipped))


def _cluster_rows(
    actual: np.ndarray,
    baseline_loss: np.ndarray,
    candidate_loss: np.ndarray,
    clusters: np.ndarray,
) -> np.ndarray:
    rows = []
    for cluster in sorted(set(clusters.tolist())):
        mask = clusters == cluster
        rows.append(
            (
                float(np.mean(actual[mask])),
                float(np.mean(baseline_loss[mask])),
                float(np.mean(candidate_loss[mask])),
            )
        )
    return np.asarray(rows, dtype=float)
