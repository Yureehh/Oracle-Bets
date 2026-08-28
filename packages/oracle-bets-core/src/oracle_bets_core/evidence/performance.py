"""Prediction, paper strategy, uncertainty, activation, and readiness reports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import numpy as np

from oracle_bets_core.evidence.settlement import SettlementResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from oracle_bets_core.evidence.contracts import DecisionMode

_PROBABILITY_EPSILON = 1e-12
_DEFAULT_BOOTSTRAP_SAMPLES = 10_000
_MIN_BOOTSTRAP_SAMPLES = 100


@dataclass(frozen=True)
class PredictionQuality:
    count: int
    log_loss: float
    brier: float
    calibration_error: float


def prediction_quality(
    actuals: tuple[int, ...] | list[int],
    probabilities: tuple[float, ...] | list[float],
    *,
    bins: int = 10,
) -> PredictionQuality:
    """Calculate proper probability scores and equal-width calibration error."""
    actual = np.asarray(actuals, dtype=float)
    probability = np.asarray(probabilities, dtype=float)
    if (
        actual.ndim != 1
        or probability.ndim != 1
        or actual.size == 0
        or actual.shape != probability.shape
    ):
        raise ValueError("actuals and probabilities require equal non-empty vectors")
    if not np.all(np.isin(actual, [0, 1])):
        raise ValueError("actuals must be binary")
    if not np.all(np.isfinite(probability)) or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if bins <= 0:
        raise ValueError("bins must be positive")

    clipped = np.clip(probability, _PROBABILITY_EPSILON, 1 - _PROBABILITY_EPSILON)
    log_loss = float(
        -np.mean(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped))
    )
    brier = float(np.mean((probability - actual) ** 2))
    boundaries = np.linspace(0, 1, bins + 1)
    bin_ids = np.minimum(np.digitize(probability, boundaries[1:-1]), bins - 1)
    calibration_error = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if np.any(mask):
            calibration_error += float(
                np.mean(mask) * abs(np.mean(probability[mask]) - np.mean(actual[mask]))
            )
    return PredictionQuality(
        count=int(actual.size),
        log_loss=log_loss,
        brier=brier,
        calibration_error=calibration_error,
    )


@dataclass(frozen=True)
class SettledPerformanceRow:
    """One chronological settled position for grouped strategy reporting."""

    position_id: str
    settled_order: int
    stake_units: Decimal
    pnl_units: Decimal
    result: SettlementResult
    league: str
    market: str
    strategy: str
    model_version: str
    mode: DecisionMode
    edge_band: str
    probability_clv: float | None
    evidence_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "position_id",
            "league",
            "market",
            "strategy",
            "model_version",
            "edge_band",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} cannot be empty")
        if self.settled_order < 0:
            raise ValueError("settled_order cannot be negative")
        if self.stake_units <= 0:
            raise ValueError("stake_units must be positive")
        if self.probability_clv is not None and not math.isfinite(self.probability_clv):
            raise ValueError("probability_clv must be finite when available")


@dataclass(frozen=True)
class PerformanceMetrics:
    settled_count: int
    graded_count: int
    wins: int
    losses: int
    pushes: int
    voids: int
    turnover_units: Decimal
    pnl_units: Decimal
    roi: float
    hit_rate: float | None
    mean_probability_clv: float | None
    maximum_drawdown_units: Decimal
    maximum_drawdown_fraction: float


def aggregate_performance(
    rows: tuple[SettledPerformanceRow, ...] | list[SettledPerformanceRow],
    *,
    initial_bankroll_units: Decimal = Decimal(100),
) -> PerformanceMetrics:
    """Aggregate accounting in chronological settlement order."""
    if initial_bankroll_units <= 0:
        raise ValueError("initial_bankroll_units must be positive")
    ordered = sorted(rows, key=lambda row: (row.settled_order, row.position_id))
    turnover = sum((row.stake_units for row in ordered), Decimal(0))
    pnl = sum((row.pnl_units for row in ordered), Decimal(0))
    counts = {
        result: sum(row.result is result for row in ordered)
        for result in SettlementResult
    }
    graded = counts[SettlementResult.WIN] + counts[SettlementResult.LOSS]
    clv_values = [
        row.probability_clv for row in ordered if row.probability_clv is not None
    ]

    equity = initial_bankroll_units
    peak = equity
    maximum_drawdown = Decimal(0)
    maximum_drawdown_fraction = 0.0
    for row in ordered:
        equity += row.pnl_units
        peak = max(peak, equity)
        drawdown = peak - equity
        maximum_drawdown = max(maximum_drawdown, drawdown)
        if peak > 0:
            maximum_drawdown_fraction = max(
                maximum_drawdown_fraction,
                float(drawdown / peak),
            )

    return PerformanceMetrics(
        settled_count=len(ordered),
        graded_count=graded,
        wins=counts[SettlementResult.WIN],
        losses=counts[SettlementResult.LOSS],
        pushes=counts[SettlementResult.PUSH],
        voids=counts[SettlementResult.VOID],
        turnover_units=turnover,
        pnl_units=pnl,
        roi=float(pnl / turnover) if turnover else 0.0,
        hit_rate=counts[SettlementResult.WIN] / graded if graded else None,
        mean_probability_clv=(float(np.mean(clv_values)) if clv_values else None),
        maximum_drawdown_units=maximum_drawdown,
        maximum_drawdown_fraction=maximum_drawdown_fraction,
    )


def group_performance(
    rows: tuple[SettledPerformanceRow, ...] | list[SettledPerformanceRow],
    *,
    key: Callable[[SettledPerformanceRow], str],
) -> dict[str, PerformanceMetrics]:
    """Aggregate arbitrary explicit cohorts without blending them silently."""
    groups: dict[str, list[SettledPerformanceRow]] = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    return {
        group: aggregate_performance(group_rows)
        for group, group_rows in sorted(groups.items())
    }


@dataclass(frozen=True)
class BootstrapInterval:
    point: float
    lower: float
    upper: float
    confidence: float
    samples: int


def bootstrap_roi(
    rows: tuple[SettledPerformanceRow, ...] | list[SettledPerformanceRow],
    *,
    confidence: float = 0.95,
    samples: int = _DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 7,
) -> BootstrapInterval | None:
    """Return deterministic position-level ROI uncertainty."""
    if not rows:
        return None
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if samples < _MIN_BOOTSTRAP_SAMPLES:
        raise ValueError(f"bootstrap samples must be at least {_MIN_BOOTSTRAP_SAMPLES}")
    stakes = np.asarray([float(row.stake_units) for row in rows])
    pnl = np.asarray([float(row.pnl_units) for row in rows])
    point = float(np.sum(pnl) / np.sum(stakes))
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(rows), size=(samples, len(rows)))
    sampled_stakes = np.sum(stakes[indices], axis=1)
    sampled_pnl = np.sum(pnl[indices], axis=1)
    bootstrapped = np.divide(
        sampled_pnl,
        sampled_stakes,
        out=np.zeros_like(sampled_pnl),
        where=sampled_stakes > 0,
    )
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(bootstrapped, [tail, 1 - tail])
    return BootstrapInterval(
        point=point,
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        samples=samples,
    )


def fixture_clustered_bootstrap_roi(
    fixture_ids: tuple[str, ...] | list[str],
    stakes: tuple[float, ...] | list[float],
    pnl: tuple[float, ...] | list[float],
    *,
    confidence: float = 0.95,
    samples: int = _DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 7,
) -> BootstrapInterval | None:
    """Bootstrap ROI by fixture so correlated tickets are resampled together."""
    if not (len(fixture_ids) == len(stakes) == len(pnl)):
        raise ValueError("fixture IDs, stakes, and pnl require equal vectors")
    if not fixture_ids:
        return None
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if samples < _MIN_BOOTSTRAP_SAMPLES:
        raise ValueError(f"bootstrap samples must be at least {_MIN_BOOTSTRAP_SAMPLES}")
    clusters: dict[str, tuple[float, float]] = {}
    for fixture_id, stake, profit in zip(fixture_ids, stakes, pnl, strict=True):
        if not fixture_id or not math.isfinite(stake) or stake <= 0:
            raise ValueError("fixture IDs and positive finite stakes are required")
        if not math.isfinite(profit):
            raise ValueError("pnl must be finite")
        previous_stake, previous_pnl = clusters.get(fixture_id, (0.0, 0.0))
        clusters[fixture_id] = (previous_stake + stake, previous_pnl + profit)
    values = tuple(clusters.values())
    cluster_stakes = np.asarray([value[0] for value in values])
    cluster_pnl = np.asarray([value[1] for value in values])
    point = float(np.sum(cluster_pnl) / np.sum(cluster_stakes))
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(samples, len(values)))
    sampled_stakes = np.sum(cluster_stakes[indices], axis=1)
    sampled_pnl = np.sum(cluster_pnl[indices], axis=1)
    bootstrapped = sampled_pnl / sampled_stakes
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(bootstrapped, [tail, 1 - tail])
    return BootstrapInterval(
        point=point,
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        samples=samples,
    )


def summarize_unified_bets(rows: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    """Summarize settled unified bets without mixing modes or currencies."""
    currencies: dict[str, dict[str, Any]] = {}
    for row in rows:
        currency = str(row["currency"])
        bucket = currencies.setdefault(
            currency,
            {
                "settled": 0,
                "graded": 0,
                "turnover": Decimal(0),
                "pnl": Decimal(0),
                "rows": [],
            },
        )
        settlement = row["settlement"] or {}
        bucket["settled"] += 1
        bucket["graded"] += int(settlement.get("result") in {"win", "loss"})
        bucket["turnover"] += Decimal(str(row["stake_amount"]))
        bucket["pnl"] += Decimal(str(settlement.get("pnl_amount") or 0))
        bucket["rows"].append(row)
    for bucket in currencies.values():
        turnover = bucket["turnover"]
        bucket["roi"] = float(bucket["pnl"] / turnover) if turnover else None
        bucket["turnover"] = str(bucket["turnover"])
        bucket["pnl"] = str(bucket["pnl"])
        bucket.update(_unified_performance_evidence(bucket.pop("rows")))
    return {
        "mode": mode,
        "settled": len(rows),
        "currencies": currencies,
        "cohorts": _ticket_cohorts(rows),
    }


def _unified_performance_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["settlement"]["event_at"])
    graded = [
        row
        for row in ordered
        if row["settlement"]["result"] in {"win", "loss"}
        and row["payload"].get("model_probability") is not None
    ]
    quality = None
    if graded:
        scores = prediction_quality(
            [int(row["settlement"]["result"] == "win") for row in graded],
            [float(row["payload"]["model_probability"]) for row in graded],
        )
        quality = {
            "count": scores.count,
            "log_loss": scores.log_loss,
            "brier": scores.brier,
            "calibration_error": scores.calibration_error,
        }
    clv = [
        1 / float(row["settlement"]["closing_odds"]) - 1 / float(row["accepted_odds"])
        for row in ordered
        if row["settlement"].get("closing_odds") is not None
    ]
    equity = peak = Decimal(0)
    maximum_drawdown = Decimal(0)
    for row in ordered:
        equity += Decimal(str(row["settlement"].get("pnl_amount") or 0))
        peak = max(peak, equity)
        maximum_drawdown = max(maximum_drawdown, peak - equity)
    interval = fixture_clustered_bootstrap_roi(
        [str(row["fixture_id"]) for row in ordered],
        [float(row["stake_amount"]) for row in ordered],
        [float(row["settlement"].get("pnl_amount") or 0) for row in ordered],
    )
    paths: dict[str, dict[str, Decimal]] = {}
    for row in ordered:
        fractions = (row["payload"].get("sizing") or {}).get("bankroll_fractions", {})
        for name, pnl in row["settlement"].get("counterfactual_pnl", {}).items():
            bucket = paths.setdefault(name, {"turnover": Decimal(0), "pnl": Decimal(0)})
            bucket["turnover"] += Decimal(str(row["bankroll_before"])) * Decimal(
                str(fractions.get(name, 0))
            )
            bucket["pnl"] += Decimal(str(pnl))
    return {
        "fixture_clusters": len({str(row["fixture_id"]) for row in ordered}),
        "prediction_quality": quality,
        "clv": {
            "complete": len(clv),
            "missing": len(ordered) - len(clv),
            "mean_probability": sum(clv) / len(clv) if clv else None,
        },
        "maximum_drawdown": format(maximum_drawdown.normalize(), "f"),
        "fixture_clustered_roi_95": (
            {
                "point": interval.point,
                "lower": interval.lower,
                "upper": interval.upper,
                "confidence": interval.confidence,
                "samples": interval.samples,
            }
            if interval
            else None
        ),
        "counterfactual_paths": {
            name: {
                "turnover": format(values["turnover"].normalize(), "f"),
                "pnl": format(values["pnl"].normalize(), "f"),
                "roi": (
                    float(values["pnl"] / values["turnover"])
                    if values["turnover"]
                    else None
                ),
            }
            for name, values in paths.items()
        },
    }


def _ticket_cohorts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = {
        "lane": lambda row: row["payload"].get("lane"),
        "target": lambda row: row.get("target"),
        "provider": lambda row: row.get("provider"),
        "strategy": lambda row: row["payload"].get("strategy_version"),
    }
    output: dict[str, Any] = {}
    for dimension, key in keys.items():
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            value = key(row)
            if value is not None:
                groups.setdefault(str(value), []).append(row)
        output[dimension] = {
            value: {
                "tickets": len(group),
                "fixture_clusters": len({str(row["fixture_id"]) for row in group}),
            }
            for value, group in sorted(groups.items())
        }
    return output
