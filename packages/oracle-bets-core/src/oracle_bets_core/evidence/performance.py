"""Prediction, paper strategy, uncertainty, activation, and readiness reports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

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
