"""Betting and prediction-market math with no execution side effects."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt


@dataclass(frozen=True)
class EdgeSignal:
    """Decision-support output for a market quote."""

    model_probability: float
    implied_probability: float
    fair_odds: float
    market_odds: float
    edge: float
    half_kelly_fraction: float


@dataclass(frozen=True)
class OverUnderSignal:
    """Decision-support output for an over/under quote."""

    mean: float
    line: float
    sigma: float
    over_probability: float
    under_probability: float
    over_fair_odds: float
    under_fair_odds: float
    over_edge: float | None = None
    under_edge: float | None = None
    over_half_kelly_fraction: float | None = None
    under_half_kelly_fraction: float | None = None


PROBABILITY_EPSILON = 1e-6


def _clip_probability(probability: float) -> float:
    return min(1.0 - PROBABILITY_EPSILON, max(PROBABILITY_EPSILON, probability))


def probability_from_decimal_odds(odds: float) -> float:
    if odds <= 1:
        msg = "Decimal odds must be greater than 1."
        raise ValueError(msg)
    return 1.0 / odds


def decimal_odds_from_probability(probability: float) -> float:
    if not 0 < probability < 1:
        msg = "Probability must be in (0, 1)."
        raise ValueError(msg)
    return 1.0 / probability


def expected_edge(decimal_odds: float, win_probability: float) -> float:
    if decimal_odds <= 1:
        msg = "Decimal odds must be greater than 1."
        raise ValueError(msg)
    if not 0 <= win_probability <= 1:
        msg = "Win probability must be in [0, 1]."
        raise ValueError(msg)
    return win_probability * decimal_odds - 1.0


def kelly_fraction(
    decimal_odds: float, win_probability: float, *, fraction: float = 0.5
) -> float:
    """Return clipped fractional-Kelly stake fraction for decimal odds."""
    if decimal_odds <= 1:
        msg = "Decimal odds must be greater than 1."
        raise ValueError(msg)
    if not 0 <= win_probability <= 1:
        msg = "Win probability must be in [0, 1]."
        raise ValueError(msg)
    if fraction < 0:
        msg = "Kelly fraction multiplier must be non-negative."
        raise ValueError(msg)

    b = decimal_odds - 1.0
    q = 1.0 - win_probability
    full = (b * win_probability - q) / b
    return max(0.0, full * fraction)


def build_edge_signal(
    *,
    model_probability: float,
    market_odds: float,
    kelly_multiplier: float = 0.5,
) -> EdgeSignal:
    implied = probability_from_decimal_odds(market_odds)
    return EdgeSignal(
        model_probability=model_probability,
        implied_probability=implied,
        fair_odds=decimal_odds_from_probability(model_probability),
        market_odds=market_odds,
        edge=expected_edge(market_odds, model_probability),
        half_kelly_fraction=kelly_fraction(
            market_odds, model_probability, fraction=kelly_multiplier
        ),
    )


def normal_cdf(value: float) -> float:
    """Normal CDF without requiring SciPy in the core package."""
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def price_over_under(
    *,
    mean: float,
    line: float,
    sigma: float,
    over_odds: float | None = None,
    under_odds: float | None = None,
    kelly_multiplier: float = 0.5,
) -> OverUnderSignal:
    """
    Price an over/under market from a regression mean and residual uncertainty.

    The model predicts the central estimate. The residual sigma turns that central
    estimate into a probability distribution, so line pricing reflects historical
    model error instead of treating the mean as a deterministic outcome.
    """
    if sigma <= 0:
        msg = "Residual sigma must be positive."
        raise ValueError(msg)
    if kelly_multiplier < 0:
        msg = "Kelly fraction multiplier must be non-negative."
        raise ValueError(msg)

    under_probability = _clip_probability(normal_cdf((line - mean) / sigma))
    over_probability = _clip_probability(1.0 - under_probability)
    over_edge = (
        expected_edge(over_odds, over_probability) if over_odds is not None else None
    )
    under_edge = (
        expected_edge(under_odds, under_probability) if under_odds is not None else None
    )
    over_half_kelly = (
        kelly_fraction(over_odds, over_probability, fraction=kelly_multiplier)
        if over_odds is not None
        else None
    )
    under_half_kelly = (
        kelly_fraction(under_odds, under_probability, fraction=kelly_multiplier)
        if under_odds is not None
        else None
    )

    return OverUnderSignal(
        mean=mean,
        line=line,
        sigma=sigma,
        over_probability=over_probability,
        under_probability=under_probability,
        over_fair_odds=decimal_odds_from_probability(over_probability),
        under_fair_odds=decimal_odds_from_probability(under_probability),
        over_edge=over_edge,
        under_edge=under_edge,
        over_half_kelly_fraction=over_half_kelly,
        under_half_kelly_fraction=under_half_kelly,
    )
