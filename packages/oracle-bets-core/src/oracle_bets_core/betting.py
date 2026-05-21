"""Betting and prediction-market math with no execution side effects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EdgeSignal:
    """Decision-support output for a market quote."""

    model_probability: float
    implied_probability: float
    fair_odds: float
    market_odds: float
    edge: float
    half_kelly_fraction: float


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
