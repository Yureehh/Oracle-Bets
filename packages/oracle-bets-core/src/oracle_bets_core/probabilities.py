"""Pure probability distributions with no market-price dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

PROBABILITY_EPSILON = 1e-6


@dataclass(frozen=True)
class PropLineProbability:
    """Probability-only view of one scalar forecast against a market line."""

    mean: float
    line: float
    sigma: float
    over_probability: float
    under_probability: float


def probability_over_under(
    *, mean: float, line: float, sigma: float
) -> PropLineProbability:
    """Convert a normal residual model into complementary line probabilities."""
    if sigma <= 0:
        raise ValueError("Residual sigma must be positive.")
    under = _clip_probability(_normal_cdf((line - mean) / sigma))
    return PropLineProbability(
        mean=mean,
        line=line,
        sigma=sigma,
        over_probability=1.0 - under,
        under_probability=under,
    )


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _clip_probability(probability: float) -> float:
    return min(
        1.0 - PROBABILITY_EPSILON,
        max(PROBABILITY_EPSILON, probability),
    )
