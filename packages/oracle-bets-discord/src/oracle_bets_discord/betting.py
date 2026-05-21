"""Discord-facing betting helper functions."""

from __future__ import annotations

from oracle_bets_core.betting import (
    decimal_odds_from_probability,
    kelly_fraction,
    probability_from_decimal_odds,
)


def calculate_odds(win_probability: float, to_decimal: bool) -> float | str:
    """Convert a probability in (0, 1) to decimal or fractional odds."""
    if not (0 < win_probability < 1):
        return "Odds are undefined for win probabilities of 0% or 100%."
    if to_decimal:
        return round(decimal_odds_from_probability(win_probability), 2)
    return round(win_probability / (1 - win_probability), 2)


def calculate_prob(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return round(probability_from_decimal_odds(odds), 4)


def convert_odds(odds: float | str) -> float:
    """Accept a decimal probability or a percent string and return [0, 1]."""
    if isinstance(odds, str) and odds.endswith("%"):
        return float(odds.strip("%")) / 100.0
    return float(odds)


def calculate_kelly_criterion(
    bookmaker_odds: float, win_probability: float, *, half: bool = True
) -> float:
    """Return clipped Kelly stake fraction for Discord commands."""
    fraction = 0.5 if half else 1.0
    return round(kelly_fraction(bookmaker_odds, win_probability, fraction=fraction), 4)
