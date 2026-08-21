"""Property checks for probability and bankroll math invariants."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from oracle_bets_core.betting import (
    decimal_odds_from_probability,
    kelly_fraction,
    probability_from_decimal_odds,
)

FINITE_PROBABILITY = st.floats(
    min_value=1e-6,
    max_value=1 - 1e-6,
    allow_nan=False,
    allow_infinity=False,
)
PROPERTY_SETTINGS = settings(max_examples=100, deadline=None, derandomize=True)


@PROPERTY_SETTINGS
@given(FINITE_PROBABILITY)
def test_probability_and_decimal_odds_are_round_trip_inverses(probability):
    odds = decimal_odds_from_probability(probability)

    assert probability_from_decimal_odds(odds) == pytest.approx(probability)


@PROPERTY_SETTINGS
@given(
    odds=st.floats(
        min_value=1.000001,
        max_value=1000,
        allow_nan=False,
        allow_infinity=False,
    ),
    probability=st.floats(
        min_value=0,
        max_value=1,
        allow_nan=False,
        allow_infinity=False,
    ),
    multiplier=st.floats(
        min_value=0,
        max_value=1,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_fractional_kelly_never_exceeds_the_selected_bankroll_fraction(
    odds,
    probability,
    multiplier,
):
    stake_fraction = kelly_fraction(
        odds,
        probability,
        fraction=multiplier,
    )

    assert 0 <= stake_fraction <= multiplier
