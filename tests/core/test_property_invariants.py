"""Property checks for probability and bankroll math invariants."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from lol_bets.inference.series import derive_series_distribution
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


@PROPERTY_SETTINGS
@given(st.lists(FINITE_PROBABILITY, min_size=3, max_size=3))
def test_bo3_path_distribution_is_normalized_and_symmetric(probabilities):
    original = derive_series_distribution(3, probabilities)
    mirrored = derive_series_distribution(
        3,
        [1 - probability for probability in probabilities],
    )

    assert math.fsum(original.score_probabilities.values()) == pytest.approx(1)
    assert math.fsum(original.total_maps_probabilities.values()) == pytest.approx(1)
    assert original.team_a_win == pytest.approx(mirrored.team_b_win)
    assert original.team_b_win == pytest.approx(mirrored.team_a_win)
