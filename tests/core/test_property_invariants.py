"""Property checks for probability and bankroll math invariants."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from lol_bets.operations.manual_market import normalize_market_urls
from lol_bets.operations.market_strategies import enumerate_series_paths
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
@given(probability=FINITE_PROBABILITY, best_of=st.sampled_from((3, 5)))
def test_legal_series_path_probabilities_sum_to_one(probability, best_of):
    distribution = enumerate_series_paths(probability, best_of=best_of)

    assert all(0 <= value <= 1 for value in distribution.exact_score.values())
    assert 0 <= distribution.team_a_win <= 1
    assert sum(distribution.exact_score.values()) == pytest.approx(1.0)
    assert sum(distribution.total_maps.values()) == pytest.approx(1.0)
    assert sum(distribution.map_differential.values()) == pytest.approx(1.0)


@PROPERTY_SETTINGS
@given(probability=FINITE_PROBABILITY, best_of=st.sampled_from((3, 5)))
def test_series_path_swap_complements_winners_and_preserves_totals(
    probability,
    best_of,
):
    direct = enumerate_series_paths(probability, best_of=best_of)
    swapped = enumerate_series_paths(1 - probability, best_of=best_of)

    assert direct.team_a_win == pytest.approx(1 - swapped.team_a_win)
    assert direct.total_maps == pytest.approx(swapped.total_maps)
    assert direct.exact_score == pytest.approx(
        {
            (b_wins, a_wins): value
            for (a_wins, b_wins), value in swapped.exact_score.items()
        }
    )
    assert direct.map_differential == pytest.approx(
        {-difference: value for difference, value in swapped.map_differential.items()}
    )


@PROPERTY_SETTINGS
@given(
    st.lists(
        st.sampled_from(
            (
                "https://polymarket.com/esports/league-of-legends/lck/lol-a-b-2026-08-20",
                "https://thunderpick.io/en/esports/lol/team-a-vs-team-b",
            )
        ),
        min_size=1,
        max_size=2,
    )
)
def test_polymarket_url_normalization_is_idempotent(urls):
    normalized = normalize_market_urls(("\n, ".join(urls),))

    assert normalize_market_urls((" ".join(normalized),)) == normalized
