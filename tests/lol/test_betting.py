import pytest
from oracle_bets_core.betting import (
    build_edge_signal,
    decimal_odds_from_probability,
    expected_edge,
    kelly_fraction,
    price_over_under,
    probability_from_decimal_odds,
)

EVEN_ODDS_PROBABILITY = 0.5
EVEN_DECIMAL_ODDS = 2.0
MARKET_ODDS = 2.1
MODEL_PROBABILITY = 0.55
EXPECTED_EDGE = 0.155
EXPECTED_HALF_KELLY = 0.0705
EXPECTED_IMPLIED = 0.4762
PROP_MEAN = 27.5
PROP_LINE = 26.5
PROP_SIGMA = 2.0


def test_betting_math_for_positive_edge():
    assert probability_from_decimal_odds(EVEN_DECIMAL_ODDS) == EVEN_ODDS_PROBABILITY
    assert decimal_odds_from_probability(EVEN_ODDS_PROBABILITY) == EVEN_DECIMAL_ODDS
    assert round(expected_edge(MARKET_ODDS, MODEL_PROBABILITY), 3) == EXPECTED_EDGE
    assert (
        round(kelly_fraction(MARKET_ODDS, MODEL_PROBABILITY), 4) == EXPECTED_HALF_KELLY
    )


def test_edge_signal_contains_half_kelly():
    signal = build_edge_signal(
        model_probability=MODEL_PROBABILITY, market_odds=MARKET_ODDS
    )

    assert round(signal.implied_probability, 4) == EXPECTED_IMPLIED
    assert round(signal.edge, 3) == EXPECTED_EDGE
    assert round(signal.half_kelly_fraction, 4) == EXPECTED_HALF_KELLY


def test_over_under_pricing_uses_residual_distribution():
    signal = price_over_under(
        mean=PROP_MEAN,
        line=PROP_LINE,
        sigma=PROP_SIGMA,
        over_odds=1.85,
    )

    assert signal.over_probability > EVEN_ODDS_PROBABILITY
    assert signal.under_probability < EVEN_ODDS_PROBABILITY
    assert signal.over_fair_odds < EVEN_DECIMAL_ODDS
    assert signal.over_edge is not None
    assert signal.over_half_kelly_fraction is not None


def test_kelly_fraction_clamps_negative_edge_to_zero():
    # Model probability below break-even must never suggest a stake.
    assert kelly_fraction(1.5, 0.10) == 0.0
    assert kelly_fraction(2.0, 0.0) == 0.0


def test_kelly_fraction_multiplier_scales_linearly():
    full = kelly_fraction(MARKET_ODDS, MODEL_PROBABILITY, fraction=1.0)
    half = kelly_fraction(MARKET_ODDS, MODEL_PROBABILITY, fraction=0.5)
    quarter = kelly_fraction(MARKET_ODDS, MODEL_PROBABILITY, fraction=0.25)

    assert full > 0
    assert half == pytest.approx(full * 0.5)
    assert quarter == pytest.approx(full * 0.25)


def test_kelly_fraction_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="odds"):
        kelly_fraction(1.0, 0.5)
    with pytest.raises(ValueError, match="probability"):
        kelly_fraction(2.0, 1.5)
    with pytest.raises(ValueError, match="non-negative"):
        kelly_fraction(2.0, 0.5, fraction=-0.1)


def test_expected_edge_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="odds"):
        expected_edge(0.9, 0.5)
    with pytest.raises(ValueError, match="probability"):
        expected_edge(2.0, -0.1)


def test_decimal_odds_conversion_rejects_boundaries():
    with pytest.raises(ValueError, match="greater than 1"):
        probability_from_decimal_odds(1.0)
    with pytest.raises(ValueError, match="Probability"):
        decimal_odds_from_probability(0.0)
    with pytest.raises(ValueError, match="Probability"):
        decimal_odds_from_probability(1.0)


def test_price_over_under_clips_extreme_probabilities():
    # Mean far above the line: over probability must clip below 1.0 so the
    # fair odds remain finite and defined.
    signal = price_over_under(mean=100.0, line=10.0, sigma=1.0)

    assert 0.0 < signal.under_probability < 1.0
    assert 0.0 < signal.over_probability < 1.0
    assert signal.over_fair_odds >= 1.0
    assert signal.under_fair_odds > 1.0


def test_price_over_under_rejects_non_positive_sigma():
    with pytest.raises(ValueError, match="sigma"):
        price_over_under(mean=25.0, line=26.5, sigma=0.0)


def test_price_over_under_probabilities_sum_to_one():
    signal = price_over_under(mean=PROP_MEAN, line=PROP_LINE, sigma=PROP_SIGMA)

    assert signal.over_probability + signal.under_probability == pytest.approx(1.0)
