from oracle_bets_core.betting import (
    build_edge_signal,
    decimal_odds_from_probability,
    expected_edge,
    kelly_fraction,
    probability_from_decimal_odds,
)

EVEN_ODDS_PROBABILITY = 0.5
EVEN_DECIMAL_ODDS = 2.0
MARKET_ODDS = 2.1
MODEL_PROBABILITY = 0.55
EXPECTED_EDGE = 0.155
EXPECTED_HALF_KELLY = 0.0705
EXPECTED_IMPLIED = 0.4762


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
