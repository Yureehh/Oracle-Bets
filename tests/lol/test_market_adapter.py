from oracle_bets_core.markets import PolymarketGammaAdapter

EXPECTED_QUOTE_COUNT = 2
T1_PRICE = 0.62


def test_polymarket_adapter_normalizes_stringified_outcomes():
    adapter = PolymarketGammaAdapter()
    quotes = adapter._quotes_from_market(
        {
            "id": "123",
            "question": "LoL: T1 vs G2",
            "outcomes": '["T1", "G2"]',
            "outcomePrices": '["0.62", "0.38"]',
            "liquidity": "1000",
            "volume": "5000",
            "slug": "lol-t1-vs-g2",
        }
    )

    assert len(quotes) == EXPECTED_QUOTE_COUNT
    assert quotes[0].outcome == "T1"
    assert quotes[0].implied_probability == T1_PRICE
    assert quotes[0].url == "https://polymarket.com/event/lol-t1-vs-g2"
