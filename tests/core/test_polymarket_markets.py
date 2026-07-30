from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
import requests
from oracle_bets_core.markets import (
    MarketDataError,
    MarketFixture,
    OrderBook,
    PolymarketClobClient,
    PolymarketGammaAdapter,
    capture_order_book_pair,
    match_market,
    select_best_market,
    walk_buy_book,
)

START = datetime(2026, 7, 27, 12, tzinfo=UTC)
EXPECTED_MARKETS = 2
MAX_RETRIES = 3
REQUEST_TIMEOUT = 4
T1_ASSET_ID = "asset-t1"


def _market(
    *,
    market_id: str = "market-1",
    question: str = "T1 vs Gen.G: Match Winner",
    outcomes: str = '["T1", "Gen.G"]',
    tokens: str = f'["{T1_ASSET_ID}", "asset-geng"]',
    start: str = "2026-07-27T12:00:00Z",
    best_of: int = 3,
) -> dict:
    return {
        "id": market_id,
        "question": question,
        "slug": market_id,
        "outcomes": outcomes,
        "outcomePrices": '["0.60", "0.40"]',
        "clobTokenIds": tokens,
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "eventStartTime": start,
        "bestOf": best_of,
        "resolutionSource": "Official match result",
        "liquidity": "1000",
        "volume": "5000",
    }


class _Response:
    def __init__(self, payload, *, status_error: bool = False):
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error:
            raise requests.HTTPError("provider error with unsafe response body")

    def json(self):
        return self.payload


class _PagedSession:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append((url, params, timeout))
        return _Response(self.pages.get(params["page"], {"events": []}))


def test_public_search_pages_and_maps_outcomes_to_clob_tokens():
    session = _PagedSession(
        {
            1: {
                "events": [
                    {
                        "id": "event-1",
                        "title": "T1 vs Gen.G (BO3)",
                        "markets": [_market()],
                    }
                ]
            },
            2: {
                "events": [
                    {
                        "id": "event-2",
                        "title": "G2 vs Fnatic (BO3)",
                        "markets": [_market(market_id="market-2")],
                    }
                ]
            },
        }
    )
    adapter = PolymarketGammaAdapter(session=session)

    markets = adapter.search_markets("League of Legends", limit=2)

    assert [market.market_id for market in markets] == ["market-1", "market-2"]
    assert markets[0].outcomes[0].token_id == T1_ASSET_ID
    assert markets[0].outcomes[1].displayed_price == Decimal("0.40")
    assert session.calls[0][0].endswith("/public-search")
    assert session.calls[0][1]["limit_per_type"] == EXPECTED_MARKETS
    assert [call[1]["page"] for call in session.calls] == [1, 2]


def test_market_schema_drift_fails_when_outcome_token_orientation_is_unknown():
    adapter = PolymarketGammaAdapter()

    with pytest.raises(MarketDataError, match="outcomes and CLOB token"):
        adapter._market_from_payload(
            _market(tokens='["only-one-token"]'),
            event={"id": "event-1", "title": "T1 vs Gen.G"},
        )


def test_exact_market_match_resolves_reversed_selection_order():
    adapter = PolymarketGammaAdapter()
    market = adapter._market_from_payload(
        _market(
            outcomes='["Gen.G", "T1"]',
            tokens=f'["asset-geng", "{T1_ASSET_ID}"]',
        ),
        event={"id": "event-1", "title": "LCK: T1 vs Gen.G (BO3)"},
    )
    fixture = MarketFixture(
        fixture_id="fixture-1",
        competition_names=("LCK",),
        team_a_id="t1",
        team_b_id="geng",
        team_a_names=("T1",),
        team_b_names=("Gen.G", "Gen G"),
        start_time=START,
        best_of=3,
        resolution_rule_terms=("official match result",),
    )

    assessment = match_market(fixture, market)

    assert assessment.matched
    assert dict(assessment.selection_tokens) == {
        "t1": T1_ASSET_ID,
        "geng": "asset-geng",
    }
    assert assessment.reasons == ()


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"best_of": 5}, "best_of_mismatch"),
        ({"start": "2026-07-28T12:00:00Z"}, "start_time_mismatch"),
    ],
)
def test_market_match_rejects_wrong_fixture_facts(overrides, reason):
    adapter = PolymarketGammaAdapter()
    market = adapter._market_from_payload(
        _market(**overrides),
        event={"id": "event-1", "title": "LCK: T1 vs Gen.G"},
    )
    fixture = MarketFixture(
        fixture_id="fixture-1",
        competition_names=("LCK",),
        team_a_id="t1",
        team_b_id="geng",
        team_a_names=("T1",),
        team_b_names=("Gen.G",),
        start_time=START,
        best_of=3,
        resolution_rule_terms=("official match result",),
    )

    assessment = match_market(fixture, market)

    assert not assessment.matched
    assert reason in assessment.reasons


def test_missing_resolution_source_is_visible_warning_not_silent_rejection():
    payload = _market()
    payload["resolutionSource"] = ""
    market = PolymarketGammaAdapter()._market_from_payload(
        payload,
        event={"id": "event-1", "title": "LCK: T1 vs Gen.G"},
    )
    fixture = MarketFixture(
        fixture_id="fixture-1",
        competition_names=("LCK",),
        team_a_id="t1",
        team_b_id="geng",
        team_a_names=("T1",),
        team_b_names=("Gen.G",),
        start_time=START,
        best_of=3,
        resolution_rule_terms=("official match result",),
    )

    assessment = match_market(fixture, market)

    assert assessment.matched
    assert assessment.warnings == ("market_rules_uncertain",)


@pytest.mark.parametrize(
    ("event_title", "question", "resolution_source", "reason"),
    [
        (
            "LEC: T1 vs Gen.G",
            "T1 vs Gen.G: Match Winner",
            "Official match result",
            "competition_mismatch",
        ),
        (
            "LCK: T1 vs Gen.G",
            "T1 vs Gen.G total maps",
            "Official match result",
            "market_meaning_mismatch",
        ),
        (
            "LCK: T1 vs Gen.G",
            "T1 vs Gen.G: Match Winner",
            "Community vote",
            "resolution_rules_mismatch",
        ),
    ],
)
def test_market_match_rejects_wrong_competition_meaning_or_rules(
    event_title,
    question,
    resolution_source,
    reason,
):
    payload = _market(question=question)
    payload["resolutionSource"] = resolution_source
    market = PolymarketGammaAdapter()._market_from_payload(
        payload,
        event={"id": "event-1", "title": event_title},
    )
    fixture = MarketFixture(
        fixture_id="fixture-1",
        competition_names=("LCK",),
        team_a_id="t1",
        team_b_id="geng",
        team_a_names=("T1",),
        team_b_names=("Gen.G",),
        start_time=START,
        best_of=3,
        resolution_rule_terms=("official match result",),
    )

    assessment = match_market(fixture, market)

    assert not assessment.matched
    assert reason in assessment.reasons


def test_best_market_uses_liquidity_and_retains_every_candidate_assessment():
    adapter = PolymarketGammaAdapter()
    markets = []
    for market_id, liquidity, title in (
        ("low", "100", "LCK: T1 vs Gen.G"),
        ("high", "1000", "LCK: T1 vs Gen.G"),
        ("wrong", "9000", "LEC: T1 vs Gen.G"),
    ):
        payload = _market(market_id=market_id)
        payload["liquidity"] = liquidity
        markets.append(
            adapter._market_from_payload(
                payload,
                event={"id": f"event-{market_id}", "title": title},
            )
        )
    fixture = MarketFixture(
        fixture_id="fixture-1",
        competition_names=("LCK",),
        team_a_id="t1",
        team_b_id="geng",
        team_a_names=("T1",),
        team_b_names=("Gen.G",),
        start_time=START,
        best_of=3,
        resolution_rule_terms=("official match result",),
    )

    selection = select_best_market(fixture, markets)

    assert selection.selected_market_id == "high"
    assert [item.market_id for item in selection.assessments] == [
        "low",
        "high",
        "wrong",
    ]
    assert selection.assessments[-1].reasons == ("competition_mismatch",)


def _book_payload(**overrides):
    payload = {
        "market": "condition-1",
        "asset_id": T1_ASSET_ID,
        "timestamp": "1785153600000",
        "hash": "book-hash",
        "bids": [
            {"price": "0.55", "size": "20"},
            {"price": "0.57", "size": "10"},
        ],
        "asks": [
            {"price": "0.64", "size": "20"},
            {"price": "0.62", "size": "10"},
        ],
        "min_order_size": "5",
        "tick_size": "0.01",
        "neg_risk": False,
        "last_trade_price": "0.60",
    }
    payload.update(overrides)
    return payload


def test_order_book_walk_uses_best_asks_and_reports_partial_depth():
    book = OrderBook.from_payload(
        _book_payload(),
        expected_token_id=T1_ASSET_ID,
    )

    complete = walk_buy_book(book, Decimal(25))
    partial = walk_buy_book(book, Decimal(40))

    assert complete.complete
    assert complete.filled_shares == Decimal(25)
    assert complete.total_cost == Decimal("15.8")
    assert complete.average_price == Decimal("0.632")
    assert complete.worst_price == Decimal("0.64")
    assert complete.decimal_odds == pytest.approx(1 / 0.632)
    assert not partial.complete
    assert partial.filled_shares == Decimal(30)
    assert partial.unfilled_shares == Decimal(10)


def test_crossed_or_malformed_order_book_is_rejected():
    with pytest.raises(MarketDataError, match="crossed"):
        OrderBook.from_payload(
            _book_payload(
                bids=[{"price": "0.65", "size": "10"}],
                asks=[{"price": "0.64", "size": "10"}],
            ),
            expected_token_id=T1_ASSET_ID,
        )

    with pytest.raises(MarketDataError, match="price"):
        OrderBook.from_payload(
            _book_payload(asks=[{"price": "not-a-price", "size": "10"}]),
            expected_token_id=T1_ASSET_ID,
        )


class _BookClient:
    def __init__(self):
        self.calls = 0

    def get_order_book(self, token_id):
        self.calls += 1
        return OrderBook.from_payload(
            _book_payload(hash=f"hash-{self.calls}"),
            expected_token_id=token_id,
        )


def test_two_book_observations_are_separate_and_size_aware():
    client = _BookClient()
    sleeps = []
    times = iter((START, START + timedelta(seconds=45)))

    observations = capture_order_book_pair(
        client,
        token_id=T1_ASSET_ID,
        requested_shares=Decimal(25),
        interval_seconds=45,
        sleeper=sleeps.append,
        clock=lambda: next(times),
    )

    assert [item.sequence_number for item in observations] == [1, 2]
    assert [item.book.book_hash for item in observations] == ["hash-1", "hash-2"]
    assert all(item.fill.complete for item in observations)
    assert sleeps == [45]


class _RetrySession:
    def __init__(self):
        self.calls = 0

    def get(self, _url, *, params, timeout):
        self.calls += 1
        if self.calls < MAX_RETRIES:
            raise requests.ConnectionError("secret response detail")
        assert params == {"token_id": T1_ASSET_ID}
        assert timeout == REQUEST_TIMEOUT
        return _Response(_book_payload())


def test_clob_client_retries_public_read_and_uses_short_cache():
    session = _RetrySession()
    clock_values = iter((10.0, 10.5))
    client = PolymarketClobClient(
        session=session,
        timeout=REQUEST_TIMEOUT,
        attempts=MAX_RETRIES,
        cache_seconds=2,
        monotonic_clock=lambda: next(clock_values),
        retry_sleep=lambda _seconds: None,
    )

    first = client.get_order_book(T1_ASSET_ID)
    second = client.get_order_book(T1_ASSET_ID)

    assert first == second
    assert session.calls == MAX_RETRIES
