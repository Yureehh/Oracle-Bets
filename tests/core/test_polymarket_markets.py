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
    SupportedMarketType,
    capture_minimum_order_book_batch,
    capture_order_book_batch,
    capture_order_book_pair,
    confirmed_executable_fill,
    match_market,
    select_best_market,
    walk_buy_book,
    walk_buy_book_by_risk,
)

START = datetime(2026, 7, 27, 12, tzinfo=UTC)
EXPECTED_MARKETS = 2
MAX_RETRIES = 3
REQUEST_TIMEOUT = 4
T1_ASSET_ID = "asset-t1"
UNAVAILABLE_ASSET_ID = "asset-unavailable"
PARTIAL_ASSET_ID = "asset-partial"
MISSING_ASSET_ID = "asset-missing"
STALE_ASSET_ID = "asset-stale"
EXPECTED_SCHEMA_RETRY_CALLS = 2


def _market(
    *,
    market_id: str = "market-1",
    question: str = "T1 vs Gen.G: Match Winner",
    outcomes: str = '["T1", "Gen.G"]',
    tokens: str = f'["{T1_ASSET_ID}", "asset-geng"]',
    start: str = "2026-07-27T12:00:00Z",
    best_of: int = 3,
    sports_market_type: str = "moneyline",
    group_item_title: str = "Match Winner",
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
        "sportsMarketType": sports_market_type,
        "groupItemTitle": group_item_title,
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
    ("requested_type", "game_number", "sports_type", "group_title"),
    [
        (SupportedMarketType.MAP_WINNER, 1, "child_moneyline", "Game 1"),
        (SupportedMarketType.MAP_WINNER, 2, "child_moneyline", "Game 2"),
        (SupportedMarketType.MAP_WINNER, 3, "child_moneyline", "Game 3"),
        (SupportedMarketType.SERIES_WINNER, None, "moneyline", "Match Winner"),
    ],
)
def test_typed_winner_markets_cannot_be_confused(
    requested_type,
    game_number,
    sports_type,
    group_title,
):
    market = PolymarketGammaAdapter()._market_from_payload(
        _market(
            question="T1 vs Gen.G",
            sports_market_type=sports_type,
            group_item_title=group_title,
        ),
        event={"id": "event-1", "title": "LCK: T1 vs Gen.G (BO3)"},
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
        market_type=requested_type,
        game_number=game_number,
    )

    assessment = match_market(fixture, market)

    assert assessment.matched


def test_typed_total_maps_market_resolves_over_under_orientation():
    payload = _market(
        question="T1 vs Gen.G total maps 2.5",
        outcomes='["Over", "Under"]',
        tokens='["asset-over", "asset-under"]',
        sports_market_type="totals",
        group_item_title="Total Maps 2.5",
    )
    payload["line"] = "2.5"
    market = PolymarketGammaAdapter()._market_from_payload(
        payload,
        event={"id": "event-1", "title": "LCK: T1 vs Gen.G (BO3)"},
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
        market_type=SupportedMarketType.SERIES_TOTAL_MAPS,
        total_line=Decimal("2.5"),
    )

    assessment = match_market(fixture, market)

    assert assessment.matched
    assert dict(assessment.selection_tokens) == {
        "over": "asset-over",
        "under": "asset-under",
    }


def test_unsupported_or_wrong_typed_market_is_rejected():
    market = PolymarketGammaAdapter()._market_from_payload(
        _market(sports_market_type="spreads", group_item_title="Map Handicap"),
        event={"id": "event-1", "title": "LCK: T1 vs Gen.G (BO3)"},
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
    )

    assessment = match_market(fixture, market)

    assert not assessment.matched
    assert "sports_market_type_mismatch" in assessment.reasons


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


def test_order_book_walk_by_risk_spends_exact_intended_amount():
    book = OrderBook.from_payload(_book_payload(), expected_token_id=T1_ASSET_ID)

    fill = walk_buy_book_by_risk(book, Decimal(10))

    assert fill.complete
    assert fill.requested_risk == Decimal(10)
    assert fill.total_cost == Decimal(10)
    assert fill.filled_shares == Decimal("15.9375")
    assert fill.decimal_odds == pytest.approx(1.59375)


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
            _book_payload(asset_id=token_id, hash=f"hash-{self.calls}"),
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


def test_two_risk_observations_use_worse_complete_executable_price():
    client = _BookClient()
    sleeps = []
    times = iter((START, START + timedelta(seconds=45)))

    observations = capture_order_book_pair(
        client,
        token_id=T1_ASSET_ID,
        intended_risk_amount=Decimal(10),
        interval_seconds=45,
        sleeper=sleeps.append,
        clock=lambda: next(times),
    )
    confirmed = confirmed_executable_fill(observations)

    assert confirmed.complete
    assert confirmed.total_cost == Decimal(10)
    assert sleeps == [45]


def test_batch_book_capture_waits_once_for_all_tokens():
    client = _BookClient()
    sleeps = []
    times = iter(
        (
            START,
            START,
            START + timedelta(seconds=45),
            START + timedelta(seconds=45),
        )
    )

    observations = capture_order_book_batch(
        client,
        token_ids=(T1_ASSET_ID, "asset-geng"),
        intended_risk_amount=Decimal(10),
        sleeper=sleeps.append,
        clock=lambda: next(times),
    )

    assert set(observations) == {T1_ASSET_ID, "asset-geng"}
    assert client.calls == len(observations) * 2
    assert sleeps == [45]
    assert all(
        confirmed_executable_fill(pair).complete for pair in observations.values()
    )


class _ChangingMinimumBookClient:
    def __init__(self):
        self.calls: dict[str, int] = {}

    def get_order_book(self, token_id):
        call = self.calls.get(token_id, 0) + 1
        self.calls[token_id] = call
        minimum = "5" if call == 1 else "7"
        return OrderBook.from_payload(
            _book_payload(
                asset_id=token_id,
                hash=f"{token_id}-{call}",
                min_order_size=minimum,
                bids=[{"price": "0.40", "size": "20"}],
                asks=[{"price": "0.50", "size": "20"}],
            ),
            expected_token_id=token_id,
        )


def test_minimum_order_batch_uses_larger_minimum_from_both_observations():
    client = _ChangingMinimumBookClient()
    sleeps = []
    times = iter((START, START + timedelta(seconds=45)))

    result = capture_minimum_order_book_batch(
        client,
        token_ids=(T1_ASSET_ID,),
        sleeper=sleeps.append,
        clock=lambda: next(times),
    )

    pair = result.observations[T1_ASSET_ID]
    assert not result.failures
    assert sleeps == [45]
    assert {item.fill.requested_shares for item in pair} == {Decimal(7)}
    assert {item.fill.total_cost for item in pair} == {Decimal("3.50")}
    assert {item.book.minimum_order_size for item in pair} == {
        Decimal(5),
        Decimal(7),
    }


def test_minimum_order_batch_isolates_unavailable_and_partial_tokens():
    class SelectiveClient(_ChangingMinimumBookClient):
        def get_order_book(self, token_id):
            if token_id == UNAVAILABLE_ASSET_ID:
                raise RuntimeError("provider unavailable")
            book = super().get_order_book(token_id)
            if token_id == PARTIAL_ASSET_ID:
                return OrderBook(
                    condition_id=book.condition_id,
                    token_id=book.token_id,
                    timestamp=book.timestamp,
                    book_hash=book.book_hash,
                    bids=book.bids,
                    asks=(
                        book.asks[0].__class__(
                            price=Decimal("0.50"),
                            size=Decimal(2),
                        ),
                    ),
                    minimum_order_size=book.minimum_order_size,
                    tick_size=book.tick_size,
                    negative_risk=book.negative_risk,
                    last_trade_price=book.last_trade_price,
                )
            return book

    client = SelectiveClient()
    times = iter(
        (
            START,
            START,
            START + timedelta(seconds=45),
            START + timedelta(seconds=45),
        )
    )

    result = capture_minimum_order_book_batch(
        client,
        token_ids=(T1_ASSET_ID, UNAVAILABLE_ASSET_ID, PARTIAL_ASSET_ID),
        sleeper=lambda _seconds: None,
        clock=lambda: next(times),
    )

    assert T1_ASSET_ID in result.observations
    assert result.failures[UNAVAILABLE_ASSET_ID].reason == "book_unavailable"
    assert result.failures[PARTIAL_ASSET_ID].reason == "insufficient_depth"
    assert result.observations[PARTIAL_ASSET_ID][0].fill.filled_shares == Decimal(2)


def test_minimum_order_batch_distinguishes_missing_and_stale_timestamps():
    missing = OrderBook.from_payload(
        _book_payload(asset_id=MISSING_ASSET_ID, timestamp=None),
        expected_token_id=MISSING_ASSET_ID,
    )
    stale = OrderBook.from_payload(
        _book_payload(
            asset_id=STALE_ASSET_ID,
            timestamp=str(int((START - timedelta(minutes=10)).timestamp() * 1000)),
        ),
        expected_token_id=STALE_ASSET_ID,
    )

    class TimestampClient:
        def get_order_book(self, token_id):
            return missing if token_id == MISSING_ASSET_ID else stale

    result = capture_minimum_order_book_batch(
        TimestampClient(),
        token_ids=(MISSING_ASSET_ID, STALE_ASSET_ID),
        sleeper=lambda _seconds: None,
        clock=lambda: START,
    )

    assert result.failures[MISSING_ASSET_ID].reason == "missing_book_timestamp"
    assert result.failures[STALE_ASSET_ID].reason == "stale_book"


def test_order_book_accepts_iso_timestamp():
    payload = _book_payload(timestamp="2026-07-27T12:00:00+00:00")

    book = OrderBook.from_payload(payload, expected_token_id=T1_ASSET_ID)

    assert book.timestamp == START


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


def test_clob_client_retries_transient_missing_timestamp_without_caching_it():
    class SchemaRetrySession:
        def __init__(self):
            self.calls = 0

        def get(self, _url, *, params, timeout):
            self.calls += 1
            assert params == {"token_id": T1_ASSET_ID}
            assert timeout == REQUEST_TIMEOUT
            timestamp = None if self.calls == 1 else _book_payload()["timestamp"]
            return _Response(_book_payload(timestamp=timestamp))

    session = SchemaRetrySession()
    client = PolymarketClobClient(
        session=session,
        timeout=REQUEST_TIMEOUT,
        attempts=2,
        retry_sleep=lambda _seconds: None,
    )

    book = client.get_order_book(T1_ASSET_ID)

    assert book.timestamp is not None
    assert session.calls == EXPECTED_SCHEMA_RETRY_CALLS


def test_polymarket_clients_expose_read_methods_only():
    clob_methods = {
        name
        for name, value in vars(PolymarketClobClient).items()
        if callable(value) and not name.startswith("_")
    }
    gamma_methods = {
        name
        for name, value in vars(PolymarketGammaAdapter).items()
        if callable(value) and not name.startswith("_")
    }

    assert clob_methods == {"get_order_book"}
    assert gamma_methods == {"search", "search_markets"}
