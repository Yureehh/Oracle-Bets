from datetime import UTC, datetime, timedelta
from decimal import Decimal

from lol_bets.daily import (
    _discover_typed_markets,
    _format_market_quote_messages,
    build_prediction_snapshot_rows,
)
from lol_bets.operations.market_actions import evaluate_daily_market_actions
from oracle_bets_core.markets import OrderBook, PolymarketGammaAdapter
from oracle_bets_core.operations.paper import ActionState
from oracle_bets_core.pd import pd

START = datetime(2026, 8, 2, 12, tzinfo=UTC)
FIXTURE_START = START + timedelta(hours=36)
EXPECTED_FIRST_BOOKS = 2
MARKET_SEARCH_LIMIT = 100


def _market(adapter, market_id, market_type, title, outcomes, tokens, *, line=None):
    raw = {
        "id": market_id,
        "question": f"T1 vs Gen.G: {title}",
        "slug": market_id,
        "outcomes": outcomes,
        "outcomePrices": '["0.50", "0.50"]',
        "clobTokenIds": tokens,
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "eventStartTime": FIXTURE_START.isoformat(),
        "bestOf": 3,
        "sportsMarketType": market_type,
        "groupItemTitle": title,
        "resolutionSource": "https://liquipedia.net/leagueoflegends/Main_Page",
        "liquidity": "1000",
        "line": line,
    }
    return adapter._market_from_payload(
        raw,
        event={"id": "event-1", "title": "LCK T1 vs Gen.G"},
    )


class _Books:
    def __init__(self):
        self.calls = 0

    def get_order_book(self, token_id):
        self.calls += 1
        return OrderBook.from_payload(
            {
                "market": "condition-1",
                "asset_id": token_id,
                "timestamp": str(int(START.timestamp() * 1000)),
                "hash": f"book-{self.calls}",
                "bids": [{"price": "0.49", "size": "100"}],
                "asks": [{"price": "0.50", "size": "100"}],
                "min_order_size": "5",
                "tick_size": "0.01",
                "neg_risk": False,
            },
            expected_token_id=token_id,
        )


def test_daily_typed_markets_batch_quotes_and_cap_correlated_fixture():
    schedule = pd.DataFrame(
        [
            {
                "match_key": "pandascore:1",
                "league": "LCK",
                "team_a": "T1",
                "team_b": "Gen.G",
                "start_utc": FIXTURE_START,
                "best_of": 3,
            }
        ]
    )
    snapshots = build_prediction_snapshot_rows(
        schedule.iloc[0],
        team_a_name="T1",
        team_b_name="Gen.G",
        match_type="bo3",
        team_a_win=0.70,
        team_b_win=0.30,
        team_a_lower=0.65,
        team_a_upper=0.75,
        team_b_lower=0.25,
        team_b_upper=0.35,
        probability_source="calibrated",
        uncertainty_method="held_out",
        uncertainty_confidence=0.90,
        uncertainty_sample_count=100,
        drivers=["Team rating strength pushed the model toward T1"],
        rating_baseline_team_a=0.60,
        full_model_team_a=0.72,
        prop_values={},
        lineup_ready=True,
        roster_ready=True,
    )
    adapter = PolymarketGammaAdapter()
    markets = [
        _market(
            adapter,
            "game-1",
            "child_moneyline",
            "Game 1 Winner",
            '["T1","Gen.G"]',
            '["g1-t1","g1-geng"]',
        ),
        _market(
            adapter,
            "match",
            "moneyline",
            "Match Winner",
            '["T1","Gen.G"]',
            '["m-t1","m-geng"]',
        ),
        _market(
            adapter,
            "totals",
            "totals",
            "Total Maps 2.5",
            '["Over","Under"]',
            '["over","under"]',
            line="2.5",
        ),
    ]
    books = _Books()
    sleeps = []
    clock_calls = 0

    def clock():
        nonlocal clock_calls
        clock_calls += 1
        return (
            START
            if clock_calls <= EXPECTED_FIRST_BOOKS
            else START + timedelta(seconds=45)
        )

    result = evaluate_daily_market_actions(
        schedule=schedule,
        snapshot_rows=snapshots,
        markets={"pandascore:1": markets},
        clob_client=books,
        model_healthy=True,
        run_key="daily-lol-2026-08-02",
        sleeper=sleeps.append,
        clock=clock,
    )

    quoted = [row for row in result.actions if row.get("decimal_odds")]
    assert sleeps == [45]
    assert books.calls == len({row["token_id"] for row in quoted}) * 2
    assert result.reviews
    assert sum(row["state"] == ActionState.PAPER_ACTIONABLE for row in quoted) == 1
    assert any(row["reason"] == "selection_not_model_favorite" for row in quoted)
    assert all(Decimal(str(row["decimal_odds"])) > 1 for row in quoted)
    assert all(row["requested_shares"] == "5" for row in quoted)
    assert all(row["hypothetical_cost"] == "2.50" for row in quoted)
    assert all(
        observation["minimum_order_size"] == "5"
        for row in quoted
        for observation in row["observations"]
    )

    rerun_snapshots = [
        dict(row, run_ts=START + timedelta(hours=1)) for row in snapshots
    ]
    clock_calls = 0
    rerun = evaluate_daily_market_actions(
        schedule=schedule,
        snapshot_rows=rerun_snapshots,
        markets={"pandascore:1": markets},
        clob_client=_Books(),
        model_healthy=True,
        run_key="daily-lol-2026-08-02",
        sleeper=lambda _seconds: None,
        clock=clock,
    )
    assert {row["proposal_id"] for row in rerun.actions} == {
        row["proposal_id"] for row in result.actions
    }


def test_daily_discovers_supported_markets_with_fixture_specific_query():
    schedule = pd.DataFrame(
        [
            {
                "match_key": "pandascore:1",
                "league": "LCK",
                "team_a": "T1",
                "team_b": "Gen.G",
                "start_utc": START,
                "best_of": 3,
            }
        ]
    )
    adapter = PolymarketGammaAdapter()
    supported = _market(
        adapter,
        "match",
        "moneyline",
        "Match Winner",
        '["T1","Gen.G"]',
        '["m-t1","m-geng"]',
    )
    unsupported = _market(
        adapter,
        "handicap",
        "map_handicap",
        "Game Handicap",
        '["T1","Gen.G"]',
        '["h-t1","h-geng"]',
    )

    class _Search:
        def __init__(self):
            self.calls = []

        def search_markets(self, query, *, limit):
            self.calls.append((query, limit))
            return [unsupported, supported]

    search = _Search()
    discovered = _discover_typed_markets(schedule, search)

    assert search.calls == [("T1 Gen.G", 100)]
    assert discovered.markets == {"pandascore:1": (supported,)}
    assert discovered.failures == {}


def test_daily_market_discovery_uses_canonical_team_alias():
    schedule = pd.DataFrame(
        [
            {
                "match_key": "pandascore:alias",
                "league": "LPL",
                "team_a": "AG.AL",
                "team_b": "T1",
                "start_utc": START,
                "best_of": 3,
            }
        ]
    )

    class Search:
        def __init__(self):
            self.calls = []

        def search_markets(self, query, *, limit):
            self.calls.append((query, limit))
            return []

    search = Search()
    _discover_typed_markets(schedule, search)

    assert search.calls == [("Anyone's Legend T1", MARKET_SEARCH_LIMIT)]


def test_daily_discovery_keeps_provider_failure_distinct_from_no_market():
    schedule = pd.DataFrame(
        [
            {
                "match_key": "pandascore:1",
                "league": "LCK",
                "team_a": "T1",
                "team_b": "Gen.G",
                "start_utc": START,
                "best_of": 3,
            }
        ]
    )

    class _FailedSearch:
        @staticmethod
        def search_markets(_query, *, limit):
            assert limit == MARKET_SEARCH_LIMIT
            raise RuntimeError("provider secret detail")

    discovery = _discover_typed_markets(schedule, _FailedSearch())

    assert discovery.markets == {"pandascore:1": ()}
    assert discovery.failures == {"pandascore:1": "RuntimeError"}


def test_market_quote_messages_keep_no_edge_blocked_and_failed_outcomes_visible():
    messages = _format_market_quote_messages(
        (
            {
                "fixture_key": "pandascore:1",
                "team_a": "T1",
                "team_b": "Gen.G",
                "selection": "T1",
                "target": "series_winner",
                "decimal_odds": 1.8,
                "conservative_edge": -0.02,
                "state": "no_edge",
            },
            {
                "fixture_key": "pandascore:1",
                "team_a": "T1",
                "team_b": "Gen.G",
                "selection": "Gen.G",
                "target": "series_winner",
                "decimal_odds": 2.2,
                "conservative_edge": 0.08,
                "state": "blocked",
            },
            {
                "fixture_key": "pandascore:1",
                "team_a": "T1",
                "team_b": "Gen.G",
                "selection": "over",
                "target": "series_total_maps",
                "total_line": 2.5,
                "decimal_odds": None,
                "state": "blocked",
                "reason": "insufficient_depth",
            },
        )
    )

    rendered = "\n".join(messages)
    assert "1.800 odds" in rendered
    assert "no_edge" in rendered
    assert "2.200 odds" in rendered
    assert "blocked" in rendered
    assert "unavailable (insufficient_depth)" in rendered
