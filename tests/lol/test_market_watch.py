from datetime import UTC, datetime, timedelta
from decimal import Decimal

from lol_bets.operations.market_watch import _horizon_label, observe_winner_markets
from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.markets import (
    MarketOutcome,
    OrderBook,
    OrderLevel,
    PolymarketMarket,
)
from oracle_bets_core.pd import pd

NOW = datetime(2026, 8, 12, 8, tzinfo=UTC)
MARKET_SEARCH_LIMIT = 50


def test_market_watch_records_read_only_evidence_and_one_report_pair(tmp_path):
    schedule_path = tmp_path / "schedule.parquet"
    start = NOW + timedelta(hours=36)
    pd.DataFrame(
        [
            {
                "match_key": "pandascore:1",
                "league": "LCK",
                "team_a": "T1",
                "team_b": "Gen.G",
                "start_utc": start,
                "best_of": 3,
            }
        ]
    ).to_parquet(schedule_path)
    market = PolymarketMarket(
        event_id="event-1",
        market_id="market-1",
        event_title="LCK: T1 vs Gen.G",
        question="T1 vs Gen.G Match Winner",
        slug="t1-geng",
        outcomes=(
            MarketOutcome("T1", "t1-token", Decimal("0.55")),
            MarketOutcome("Gen.G", "geng-token", Decimal("0.45")),
        ),
        event_start_time=start,
        best_of=3,
        sports_market_type="moneyline",
        group_item_title="Match Winner",
        game_number=None,
        total_line=None,
        active=True,
        closed=False,
        accepting_orders=True,
        resolution_source="https://liquipedia.net/leagueoflegends/Main_Page",
        liquidity=Decimal(100),
        volume=Decimal(200),
    )

    class Gamma:
        @staticmethod
        def search_markets(_query, *, limit):
            assert limit == MARKET_SEARCH_LIMIT
            return [market]

    class Clob:
        @staticmethod
        def get_order_book(token_id):
            return OrderBook(
                condition_id="market-1",
                token_id=token_id,
                timestamp=NOW,
                book_hash=f"hash-{token_id}",
                bids=(OrderLevel(Decimal("0.50"), Decimal(10)),),
                asks=(OrderLevel(Decimal("0.55"), Decimal(10)),),
                minimum_order_size=Decimal(1),
                tick_size=Decimal("0.01"),
                negative_risk=False,
                last_trade_price=Decimal("0.53"),
            )

    store = EvidenceStore(tmp_path / "evidence.db")
    report = observe_winner_markets(
        store=store,
        schedule_path=schedule_path,
        gamma=Gamma(),
        clob=Clob(),
        observed_at=NOW,
        report_root=tmp_path / "reports",
    )

    assert len(report["observations"]) == 1
    assert report["observations"][0]["horizon"] == "36h"
    assert report["observations"][0]["first_seen"] is True
    assert store.count(EvidenceTable.SOURCE_SNAPSHOTS) == 1
    assert store.count(EvidenceTable.PROPOSALS) == 0
    assert (tmp_path / "reports" / "20260812T080000Z.json").is_file()
    assert (tmp_path / "reports" / "20260812T080000Z.md").is_file()


def test_market_watch_labels_only_near_predeclared_horizons() -> None:
    assert _horizon_label(36.5) == "36h"
    assert _horizon_label(20.0) is None
    assert _horizon_label(0.25) == "close"
