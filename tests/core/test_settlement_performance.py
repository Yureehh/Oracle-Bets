from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from oracle_bets_core.evidence.contracts import DecisionMode
from oracle_bets_core.evidence.performance import (
    SettledPerformanceRow,
    aggregate_performance,
    bootstrap_roi,
    group_performance,
    prediction_quality,
)
from oracle_bets_core.evidence.settlement import (
    MarketCloseSnapshot,
    PositionTerms,
    SettlementError,
    SettlementResult,
    SettlementSource,
    calculate_clv,
    reconcile_settlement,
    select_fixed_close,
)

NOW = datetime(2026, 7, 27, 12, tzinfo=UTC)
EXPECTED_SETTLED = 4
EXPECTED_GRADED = 2
EXPECTED_HIT_RATE = 0.5
MAX_EXPECTED_LOG_LOSS = 0.4
MAX_EXPECTED_BRIER = 0.1


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (SettlementResult.WIN, Decimal("9.0")),
        (SettlementResult.LOSS, Decimal(-10)),
        (SettlementResult.PUSH, Decimal(0)),
        (SettlementResult.VOID, Decimal(0)),
    ],
)
def test_paper_settlement_accounting_invariants(result, expected):
    record = reconcile_settlement(
        PositionTerms("position-1", "10", "1.90"),
        settled_at=NOW,
        provider_result=result,
        provider_reference="provider-settlement-1",
    )

    assert record.pnl_units == expected
    assert record.source is SettlementSource.PROVIDER


def test_provider_is_preferred_but_source_conflicts_stop_settlement():
    provider = reconcile_settlement(
        PositionTerms("position-1", "10", "1.90"),
        settled_at=NOW,
        provider_result=SettlementResult.WIN,
        provider_reference="provider-1",
        internal_result=SettlementResult.WIN,
        internal_reference="game-1",
        internal_verified=True,
    )
    internal = reconcile_settlement(
        PositionTerms("position-2", "10", "1.90"),
        settled_at=NOW,
        provider_result=None,
        internal_result=SettlementResult.LOSS,
        internal_reference="game-2",
        internal_verified=True,
    )

    assert provider.source is SettlementSource.PROVIDER
    assert internal.source is SettlementSource.VERIFIED_INTERNAL
    assert internal.warnings == ("internal_settlement",)
    with pytest.raises(SettlementError, match="conflict"):
        reconcile_settlement(
            PositionTerms("position-3", "10", "1.90"),
            settled_at=NOW,
            provider_result=SettlementResult.WIN,
            provider_reference="provider-3",
            internal_result=SettlementResult.LOSS,
            internal_reference="game-3",
            internal_verified=True,
        )


def _close(snapshot_id, seconds_before, odds, available="10", warnings=()):
    return MarketCloseSnapshot(
        snapshot_id=snapshot_id,
        observed_at=NOW - timedelta(seconds=seconds_before),
        decimal_odds=odds,
        available_stake_units=available,
        source="polymarket-clob-v2",
        warnings=warnings,
    )


def test_fixed_close_is_latest_fillable_prestart_snapshot():
    snapshots = (
        _close("old", 120, "1.85"),
        _close("thin", 30, "1.80", available="0.2"),
        _close("chosen", 60, "1.82"),
        MarketCloseSnapshot(
            snapshot_id="late",
            observed_at=NOW + timedelta(seconds=1),
            decimal_odds="1.75",
            available_stake_units="10",
            source="polymarket-clob-v2",
        ),
    )

    close = select_fixed_close(
        snapshots,
        event_start=NOW,
        intended_stake_units="1",
    )

    assert close is not None
    assert close.snapshot_id == "chosen"
    clv = calculate_clv(entry_decimal_odds="2.00", close=close)
    assert clv.available
    assert clv.probability_clv > 0
    assert clv.odds_ratio_clv > 0


def test_missing_close_is_reported_not_fabricated():
    clv = calculate_clv(entry_decimal_odds="2.00", close=None)

    assert not clv.available
    assert clv.probability_clv is None
    assert clv.reason == "closing_snapshot_unavailable"


def _row(
    position_id,
    order,
    result,
    pnl,
    *,
    league="LCK",
    strategy="quarter_kelly",
    clv=0.02,
    warnings=(),
):
    return SettledPerformanceRow(
        position_id=position_id,
        settled_order=order,
        stake_units=Decimal(1),
        pnl_units=Decimal(str(pnl)),
        result=result,
        league=league,
        market="series_winner",
        strategy=strategy,
        model_version="champion-1",
        mode=DecisionMode.PREMATCH,
        edge_band="0.05-0.10",
        probability_clv=clv,
        evidence_warnings=warnings,
    )


def test_performance_reconciles_turnover_pnl_roi_hit_rate_and_drawdown():
    rows = (
        _row("p1", 1, SettlementResult.WIN, 1),
        _row("p2", 2, SettlementResult.LOSS, -1),
        _row("p3", 3, SettlementResult.PUSH, 0),
        _row("p4", 4, SettlementResult.VOID, 0),
    )

    metrics = aggregate_performance(rows)

    assert metrics.settled_count == EXPECTED_SETTLED
    assert metrics.graded_count == EXPECTED_GRADED
    assert metrics.turnover_units == Decimal(4)
    assert metrics.pnl_units == Decimal(0)
    assert metrics.roi == 0
    assert metrics.hit_rate == EXPECTED_HIT_RATE
    assert metrics.mean_probability_clv == pytest.approx(0.02)
    assert metrics.maximum_drawdown_units == Decimal(1)


def test_prediction_quality_and_grouped_strategy_metrics_are_explicit():
    quality = prediction_quality(
        [1, 0, 1, 0],
        [0.8, 0.2, 0.7, 0.3],
        bins=2,
    )
    groups = group_performance(
        (
            _row("lck", 1, SettlementResult.WIN, 1, league="LCK"),
            _row("lec", 2, SettlementResult.LOSS, -1, league="LEC"),
        ),
        key=lambda row: row.league,
    )

    assert quality.log_loss < MAX_EXPECTED_LOG_LOSS
    assert quality.brier < MAX_EXPECTED_BRIER
    assert set(groups) == {"LCK", "LEC"}
    assert groups["LCK"].roi == 1
    assert groups["LEC"].roi == -1


def test_roi_bootstrap_is_repeatable():
    rows = tuple(
        _row(
            f"p{index}",
            index,
            SettlementResult.WIN if index % 3 else SettlementResult.LOSS,
            0.9 if index % 3 else -1,
        )
        for index in range(1, 31)
    )

    first = bootstrap_roi(rows, samples=1000, seed=11)
    second = bootstrap_roi(rows, samples=1000, seed=11)

    assert first == second
    assert first is not None
    assert first.lower < first.point < first.upper
