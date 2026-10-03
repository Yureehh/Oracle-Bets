"""Accounting and leakage checks for offline fractional-Kelly research."""

from datetime import UTC, datetime, timedelta

import pytest
from oracle_bets_core.evidence.portfolio import (
    JointScenarios,
    KellyPolicy,
    Opportunity,
    RiskTarget,
    allocate_portfolio,
    evaluate_policy,
    select_policy,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


def opportunity(key, *, day=0, fixture=None, probability=0.6, result=1.0, duration=1):
    decision = START + timedelta(days=day)
    return Opportunity(
        key,
        fixture or key,
        decision,
        decision + timedelta(days=duration),
        probability,
        2.0,
        result,
    )


def policy(fraction=0.5, ticket=0.5, fixture=0.5, total=0.8):
    return KellyPolicy("test", fraction, ticket, fixture, total)


def test_hand_calculated_kelly_and_no_edge():
    rows = (opportunity("edge"), opportunity("none", probability=0.4))
    stakes = allocate_portfolio(rows, policy(), equity=100)
    assert stakes == pytest.approx({"edge": 10, "none": 0})


def test_caps_group_same_fixture_and_reserve_existing_exposure():
    rows = (opportunity("a", fixture="match"), opportunity("b", fixture="match"))
    stakes = allocate_portfolio(
        rows,
        policy(ticket=0.08, fixture=0.12, total=0.15),
        equity=100,
        open_stakes={"match": 10, "other": 3},
    )
    assert sum(stakes.values()) == pytest.approx(2)
    assert stakes["a"] == pytest.approx(stakes["b"])


def test_joint_correlated_outcomes_do_not_receive_independent_full_kelly():
    rows = (opportunity("a", fixture="match"), opportunity("b", fixture="match"))
    scenarios = JointScenarios(
        ("a", "b"), (0.6, 0.4), ((1.0, 1.0), (-1.0, -1.0)), START
    )
    stakes = allocate_portfolio(
        rows, policy(fraction=1), equity=100, scenarios=scenarios
    )
    assert sum(stakes.values()) == pytest.approx(20, abs=1e-5)


def test_joint_mutually_exclusive_tickets_with_no_edge_get_zero():
    rows = (opportunity("a"), opportunity("b"))
    scenarios = JointScenarios(
        ("a", "b"), (0.5, 0.5), ((1.0, -1.0), (-1.0, 1.0)), START
    )
    assert (
        sum(
            allocate_portfolio(rows, policy(), equity=100, scenarios=scenarios).values()
        )
        == 0
    )


def test_future_joint_probabilities_rejected():
    scenarios = JointScenarios(
        ("a",), (0.6, 0.4), ((1.0,), (-1.0,)), START + timedelta(seconds=1)
    )
    with pytest.raises(ValueError, match="available"):
        allocate_portfolio(
            (opportunity("a"),), policy(), equity=100, scenarios=scenarios
        )


def test_settlement_prevents_early_profit_reinvestment_and_exposure_recycling():
    rows = (
        opportunity("a", duration=3),
        opportunity("b", day=1),
        opportunity("c", day=3),
    )
    result = evaluate_policy(
        rows, policy(fraction=1, ticket=0.2, fixture=0.2, total=0.2)
    )
    assert result.stakes == pytest.approx({"a": 20, "b": 0, "c": 24})
    assert result.final_equity == pytest.approx(144)
    assert result.maximum_open_exposure == pytest.approx(0.2)


def test_simultaneous_settlements_do_not_create_artificial_drawdown():
    result = evaluate_policy((opportunity("a"), opportunity("b", result=-1)), policy())
    assert result.final_equity == pytest.approx(100)
    assert result.maximum_drawdown == 0


def test_same_fixture_cross_split_and_unsettled_development_are_rejected():
    with pytest.raises(ValueError, match="fixture"):
        select_policy(
            (opportunity("a", fixture="shared"),),
            (opportunity("b", day=20, fixture="shared"),),
        )
    with pytest.raises(ValueError, match="settled"):
        select_policy((opportunity("a", duration=30),), (opportunity("b", day=20),))


def test_no_evidence_and_small_evidence_are_not_strategy_claims():
    assert select_policy((), ()).status == "unavailable"
    report = select_policy((opportunity("a"),), (opportunity("b", day=20),))
    assert report.status == "insufficient_evidence"
    assert report.selected_policy is None


def test_deterministic_block_bootstrap_and_no_holdout_policy_selection():
    development = tuple(
        opportunity(f"d{i}", day=8 * i, result=-1 if i % 4 == 0 else 1)
        for i in range(24)
    )
    holdout = tuple(opportunity(f"h{i}", day=220 + 8 * i) for i in range(24))
    target = RiskTarget(min_fixtures=20, min_blocks=8, bootstrap_samples=100, seed=17)
    first = select_policy(development, holdout, target=target)
    assert first == select_policy(development, holdout, target=target)
    losing_holdout = tuple(
        opportunity(f"h{i}", day=220 + 8 * i, result=-1) for i in range(24)
    )
    second = select_policy(development, losing_holdout, target=target)
    assert first.status == "evaluated"
    assert first.selected_policy == second.selected_policy
    assert first.holdout.final_equity > second.holdout.final_equity
    assert first.holdout.bootstrap is not None


def test_overlapping_long_positions_do_not_count_as_independent_blocks():
    rows = tuple(opportunity(str(i), day=8 * i, duration=30) for i in range(24))
    result = evaluate_policy(rows, policy(), target=RiskTarget(min_fixtures=20))
    assert result.block_count == 1
    assert result.bootstrap is None


@pytest.mark.parametrize("probability", [float("nan"), -0.1, 1.1])
def test_invalid_probabilities_rejected(probability):
    with pytest.raises(ValueError, match="probability"):
        opportunity("bad", probability=probability)


def test_joint_optimizer_can_reallocate_a_binding_shared_budget():
    rows = (
        opportunity("weak", probability=0.55),
        opportunity("strong", probability=0.8),
    )
    scenarios = JointScenarios(
        ("weak", "strong"),
        (0.55, 0.25, 0.2),
        ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0)),
        START,
    )
    capped = policy(fraction=1, ticket=0.1, fixture=0.1, total=0.1)
    stakes = allocate_portfolio(rows, capped, equity=100, scenarios=scenarios)
    assert stakes == pytest.approx({"weak": 0, "strong": 10}, abs=1e-5)


def test_heldout_losses_are_reported_without_changing_locked_policy():
    development = tuple(opportunity(f"d{i}", day=8 * i) for i in range(24))
    holdout = tuple(opportunity(f"h{i}", day=220 + 8 * i, result=-1) for i in range(24))
    report = select_policy(
        development, holdout, target=RiskTarget(min_fixtures=20, bootstrap_samples=100)
    )
    assert report.status == "evaluated"
    assert report.holdout.log_growth < 0
    assert report.holdout.maximum_drawdown > report.target.drawdown_limit


def test_allocations_do_not_read_realized_outcomes():
    winners = (opportunity("a"), opportunity("b"))
    losers = (opportunity("a", result=-1), opportunity("b", result=-1))
    assert allocate_portfolio(winners, policy(), equity=100) == allocate_portfolio(
        losers, policy(), equity=100
    )


def test_unsettled_quote_can_be_sized_but_cannot_enter_historical_replay():
    quote = Opportunity("quote", "fixture", START, None, 0.6, 2.0, None)
    assert allocate_portfolio((quote,), policy(), equity=100) == pytest.approx(
        {"quote": 10}
    )
    with pytest.raises(ValueError, match="requires settled outcomes"):
        evaluate_policy((quote,), policy())


def test_losses_can_raise_open_exposure_but_never_release_unsettled_stakes():
    rows = (
        opportunity("a", result=-1),
        opportunity("b", duration=3),
        opportunity("c", day=1),
    )
    capped = policy(fraction=1, ticket=0.1, fixture=0.1, total=0.2)
    result = evaluate_policy(rows, capped)
    assert result.stakes == pytest.approx({"a": 10, "b": 10, "c": 8})
    assert result.final_equity == pytest.approx(108)


def test_a_losing_development_population_does_not_select_any_policy():
    development = tuple(opportunity(f"d{i}", day=8 * i, result=-1) for i in range(24))
    holdout = tuple(opportunity(f"h{i}", day=220 + 8 * i) for i in range(24))
    report = select_policy(
        development,
        holdout,
        target=RiskTarget(min_fixtures=20, bootstrap_samples=100),
    )
    assert report.status == "no_eligible_policy"
    assert report.selected_policy is None
    assert report.holdout is None
