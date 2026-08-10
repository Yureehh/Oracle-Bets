from oracle_bets_core.operations.paper import (
    MAXIMUM_DAILY_UNITS,
    RESEARCH_PROP_UNITS,
    ActionGateInput,
    ActionState,
    apply_action_gate,
    select_fixture_actions,
)

EXPECTED_EDGE = 0.18


def _gate(**overrides):
    values = {
        "proposal_id": "proposal-1",
        "fixture_id": "fixture-1",
        "target": "map_winner",
        "league": "LCK",
        "probability": 0.62,
        "probability_lower": 0.59,
        "decimal_odds": 2.0,
        "model_healthy": True,
        "roster_ready": True,
        "uncertainty_available": True,
        "market_supported": True,
        "quote_valid": True,
    }
    values.update(overrides)
    return apply_action_gate(ActionGateInput(**values))


def test_conservative_edge_and_quarter_kelly_create_capped_paper_action():
    decision = _gate()

    assert decision.state is ActionState.PAPER_ACTIONABLE
    assert decision.conservative_edge == EXPECTED_EDGE
    assert 0 < decision.stake_units <= 1


def test_edge_below_five_percent_is_no_edge():
    decision = _gate(probability_lower=0.51, decimal_odds=2.0)

    assert decision.state is ActionState.NO_EDGE
    assert decision.stake_units == 0


def test_lcp_cblol_and_unknown_rosters_are_blocked():
    assert _gate(league="LCP").reason == "league_not_actionable"
    assert _gate(league="CBLOL").reason == "league_not_actionable"
    assert _gate(roster_ready=False).reason == "roster_unstable"


def test_weak_scalar_props_remain_research_only_at_fixed_stake():
    decision = _gate(target="total_kills", evidence_status="weak_signal")

    assert decision.state is ActionState.RESEARCH_ONLY
    assert decision.stake_units == RESEARCH_PROP_UNITS


def test_only_largest_edge_per_fixture_and_three_daily_units_survive():
    decisions = [
        _gate(proposal_id="a", fixture_id="one", probability_lower=0.60),
        _gate(
            proposal_id="b", fixture_id="one", probability=0.70, probability_lower=0.65
        ),
        _gate(
            proposal_id="c", fixture_id="two", probability=0.70, probability_lower=0.64
        ),
        _gate(
            proposal_id="d",
            fixture_id="three",
            probability=0.70,
            probability_lower=0.63,
        ),
        _gate(proposal_id="e", fixture_id="four", probability_lower=0.62),
    ]

    selected = select_fixture_actions(decisions)

    assert [item.proposal_id for item in selected] == ["b", "c", "d"]
    assert sum(item.stake_units for item in selected) <= MAXIMUM_DAILY_UNITS
