from oracle_bets_core.operations.paper import (
    ActionGateInput,
    ActionState,
    apply_action_gate,
    select_fixture_actions,
)

EXPECTED_EDGE = 0.18
EXPECTED_SHORT_ODDS_EDGE = 0.1175


def _gate(**overrides):
    values = {
        "proposal_id": "proposal-1",
        "fixture_id": "fixture-1",
        "target": "series_winner",
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


def test_conservative_edge_creates_flat_one_unit_paper_action():
    decision = _gate()

    assert decision.state is ActionState.PAPER_ACTIONABLE
    assert decision.conservative_edge == EXPECTED_EDGE
    assert decision.stake_units == 1
    assert 0 < decision.counterfactual_quarter_kelly_units <= 1


def test_edge_below_five_percent_is_no_edge():
    decision = _gate(probability_lower=0.51, decimal_odds=2.0)

    assert decision.state is ActionState.NO_EDGE
    assert decision.stake_units == 0


def test_conservative_bound_may_cross_fifty_when_executable_edge_is_real():
    decision = _gate(
        probability=0.58,
        probability_lower=0.48,
        decimal_odds=2.30,
        market_probability=1 / 2.30,
    )

    assert decision.state is ActionState.PAPER_ACTIONABLE


def test_g2_navi_style_extreme_disagreement_is_quarantined_not_hardcoded():
    decision = _gate(
        probability=0.58,
        probability_lower=0.52,
        decimal_odds=4.0,
        market_probability=0.25,
    )

    assert decision.state is ActionState.BLOCKED
    assert decision.reason == "model_market_disagreement_quarantine"


def test_weak_point_favorite_is_blocked_even_when_market_odds_are_long():
    decision = _gate(
        probability=0.57,
        probability_lower=0.48,
        decimal_odds=2.30,
        market_probability=1 / 2.30,
    )

    assert decision.state is ActionState.BLOCKED
    assert decision.reason == "favorite_probability_below_threshold"


def test_lcp_cblol_and_unknown_rosters_are_blocked():
    assert _gate(league="LCP").reason == "league_not_actionable"
    assert _gate(league="CBLOL").reason == "league_not_actionable"
    assert _gate(roster_ready=False).reason == "roster_unstable"


def test_scalar_props_are_shadow_only_and_never_actionable():
    decision = _gate(target="total_kills", evidence_status="weak_signal")

    assert decision.state is ActionState.BLOCKED
    assert decision.reason == "winner_only_strategy"
    assert decision.stake_units == 0


def test_only_largest_edge_per_fixture_survives_with_daily_exposure_cap():
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
    assert all(item.stake_units == 1 for item in selected)


def test_daily_exposure_cap_includes_existing_paper_positions():
    decisions = [
        _gate(
            proposal_id="a",
            fixture_id="one",
            probability=0.70,
            probability_lower=0.65,
        ),
        _gate(
            proposal_id="b",
            fixture_id="two",
            probability=0.70,
            probability_lower=0.64,
        ),
    ]

    selected = select_fixture_actions(decisions, existing_exposure_units=2.0)

    assert [item.proposal_id for item in selected] == ["a"]


def test_short_odds_are_allowed_when_the_conservative_edge_is_real():
    decision = _gate(decimal_odds=1.49, probability=0.8, probability_lower=0.75)

    assert decision.state is ActionState.PAPER_ACTIONABLE
    assert decision.conservative_edge == EXPECTED_SHORT_ODDS_EDGE
