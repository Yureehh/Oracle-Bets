"""Deterministic paper-action gates and correlated exposure selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from oracle_bets_core.betting import expected_edge, kelly_fraction
from oracle_bets_core.league_selection import actionable_leagues

MINIMUM_CONSERVATIVE_EDGE = 0.05
KELLY_MULTIPLIER = 0.25
MAXIMUM_POSITION_UNITS = 1.0
MAXIMUM_DAILY_UNITS = 3.0
RESEARCH_PROP_UNITS = 0.25
DEFAULT_BANKROLL_UNITS = 100.0
RESEARCH_PROP_TARGETS = frozenset({"gamelength", "total_kills", "total_towers"})


class ActionState(StrEnum):
    """Stable result of applying every deterministic model and risk gate."""

    BLOCKED = "blocked"
    NO_EDGE = "no_edge"
    RESEARCH_ONLY = "research_only"
    PAPER_ACTIONABLE = "paper_actionable"


@dataclass(frozen=True)
class ActionGateInput:
    proposal_id: str
    fixture_id: str
    target: str
    league: str
    probability: float
    probability_lower: float
    decimal_odds: float
    model_healthy: bool
    roster_ready: bool
    uncertainty_available: bool
    market_supported: bool
    quote_valid: bool
    evidence_status: str = "meets_basic_sanity"
    bankroll_units: float = DEFAULT_BANKROLL_UNITS


@dataclass(frozen=True)
class ActionGateDecision:
    proposal_id: str
    fixture_id: str
    state: ActionState
    reason: str | None
    conservative_edge: float | None
    stake_units: float


def apply_action_gate(value: ActionGateInput) -> ActionGateDecision:
    """Apply model, league, roster, quote, edge, and stake gates in order."""
    blocked_reason = _blocked_reason(value)
    if blocked_reason:
        return _decision(value, ActionState.BLOCKED, blocked_reason)

    conservative_edge = round(
        expected_edge(value.decimal_odds, value.probability_lower),
        12,
    )
    if value.target in RESEARCH_PROP_TARGETS:
        return _decision(
            value,
            ActionState.RESEARCH_ONLY,
            "weak_prop_evidence",
            edge=conservative_edge,
            stake=RESEARCH_PROP_UNITS,
        )
    if conservative_edge < MINIMUM_CONSERVATIVE_EDGE:
        return _decision(
            value,
            ActionState.NO_EDGE,
            "insufficient_conservative_edge",
            edge=conservative_edge,
        )
    stake = min(
        MAXIMUM_POSITION_UNITS,
        value.bankroll_units
        * kelly_fraction(
            value.decimal_odds,
            value.probability_lower,
            fraction=KELLY_MULTIPLIER,
        ),
    )
    return _decision(
        value,
        ActionState.PAPER_ACTIONABLE,
        None,
        edge=conservative_edge,
        stake=stake,
    )


def select_fixture_actions(
    decisions: list[ActionGateDecision] | tuple[ActionGateDecision, ...],
    *,
    maximum_daily_units: float = MAXIMUM_DAILY_UNITS,
) -> tuple[ActionGateDecision, ...]:
    """Keep one highest-edge action per fixture under the daily exposure cap."""
    if maximum_daily_units <= 0:
        raise ValueError("maximum_daily_units must be positive")
    best_by_fixture: dict[str, ActionGateDecision] = {}
    actionable = (
        decision
        for decision in decisions
        if decision.state is ActionState.PAPER_ACTIONABLE
    )
    for decision in sorted(
        actionable,
        key=lambda item: (item.conservative_edge or float("-inf"), item.proposal_id),
        reverse=True,
    ):
        best_by_fixture.setdefault(decision.fixture_id, decision)

    selected: list[ActionGateDecision] = []
    exposure = 0.0
    for decision in best_by_fixture.values():
        if exposure + decision.stake_units > maximum_daily_units:
            continue
        selected.append(decision)
        exposure += decision.stake_units
    return tuple(selected)


def _blocked_reason(value: ActionGateInput) -> str | None:  # noqa: PLR0911
    if not value.model_healthy:
        return "model_unhealthy"
    if value.league not in actionable_leagues():
        return "league_not_actionable"
    if not value.roster_ready:
        return "roster_unstable"
    if not value.uncertainty_available:
        return "uncertainty_unavailable"
    if not value.market_supported:
        return "unsupported_market"
    if not value.quote_valid or value.decimal_odds <= 1:
        return "invalid_quote"
    if not 0 <= value.probability_lower <= value.probability <= 1:
        return "invalid_probability"
    if value.bankroll_units <= 0:
        return "invalid_bankroll"
    if (
        value.target not in RESEARCH_PROP_TARGETS
        and value.evidence_status != "meets_basic_sanity"
    ):
        return "target_evidence_unhealthy"
    return None


def _decision(
    value: ActionGateInput,
    state: ActionState,
    reason: str | None,
    *,
    edge: float | None = None,
    stake: float = 0.0,
) -> ActionGateDecision:
    return ActionGateDecision(
        proposal_id=value.proposal_id,
        fixture_id=value.fixture_id,
        state=state,
        reason=reason,
        conservative_edge=edge,
        stake_units=round(stake, 6),
    )
