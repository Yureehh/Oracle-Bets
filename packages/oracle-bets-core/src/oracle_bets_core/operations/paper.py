"""Deterministic independent winner-only paper-action gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from oracle_bets_core.betting import expected_edge, kelly_fraction
from oracle_bets_core.league_selection import actionable_leagues

MINIMUM_CONSERVATIVE_EDGE = 0.05
MINIMUM_FAVORITE_PROBABILITY = 0.525
MINIMUM_DECIMAL_ODDS = 1.5
MINIMUM_ENTRY_HOURS = 24.0
MAXIMUM_ENTRY_HOURS = 48.0
MAXIMUM_MODEL_MARKET_DISAGREEMENT = 0.20
RATING_BASELINE_CONTRADICTION = 0.45
KELLY_MULTIPLIER = 0.25
MAXIMUM_POSITION_UNITS = 1.0
MAXIMUM_DAILY_EXPOSURE_UNITS = 3.0
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
    is_model_favorite: bool = True
    hours_to_start: float = 36.0
    market_probability: float | None = None
    rating_baseline_probability: float = 0.5
    attribution_stable: bool = True


@dataclass(frozen=True)
class ActionGateDecision:
    proposal_id: str
    fixture_id: str
    state: ActionState
    reason: str | None
    conservative_edge: float | None
    stake_units: float
    counterfactual_quarter_kelly_units: float = 0.0


def apply_action_gate(value: ActionGateInput) -> ActionGateDecision:
    """Apply model, league, roster, quote, edge, and stake gates in order."""
    blocked_reason = _blocked_reason(value)
    if blocked_reason:
        return _decision(value, ActionState.BLOCKED, blocked_reason)

    conservative_edge = round(
        expected_edge(value.decimal_odds, value.probability_lower),
        12,
    )
    if conservative_edge < MINIMUM_CONSERVATIVE_EDGE:
        return _decision(
            value,
            ActionState.NO_EDGE,
            "insufficient_conservative_edge",
            edge=conservative_edge,
        )
    counterfactual = min(
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
        stake=1.0,
        counterfactual=counterfactual,
    )


def select_fixture_actions(
    decisions: list[ActionGateDecision] | tuple[ActionGateDecision, ...],
    *,
    existing_exposure_units: float = 0.0,
) -> tuple[ActionGateDecision, ...]:
    """Keep at most one flat-unit winner action per fixture, without a portfolio claim."""
    if not math.isfinite(existing_exposure_units) or existing_exposure_units < 0.0:
        raise ValueError("existing exposure must be nonnegative and finite")
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
    exposure = existing_exposure_units
    for decision in best_by_fixture.values():
        if exposure + decision.stake_units > MAXIMUM_DAILY_EXPOSURE_UNITS:
            continue
        selected.append(decision)
        exposure += decision.stake_units
    return tuple(selected)


def _blocked_reason(value: ActionGateInput) -> str | None:  # noqa: PLR0911, PLR0912
    if not value.model_healthy:
        return "model_unhealthy"
    if value.target != "series_winner":
        return "winner_only_strategy"
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
    if value.decimal_odds < MINIMUM_DECIMAL_ODDS:
        return "odds_below_1.50"
    if not 0 <= value.probability_lower <= value.probability <= 1:
        return "invalid_probability"
    if not value.is_model_favorite:
        return "selection_not_model_favorite"
    if value.probability < MINIMUM_FAVORITE_PROBABILITY:
        return "favorite_probability_below_threshold"
    if not MINIMUM_ENTRY_HOURS <= value.hours_to_start <= MAXIMUM_ENTRY_HOURS:
        return "outside_48_to_24_hour_entry_window"
    if (
        value.market_probability is not None
        and abs(value.probability - value.market_probability)
        >= MAXIMUM_MODEL_MARKET_DISAGREEMENT
    ):
        return "model_market_disagreement_quarantine"
    if value.rating_baseline_probability <= RATING_BASELINE_CONTRADICTION:
        return "rating_baseline_contradiction"
    if not value.attribution_stable:
        return "unstable_feature_attribution"
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
    counterfactual: float = 0.0,
) -> ActionGateDecision:
    return ActionGateDecision(
        proposal_id=value.proposal_id,
        fixture_id=value.fixture_id,
        state=state,
        reason=reason,
        conservative_edge=edge,
        stake_units=round(stake, 6),
        counterfactual_quarter_kelly_units=round(counterfactual, 6),
    )
