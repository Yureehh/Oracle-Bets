"""Pure prematch LoL market probability transformations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from oracle_bets_core.betting import expected_edge, kelly_fraction

RECOMMENDATION_MINIMUM_FAVORITE = 0.51
ONE_UNIT_BANKROLL_FRACTION = 0.01
MARKET_POLICY_VERSION = "lol-market-policy-v1"


@dataclass(frozen=True)
class SeriesPathDistribution:
    """Legal terminal BO-series outcomes derived from one map probability."""

    team_a_win: float
    exact_score: dict[tuple[int, int], float]
    total_maps: dict[int, float]
    map_differential: dict[int, float]


class DecisionClass(StrEnum):
    RECOMMENDED = "recommended"
    EXPLORATION = "exploration"
    NOT_COMPARABLE = "not_comparable"


@dataclass(frozen=True)
class SizingPaths:
    """Frozen bankroll fractions and 1%-bankroll units for one quote."""

    flat_1u: float
    full_kelly: float
    half_kelly: float
    quarter_kelly: float
    selected_path: str | None

    def to_dict(self) -> dict[str, Any]:
        fractions = {
            "flat_1u": self.flat_1u,
            "full_kelly": self.full_kelly,
            "half_kelly": self.half_kelly,
            "quarter_kelly": self.quarter_kelly,
        }
        return {
            "bankroll_fractions": fractions,
            "stake_units": {
                name: value / ONE_UNIT_BANKROLL_FRACTION
                for name, value in fractions.items()
            },
            "selected_path": self.selected_path,
        }


@dataclass(frozen=True)
class MarketDecision:
    decision_id: str
    classification: DecisionClass
    reason_codes: tuple[str, ...]
    point_ev: float | None
    conservative_ev: float | None
    sizing: SizingPaths
    policy_version: str = MARKET_POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "classification": self.classification.value,
            "reason_codes": list(self.reason_codes),
            "point_ev": self.point_ev,
            "conservative_ev": self.conservative_ev,
            "sizing": self.sizing.to_dict(),
            "policy_version": self.policy_version,
        }


def decide_market(
    *,
    semantic_fingerprint: str,
    target: str,
    readiness: str,
    probability: float | None,
    conservative_probability: float | None,
    decimal_odds: float | None,
    is_model_favorite: bool,
    hard_blocks: tuple[str, ...] = (),
) -> MarketDecision:
    """Classify one frozen forecast/quote without provider or model side effects."""
    reasons = list(dict.fromkeys(hard_blocks))
    if readiness == "display_only":
        reasons.append("strategy_display_only")
    if probability is None:
        reasons.append("model_probability_unavailable")
    if decimal_odds is None:
        reasons.append("executable_quote_unavailable")
    if reasons:
        return _decision(
            semantic_fingerprint,
            DecisionClass.NOT_COMPARABLE,
            reasons,
            point_ev=None,
            conservative_ev=None,
            sizing=_zero_sizing(),
        )

    assert probability is not None
    assert decimal_odds is not None
    point_ev = expected_edge(decimal_odds, probability)
    conservative_ev = (
        expected_edge(decimal_odds, conservative_probability)
        if conservative_probability is not None
        else None
    )
    recommendation_reasons: list[str] = []
    if target != "series_winner":
        recommendation_reasons.append("target_exploration_only")
    if readiness != "recommendation_active":
        recommendation_reasons.append("cohort_not_recommendation_active")
    if not is_model_favorite:
        recommendation_reasons.append("selection_not_model_favorite")
    if probability < RECOMMENDATION_MINIMUM_FAVORITE:
        recommendation_reasons.append("favorite_probability_below_0.51")
    if point_ev <= 0:
        recommendation_reasons.append("point_ev_non_positive")
    if conservative_ev is None:
        recommendation_reasons.append("conservative_probability_unavailable")
    elif conservative_ev <= 0:
        recommendation_reasons.append("conservative_ev_non_positive")
    classification = (
        DecisionClass.RECOMMENDED
        if not recommendation_reasons
        else DecisionClass.EXPLORATION
    )
    reasons.extend(recommendation_reasons or ["all_recommendation_gates_passed"])
    sizing = sizing_paths(
        decimal_odds=decimal_odds,
        probability=probability,
        classification=classification,
    )
    return _decision(
        semantic_fingerprint,
        classification,
        reasons,
        point_ev=point_ev,
        conservative_ev=conservative_ev,
        sizing=sizing,
    )


def sizing_paths(
    *, decimal_odds: float, probability: float, classification: DecisionClass
) -> SizingPaths:
    """Calculate every requested paper sizing path from one frozen forecast."""
    full = kelly_fraction(decimal_odds, probability, fraction=1.0)
    return SizingPaths(
        flat_1u=ONE_UNIT_BANKROLL_FRACTION,
        full_kelly=full,
        half_kelly=full / 2.0,
        quarter_kelly=full / 4.0,
        selected_path=(
            "full_kelly"
            if full > 0
            else "flat_1u"
            if classification is DecisionClass.EXPLORATION
            else None
        ),
    )


def rank_market_decisions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select at most one exploration per fixture, target, and semantic period."""
    output = [dict(action) for action in actions]
    groups: dict[tuple[str, str, str], list[int]] = {}
    for index, action in enumerate(output):
        classification = action.get("classification")
        action["bet_type"] = classification
        action["exploration_sampled"] = False
        action["ticket_eligible"] = classification == DecisionClass.RECOMMENDED
        semantic = action.get("semantic_key")
        if classification != DecisionClass.EXPLORATION or not isinstance(
            semantic, dict
        ):
            continue
        groups.setdefault(
            (
                str(action.get("fixture_key") or ""),
                str(semantic.get("target")),
                str(semantic.get("period")),
            ),
            [],
        ).append(index)
    for indices in groups.values():
        selected = max(
            indices,
            key=lambda index: (
                _sortable_ev(output[index].get("point_ev")),
                str(output[index].get("semantic_fingerprint") or ""),
            ),
        )
        output[selected]["exploration_sampled"] = True
        output[selected]["ticket_eligible"] = True
        for index in indices:
            if index != selected:
                reasons = list(output[index].get("reason_codes") or [])
                reasons.append("exploration_sample_already_filled")
                output[index]["reason_codes"] = list(dict.fromkeys(reasons))
    return output


def _sortable_ev(value: Any) -> float:
    return float(value) if value is not None else float("-inf")


def _zero_sizing() -> SizingPaths:
    return SizingPaths(0.0, 0.0, 0.0, 0.0, None)


def _decision(
    semantic_fingerprint: str,
    classification: DecisionClass,
    reasons: list[str],
    *,
    point_ev: float | None,
    conservative_ev: float | None,
    sizing: SizingPaths,
) -> MarketDecision:
    payload = {
        "policy": MARKET_POLICY_VERSION,
        "semantic": semantic_fingerprint,
        "classification": classification.value,
        "reasons": list(dict.fromkeys(reasons)),
        "point_ev": point_ev,
        "conservative_ev": conservative_ev,
        "sizing": sizing.to_dict(),
    }
    decision_id = (
        "decision-"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
    )
    return MarketDecision(
        decision_id=decision_id,
        classification=classification,
        reason_codes=tuple(payload["reasons"]),
        point_ev=point_ev,
        conservative_ev=conservative_ev,
        sizing=sizing,
    )


def enumerate_series_paths(
    team_a_map_probability: float,
    *,
    best_of: int,
) -> SeriesPathDistribution:
    """Enumerate paths that stop immediately when either team clinches."""
    probability = float(team_a_map_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("map probability must be between zero and one")
    if best_of not in {1, 3, 5}:
        raise ValueError("best_of must be 1, 3, or 5")
    wins_needed = best_of // 2 + 1
    exact_score: dict[tuple[int, int], float] = {}

    def visit(a_wins: int, b_wins: int, path_probability: float) -> None:
        if wins_needed in {a_wins, b_wins}:
            score = (a_wins, b_wins)
            exact_score[score] = exact_score.get(score, 0.0) + path_probability
            return
        visit(a_wins + 1, b_wins, path_probability * probability)
        visit(a_wins, b_wins + 1, path_probability * (1.0 - probability))

    visit(0, 0, 1.0)
    totals: dict[int, float] = {}
    differentials: dict[int, float] = {}
    for (a_wins, b_wins), value in exact_score.items():
        totals[a_wins + b_wins] = totals.get(a_wins + b_wins, 0.0) + value
        difference = a_wins - b_wins
        differentials[difference] = differentials.get(difference, 0.0) + value
    return SeriesPathDistribution(
        team_a_win=sum(
            value for (a_wins, b_wins), value in exact_score.items() if a_wins > b_wins
        ),
        exact_score=exact_score,
        total_maps=totals,
        map_differential=differentials,
    )
