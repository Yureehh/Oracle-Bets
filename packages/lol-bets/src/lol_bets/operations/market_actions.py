"""Exact-event LoL market inventory, probability mapping, and read-only quotes."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

from oracle_bets_core.betting import (
    decimal_odds_from_probability,
    probability_from_decimal_odds,
)
from oracle_bets_core.io_utils import load_model
from oracle_bets_core.markets import (
    MarketDataError,
    PolymarketMarket,
    capture_current_order_books,
    market_semantic_key,
)
from oracle_bets_core.paths import (
    GAMELENGTH_PREDICTION_PROP_CALIBRATOR,
    MODEL_REGISTRY_DIR,
    MODELS_DIR,
    TOTAL_KILLS_PREDICTION_PROP_CALIBRATOR,
    TOTAL_TOWERS_PREDICTION_PROP_CALIBRATOR,
)
from oracle_bets_core.pd import pd

from lol_bets.inference.team_resolver import canonical_team_name, team_name_variants
from lol_bets.operations.market_strategies import (
    decide_market,
    enumerate_series_paths,
    rank_market_decisions,
)
from lol_bets.operations.models import ModelRegistry, resolve_serving_artifact

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from oracle_bets_core.markets import OrderBookClient

_WINNER_TYPES = {"moneyline": "series_winner", "child_moneyline": "map_winner"}
_HANDICAP_TYPES = {"map_handicap", "match_handicap"}
_PROP_MARKET_TYPES = {
    "gamelength": "gamelength_mean",
    "game_length": "gamelength_mean",
    "game_duration": "gamelength_mean",
    "kill_over_under_game": "total_kills_mean",
    "kills": "total_kills_mean",
    "towers": "total_towers_mean",
}
_PROP_CALIBRATORS = {
    "gamelength_mean": GAMELENGTH_PREDICTION_PROP_CALIBRATOR,
    "total_kills_mean": TOTAL_KILLS_PREDICTION_PROP_CALIBRATOR,
    "total_towers_mean": TOTAL_TOWERS_PREDICTION_PROP_CALIBRATOR,
}
LOL_RESOLUTION_RULE_TERMS = ("liquipedia", "leagueoflegends", "gol.gg")
_START_TOLERANCE = timedelta(hours=6)
_EVEN_PROBABILITY = 0.5
_LINE_PATTERN = re.compile(r"(?P<label>[^()]+?)\s*\((?P<line>[+-]?\d+(?:\.\d+)?)\)")


@dataclass(frozen=True)
class PendingMarketAction:
    comparison_id: str
    fixture_id: str
    fixture_key: str
    league: str
    team_a: str
    team_b: str
    target: str
    strategy_version: str | None
    probability_source: str | None
    game_number: int | None
    line: float | None
    selection: str
    probability: float | None
    probability_lower: float | None
    rating_baseline_probability: float
    is_model_favorite: bool
    attribution_stable: bool
    start_time: datetime
    market_id: str
    token_id: str
    market_url: str
    resolution_source: str
    roster_ready: bool
    roster_confidence: str
    uncertainty_available: bool
    readiness: str
    semantic_key: dict[str, Any] | None
    semantic_fingerprint: str | None
    hard_blocks: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DailyMarketEvaluation:
    """Complete contract inventory and outcome-level quote/gate evidence."""

    reviews: tuple[dict[str, Any], ...]
    actions: tuple[dict[str, Any], ...]


def active_strategy_readiness() -> dict[str, str]:
    """Flatten the champion's immutable target/cohort readiness cells."""
    artifact = ModelRegistry(MODEL_REGISTRY_DIR).strategy_readiness() or {}
    cells = artifact.get("cells")
    if not isinstance(cells, list):
        return {}
    return {
        f"{cell['target']}|{cell['cohort']}": str(cell["state"])
        for cell in cells
        if isinstance(cell, dict)
        and all(key in cell for key in ("target", "cohort", "state"))
    }


def evaluate_daily_market_actions(
    *,
    schedule: pd.DataFrame,
    snapshot_rows: Sequence[dict[str, Any]],
    markets: Sequence[PolymarketMarket] | Mapping[str, Sequence[PolymarketMarket]],
    clob_client: OrderBookClient,
    model_healthy: bool,
    strategy_readiness: Mapping[str, str] | None = None,
    run_key: str | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    rank: bool = True,
) -> DailyMarketEvaluation:
    """Inventory every exact-event contract and quote every outcome once."""
    grouped = _group_markets(schedule, markets)
    snapshots = _snapshot_index(snapshot_rows)
    reviewed_at = clock()
    pending: list[PendingMarketAction] = []
    reviews: list[dict[str, Any]] = []
    inventory_actions: list[dict[str, Any]] = []
    for _, fixture in schedule.iterrows():
        fixture_key = str(fixture.get("match_key") or "").strip()
        fixture_snapshots = snapshots.get(fixture_key, {})
        for market in grouped.get(fixture_key, ()):
            mapped, review = _interpret_market(
                fixture,
                market,
                fixture_snapshots,
                reviewed_at=reviewed_at,
                model_healthy=model_healthy,
                strategy_readiness=strategy_readiness or {},
                run_key=run_key,
            )
            pending.extend(mapped)
            reviews.append(review)
            if review["target"] == "unknown":
                inventory_actions.extend(
                    _unsupported_inventory_actions(
                        fixture,
                        market,
                        reviewed_at=reviewed_at,
                        run_key=run_key,
                    )
                )
    if not pending:
        actions = _optional_ranking(inventory_actions, rank=rank)
        return DailyMarketEvaluation(tuple(reviews), tuple(actions))

    batch = capture_current_order_books(
        clob_client,
        token_ids=tuple(item.token_id for item in pending),
        clock=clock,
    )
    action_rows: list[dict[str, Any]] = []
    for item in pending:
        observation = batch.observations.get(item.token_id)
        failure = batch.failures.get(item.token_id)
        provider_warnings = batch.warnings.get(item.token_id, ())
        if observation is None:
            hard_blocks = tuple(
                dict.fromkeys(
                    (
                        *item.hard_blocks,
                        failure.reason if failure else "book_unavailable",
                    )
                )
            )
            action_rows.append(
                _action_payload(
                    item,
                    state="not_comparable",
                    reason=hard_blocks[0],
                    hard_blocks=hard_blocks,
                    warnings=(*item.warnings, *provider_warnings),
                    observation=None,
                )
            )
            continue
        odds = float(observation.fill.decimal_odds or 0.0)
        if item.probability is None or item.probability_lower is None:
            hard_blocks = tuple(
                dict.fromkeys((*item.hard_blocks, "unsupported_probability_mapping"))
            )
            action_rows.append(
                _action_payload(
                    item,
                    state="not_comparable",
                    reason=hard_blocks[0],
                    hard_blocks=hard_blocks,
                    warnings=(*item.warnings, *provider_warnings),
                    observation=observation,
                )
            )
            continue
        warnings = [*item.warnings, *provider_warnings]
        if not model_healthy:
            warnings.append("model_unhealthy")
        if not item.roster_ready or item.roster_confidence in {"low", "unknown"}:
            warnings.append("roster_confidence_low_or_unknown")
        if not item.uncertainty_available:
            warnings.append("uncertainty_unavailable")
        market_probability = probability_from_decimal_odds(odds)
        if abs(item.probability - market_probability) >= 0.20:  # noqa: PLR2004
            warnings.append("model_market_disagreement_high")
        elif abs(item.probability - market_probability) >= 0.10:  # noqa: PLR2004
            warnings.append("model_market_disagreement_attention")
        action_rows.append(
            _action_payload(
                item,
                state="quoted" if not item.hard_blocks else "not_comparable",
                reason=item.hard_blocks[0] if item.hard_blocks else None,
                hard_blocks=item.hard_blocks,
                warnings=tuple(warnings),
                observation=observation,
            )
        )
    actions = [*action_rows, *inventory_actions]
    actions = _optional_ranking(actions, rank=rank)
    return DailyMarketEvaluation(tuple(reviews), tuple(actions))


def _interpret_market(
    fixture: pd.Series,
    market: PolymarketMarket,
    snapshots: dict[str, list[dict[str, Any]]],
    *,
    reviewed_at: datetime,
    model_healthy: bool,
    strategy_readiness: Mapping[str, str],
    run_key: str | None,
) -> tuple[list[PendingMarketAction], dict[str, Any]]:
    team_a = str(fixture.get("team_a") or "").strip()
    team_b = str(fixture.get("team_b") or "").strip()
    fixture_key = str(fixture.get("match_key") or f"{team_a}:{team_b}")
    start = _utc_datetime(fixture.get("start_utc"))
    best_of = int(fixture.get("best_of") or 1)
    target, strategy, source = _market_classification(market)
    hard_blocks = _contract_blocks(
        market,
        start=start,
        best_of=best_of,
        reviewed_at=reviewed_at,
    )
    if not model_healthy:
        hard_blocks = (*hard_blocks, "model_unhealthy")
    probabilities, mapping_warnings = _market_probabilities(
        market,
        target=target,
        team_a=team_a,
        team_b=team_b,
        best_of=best_of,
        snapshots=snapshots,
    )
    if probabilities is None:
        hard_blocks = (*hard_blocks, "unsupported_probability_mapping")
        probabilities = {}
    actions: list[PendingMarketAction] = []
    for outcome in market.outcomes:
        mapped = probabilities.get(outcome.name.casefold())
        probability, lower, baseline, favorite = mapped or (None, None, 0.5, False)
        line = _outcome_line(
            market,
            target=target,
            outcome=outcome.name,
            team_a=team_a,
            team_b=team_b,
        )
        semantic = None
        outcome_blocks = list(hard_blocks)
        try:
            semantic = market_semantic_key(
                target=target,
                selection=_semantic_selection(
                    outcome.name, team_a=team_a, team_b=team_b
                ),
                game_number=market.game_number,
                line=line,
            )
        except MarketDataError as error:
            outcome_blocks.append(f"semantic_contract:{error}")
        comparison_id = _stable_id(
            "comparison",
            f"{run_key or reviewed_at.isoformat()}|{fixture_key}|"
            f"{market.market_id}|{outcome.token_id}",
        )
        actions.append(
            PendingMarketAction(
                comparison_id=comparison_id,
                fixture_id=fixture_key,
                fixture_key=fixture_key,
                league=str(fixture.get("league") or ""),
                team_a=team_a,
                team_b=team_b,
                target=target,
                strategy_version=strategy,
                probability_source=source,
                game_number=market.game_number,
                line=line,
                selection=outcome.name,
                probability=probability,
                probability_lower=lower,
                rating_baseline_probability=baseline,
                is_model_favorite=favorite,
                attribution_stable=_attribution_stable(snapshots),
                start_time=start,
                market_id=market.market_id,
                token_id=outcome.token_id,
                market_url=market.url,
                resolution_source=market.resolution_source,
                roster_ready=_roster_ready(snapshots),
                roster_confidence=_roster_confidence(snapshots),
                uncertainty_available=lower is not None,
                readiness=_readiness_state(
                    target,
                    league=str(fixture.get("league") or ""),
                    game_number=market.game_number,
                    readiness=strategy_readiness,
                ),
                semantic_key=semantic.to_dict() if semantic else None,
                semantic_fingerprint=semantic.fingerprint if semantic else None,
                hard_blocks=tuple(dict.fromkeys(outcome_blocks)),
                warnings=tuple(mapping_warnings),
            )
        )
    review = {
        "fixture_id": fixture_key,
        "market_id": market.market_id,
        "question": market.question,
        "sports_market_type": market.sports_market_type,
        "group_item_title": market.group_item_title,
        "target": target,
        "strategy_version": strategy,
        "probability_source": source,
        "game_number": market.game_number,
        "line": float(market.total_line) if market.total_line is not None else None,
        "outcomes": [asdict(outcome) for outcome in market.outcomes],
        "hard_blocks": list(dict.fromkeys(hard_blocks)),
        "warnings": list(mapping_warnings),
        "model_healthy": model_healthy,
        "direct_series_map_path_disagreement_pp": _series_disagreement(
            snapshots,
            team_a=team_a,
            team_b=team_b,
            best_of=best_of,
        ),
    }
    if target == "unknown":
        return [], review
    return actions, review


def _unsupported_inventory_actions(
    fixture: pd.Series,
    market: PolymarketMarket,
    *,
    reviewed_at: datetime,
    run_key: str | None,
) -> list[dict[str, Any]]:
    """Retain one compact unsupported-contract record without quoting outcomes."""
    fixture_key = str(fixture.get("match_key") or "")
    return [
        {
            "comparison_id": _stable_id(
                "comparison",
                f"{run_key or reviewed_at.isoformat()}|{fixture_key}|{market.market_id}",
            ),
            "fixture_id": fixture_key,
            "fixture_key": fixture_key,
            "league": str(fixture.get("league") or ""),
            "team_a": str(fixture.get("team_a") or ""),
            "team_b": str(fixture.get("team_b") or ""),
            "provider": "polymarket",
            "target": "unknown",
            "strategy_version": None,
            "probability_source": None,
            "game_number": market.game_number,
            "line": float(market.total_line) if market.total_line is not None else None,
            "selection": None,
            "selections": [outcome.name for outcome in market.outcomes],
            "probability": None,
            "probability_lower": None,
            "start_time": _utc_datetime(fixture.get("start_utc")),
            "market_id": market.market_id,
            "token_id": None,
            "token_ids": [outcome.token_id for outcome in market.outcomes],
            "market_url": market.url,
            "resolution_source": market.resolution_source,
            "state": "not_comparable",
            "classification": "not_comparable",
            "reason": "no_model_target",
            "reason_codes": ["unsupported_probability_mapping"],
            "hard_blocks": ["unsupported_probability_mapping"],
            "warnings": ["no_model_target"],
            "point_edge": None,
            "decimal_odds": None,
            "quote_basis": "not_quoted",
            "observations": [],
        }
    ]


def price_manual_lines(
    *,
    fixture: pd.Series,
    snapshot_rows: Sequence[dict[str, Any]],
    lines: Sequence[dict[str, Any]],
    reviewed_at: datetime,
    strategy_readiness: Mapping[str, str] | None = None,
    rank: bool = True,
) -> list[dict[str, Any]]:
    """Price owner-entered bookmaker lines without contacting the provider."""
    fixture_key = str(fixture.get("match_key") or "")
    snapshots = _snapshot_index(snapshot_rows).get(fixture_key, {})
    team_a = str(fixture.get("team_a") or "").strip()
    team_b = str(fixture.get("team_b") or "").strip()
    best_of = int(fixture.get("best_of") or 1)
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(lines, start=1):
        target = str(raw.get("target") or "").strip()
        selection = str(raw.get("selection") or "").strip()
        odds = float(raw.get("decimal_odds") or 0)
        line = float(raw["line"]) if raw.get("line") is not None else None
        if not target or not selection or not math.isfinite(odds) or odds <= 1:
            raise ValueError(
                "Manual lines require target, selection, and finite odds > 1."
            )
        if line is not None and not math.isfinite(line):
            raise ValueError("Manual market lines must be finite.")
        probability, probability_lower, warning = _manual_probabilities(
            target=target,
            selection=selection,
            line=line,
            snapshots=snapshots,
            team_a=team_a,
            team_b=team_b,
            best_of=best_of,
        )
        hard_blocks: list[str] = []
        semantic = None
        try:
            semantic = market_semantic_key(
                target=target,
                selection=_semantic_selection(selection, team_a=team_a, team_b=team_b),
                game_number=(
                    int(raw["game_number"])
                    if raw.get("game_number") is not None
                    else None
                ),
                line=line,
            )
        except (MarketDataError, TypeError, ValueError) as error:
            hard_blocks.append(f"semantic_contract:{error}")
        market_id = str(raw.get("market_id") or f"thunderpick-{index}")
        token_id = _stable_id(
            "manual-selection",
            f"{fixture_key}|{market_id}|{selection}|{line}|{odds}",
        )
        readiness = _readiness_state(
            target,
            league=str(fixture.get("league") or ""),
            game_number=(
                int(raw["game_number"]) if raw.get("game_number") is not None else None
            ),
            readiness=strategy_readiness or {},
        )
        decision = decide_market(
            semantic_fingerprint=semantic.fingerprint if semantic else token_id,
            target=target,
            readiness=readiness,
            probability=probability,
            conservative_probability=probability_lower,
            decimal_odds=odds,
            is_model_favorite=(
                probability is not None and probability >= _EVEN_PROBABILITY
            ),
            hard_blocks=tuple(hard_blocks),
        )
        decision_payload = decision.to_dict()
        selected_path = decision_payload["sizing"]["selected_path"]
        stake_units = (
            decision_payload["sizing"]["stake_units"][selected_path]
            if selected_path
            else 0.0
        )
        output.append(
            {
                "comparison_id": _stable_id("comparison", token_id),
                "fixture_id": fixture_key,
                "fixture_key": fixture_key,
                "league": str(fixture.get("league") or ""),
                "team_a": team_a,
                "team_b": team_b,
                "provider": "thunderpick",
                "target": target,
                "strategy_version": None,
                "probability_source": "owner_entered_line",
                "game_number": raw.get("game_number"),
                "line": line,
                "selection": selection,
                "probability": probability,
                "probability_lower": probability_lower,
                "rating_baseline_probability": _EVEN_PROBABILITY,
                "is_model_favorite": (
                    probability is not None and probability >= _EVEN_PROBABILITY
                ),
                "attribution_stable": True,
                "start_time": _utc_datetime(fixture.get("start_utc")),
                "market_id": market_id,
                "token_id": token_id,
                "market_url": str(raw.get("url") or ""),
                "resolution_source": "owner_entered",
                "roster_ready": _roster_ready(snapshots),
                "roster_confidence": _roster_confidence(snapshots),
                "uncertainty_available": probability is not None,
                "readiness": readiness,
                "semantic_key": semantic.to_dict() if semantic else None,
                "semantic_fingerprint": semantic.fingerprint if semantic else None,
                "state": decision.classification.value,
                "reason": decision.reason_codes[0]
                if decision.reason_codes
                else warning,
                "hard_blocks": hard_blocks,
                "warnings": [warning] if warning else ["owner_entered_bookmaker_line"],
                "point_edge": decision.point_ev,
                "conservative_edge": decision.conservative_ev,
                "stake_units": stake_units,
                "correlation_rank": None,
                "decimal_odds": odds,
                "fair_decimal_odds": decimal_odds_from_probability(probability)
                if probability
                else None,
                "requested_shares": None,
                "hypothetical_cost": None,
                "quote_basis": "owner_entered",
                "observations": [
                    {
                        "sequence_number": 1,
                        "observed_at": raw.get("observed_at")
                        or reviewed_at.isoformat(),
                        "provider_timestamp": None,
                        "book_hash": _stable_id("manual-quote", token_id),
                        "token_id": token_id,
                        "complete": True,
                        "minimum_order_size": None,
                        "requested_shares": None,
                        "filled_shares": None,
                        "unfilled_shares": None,
                        "hypothetical_cost": None,
                        "average_price": str(probability_from_decimal_odds(odds)),
                        "decimal_odds": odds,
                        "depth": {},
                    }
                ],
                "note": str(raw.get("note") or "").strip() or None,
                "owner_terms": str(raw.get("terms") or "").strip() or None,
            }
            | decision_payload
        )
    return _optional_ranking(output, rank=rank)


def _optional_ranking(
    actions: list[dict[str, Any]], *, rank: bool
) -> list[dict[str, Any]]:
    return rank_market_decisions(actions) if rank else actions


def _manual_probabilities(  # noqa: PLR0911, PLR0912
    *,
    target: str,
    selection: str,
    line: float | None,
    snapshots: dict[str, list[dict[str, Any]]],
    team_a: str,
    team_b: str,
    best_of: int,
) -> tuple[float | None, float | None, str | None]:
    if target in {"series_winner", "map_winner"}:
        rows = snapshots.get(target, [])
        side = _team_side(selection, team_a=team_a, team_b=team_b)
        wanted = team_a if side == "a" else team_b if side == "b" else None
        row = next(
            (
                item
                for item in rows
                if wanted
                and canonical_team_name(str(item.get("selection") or "")).casefold()
                == canonical_team_name(wanted).casefold()
            ),
            None,
        )
        if row is None:
            return None, None, "model_probability_unavailable"
        probability = float(row["model_value"])
        return probability, float(row.get("probability_lower", probability)), None
    if target in _PROP_CALIBRATORS:
        if line is None:
            return None, None, "prop_line_missing"
        rows = snapshots.get(target, [])
        if len(rows) != 1:
            return None, None, "prop_forecast_unavailable"
        try:
            signal = _prop_calibrator(target).price(
                mean=float(rows[0]["model_value"]),
                line=line,
                metadata={
                    "league": rows[0].get("league"),
                    "bo_format": rows[0].get("match_type"),
                },
            )
        except Exception:
            return None, None, "prop_calibrator_unavailable"
        side = selection.casefold()
        if side == "over":
            probability = float(signal.over_probability)
            return probability, probability, None
        if side == "under":
            probability = float(signal.under_probability)
            return probability, probability, None
        return None, None, "prop_outcome_orientation_unsupported"
    if target == "series_total_maps" and line is not None:
        if line.is_integer():
            return None, None, "total_push_probability_not_supported"
        map_rows = snapshots.get("map_winner", [])
        team_a_row = next(
            (
                row
                for row in map_rows
                if canonical_team_name(str(row.get("selection") or "")).casefold()
                == canonical_team_name(team_a).casefold()
            ),
            None,
        )
        if team_a_row is None:
            return None, None, "map_probability_unavailable"
        distribution = enumerate_series_paths(
            float(team_a_row["model_value"]), best_of=best_of
        )
        side = selection.casefold()
        if side not in {"over", "under"}:
            return None, None, "total_outcome_orientation_unsupported"
        probability = sum(
            value
            for maps, value in distribution.total_maps.items()
            if (maps > line if side == "over" else maps < line)
        )
        return probability, probability, "derived_map_path_v1"
    if target == "series_handicap" and line is not None:
        map_rows = snapshots.get("map_winner", [])
        team_a_row = next(
            (
                row
                for row in map_rows
                if canonical_team_name(str(row.get("selection") or "")).casefold()
                == canonical_team_name(team_a).casefold()
            ),
            None,
        )
        side = _team_side(selection, team_a=team_a, team_b=team_b)
        if team_a_row is None or side is None:
            return None, None, "map_probability_unavailable"
        distribution = enumerate_series_paths(
            float(team_a_row["model_value"]), best_of=best_of
        )
        if any(
            (difference if side == "a" else -difference) + line == 0
            for difference in distribution.map_differential
        ):
            return None, None, "handicap_push_probability_not_supported"
        probability = sum(
            value
            for difference, value in distribution.map_differential.items()
            if ((difference if side == "a" else -difference) + line) > 0
        )
        return probability, probability, "derived_map_path_v1"
    return None, None, "model_probability_unavailable"


def _market_classification(
    market: PolymarketMarket,
) -> tuple[str, str | None, str | None]:
    market_type = market.sports_market_type
    if market_type == "moneyline" and market.game_number is None:
        return "series_winner", "series_direct_v2", "direct_series_model"
    if market_type == "child_moneyline" and market.game_number is not None:
        return "map_winner", "map_prematch_v1", "prematch_map_model"
    prop_target = _prop_target(market)
    if prop_target is not None:
        return prop_target, None, "calibrated_regression_artifact"
    if market_type == "totals":
        return (
            "series_total_maps",
            "series_totals_map_path_v1",
            "derived_map_path_v1",
        )
    if market_type in _HANDICAP_TYPES:
        return (
            "series_handicap",
            "series_handicap_map_path_v1",
            "derived_map_path_v1",
        )
    return "unknown", None, None


def _outcome_line(
    market: PolymarketMarket,
    *,
    target: str,
    outcome: str,
    team_a: str,
    team_b: str,
) -> float | None:
    if target == "series_handicap":
        lines = _handicap_lines(market, team_a=team_a, team_b=team_b) or {}
        side = _team_side(outcome, team_a=team_a, team_b=team_b)
        return lines.get(side or "")
    return float(market.total_line) if market.total_line is not None else None


def _semantic_selection(value: str, *, team_a: str, team_b: str) -> str:
    side = _team_side(value, team_a=team_a, team_b=team_b)
    if side == "a":
        return canonical_team_name(team_a)
    if side == "b":
        return canonical_team_name(team_b)
    return value


def _readiness_state(
    target: str,
    *,
    league: str,
    game_number: int | None,
    readiness: Mapping[str, str],
) -> str:
    normalized_target = {
        "gamelength_mean": "gamelength",
        "total_kills_mean": "total_kills",
        "total_towers_mean": "total_towers",
    }.get(target, target)
    if normalized_target == "series_winner":
        states = (
            readiness.get("series_winner|actionable_tier1_plus_erls"),
            readiness.get(f"series_winner|league:{league}"),
        )
        if states == ("recommendation_active", "recommendation_active"):
            return "recommendation_active"
        if "display_only" in states:
            return "display_only"
        return "exploration_only"
    if normalized_target == "map_winner":
        normalized_target = (
            "map_winner:map_1" if game_number == 1 else "map_winner:later_maps"
        )
    return readiness.get(
        f"{normalized_target}|all_actionable",
        "exploration_only",
    )


def _prop_target(market: PolymarketMarket) -> str | None:
    """Recognize only scalar over/under contracts supported by an artifact."""
    outcome_names = {outcome.name.casefold() for outcome in market.outcomes}
    if outcome_names != {"over", "under"} or market.total_line is None:
        return None
    if market.sports_market_type in _PROP_MARKET_TYPES:
        return _PROP_MARKET_TYPES[market.sports_market_type]
    text = f"{market.group_item_title} {market.question}".casefold()
    for terms, target in (
        (("game length", "duration"), "gamelength_mean"),
        (("kill",), "total_kills_mean"),
        (("tower",), "total_towers_mean"),
    ):
        if any(term in text for term in terms):
            return target
    return None


def _market_probabilities(  # noqa: PLR0911
    market: PolymarketMarket,
    *,
    target: str,
    team_a: str,
    team_b: str,
    best_of: int,
    snapshots: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, tuple[float, float, float, bool]] | None, tuple[str, ...]]:
    if target in _WINNER_TYPES.values():
        snapshot_target = "series_winner" if target == "series_winner" else "map_winner"
        values = _team_snapshot_values(
            snapshots.get(snapshot_target, []),
            team_a=team_a,
            team_b=team_b,
        )
        if values is None:
            return None, ("model_snapshot_unavailable",)
        mapped = _winner_outcome_probabilities(
            market,
            values,
            team_a=team_a,
            team_b=team_b,
        )
        return mapped, (("experimental_strategy",) if target == "map_winner" else ())
    map_values = _team_snapshot_values(
        snapshots.get("map_winner", []),
        team_a=team_a,
        team_b=team_b,
    )
    if target in {"series_total_maps", "series_handicap"} and map_values is None:
        return None, ("map_model_snapshot_unavailable",)
    if target == "series_total_maps":
        assert map_values is not None
        return _total_probabilities(market, map_values, best_of), (
            "derived_path_approximation",
            "experimental_strategy",
        )
    if target == "series_handicap":
        assert map_values is not None
        return _handicap_probabilities(
            market,
            map_values,
            team_a=team_a,
            team_b=team_b,
            best_of=best_of,
        ), ("derived_path_approximation", "experimental_strategy")
    if target in _PROP_CALIBRATORS:
        probabilities, reason = _prop_probabilities(market, target, snapshots)
        warnings = ("display_only_prop", *((reason,) if reason else ()))
        return probabilities, warnings
    return None, ("no_model_target",)


def _prop_probabilities(
    market: PolymarketMarket,
    target: str,
    snapshots: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, tuple[float, float, float, bool]] | None, str | None]:
    if market.total_line is None:
        return None, "prop_line_missing"
    rows = snapshots.get(target, [])
    if len(rows) != 1:
        return None, "prop_forecast_unavailable"
    row = rows[0]
    try:
        signal = _prop_calibrator(target).price(
            mean=float(row["model_value"]),
            line=float(market.total_line),
            metadata={
                "league": row.get("league"),
                "bo_format": row.get("match_type"),
            },
        )
    except Exception:
        return None, "prop_calibrator_unavailable"
    probabilities = {
        "over": float(signal.over_probability),
        "under": float(signal.under_probability),
    }
    if {outcome.name.casefold() for outcome in market.outcomes} != set(probabilities):
        return None, "prop_outcome_orientation_unsupported"
    return (
        {
            outcome.name.casefold(): (
                probabilities[outcome.name.casefold()],
                probabilities[outcome.name.casefold()],
                0.5,
                True,
            )
            for outcome in market.outcomes
        },
        None,
    )


@lru_cache(maxsize=len(_PROP_CALIBRATORS))
def _prop_calibrator(target: str) -> Any:
    return load_model(
        resolve_serving_artifact(
            _PROP_CALIBRATORS[target],
            registry_root=MODEL_REGISTRY_DIR,
            legacy_root=MODELS_DIR,
        )
    )


def _team_snapshot_values(
    rows: list[dict[str, Any]], *, team_a: str, team_b: str
) -> dict[str, tuple[float, float, float, bool]] | None:
    output: dict[str, tuple[float, float, float, bool]] = {}
    for row in rows:
        side = _team_side(
            str(row.get("selection") or ""),
            team_a=team_a,
            team_b=team_b,
        )
        if side is None:
            continue
        point = float(row["model_value"])
        lower = float(row.get("probability_lower", point))
        baseline = float(row.get("rating_baseline_probability") or 0.5)
        output[side] = (point, lower, baseline, False)
    if set(output) != {"a", "b"}:
        return None
    favorite = "a" if output["a"][0] >= output["b"][0] else "b"
    return {
        side: (point, lower, baseline, side == favorite)
        for side, (point, lower, baseline, _) in output.items()
    }


def _winner_outcome_probabilities(
    market: PolymarketMarket,
    values: dict[str, tuple[float, float, float, bool]],
    *,
    team_a: str,
    team_b: str,
) -> dict[str, tuple[float, float, float, bool]] | None:
    output: dict[str, tuple[float, float, float, bool]] = {}
    for outcome in market.outcomes:
        side = _team_side(outcome.name, team_a=team_a, team_b=team_b)
        if side is None:
            return None
        output[outcome.name.casefold()] = values[side]
    return output if len(output) == len(market.outcomes) else None


def _total_probabilities(
    market: PolymarketMarket,
    map_values: dict[str, tuple[float, float, float, bool]],
    best_of: int,
) -> dict[str, tuple[float, float, float, bool]] | None:
    if market.total_line is None or float(market.total_line).is_integer():
        return None
    point = map_values["a"][0]
    lower = map_values["a"][1]
    upper = 1.0 - map_values["b"][1]
    distributions = [
        enumerate_series_paths(value, best_of=best_of)
        for value in (lower, point, upper)
    ]
    line = float(market.total_line)
    output: dict[str, tuple[float, float, float, bool]] = {}
    for outcome in market.outcomes:
        side = outcome.name.casefold()
        if side not in {"over", "under"}:
            return None
        probabilities = [
            sum(
                value
                for total, value in distribution.total_maps.items()
                if (total > line if side == "over" else total < line)
            )
            for distribution in distributions
        ]
        output[side] = (probabilities[1], min(probabilities), 0.5, True)
    return output


def _handicap_probabilities(
    market: PolymarketMarket,
    map_values: dict[str, tuple[float, float, float, bool]],
    *,
    team_a: str,
    team_b: str,
    best_of: int,
) -> dict[str, tuple[float, float, float, bool]] | None:
    lines = _handicap_lines(market, team_a=team_a, team_b=team_b)
    if lines is None or any(float(line).is_integer() for line in lines.values()):
        return None
    p_values = (
        map_values["a"][1],
        map_values["a"][0],
        1.0 - map_values["b"][1],
    )
    distributions = [
        enumerate_series_paths(value, best_of=best_of) for value in p_values
    ]
    output: dict[str, tuple[float, float, float, bool]] = {}
    for outcome in market.outcomes:
        side = _team_side(outcome.name, team_a=team_a, team_b=team_b)
        if side is None:
            return None
        handicap = lines[side]
        probabilities = [
            sum(
                value
                for difference, value in distribution.map_differential.items()
                if ((difference if side == "a" else -difference) + handicap) > 0
            )
            for distribution in distributions
        ]
        output[outcome.name.casefold()] = (
            probabilities[1],
            min(probabilities),
            0.5,
            True,
        )
    return output


def _handicap_lines(
    market: PolymarketMarket, *, team_a: str, team_b: str
) -> dict[str, float] | None:
    text = f"{market.group_item_title} {market.question}"
    output: dict[str, float] = {}
    for match in _LINE_PATTERN.finditer(text):
        side = _team_side(match.group("label"), team_a=team_a, team_b=team_b)
        if side is not None:
            output[side] = float(match.group("line"))
    return output if set(output) == {"a", "b"} else None


def _contract_blocks(
    market: PolymarketMarket,
    *,
    start: datetime,
    best_of: int,
    reviewed_at: datetime,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not market.active or market.closed or not market.accepting_orders:
        reasons.append("market_not_open")
    if start <= reviewed_at:
        reasons.append("fixture_started")
    if market.event_start_time is None:
        reasons.append("start_time_missing")
    elif abs(market.event_start_time - start) > _START_TOLERANCE:
        reasons.append("start_time_mismatch")
    if market.best_of is not None and market.best_of != best_of:
        reasons.append("best_of_mismatch")
    resolution = market.resolution_source.casefold()
    if not resolution:
        reasons.append("resolution_rules_missing")
    elif not any(term in resolution for term in LOL_RESOLUTION_RULE_TERMS):
        reasons.append("resolution_rules_ambiguous")
    return tuple(reasons)


def _group_markets(
    schedule: pd.DataFrame,
    markets: Sequence[PolymarketMarket] | Mapping[str, Sequence[PolymarketMarket]],
) -> dict[str, tuple[PolymarketMarket, ...]]:
    if isinstance(markets, Mapping):
        return {
            str(key): tuple(cast("Sequence[PolymarketMarket]", value))
            for key, value in markets.items()
        }
    fixture_keys = [str(row.get("match_key") or "") for _, row in schedule.iterrows()]
    return {key: tuple(markets) for key in fixture_keys}


def _snapshot_index(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    output: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        fixture = str(row.get("source_match_key") or "")
        target = str(row.get("market") or "")
        output.setdefault(fixture, {}).setdefault(target, []).append(row)
    return output


def _team_side(value: str, *, team_a: str, team_b: str) -> str | None:
    normalized = _name_tokens(value)
    matches_a = any(
        _name_tokens(alias) <= normalized for alias in team_name_variants(team_a)
    )
    matches_b = any(
        _name_tokens(alias) <= normalized for alias in team_name_variants(team_b)
    )
    if matches_a == matches_b:
        canonical = canonical_team_name(value).casefold()
        if canonical == canonical_team_name(team_a).casefold():
            return "a"
        if canonical == canonical_team_name(team_b).casefold():
            return "b"
        return None
    return "a" if matches_a else "b"


def _name_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def _roster_ready(snapshots: dict[str, list[dict[str, Any]]]) -> bool:
    rows = snapshots.get("series_winner") or snapshots.get("map_winner") or []
    return bool(rows) and all(row.get("roster_ready") is True for row in rows)


def _roster_confidence(snapshots: dict[str, list[dict[str, Any]]]) -> str:
    rows = snapshots.get("series_winner") or snapshots.get("map_winner") or []
    values = {
        str(row.get(field) or "unknown")
        for row in rows
        for field in ("team_a_roster_confidence", "team_b_roster_confidence")
    }
    if not values:
        return "high" if _roster_ready(snapshots) else "unknown"
    rank = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
    return min(values, key=lambda value: rank.get(value, 0))


def _attribution_stable(snapshots: dict[str, list[dict[str, Any]]]) -> bool:
    rows = snapshots.get("series_winner", [])
    return bool(rows) and all(row.get("attribution_stable") is True for row in rows)


def _series_disagreement(
    snapshots: dict[str, list[dict[str, Any]]],
    *,
    team_a: str,
    team_b: str,
    best_of: int,
) -> float | None:
    direct = _team_snapshot_values(
        snapshots.get("series_winner", []), team_a=team_a, team_b=team_b
    )
    maps = _team_snapshot_values(
        snapshots.get("map_winner", []), team_a=team_a, team_b=team_b
    )
    if direct is None or maps is None:
        return None
    derived = enumerate_series_paths(maps["a"][0], best_of=best_of).team_a_win
    return round(abs(direct["a"][0] - derived) * 100.0, 4)


def _utc_datetime(value: Any) -> datetime:
    parsed = cast("pd.Timestamp", pd.Timestamp(value))
    if pd.isna(parsed):
        raise ValueError("market fixture start time is required")
    result = parsed.to_pydatetime()
    return (
        result.replace(tzinfo=UTC) if result.tzinfo is None else result.astimezone(UTC)
    )


def _action_payload(
    item: PendingMarketAction,
    *,
    state: str,
    reason: str | None,
    hard_blocks: tuple[str, ...],
    warnings: tuple[str, ...],
    observation: Any,
) -> dict[str, Any]:
    quoted_odds = (
        float(observation.fill.decimal_odds)
        if observation is not None and observation.fill.decimal_odds is not None
        else None
    )
    payload = asdict(item) | {
        "state": state,
        "reason": reason,
        "hard_blocks": list(dict.fromkeys(hard_blocks)),
        "warnings": list(dict.fromkeys(warnings)),
        "point_edge": None,
        "conservative_edge": None,
        "stake_units": 0.0,
        "correlation_rank": None,
        "decimal_odds": None,
        "fair_decimal_odds": decimal_odds_from_probability(item.probability)
        if item.probability
        else None,
        "requested_shares": None,
        "hypothetical_cost": None,
        "observations": [],
    }
    if observation is not None:
        payload.update(
            {
                "decimal_odds": observation.fill.decimal_odds,
                "requested_shares": str(observation.fill.requested_shares),
                "hypothetical_cost": str(observation.fill.total_cost),
                "observations": [_observation_payload(observation)],
            }
        )
    decision = decide_market(
        semantic_fingerprint=item.semantic_fingerprint or item.comparison_id,
        target=item.target,
        readiness=item.readiness,
        probability=item.probability,
        conservative_probability=item.probability_lower,
        decimal_odds=quoted_odds,
        is_model_favorite=item.is_model_favorite,
        hard_blocks=hard_blocks,
    )
    decision_payload = decision.to_dict()
    selected_path = decision_payload["sizing"]["selected_path"]
    stake_units = (
        decision_payload["sizing"]["stake_units"][selected_path]
        if selected_path
        else 0.0
    )
    payload.update(decision_payload)
    payload.update(
        {
            "state": decision.classification.value,
            "reason": decision.reason_codes[0] if decision.reason_codes else None,
            "point_edge": decision.point_ev,
            "conservative_edge": decision.conservative_ev,
            "stake_units": stake_units,
        }
    )
    return payload


def _observation_payload(observation: Any) -> dict[str, Any]:
    return {
        "sequence_number": observation.sequence_number,
        "observed_at": observation.observed_at.isoformat(),
        "provider_timestamp": (
            observation.book.timestamp.isoformat()
            if observation.book.timestamp
            else None
        ),
        "book_hash": observation.book.book_hash,
        "token_id": observation.book.token_id,
        "complete": observation.fill.complete,
        "minimum_order_size": str(observation.book.minimum_order_size),
        "requested_shares": str(observation.fill.requested_shares),
        "filled_shares": str(observation.fill.filled_shares),
        "unfilled_shares": str(observation.fill.unfilled_shares),
        "hypothetical_cost": str(observation.fill.total_cost),
        "average_price": str(observation.fill.average_price),
        "decimal_odds": observation.fill.decimal_odds,
        "depth": {
            "bids": [asdict(level) for level in observation.book.bids],
            "asks": [asdict(level) for level in observation.book.asks],
        },
    }


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode()).hexdigest()[:24]}"
