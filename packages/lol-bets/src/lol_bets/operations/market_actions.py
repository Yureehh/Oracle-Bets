"""Typed, read-only Polymarket evaluation for daily LoL predictions."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from oracle_bets_core.markets import (
    MarketFixture,
    PolymarketMarket,
    SupportedMarketType,
    capture_minimum_order_book_batch,
    confirmed_executable_fill,
    select_best_market,
)
from oracle_bets_core.operations.paper import (
    ActionGateDecision,
    ActionGateInput,
    ActionState,
    apply_action_gate,
    select_fixture_actions,
)
from oracle_bets_core.pd import pd

from lol_bets.inference.team_resolver import team_name_variants

_EXPECTED_WINNER_ROWS = 2
LOL_RESOLUTION_RULE_TERMS = ("liquipedia leagueoflegends",)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from oracle_bets_core.markets import OrderBookClient


@dataclass(frozen=True)
class PendingMarketAction:
    proposal_id: str
    fixture_id: str
    fixture_key: str
    league: str
    team_a: str
    team_b: str
    target: str
    game_number: int | None
    total_line: float | None
    selection: str
    probability: float
    probability_lower: float
    rating_baseline_probability: float
    full_model_probability: float
    is_model_favorite: bool
    attribution_stable: bool
    start_time: datetime
    market_id: str
    token_id: str
    market_url: str
    resolution_source: str
    roster_ready: bool
    uncertainty_available: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DailyMarketEvaluation:
    """Contract matches and outcome-level executable quotes kept separate."""

    reviews: tuple[dict[str, Any], ...]
    actions: tuple[dict[str, Any], ...]


def evaluate_daily_market_actions(  # noqa: PLR0912, PLR0915
    *,
    schedule: pd.DataFrame,
    snapshot_rows: Sequence[dict[str, Any]],
    markets: (Sequence[PolymarketMarket] | Mapping[str, Sequence[PolymarketMarket]]),
    clob_client: OrderBookClient,
    model_healthy: bool,
    run_key: str | None = None,
    interval_seconds: int = 45,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> DailyMarketEvaluation:
    """Match supported contracts, batch executable quotes, and apply paper gates."""
    pending: list[PendingMarketAction] = []
    review_rows: list[dict[str, Any]] = []
    flat_markets: tuple[PolymarketMarket, ...] = ()
    if isinstance(markets, Mapping):
        grouped_markets = cast(
            "Mapping[str, Sequence[PolymarketMarket]]",
            markets,
        )
        all_markets = [
            market for values in grouped_markets.values() for market in values
        ]
    else:
        grouped_markets = None
        flat_markets = tuple(markets)
        all_markets = list(flat_markets)
    market_by_id = {market.market_id: market for market in all_markets}
    winner_rows_by_fixture: dict[str, list[dict[str, Any]]] = {}
    for row in snapshot_rows:
        if row.get("market") != "series_winner":
            continue
        source_key = str(row.get("source_match_key") or "").strip()
        winner_rows_by_fixture.setdefault(source_key, []).append(row)
    for _, fixture_row in schedule.iterrows():
        fixture_key = str(fixture_row.get("match_key") or "").strip()
        if grouped_markets is not None:
            fixture_markets = tuple(grouped_markets.get(fixture_key, ()))
        else:
            fixture_markets = flat_markets
        winner_rows = winner_rows_by_fixture.get(fixture_key, [])
        if len(winner_rows) != _EXPECTED_WINNER_ROWS:
            continue
        requests = _market_requests(fixture_row, winner_rows, fixture_markets)
        favorite_probability = max(float(row["model_value"]) for row in winner_rows)
        for fixture, probabilities in requests:
            selection = select_best_market(fixture, list(fixture_markets))
            review = {
                "fixture_id": fixture.fixture_id,
                "fixture_key": fixture_key,
                "target": fixture.market_type.value,
                "game_number": fixture.game_number,
                "total_line": float(fixture.total_line) if fixture.total_line else None,
                "selected_market_id": selection.selected_market_id,
                "assessments": [asdict(item) for item in selection.assessments],
            }
            if selection.selected_market_id is None:
                review_rows.append(
                    review | {"state": "blocked", "reason": "market_not_found"}
                )
                continue
            market = market_by_id[selection.selected_market_id]
            review["resolution_source"] = market.resolution_source
            assessment = next(
                item
                for item in selection.assessments
                if item.market_id == market.market_id
            )
            for selection_id, token_id in assessment.selection_tokens:
                probability = probabilities.get(selection_id)
                if probability is None:
                    continue
                point, lower = probability
                proposal_id = _stable_id(
                    "proposal",
                    f"{run_key or winner_rows[0].get('run_ts')}|{fixture.fixture_id}|"
                    f"{market.market_id}|{selection_id}",
                )
                pending.append(
                    PendingMarketAction(
                        proposal_id=proposal_id,
                        fixture_id=fixture.fixture_id,
                        fixture_key=fixture_key,
                        league=str(fixture_row.get("league") or ""),
                        team_a=str(fixture_row.get("team_a") or ""),
                        team_b=str(fixture_row.get("team_b") or ""),
                        target=fixture.market_type.value,
                        game_number=fixture.game_number,
                        total_line=(
                            float(fixture.total_line) if fixture.total_line else None
                        ),
                        selection=selection_id,
                        probability=point,
                        probability_lower=lower,
                        rating_baseline_probability=float(
                            next(
                                row.get("rating_baseline_probability", 0.5)
                                for row in winner_rows
                                if str(row["selection"]) == selection_id
                            )
                            or 0.5
                        ),
                        full_model_probability=float(
                            next(
                                row.get("full_model_probability", point)
                                for row in winner_rows
                                if str(row["selection"]) == selection_id
                            )
                            or point
                        ),
                        is_model_favorite=point == favorite_probability,
                        attribution_stable=all(
                            row.get("attribution_stable") is True for row in winner_rows
                        ),
                        start_time=fixture.start_time,
                        market_id=market.market_id,
                        token_id=token_id,
                        market_url=market.url,
                        resolution_source=market.resolution_source,
                        roster_ready=all(
                            row.get("roster_ready") is True for row in winner_rows
                        ),
                        uncertainty_available=all(
                            row.get("uncertainty_method") for row in winner_rows
                        ),
                        warnings=assessment.warnings,
                    )
                )
            review_rows.append(review | {"state": "matched", "reason": None})

    if not pending:
        return DailyMarketEvaluation(tuple(review_rows), ())
    batch = capture_minimum_order_book_batch(
        clob_client,
        token_ids=tuple(item.token_id for item in pending),
        interval_seconds=interval_seconds,
        sleeper=sleeper,
        clock=clock,
    )
    decisions: list[ActionGateDecision] = []
    action_rows: list[dict[str, Any]] = []
    for item in pending:
        pair = batch.observations.get(item.token_id)
        failure = batch.failures.get(item.token_id)
        if failure is not None or pair is None:
            action_rows.append(
                asdict(item)
                | {
                    "state": ActionState.BLOCKED.value,
                    "reason": failure.reason if failure else "book_unavailable",
                    "detail": failure.detail if failure else "book capture unavailable",
                    "conservative_edge": None,
                    "stake_units": 0.0,
                    "decimal_odds": None,
                    "requested_shares": (
                        str(pair[0].fill.requested_shares) if pair else None
                    ),
                    "hypothetical_cost": None,
                    "observations": (
                        [_observation_payload(observation) for observation in pair]
                        if pair
                        else []
                    ),
                }
            )
            continue
        fill = confirmed_executable_fill(pair)
        odds = float(fill.decimal_odds or 0.0)
        decision = apply_action_gate(
            ActionGateInput(
                proposal_id=item.proposal_id,
                fixture_id=item.fixture_id,
                target=item.target,
                league=item.league,
                probability=item.probability,
                probability_lower=item.probability_lower,
                decimal_odds=odds,
                model_healthy=model_healthy,
                roster_ready=item.roster_ready,
                uncertainty_available=item.uncertainty_available,
                market_supported=True,
                quote_valid=fill.complete,
                is_model_favorite=item.is_model_favorite,
                hours_to_start=(item.start_time - clock()).total_seconds() / 3600,
                market_probability=(1.0 / odds if odds > 1.0 else None),
                rating_baseline_probability=item.rating_baseline_probability,
                attribution_stable=item.attribution_stable,
            )
        )
        decisions.append(decision)
        action_rows.append(
            asdict(item)
            | {
                "state": decision.state.value,
                "reason": decision.reason,
                "conservative_edge": decision.conservative_edge,
                "stake_units": decision.stake_units,
                "counterfactual_quarter_kelly_units": (
                    decision.counterfactual_quarter_kelly_units
                ),
                "decimal_odds": odds,
                "requested_shares": str(fill.requested_shares),
                "hypothetical_cost": str(fill.total_cost),
                "observations": [
                    _observation_payload(observation) for observation in pair
                ],
            }
        )
    selected_ids = {
        decision.proposal_id for decision in select_fixture_actions(decisions)
    }
    for index, row in enumerate(action_rows):
        if (
            row["state"] == ActionState.PAPER_ACTIONABLE
            and row["proposal_id"] not in selected_ids
        ):
            action_rows[index] = row | {
                "state": ActionState.BLOCKED.value,
                "reason": "correlated_fixture_exposure",
                "stake_units": 0.0,
            }
    return DailyMarketEvaluation(tuple(review_rows), tuple(action_rows))


def _market_requests(
    fixture_row: pd.Series,
    winner_rows: list[dict[str, Any]],
    markets: Sequence[PolymarketMarket],
) -> list[tuple[MarketFixture, dict[str, tuple[float, float]]]]:
    del markets
    team_a = str(fixture_row.get("team_a") or "").strip()
    team_b = str(fixture_row.get("team_b") or "").strip()
    fixture_key = str(fixture_row.get("match_key") or f"{team_a}:{team_b}")
    best_of = int(fixture_row.get("best_of") or 1)
    start = pd.Timestamp(fixture_row.get("start_utc")).to_pydatetime()
    if pd.isna(start):
        raise ValueError("market fixture start time is required")
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    team_probabilities = {
        str(row["selection"]): (
            float(row["model_value"]),
            float(row.get("probability_lower", row["model_value"])),
        )
        for row in winner_rows
    }
    common: dict[str, Any] = {
        "fixture_id": fixture_key,
        "competition_names": (str(fixture_row.get("league") or ""),),
        "team_a_id": team_a,
        "team_b_id": team_b,
        "team_a_names": team_name_variants(team_a),
        "team_b_names": team_name_variants(team_b),
        "start_time": start,
        "best_of": best_of,
        "resolution_rule_terms": LOL_RESOLUTION_RULE_TERMS,
    }
    return [
        (
            MarketFixture(**common, market_type=SupportedMarketType.SERIES_WINNER),
            team_probabilities,
        )
    ]


def _observation_payload(observation: Any) -> dict[str, Any]:
    return {
        "sequence_number": observation.sequence_number,
        "observed_at": observation.observed_at.isoformat(),
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
        "book": {
            "bids": [asdict(level) for level in observation.book.bids],
            "asks": [asdict(level) for level in observation.book.asks],
        },
    }


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode()).hexdigest()[:24]}"
