"""Read-only market discovery, exact matching, and executable book pricing."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

import requests

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_DEFAULT_MATCH_TOLERANCE = timedelta(hours=6)
_MIN_OBSERVATION_SECONDS = 30
_MAX_OBSERVATION_SECONDS = 60
_MAX_SEARCH_PAGES = 20
_EXPECTED_BINARY_OUTCOMES = 2


class MarketDataError(ValueError):
    """Raised when provider data cannot be interpreted without guessing."""


class MarketReadError(RuntimeError):
    """Raised when a sanitized read-only provider request fails."""


class SupportedMarketType(StrEnum):
    """Prematch contract meanings the LoL model can price without guessing."""

    MAP_WINNER = "map_winner"
    SERIES_WINNER = "series_winner"
    SERIES_TOTAL_MAPS = "series_total_maps"


@dataclass(frozen=True)
class MarketQuote:
    """Backward-compatible normalized display quote used by daily reports."""

    source: str
    market_id: str
    question: str
    outcome: str
    price: float | None = None
    implied_probability: float | None = None
    liquidity: float | None = None
    volume: float | None = None
    url: str | None = None
    token_id: str | None = None
    event_id: str | None = None


class MarketAdapter(Protocol):
    """Read-only display-quote source adapter."""

    source: str

    def search(self, query: str, *, limit: int = 25) -> list[MarketQuote]:
        """Return normalized quotes matching ``query``."""


@dataclass(frozen=True)
class MarketOutcome:
    """One explicit outcome and its orientation-specific CLOB token."""

    name: str
    token_id: str
    displayed_price: Decimal | None


@dataclass(frozen=True)
class PolymarketMarket:
    """Strict current Gamma market facts needed before pricing."""

    event_id: str
    market_id: str
    event_title: str
    question: str
    slug: str
    outcomes: tuple[MarketOutcome, ...]
    event_start_time: datetime | None
    best_of: int | None
    sports_market_type: str
    group_item_title: str
    game_number: int | None
    total_line: Decimal | None
    active: bool
    closed: bool
    accepting_orders: bool
    resolution_source: str
    liquidity: Decimal | None
    volume: Decimal | None

    @property
    def url(self) -> str:
        return f"https://polymarket.com/event/{self.slug}"


class PolymarketGammaAdapter:
    """Read-only adapter for the current Gamma ``/public-search`` endpoint."""

    source = "polymarket"

    def __init__(
        self,
        *,
        base_url: str = "https://gamma-api.polymarket.com",
        session: requests.Session | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    def search(self, query: str, *, limit: int = 25) -> list[MarketQuote]:
        """Return legacy display quotes without treating them as fill prices."""
        return [
            quote
            for market in self.search_markets(query, limit=limit)
            for quote in self._quotes_from_normalized_market(market)
        ]

    def search_markets(
        self,
        query: str,
        *,
        limit: int = 25,
    ) -> list[PolymarketMarket]:
        """Page public search and retain strict, orientation-safe markets."""
        if not query.strip():
            raise ValueError("market search query cannot be empty")
        if limit <= 0:
            raise ValueError("market search limit must be positive")

        markets: list[PolymarketMarket] = []
        seen_ids: set[str] = set()
        page = 1
        while len(markets) < limit and page <= _MAX_SEARCH_PAGES:
            events = self._search_page(query=query, limit=limit, page=page)
            if not events:
                break

            for event in events:
                raw_markets = event.get("markets", [])
                for raw_market in raw_markets:
                    try:
                        market = self._market_from_payload(raw_market, event=event)
                    except MarketDataError:
                        continue
                    if market.market_id in seen_ids:
                        continue
                    markets.append(market)
                    seen_ids.add(market.market_id)
                    if len(markets) >= limit:
                        break
                if len(markets) >= limit:
                    break
            page += 1
        return markets

    def _search_page(
        self,
        *,
        query: str,
        limit: int,
        page: int,
    ) -> list[dict[str, Any]]:
        response = self.session.get(
            f"{self.base_url}/public-search",
            params={
                "q": query,
                "events_status": "active",
                "keep_closed_markets": 0,
                "limit_per_type": min(limit, 50),
                "page": page,
                "search_profiles": False,
                "search_tags": False,
            },
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            raise MarketReadError(
                "Polymarket Gamma public-search request failed."
            ) from error
        if not isinstance(payload, dict):
            raise MarketDataError("Gamma public-search response must be an object")
        events = payload.get("events", [])
        if not isinstance(events, list) or not all(
            isinstance(event, dict) for event in events
        ):
            raise MarketDataError("Gamma public-search events must be object records")
        for event in events:
            if not isinstance(event.get("markets", []), list) or not all(
                isinstance(market, dict) for market in event.get("markets", [])
            ):
                raise MarketDataError("Gamma event markets must be object records")
        return events

    def _market_from_payload(
        self,
        market: dict[str, Any],
        *,
        event: dict[str, Any],
    ) -> PolymarketMarket:
        """Parse a Gamma market and preserve outcome-to-token orientation."""
        market_id = _required_text(
            market.get("id") or market.get("conditionId"),
            "Gamma market id",
        )
        event_id = _required_text(
            event.get("id") or market.get("eventId"),
            "Gamma event id",
        )
        outcomes = _coerce_list(market.get("outcomes"))
        token_ids = _coerce_list(market.get("clobTokenIds"))
        prices = _coerce_list(market.get("outcomePrices"))
        if len(outcomes) < _EXPECTED_BINARY_OUTCOMES or len(outcomes) != len(token_ids):
            raise MarketDataError(
                "Gamma outcomes and CLOB token IDs must have equal lengths"
            )

        normalized_outcomes = tuple(
            MarketOutcome(
                name=_required_text(outcome, "Gamma outcome"),
                token_id=_required_text(token_ids[index], "Gamma CLOB token id"),
                displayed_price=(
                    _optional_decimal(prices[index], field="displayed price")
                    if index < len(prices)
                    else None
                ),
            )
            for index, outcome in enumerate(outcomes)
        )
        title = str(event.get("title") or "")
        question = _required_text(market.get("question"), "Gamma market question")
        start_value = (
            market.get("eventStartTime")
            or market.get("gameStartTime")
            or event.get("startDate")
        )
        best_of = _parse_best_of(
            market.get("bestOf"),
            f"{title} {question} {market.get('groupItemTitle') or ''}",
        )
        group_item_title = str(market.get("groupItemTitle") or "").strip()
        return PolymarketMarket(
            event_id=event_id,
            market_id=market_id,
            event_title=title,
            question=question,
            slug=_required_text(
                market.get("slug") or event.get("slug") or market_id,
                "Gamma market slug",
            ),
            outcomes=normalized_outcomes,
            event_start_time=_optional_utc(start_value),
            best_of=best_of,
            sports_market_type=str(market.get("sportsMarketType") or "")
            .strip()
            .casefold(),
            group_item_title=group_item_title,
            game_number=_parse_game_number(f"{group_item_title} {question}"),
            total_line=_parse_total_line(
                market.get("line"),
                f"{group_item_title} {question}",
            ),
            active=market.get("active") is True,
            closed=market.get("closed") is True,
            accepting_orders=market.get("acceptingOrders") is True,
            resolution_source=str(
                market.get("resolutionSource") or event.get("resolutionSource") or ""
            ).strip(),
            liquidity=_optional_decimal(
                market.get("liquidity") or market.get("liquidityNum"),
                field="liquidity",
            ),
            volume=_optional_decimal(
                market.get("volume") or market.get("volumeNum"),
                field="volume",
            ),
        )

    def _quotes_from_normalized_market(
        self,
        market: PolymarketMarket,
    ) -> list[MarketQuote]:
        return [
            MarketQuote(
                source=self.source,
                market_id=market.market_id,
                question=market.question,
                outcome=outcome.name,
                price=(
                    float(outcome.displayed_price)
                    if outcome.displayed_price is not None
                    else None
                ),
                implied_probability=(
                    float(outcome.displayed_price)
                    if outcome.displayed_price is not None
                    else None
                ),
                liquidity=(
                    float(market.liquidity) if market.liquidity is not None else None
                ),
                volume=float(market.volume) if market.volume is not None else None,
                url=market.url,
                token_id=outcome.token_id,
                event_id=market.event_id,
            )
            for outcome in market.outcomes
        ]

    def _quotes_from_market(self, market: dict[str, Any]) -> list[MarketQuote]:
        """Normalize a legacy raw market without asserting token orientation."""
        outcomes = _coerce_list(market.get("outcomes"))
        prices = _coerce_list(market.get("outcomePrices"))
        tokens = _coerce_list(market.get("clobTokenIds"))
        quotes: list[MarketQuote] = []
        for index, outcome in enumerate(outcomes):
            price = _coerce_float(prices[index] if index < len(prices) else None)
            quotes.append(
                MarketQuote(
                    source=self.source,
                    market_id=str(market.get("id") or market.get("conditionId") or ""),
                    question=str(market.get("question") or ""),
                    outcome=str(outcome),
                    price=price,
                    implied_probability=price,
                    liquidity=_coerce_float(market.get("liquidity")),
                    volume=_coerce_float(market.get("volume")),
                    url=(
                        f"https://polymarket.com/event/{market['slug']}"
                        if market.get("slug")
                        else None
                    ),
                    token_id=(str(tokens[index]) if index < len(tokens) else None),
                )
            )
        return quotes


@dataclass(frozen=True)
class MarketFixture:
    """Canonical fixture facts and exact provider-facing team aliases."""

    fixture_id: str
    competition_names: tuple[str, ...]
    team_a_id: str
    team_b_id: str
    team_a_names: tuple[str, ...]
    team_b_names: tuple[str, ...]
    start_time: datetime
    best_of: int
    market_type: SupportedMarketType = SupportedMarketType.SERIES_WINNER
    game_number: int | None = None
    total_line: Decimal | None = None
    resolution_rule_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value in (self.fixture_id, self.team_a_id, self.team_b_id):
            _required_text(value, "fixture identifier")
        if self.team_a_id == self.team_b_id:
            raise ValueError("market fixture teams must be distinct")
        if not self.competition_names:
            raise ValueError("market fixture requires competition aliases")
        if not self.team_a_names or not self.team_b_names:
            raise ValueError("market fixture requires aliases for both teams")
        if self.start_time.tzinfo is None or self.start_time.utcoffset() != timedelta(
            0
        ):
            raise ValueError("market fixture start_time must be UTC")
        if self.best_of not in {1, 2, 3, 5}:
            raise ValueError("market fixture best_of must be 1, 2, 3, or 5")
        object.__setattr__(self, "market_type", SupportedMarketType(self.market_type))
        if self.market_type is SupportedMarketType.MAP_WINNER:
            if self.game_number is None or not 1 <= self.game_number <= self.best_of:
                raise ValueError("map winner requests require a playable game_number")
        elif self.game_number is not None:
            raise ValueError("game_number is valid only for map winner requests")
        if self.market_type is SupportedMarketType.SERIES_TOTAL_MAPS:
            line = _required_decimal(self.total_line, field="total maps line")
            if line <= 0:
                raise ValueError("total maps line must be positive")
            object.__setattr__(self, "total_line", line)
        elif self.total_line is not None:
            raise ValueError("total_line is valid only for total maps requests")


@dataclass(frozen=True)
class MarketMatchAssessment:
    """Accepted or rejected exact market candidate with all visible reasons."""

    fixture_id: str
    market_id: str
    matched: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    selection_tokens: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class MarketSelection:
    """Best exact usable market plus the audit result for every candidate."""

    selected_market_id: str | None
    assessments: tuple[MarketMatchAssessment, ...]


def match_market(
    fixture: MarketFixture,
    market: PolymarketMarket,
    *,
    start_tolerance: timedelta = _DEFAULT_MATCH_TOLERANCE,
) -> MarketMatchAssessment:
    """Match teams, start, format, status, and selection orientation."""
    if start_tolerance < timedelta(0):
        raise ValueError("start_tolerance cannot be negative")
    reasons: list[str] = []
    warnings: list[str] = []

    combined_text = f"{market.event_title} {market.question}"
    if not _aliases_match_text(fixture.competition_names, combined_text):
        reasons.append("competition_mismatch")
    if not _aliases_match_text(fixture.team_a_names, combined_text):
        reasons.append("team_a_mismatch")
    if not _aliases_match_text(fixture.team_b_names, combined_text):
        reasons.append("team_b_mismatch")
    if market.event_start_time is None:
        reasons.append("start_time_missing")
    elif abs(market.event_start_time - fixture.start_time) > start_tolerance:
        reasons.append("start_time_mismatch")
    if market.best_of is None:
        reasons.append("best_of_missing")
    elif market.best_of != fixture.best_of:
        reasons.append("best_of_mismatch")
    reasons.extend(_market_contract_reasons(fixture, market))

    selection_tokens = _selection_orientation(fixture, market)
    if len(selection_tokens) != _EXPECTED_BINARY_OUTCOMES:
        reasons.append("selection_orientation_unresolved")
    if not market.resolution_source:
        reasons.append("resolution_rules_missing")
    elif not _verified_resolution_source(market.resolution_source):
        reasons.append("resolution_rules_malformed")

    return MarketMatchAssessment(
        fixture_id=fixture.fixture_id,
        market_id=market.market_id,
        matched=not reasons,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        selection_tokens=selection_tokens,
    )


def _market_contract_reasons(
    fixture: MarketFixture,
    market: PolymarketMarket,
) -> list[str]:
    reasons: list[str] = []
    if not market.active or market.closed or not market.accepting_orders:
        reasons.append("market_not_open")
    expected_sports_type = {
        SupportedMarketType.MAP_WINNER: "child_moneyline",
        SupportedMarketType.SERIES_WINNER: "moneyline",
        SupportedMarketType.SERIES_TOTAL_MAPS: "totals",
    }[fixture.market_type]
    if market.sports_market_type != expected_sports_type:
        reasons.append("sports_market_type_mismatch")
    if fixture.market_type is SupportedMarketType.MAP_WINNER:
        if market.game_number != fixture.game_number:
            reasons.append("game_number_mismatch")
    elif fixture.market_type is SupportedMarketType.SERIES_WINNER:
        if market.game_number is not None:
            reasons.append("game_number_mismatch")
        if "total" in _text_tokens(market.question):
            reasons.append("market_meaning_mismatch")
    elif market.total_line != fixture.total_line:
        reasons.append("totals_line_mismatch")
    if (
        market.resolution_source
        and fixture.resolution_rule_terms
        and not _aliases_match_text(
            fixture.resolution_rule_terms,
            market.resolution_source,
        )
    ):
        reasons.append("resolution_rules_mismatch")
    return reasons


def _verified_resolution_source(value: str) -> bool:
    """Accept an explicit provider rule description or a public HTTP(S) source."""
    normalized = value.strip()
    if len(normalized) < 8:  # noqa: PLR2004
        return False
    if "://" not in normalized:
        return True
    return normalized.casefold().startswith(("https://", "http://"))


def select_best_market(
    fixture: MarketFixture,
    markets: tuple[PolymarketMarket, ...] | list[PolymarketMarket],
    *,
    start_tolerance: timedelta = _DEFAULT_MATCH_TOLERANCE,
) -> MarketSelection:
    """Assess every candidate and choose the most liquid exact usable contract."""
    assessed = tuple(
        match_market(fixture, market, start_tolerance=start_tolerance)
        for market in markets
    )
    market_by_id = {market.market_id: market for market in markets}
    usable = [assessment for assessment in assessed if assessment.matched]
    usable.sort(
        key=lambda assessment: (
            market_by_id[assessment.market_id].liquidity or Decimal(-1),
            market_by_id[assessment.market_id].volume or Decimal(-1),
            assessment.market_id,
        ),
        reverse=True,
    )
    return MarketSelection(
        selected_market_id=usable[0].market_id if usable else None,
        assessments=assessed,
    )


def _selection_orientation(
    fixture: MarketFixture,
    market: PolymarketMarket,
) -> tuple[tuple[str, str], ...]:
    if fixture.market_type is SupportedMarketType.SERIES_TOTAL_MAPS:
        totals = {
            outcome.name.casefold(): outcome.token_id for outcome in market.outcomes
        }
        return tuple(sorted(totals.items())) if set(totals) == {"over", "under"} else ()
    selected: dict[str, str] = {}
    for outcome in market.outcomes:
        matches_a = _aliases_match_text(fixture.team_a_names, outcome.name)
        matches_b = _aliases_match_text(fixture.team_b_names, outcome.name)
        if matches_a == matches_b:
            continue
        selected[fixture.team_a_id if matches_a else fixture.team_b_id] = (
            outcome.token_id
        )
    if set(selected) == {fixture.team_a_id, fixture.team_b_id}:
        return tuple(sorted(selected.items()))

    binary = {outcome.name.casefold(): outcome.token_id for outcome in market.outcomes}
    if set(binary) != {"yes", "no"}:
        return ()
    normalized_question = " ".join(_text_tokens(market.question))
    for team_id, aliases, opponent_id in (
        (fixture.team_a_id, fixture.team_a_names, fixture.team_b_id),
        (fixture.team_b_id, fixture.team_b_names, fixture.team_a_id),
    ):
        for alias in aliases:
            normalized_alias = " ".join(_text_tokens(alias))
            if normalized_question.startswith(f"will {normalized_alias} ") and {
                "win",
                "beat",
                "defeat",
            } & set(_text_tokens(market.question)):
                return tuple(
                    sorted(
                        {
                            team_id: binary["yes"],
                            opponent_id: binary["no"],
                        }.items()
                    )
                )
    return ()


@dataclass(frozen=True)
class OrderLevel:
    """One validated price/size level in outcome-token shares."""

    price: Decimal
    size: Decimal

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> OrderLevel:
        if not isinstance(payload, dict):
            raise MarketDataError("order-book level must be an object")
        price = _required_decimal(payload.get("price"), field="order-book price")
        size = _required_decimal(payload.get("size"), field="order-book size")
        if not 0 < price < 1:
            raise MarketDataError("order-book price must be between zero and one")
        if size <= 0:
            raise MarketDataError("order-book size must be positive")
        return cls(price=price, size=size)


@dataclass(frozen=True)
class OrderBook:
    """Validated public CLOB state for one orientation-specific token."""

    condition_id: str
    token_id: str
    timestamp: datetime | None
    book_hash: str
    bids: tuple[OrderLevel, ...]
    asks: tuple[OrderLevel, ...]
    minimum_order_size: Decimal
    tick_size: Decimal
    negative_risk: bool
    last_trade_price: Decimal | None

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        expected_token_id: str,
    ) -> OrderBook:
        if not isinstance(payload, dict):
            raise MarketDataError("CLOB order book must be an object")
        token_id = _required_text(
            payload.get("asset_id") or payload.get("token_id"),
            "CLOB token id",
        )
        if token_id != expected_token_id:
            raise MarketDataError("CLOB order book returned a different token id")
        raw_bids = payload.get("bids")
        raw_asks = payload.get("asks")
        if not isinstance(raw_bids, list) or not isinstance(raw_asks, list):
            raise MarketDataError("CLOB bids and asks must be lists")

        bids = tuple(
            sorted(
                (OrderLevel.from_payload(level) for level in raw_bids),
                key=lambda level: level.price,
                reverse=True,
            )
        )
        asks = tuple(
            sorted(
                (OrderLevel.from_payload(level) for level in raw_asks),
                key=lambda level: level.price,
            )
        )
        if bids and asks and bids[0].price >= asks[0].price:
            raise MarketDataError("CLOB order book is crossed")
        minimum_order_size = _required_decimal(
            payload.get("min_order_size") or payload.get("minOrderSize"),
            field="minimum order size",
        )
        tick_size = _required_decimal(
            payload.get("tick_size") or payload.get("tickSize"),
            field="tick size",
        )
        if minimum_order_size <= 0 or tick_size <= 0:
            raise MarketDataError("CLOB order constraints must be positive")

        return cls(
            condition_id=_required_text(
                payload.get("market") or payload.get("condition_id"),
                "CLOB condition id",
            ),
            token_id=token_id,
            timestamp=_optional_clob_timestamp(payload.get("timestamp")),
            book_hash=_required_text(payload.get("hash"), "CLOB book hash"),
            bids=bids,
            asks=asks,
            minimum_order_size=minimum_order_size,
            tick_size=tick_size,
            negative_risk=payload.get("neg_risk") is True,
            last_trade_price=_optional_decimal(
                payload.get("last_trade_price") or payload.get("lastTradePrice"),
                field="last trade price",
            ),
        )


@dataclass(frozen=True)
class ExpectedFill:
    """Size-aware result of walking the current asks."""

    token_id: str
    requested_shares: Decimal
    filled_shares: Decimal
    unfilled_shares: Decimal
    total_cost: Decimal
    average_price: Decimal | None
    worst_price: Decimal | None
    complete: bool
    decimal_odds: float | None
    requested_risk: Decimal | None = None
    filled_risk: Decimal | None = None
    unfilled_risk: Decimal | None = None


def walk_buy_book(
    book: OrderBook,
    requested_shares: Decimal | str | float,
) -> ExpectedFill:
    """Walk lowest asks first for an outcome-token buy."""
    requested = _required_decimal(requested_shares, field="requested shares")
    if requested <= 0:
        raise ValueError("requested shares must be positive")

    remaining = requested
    filled = Decimal(0)
    cost = Decimal(0)
    worst_price: Decimal | None = None
    for level in book.asks:
        if remaining <= 0:
            break
        shares = min(remaining, level.size)
        filled += shares
        cost += shares * level.price
        remaining -= shares
        worst_price = level.price

    average = cost / filled if filled else None
    return ExpectedFill(
        token_id=book.token_id,
        requested_shares=requested,
        filled_shares=filled,
        unfilled_shares=remaining,
        total_cost=cost,
        average_price=average,
        worst_price=worst_price,
        complete=remaining == 0,
        decimal_odds=float(Decimal(1) / average) if average else None,
    )


def walk_buy_book_by_risk(
    book: OrderBook,
    intended_risk_amount: Decimal | str | float,
) -> ExpectedFill:
    """Walk asks until the intended currency risk is completely executable."""
    intended = _required_decimal(intended_risk_amount, field="intended risk amount")
    if intended <= 0:
        raise ValueError("intended risk amount must be positive")
    remaining_risk = intended
    filled_shares = Decimal(0)
    cost = Decimal(0)
    worst_price: Decimal | None = None
    for level in book.asks:
        if remaining_risk <= 0:
            break
        level_cost = level.size * level.price
        level_spend = min(remaining_risk, level_cost)
        filled_shares += level_spend / level.price
        cost += level_spend
        remaining_risk -= level_spend
        worst_price = level.price
    average = cost / filled_shares if filled_shares else None
    complete = remaining_risk == 0 and filled_shares >= book.minimum_order_size
    return ExpectedFill(
        token_id=book.token_id,
        requested_shares=filled_shares,
        filled_shares=filled_shares,
        unfilled_shares=Decimal(0),
        total_cost=cost,
        average_price=average,
        worst_price=worst_price,
        complete=complete,
        decimal_odds=float(filled_shares / cost) if cost else None,
        requested_risk=intended,
        filled_risk=cost,
        unfilled_risk=remaining_risk,
    )


class OrderBookClient(Protocol):
    """Read-only order-book provider surface used by observation capture."""

    def get_order_book(self, token_id: str) -> OrderBook:
        """Return current public book for one outcome token."""


@dataclass(frozen=True)
class BookObservation:
    """One timestamped book and size-aware fill calculation."""

    sequence_number: int
    observed_at: datetime
    book: OrderBook
    fill: ExpectedFill


@dataclass(frozen=True)
class BookCaptureFailure:
    """One outcome-token quote failure that does not invalidate peer tokens."""

    token_id: str
    reason: str
    detail: str
    sequence_number: int | None = None


@dataclass(frozen=True)
class MinimumOrderBookBatch:
    """Two-observation minimum-share quotes and isolated token failures."""

    observations: dict[str, tuple[BookObservation, BookObservation]]
    failures: dict[str, BookCaptureFailure]


def capture_order_book_pair(
    client: OrderBookClient,
    *,
    token_id: str,
    requested_shares: Decimal | str | float | None = None,
    intended_risk_amount: Decimal | str | float | None = None,
    interval_seconds: int = 45,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    maximum_book_age_seconds: int = 120,
) -> tuple[BookObservation, BookObservation]:
    """Capture the two confirmed 30–60 second-spaced executable books."""
    captured = capture_order_book_batch(
        client,
        token_ids=(token_id,),
        requested_shares=requested_shares,
        intended_risk_amount=intended_risk_amount,
        interval_seconds=interval_seconds,
        sleeper=sleeper,
        clock=clock,
        maximum_book_age_seconds=maximum_book_age_seconds,
    )
    return captured[token_id]


def capture_order_book_batch(
    client: OrderBookClient,
    *,
    token_ids: Sequence[str],
    requested_shares: Decimal | str | float | None = None,
    intended_risk_amount: Decimal | str | float | None = None,
    interval_seconds: int = 45,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    maximum_book_age_seconds: int = 120,
) -> dict[str, tuple[BookObservation, BookObservation]]:
    """Capture every first book, wait once, then capture every second book."""
    if not _MIN_OBSERVATION_SECONDS <= interval_seconds <= _MAX_OBSERVATION_SECONDS:
        raise ValueError("book observation interval must be between 30 and 60 seconds")
    if (requested_shares is None) == (intended_risk_amount is None):
        raise ValueError(
            "provide exactly one of requested_shares or intended_risk_amount"
        )
    if maximum_book_age_seconds <= 0:
        raise ValueError("maximum_book_age_seconds must be positive")
    unique_tokens = tuple(
        dict.fromkeys(_required_text(item, "CLOB token id") for item in token_ids)
    )
    if not unique_tokens:
        return {}
    observations: dict[str, list[BookObservation]] = {
        token: [] for token in unique_tokens
    }
    for sequence in (1, 2):
        for token_id in unique_tokens:
            book = client.get_order_book(token_id)
            observed_at = clock()
            _validate_book_observation(
                book,
                observed_at=observed_at,
                maximum_book_age_seconds=maximum_book_age_seconds,
            )
            if intended_risk_amount is not None:
                fill = walk_buy_book_by_risk(book, intended_risk_amount)
            else:
                assert requested_shares is not None
                fill = walk_buy_book(book, requested_shares)
            observations[token_id].append(
                BookObservation(
                    sequence_number=sequence,
                    observed_at=observed_at,
                    book=book,
                    fill=fill,
                )
            )
        if sequence == 1:
            sleeper(interval_seconds)
    return {token: (items[0], items[1]) for token, items in observations.items()}


def capture_minimum_order_book_batch(
    client: OrderBookClient,
    *,
    token_ids: Sequence[str],
    interval_seconds: int = 45,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    maximum_book_age_seconds: int = 120,
) -> MinimumOrderBookBatch:
    """Quote each token at the larger minimum share size across two books."""
    if not _MIN_OBSERVATION_SECONDS <= interval_seconds <= _MAX_OBSERVATION_SECONDS:
        raise ValueError("book observation interval must be between 30 and 60 seconds")
    if maximum_book_age_seconds <= 0:
        raise ValueError("maximum_book_age_seconds must be positive")
    unique_tokens = tuple(
        dict.fromkeys(_required_text(item, "CLOB token id") for item in token_ids)
    )
    if not unique_tokens:
        return MinimumOrderBookBatch(observations={}, failures={})

    captured: dict[str, list[tuple[datetime, OrderBook]]] = {
        token: [] for token in unique_tokens
    }
    failures: dict[str, BookCaptureFailure] = {}
    for sequence in (1, 2):
        active_tokens = (
            unique_tokens
            if sequence == 1
            else tuple(token for token in unique_tokens if token not in failures)
        )
        for token_id in active_tokens:
            try:
                book = client.get_order_book(token_id)
                observed_at = clock()
                _validate_book_observation(
                    book,
                    observed_at=observed_at,
                    maximum_book_age_seconds=maximum_book_age_seconds,
                )
                captured[token_id].append((observed_at, book))
            except Exception as error:  # provider failures must remain token-local
                failures[token_id] = _book_capture_failure(
                    token_id,
                    error,
                    sequence_number=sequence,
                )
        if sequence == 1 and any(captured.values()):
            sleeper(interval_seconds)

    observations: dict[str, tuple[BookObservation, BookObservation]] = {}
    for token_id, items in captured.items():
        if token_id in failures or len(items) != _EXPECTED_BINARY_OUTCOMES:
            continue
        requested_shares = max(book.minimum_order_size for _, book in items)
        first, second = items
        pair = (
            BookObservation(
                1, first[0], first[1], walk_buy_book(first[1], requested_shares)
            ),
            BookObservation(
                2, second[0], second[1], walk_buy_book(second[1], requested_shares)
            ),
        )
        observations[token_id] = pair
        if not all(observation.fill.complete for observation in pair):
            reason = (
                "empty_order_book"
                if any(not observation.book.asks for observation in pair)
                else "insufficient_depth"
            )
            failures[token_id] = BookCaptureFailure(
                token_id=token_id,
                reason=reason,
                detail=(
                    f"minimum executable size {requested_shares} shares was not "
                    "available in both observations"
                ),
            )
    return MinimumOrderBookBatch(observations=observations, failures=failures)


def _validate_book_observation(
    book: OrderBook,
    *,
    observed_at: datetime,
    maximum_book_age_seconds: int,
) -> None:
    if observed_at.tzinfo is None or observed_at.utcoffset() != timedelta(0):
        raise ValueError("book observation clock must return UTC")
    if book.timestamp is None:
        raise MarketDataError("CLOB order book timestamp is missing")
    if abs((observed_at - book.timestamp).total_seconds()) > maximum_book_age_seconds:
        raise MarketDataError("CLOB order book is stale")


def _book_capture_failure(
    token_id: str,
    error: Exception,
    *,
    sequence_number: int,
) -> BookCaptureFailure:
    detail = f"{type(error).__name__}: {error}"
    normalized = str(error).casefold()
    if isinstance(error, MarketDataError):
        if "timestamp is missing" in normalized:
            reason = "missing_book_timestamp"
        elif "stale" in normalized:
            reason = "stale_book"
        elif "crossed" in normalized:
            reason = "crossed_book"
        else:
            reason = "malformed_book"
    elif isinstance(error, ValueError) and "clock" in normalized:
        reason = "invalid_observation_clock"
    else:
        reason = "book_unavailable"
    return BookCaptureFailure(
        token_id=token_id,
        reason=reason,
        detail=detail,
        sequence_number=sequence_number,
    )


def confirmed_executable_fill(
    observations: tuple[BookObservation, BookObservation],
) -> ExpectedFill:
    """Return the worse average price only when both observations fully fill."""
    if len(observations) != _EXPECTED_BINARY_OUTCOMES or not all(
        item.fill.complete for item in observations
    ):
        raise MarketDataError("both order-book observations require complete depth")
    return max(
        (item.fill for item in observations),
        key=lambda fill: fill.average_price or Decimal(1),
    )


class PolymarketClobClient:
    """Read-only current CLOB client with retries and a tiny transient cache."""

    def __init__(
        self,
        *,
        base_url: str = "https://clob.polymarket.com",
        session: requests.Session | None = None,
        timeout: float = 10.0,
        attempts: int = 3,
        cache_seconds: float = 2.0,
        monotonic_clock: Callable[[], float] = time.monotonic,
        retry_sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout <= 0 or attempts <= 0 or cache_seconds < 0:
            raise ValueError("invalid CLOB client retry or cache settings")
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.attempts = attempts
        self.cache_seconds = cache_seconds
        self.monotonic_clock = monotonic_clock
        self.retry_sleep = retry_sleep
        self._cache: dict[str, tuple[float, OrderBook]] = {}

    def get_order_book(self, token_id: str) -> OrderBook:
        """Fetch one public order book without signer or trading credentials."""
        token_id = _required_text(token_id, "CLOB token id")
        now = self.monotonic_clock()
        cached = self._cache.get(token_id)
        if cached and now - cached[0] <= self.cache_seconds:
            return cached[1]

        for attempt in range(1, self.attempts + 1):
            try:
                response = self.session.get(
                    f"{self.base_url}/book",
                    params={"token_id": token_id},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                book = OrderBook.from_payload(
                    payload,
                    expected_token_id=token_id,
                )
                _require_book_timestamp(book)
            except (requests.RequestException, ValueError, MarketDataError) as error:
                if attempt == self.attempts:
                    if isinstance(error, MarketDataError):
                        raise
                    raise MarketReadError(
                        "Polymarket CLOB order-book request failed after "
                        f"{self.attempts} attempts."
                    ) from error
                self.retry_sleep(min(2 ** (attempt - 1), 4))
                continue

            self._cache[token_id] = (now, book)
            return book
        raise AssertionError("unreachable CLOB retry state")


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, (str, int)) or not str(value).strip():
        raise MarketDataError(f"{field} must be a non-empty value")
    return str(value).strip()


def _require_book_timestamp(book: OrderBook) -> None:
    if book.timestamp is None:
        raise MarketDataError("CLOB order book timestamp is missing")


def _required_decimal(value: Any, *, field: str) -> Decimal:
    parsed = _optional_decimal(value, field=field)
    if parsed is None:
        raise MarketDataError(f"{field} is required")
    return parsed


def _optional_decimal(value: Any, *, field: str) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise MarketDataError(f"{field} must be a finite decimal") from error
    if not parsed.is_finite():
        raise MarketDataError(f"{field} must be a finite decimal")
    return parsed


def _optional_utc(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as error:
        raise MarketDataError("Gamma event start time must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise MarketDataError("Gamma event start time must include a timezone")
    return parsed.astimezone(UTC)


def _optional_clob_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        milliseconds = float(value)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError as error:
            raise MarketDataError(
                "CLOB timestamp must be epoch milliseconds or ISO-8601"
            ) from error
        if parsed.tzinfo is None:
            raise MarketDataError(
                "CLOB ISO-8601 timestamp must include a timezone"
            ) from None
        return parsed.astimezone(UTC)
    if not math.isfinite(milliseconds):
        raise MarketDataError("CLOB timestamp must be finite")
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC)


def _parse_best_of(explicit: Any, text: str) -> int | None:
    if explicit not in (None, ""):
        try:
            value = int(explicit)
        except (TypeError, ValueError):
            return None
        return value if value in {1, 2, 3, 5} else None
    match = re.search(r"\b(?:bo|best\s+of\s+)([1235])\b", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _parse_game_number(text: str) -> int | None:
    match = re.search(r"\b(?:game|map)\s*([1-5])\b", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _parse_total_line(explicit: Any, text: str) -> Decimal | None:
    if explicit not in (None, ""):
        return _optional_decimal(explicit, field="totals line")
    match = re.search(r"\b([1-5]\.5)\b", text)
    return Decimal(match.group(1)) if match else None


def _text_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        part
        for part in "".join(
            character.casefold() if character.isalnum() else " " for character in value
        ).split()
        if part
    )


def _aliases_match_text(aliases: tuple[str, ...], text: str) -> bool:
    text_tokens = set(_text_tokens(text))
    return any(
        bool(alias_tokens) and set(alias_tokens) <= text_tokens
        for alias_tokens in (_text_tokens(alias) for alias in aliases)
    )


def _coerce_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return decoded if isinstance(decoded, list) else [decoded]
    return [value]
