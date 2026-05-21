"""Read-only prediction-market discovery adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import requests


@dataclass(frozen=True)
class MarketQuote:
    """Normalized market quote used by Oracle Bets."""

    source: str
    market_id: str
    question: str
    outcome: str
    price: float | None = None
    implied_probability: float | None = None
    liquidity: float | None = None
    volume: float | None = None
    url: str | None = None


class MarketAdapter(Protocol):
    """Read-only market source adapter."""

    source: str

    def search(self, query: str, *, limit: int = 25) -> list[MarketQuote]:
        """Return normalized quotes matching ``query``."""


class PolymarketGammaAdapter:
    """Small read-only adapter for Polymarket Gamma market discovery."""

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
        params = {"q": query, "limit": limit, "active": "true", "closed": "false"}
        response = self.session.get(
            f"{self.base_url}/markets", params=params, timeout=self.timeout
        )
        response.raise_for_status()
        payload = response.json()
        markets = payload if isinstance(payload, list) else payload.get("markets", [])
        return [
            quote for market in markets for quote in self._quotes_from_market(market)
        ]

    def _quotes_from_market(self, market: dict[str, Any]) -> list[MarketQuote]:
        outcomes = _coerce_list(market.get("outcomes"))
        prices = _coerce_list(market.get("outcomePrices"))
        quotes: list[MarketQuote] = []
        for idx, outcome in enumerate(outcomes):
            price = _coerce_float(prices[idx] if idx < len(prices) else None)
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
                    url=market.get("slug")
                    and f"https://polymarket.com/event/{market['slug']}",
                )
            )
        return quotes


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
        import json

        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return decoded if isinstance(decoded, list) else [decoded]
    return [value]
