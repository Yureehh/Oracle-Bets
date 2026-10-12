"""Declared paper capital; currency conversion is evidence, never an assumption."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from oracle_bets_core.paths import SUITE_ROOT


@dataclass(frozen=True)
class VenueCapital:
    allocation_eur: Decimal
    native_currency: str
    native_units_per_eur: Decimal | None
    conversion_cost_native: Decimal | None
    source_reference: str | None
    verified_at: datetime | None
    costs_verified: bool

    def initial_bankroll(self) -> Decimal:
        if self.native_units_per_eur is None or not self.costs_verified:
            raise ValueError(
                "Paper capital conversion and costs must be owner-verified."
            )
        return self.allocation_eur * self.native_units_per_eur - (
            self.conversion_cost_native or Decimal(0)
        )


@dataclass(frozen=True)
class PaperCapital:
    policy_version: str
    config_hash: str
    venues: dict[str, VenueCapital]

    def venue(self, provider: str, currency: str) -> VenueCapital:
        venue = self.venues.get(provider.casefold())
        if venue is None or venue.native_currency != currency.upper():
            raise ValueError(
                "Paper capital provider/currency differs from the declared allocation."
            )
        return venue


def _decimal(value: Any, *, positive: bool) -> Decimal:
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(
            "Paper capital amounts must be finite numeric values."
        ) from error
    if not number.is_finite() or number < 0 or (positive and number == 0):
        raise ValueError(
            "Paper capital amounts must be finite and positive (costs may be zero)."
        )
    return number


def load_paper_capital(path: Path | str | None = None) -> PaperCapital:
    source = (
        Path(path)
        if path is not None
        else SUITE_ROOT / "config/research/lol-paper-capital.json"
    )
    payload = json.loads(source.read_text())
    if (
        payload.get("schema_version") != 1
        or payload.get("base_currency") != "EUR"
        or _decimal(payload.get("total_allocation"), positive=True) != Decimal(500)
        or set(payload.get("venues", {})) != {"thunderpick", "polymarket"}
        or not str(payload.get("policy_version") or "").strip()
    ):
        raise ValueError(
            "Paper capital requires EUR 500 allocated across the two declared venues."
        )
    venues = {}
    for provider, raw in payload["venues"].items():
        allocation = _decimal(raw["allocation_eur"], positive=True)
        currency = str(raw["native_currency"]).upper()
        rate = (
            _decimal(raw["native_units_per_eur"], positive=True)
            if raw["native_units_per_eur"] is not None
            else None
        )
        cost = (
            _decimal(raw["conversion_cost_native"], positive=False)
            if raw["conversion_cost_native"] is not None
            else None
        )
        reference = str(raw.get("source_reference") or "").strip() or None
        timestamp = (
            datetime.fromisoformat(raw["verified_at"])
            if raw.get("verified_at")
            else None
        )
        if timestamp is not None and (
            timestamp.tzinfo is None or timestamp > datetime.now(UTC)
        ):
            raise ValueError(
                "Paper conversion timestamp must be aware and not in the future."
            )
        if (
            allocation != Decimal(250)
            or currency != {"thunderpick": "EUR", "polymarket": "PUSD"}[provider]
        ):
            raise ValueError(
                "Paper allocation is EUR 250 per venue with its declared native currency."
            )
        verified = raw.get("costs_verified") is True
        if rate is not None:
            identity = currency == "EUR" and rate == 1 and cost == 0
            if (
                not reference
                or cost is None
                or not verified
                or (not identity and timestamp is None)
            ):
                raise ValueError(
                    "A conversion requires sourced rate, timestamp and verified costs."
                )
            if allocation * rate - cost <= 0:
                raise ValueError("Conversion costs exhaust the paper allocation.")
        venues[provider] = VenueCapital(
            allocation, currency, rate, cost, reference, timestamp, verified
        )
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PaperCapital(payload["policy_version"], digest, venues)


def paper_capital_status(config: PaperCapital | None = None) -> dict[str, Any]:
    capital = config or load_paper_capital()
    venues = {}
    for provider, venue in capital.venues.items():
        try:
            initial = venue.initial_bankroll()
        except ValueError:
            initial = None
        venues[provider] = {
            "allocation_eur": str(venue.allocation_eur),
            "native_currency": venue.native_currency,
            "native_units_per_eur": str(venue.native_units_per_eur)
            if venue.native_units_per_eur is not None
            else None,
            "conversion_cost_native": str(venue.conversion_cost_native)
            if venue.conversion_cost_native is not None
            else None,
            "costs_verified": venue.costs_verified,
            "initial_native_bankroll": str(initial) if initial is not None else None,
            "entry_ready": initial is not None,
            "source_reference": venue.source_reference,
            "verified_at": venue.verified_at.isoformat() if venue.verified_at else None,
        }
    return {
        "policy_version": capital.policy_version,
        "config_hash": capital.config_hash,
        "total_allocation_eur": "500",
        "venues": venues,
    }
