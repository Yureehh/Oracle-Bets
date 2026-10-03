"""Capture paper quotes for a chosen stake; never submit provider orders."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.markets import (
    PolymarketClobClient,
    capture_confirmation_order_book,
)
from oracle_bets_core.operations.bets import (
    BetEvidenceError,
    _bet_context,
    _positive_decimal,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from oracle_bets_core.markets import OrderBookClient

_MAX_CURRENCY_LENGTH = 12


def capture_paper_quote(
    store: EvidenceStore,
    *,
    review_id: str,
    market_id: str,
    stake_amount: Decimal | str,
    currency: str,
    net_decimal_odds: Decimal | str,
    terms_attested: bool,
    actor_id: str,
    clob_client: OrderBookClient | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, Any]:
    """
    Freeze an owner-checked all-in quote and its actual capture time.

    The owner attests the source rules are unchanged, the entered net odds include
    all costs, and this stake is currently available. For Polymarket, a fresh book
    additionally bounds depth and gross odds. Net fees are owner-verified here,
    not inferred from minimum-order prices or a hardcoded category fee.
    """
    if terms_attested is not True or not actor_id.strip():
        raise BetEvidenceError(
            "Verify the current rules, full stake and all costs first."
        )
    amount = _positive_decimal(stake_amount, "quote stake")
    odds = _positive_decimal(net_decimal_odds, "net decimal odds")
    if odds <= 1:
        raise BetEvidenceError("Net decimal odds must be greater than one.")
    denomination = currency.strip().upper()
    if not denomination or len(denomination) > _MAX_CURRENCY_LENGTH:
        raise BetEvidenceError("A quote currency is required.")
    candidate, payload, _, _ = _bet_context(
        store,
        review_id=review_id,
        market_id=market_id,
        opened_at=clock(),
        prospective=True,
    )
    terms = str(
        (payload.get("semantic_key") or {}).get("resolution_terms") or ""
    ).strip()
    if not terms or not payload.get("semantic_fingerprint") or not payload.get("url"):
        raise BetEvidenceError(
            "Review the provider's source and full settlement rules first."
        )
    provider = candidate["provider"]
    book_payload: dict[str, Any] = {}
    details: dict[str, Any] = {
        "complete": True,
        "semantic_fingerprint": payload["semantic_fingerprint"],
        "resolution_terms": terms,
        "terms_verified": True,
        "terms_attested_by": actor_id,
        "source_url": payload["url"],
        "stake_currency": denomination,
        "stake_amount": str(amount),
        "quote_basis": "owner_verified_net_odds",
        "cost_verification": "owner_attested_all_in",
        "provider_timestamp": None,
    }
    if provider == "polymarket":
        # Collateral migration: https://help.polymarket.com/en/articles/14762452
        # No implicit FX conversion or USDC/pUSD portfolio mixing.
        if denomination != "PUSD":
            raise BetEvidenceError(
                "Polymarket paper quotes require a PUSD bankroll; no FX is assumed."
            )
        captured = capture_confirmation_order_book(
            clob_client or PolymarketClobClient(),
            token_id=str(candidate["provider_selection_id"]),
            review_observation=None,
            stake_amount=amount,
            clock=clock,
        )
        observation = captured.observation
        if observation is None:
            reason = captured.failure.reason if captured.failure else "unavailable"
            raise BetEvidenceError(f"Current stake quote unavailable: {reason}.")
        gross_odds = Decimal(str(observation.fill.decimal_odds))
        if odds > gross_odds:
            raise BetEvidenceError(
                f"Net odds exceed the current book price ({gross_odds}); recheck costs and reprice."
            )
        assert observation.book.timestamp is not None  # validated by capture
        details.update(
            provider_timestamp=observation.book.timestamp.isoformat(),
            book_hash=observation.book.book_hash,
            gross_book_odds=str(gross_odds),
            gross_requested_shares=str(observation.fill.requested_shares),
            quote_basis="owner_verified_net_odds_with_book_depth",
        )
        book_payload = {
            side: [
                {"price": str(level.price), "size": str(level.size)} for level in levels
            ]
            for side, levels in (
                ("asks", observation.book.asks),
                ("bids", observation.book.bids),
            )
        }
    elif provider != "thunderpick":
        raise BetEvidenceError("Paper quote capture is unsupported for this provider.")
    observed = clock()
    # A slow provider response may have crossed the fixture start or invalidation.
    _bet_context(
        store,
        review_id=review_id,
        market_id=market_id,
        opened_at=observed,
        prospective=True,
    )
    identity = json.dumps(
        [market_id, observed.isoformat(), str(odds), details], sort_keys=True
    )
    snapshot_id = "snapshot-" + hashlib.sha256(identity.encode()).hexdigest()[:24]
    quote = {
        "id": snapshot_id,
        "market_candidate_id": market_id,
        "observed_at": observed,
        "sequence_number": int(observed.timestamp() * 1_000_000),
        "intended_stake_units": "0",
        "available_stake_units": "0",
        "expected_decimal_odds": str(odds),
        "book_json": book_payload,
        "idempotency_key": snapshot_id,
        "payload_json": details,
    }
    store.append(EvidenceTable.MARKET_SNAPSHOTS, quote)
    return quote
