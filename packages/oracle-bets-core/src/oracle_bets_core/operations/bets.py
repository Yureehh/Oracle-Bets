"""Owner-recorded paper and real bets; never bookmaker transactions."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.evidence.settlement import (
    PositionTerms,
    SettlementResult,
    settlement_pnl,
)

_RESULTS = frozenset({"win", "loss", "push", "void"})
_MODES = frozenset({"paper", "real"})
_MAX_CURRENCY_LENGTH = 12
_PERCENT = Decimal(100)


class BetEvidenceError(ValueError):
    """Raised when owner-entered bet evidence is invalid or conflicting."""


def record_bet(
    store: EvidenceStore,
    *,
    review_id: str,
    market_id: str,
    mode: str,
    currency: str,
    bankroll_before: Decimal | str,
    stake_percent: Decimal | str,
    accepted_odds: Decimal | str,
    reason: str,
    stake_amount: Decimal | str | None = None,
    actor_id: str = "owner-cli",
    opened_at: datetime | None = None,
    idempotency_key: str | None = None,
) -> str:
    """Record a bet the owner already chose and placed outside Oracle Bets."""
    entry = prepare_bet(
        store,
        review_id=review_id,
        market_id=market_id,
        mode=mode,
        currency=currency,
        bankroll_before=bankroll_before,
        stake_percent=stake_percent,
        accepted_odds=accepted_odds,
        reason=reason,
        stake_amount=stake_amount,
        actor_id=actor_id,
        opened_at=opened_at,
        idempotency_key=idempotency_key,
    )
    store.append(EvidenceTable.BETS, entry)
    return str(entry["id"])


def prepare_bet(
    store: EvidenceStore,
    *,
    review_id: str,
    market_id: str,
    mode: str,
    currency: str,
    bankroll_before: Decimal | str,
    stake_percent: Decimal | str,
    accepted_odds: Decimal | str,
    reason: str,
    stake_amount: Decimal | str | None = None,
    actor_id: str = "owner-cli",
    opened_at: datetime | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Validate and build an immutable bet entry without writing it."""
    candidate = store.get(EvidenceTable.MARKET_CANDIDATES, market_id)
    if candidate is None:
        raise BetEvidenceError(f"Unknown reviewed market: {market_id}")
    if str(candidate["run_id"]) != review_id:
        raise BetEvidenceError("The market does not belong to the supplied review.")
    if mode not in _MODES:
        raise BetEvidenceError("Bet mode must be paper or real.")
    rationale = reason.strip()
    if not rationale:
        raise BetEvidenceError("A short owner rationale is required.")
    normalized_currency = currency.strip().upper()
    if not normalized_currency or len(normalized_currency) > _MAX_CURRENCY_LENGTH:
        raise BetEvidenceError("Currency must be a short non-empty code.")

    bankroll = _positive_decimal(bankroll_before, "bankroll-before")
    percent = _positive_decimal(stake_percent, "stake-percent")
    if percent > _PERCENT:
        raise BetEvidenceError("Stake percentage cannot exceed 100.")
    derived_amount = bankroll * percent / _PERCENT
    amount = (
        _positive_decimal(stake_amount, "stake-amount")
        if stake_amount is not None
        else derived_amount
    )
    if abs(amount - derived_amount) > Decimal("0.01"):
        raise BetEvidenceError(
            "Stake amount must match bankroll-before × stake-percent within 0.01."
        )
    odds = _positive_decimal(accepted_odds, "accepted-odds")
    if odds <= 1:
        raise BetEvidenceError("Accepted decimal odds must be greater than 1.")

    payload = _payload(candidate.get("payload_json"))
    probability = _optional_decimal(payload.get("probability"))
    point_ev = probability * odds - 1 if probability is not None else None
    classification = (
        "model_unavailable"
        if probability is None
        else "model_positive_ev"
        if point_ev is not None and point_ev > 0
        else "model_non_positive_ev"
    )
    opened = _as_utc(opened_at or datetime.now(UTC))
    retry_key = idempotency_key.strip() if idempotency_key else None
    identity = retry_key or "|".join(
        (
            review_id,
            market_id,
            mode,
            actor_id,
            opened.isoformat(),
            normalized_currency,
            str(bankroll),
            str(percent),
            str(odds),
            str(amount),
            rationale,
        )
    )
    bet_id = _id("bet", identity)
    return {
        "id": bet_id,
        "review_id": review_id,
        "fixture_id": candidate["fixture_id"],
        "market_candidate_id": market_id,
        "mode": mode,
        "provider": candidate["provider"],
        "target": str(payload.get("target") or "unknown"),
        "selection": str(
            payload.get("selection") or candidate.get("provider_selection_id") or ""
        ),
        "opened_at": opened,
        "currency": normalized_currency,
        "bankroll_before": bankroll,
        "stake_percent": percent,
        "stake_amount": amount,
        "accepted_odds": odds,
        "evidence_classification": classification,
        "actor_id": actor_id,
        "idempotency_key": retry_key or bet_id,
        "payload_json": {
            "reason": rationale,
            "source_url": payload.get("url"),
            "line": payload.get("line"),
            "game_number": payload.get("game_number"),
            "quoted_odds": payload.get("decimal_odds"),
            "model_probability": float(probability)
            if probability is not None
            else None,
            "fair_odds": float(1 / probability)
            if probability is not None and probability > 0
            else None,
            "point_ev": float(point_ev) if point_ev is not None else None,
            "stake_amount_derived": str(derived_amount),
            "warnings": payload.get("warnings") or [],
            "hard_blocks": payload.get("hard_blocks") or [],
            "tracking_only": True,
        },
    }


def settle_bet(
    store: EvidenceStore,
    *,
    bet_id: str,
    result: str,
    source_reference: str,
    actor_id: str = "owner-cli",
    note: str | None = None,
    settled_at: datetime | None = None,
    closing_odds: Decimal | str | None = None,
) -> str:
    """Append one owner-verified settlement; repeated identical calls are safe."""
    bet = store.get(EvidenceTable.BETS, bet_id)
    if bet is None:
        raise BetEvidenceError(f"Unknown bet: {bet_id}")
    normalized_result = result.strip().casefold()
    if normalized_result not in _RESULTS:
        raise BetEvidenceError("Result must be win, loss, push, or void.")
    reference = source_reference.strip()
    if not reference:
        raise BetEvidenceError("Settlement source reference is required.")
    normalized_note = note.strip() if note else None
    normalized_closing_odds = (
        str(_positive_decimal(closing_odds, "closing-odds"))
        if closing_odds is not None
        else None
    )
    existing = _settlement_event(store, bet_id)
    if existing is not None:
        payload = _payload(existing["payload_json"])
        if (
            payload.get("result") == normalized_result
            and payload.get("source_reference") == reference
            and payload.get("note") == normalized_note
            and payload.get("closing_odds") == normalized_closing_odds
            and str(existing["actor_id"]) == actor_id
        ):
            return str(existing["id"])
        raise BetEvidenceError("This bet already has a conflicting settlement.")

    stake = Decimal(str(bet["stake_amount"]))
    odds = Decimal(str(bet["accepted_odds"]))
    pnl = settlement_pnl(
        PositionTerms(bet_id, stake, odds),
        SettlementResult(normalized_result),
    )
    settled = _as_utc(settled_at or datetime.now(UTC))
    event_id = _id("bet-event", f"{bet_id}|settlement")
    store.append(
        EvidenceTable.BET_EVENTS,
        {
            "id": event_id,
            "bet_id": bet_id,
            "event_at": settled,
            "event_type": "settlement",
            "actor_id": actor_id,
            "idempotency_key": event_id,
            "payload_json": {
                "result": normalized_result,
                "source_reference": reference,
                "note": normalized_note,
                "pnl_amount": str(pnl),
                "currency": bet["currency"],
                "closing_odds": normalized_closing_odds,
                "owner_verified": True,
            },
        },
    )
    return event_id


def list_bets(
    store: EvidenceStore,
    *,
    state: str | None = None,
    mode: str | None = None,
    provider: str | None = None,
    target: str | None = None,
) -> list[dict[str, Any]]:
    """Return unified bets with their derived append-only lifecycle state."""
    if state not in {None, "open", "settled"}:
        raise BetEvidenceError("Bet state must be open or settled.")
    if mode not in {None, *_MODES}:
        raise BetEvidenceError("Bet mode must be paper or real.")
    settlements = {
        str(row["bet_id"]): row
        for row in store.list(EvidenceTable.BET_EVENTS)
        if row["event_type"] == "settlement"
    }
    output: list[dict[str, Any]] = []
    for raw in reversed(store.list(EvidenceTable.BETS)):
        settlement = settlements.get(str(raw["id"]))
        row = _enrich_bet(raw, settlement)
        if state and row["state"] != state:
            continue
        if mode and row["mode"] != mode:
            continue
        if provider and str(row["provider"]).casefold() != provider.casefold():
            continue
        if target and str(row["target"]).casefold() != target.casefold():
            continue
        output.append(row)
    return output


def show_bet(store: EvidenceStore, bet_id: str) -> dict[str, Any]:
    """Return one bet with its lifecycle state."""
    raw = store.get(EvidenceTable.BETS, bet_id)
    if raw is None:
        raise BetEvidenceError(f"Unknown bet: {bet_id}")
    return _enrich_bet(raw, _settlement_event(store, bet_id))


def performance_summary(
    store: EvidenceStore,
    *,
    mode: str,
    since: datetime | None = None,
) -> dict[str, Any]:
    """Summarize one evidence mode without combining currencies."""
    rows = performance_rows(store, mode=mode, since=since)
    return summarize_bets(rows, mode=mode)


def performance_rows(
    store: EvidenceStore,
    *,
    mode: str,
    since: datetime | None = None,
) -> list[dict[str, Any]]:
    """Load one filtered settled-bet snapshot for summaries and charts."""
    rows = list_bets(store, state="settled", mode=mode)
    if since is not None:
        cutoff = _as_utc(since)
        rows = [
            row
            for row in rows
            if _as_utc(_datetime(row["settlement"]["event_at"])) >= cutoff
        ]
    return rows


def summarize_bets(rows: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    """Summarize already-loaded settled bets without another database read."""
    currencies: dict[str, dict[str, Any]] = {}
    for row in rows:
        currency = str(row["currency"])
        bucket = currencies.setdefault(
            currency,
            {"settled": 0, "graded": 0, "turnover": Decimal(0), "pnl": Decimal(0)},
        )
        settlement = row["settlement"] or {}
        result = settlement.get("result")
        bucket["settled"] += 1
        bucket["graded"] += int(result in {"win", "loss"})
        bucket["turnover"] += Decimal(str(row["stake_amount"]))
        bucket["pnl"] += Decimal(str(settlement.get("pnl_amount") or 0))
    for bucket in currencies.values():
        turnover = bucket["turnover"]
        bucket["roi"] = float(bucket["pnl"] / turnover) if turnover else None
        bucket["turnover"] = str(bucket["turnover"])
        bucket["pnl"] = str(bucket["pnl"])
    return {"mode": mode, "settled": len(rows), "currencies": currencies}


def count_open_bets(store: EvidenceStore) -> int:
    return len(list_bets(store, state="open"))


def _settlement_event(store: EvidenceStore, bet_id: str) -> dict[str, Any] | None:
    return store.get(EvidenceTable.BET_EVENTS, _id("bet-event", f"{bet_id}|settlement"))


def _enrich_bet(
    raw: dict[str, Any], settlement: dict[str, Any] | None
) -> dict[str, Any]:
    row = dict(raw)
    row["payload"] = _payload(row.pop("payload_json"))
    row["state"] = "settled" if settlement else "open"
    row["settlement"] = (
        _payload(settlement["payload_json"]) | {"event_at": settlement["event_at"]}
        if settlement
        else None
    )
    return row


def _payload(value: Any) -> dict[str, Any]:
    try:
        result = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return result if isinstance(result, dict) else {}


def _positive_decimal(value: Any, name: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise BetEvidenceError(f"{name} must be numeric.") from exc
    if not number.is_finite() or number <= 0:
        raise BetEvidenceError(f"{name} must be positive.")
    return number


def _optional_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() and 0 <= number <= 1 else None


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise BetEvidenceError("Bet timestamps must be timezone-aware.")
    return value.astimezone(UTC)


def _datetime(value: Any) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))


def _id(prefix: str, identity: str) -> str:
    return f"{prefix}-{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
