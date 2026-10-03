"""Owner-recorded paper and real bets; never bookmaker transactions."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from typing import Any

from oracle_bets_core.config import QUOTE_TTL_SECONDS
from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.evidence.performance import summarize_unified_bets
from oracle_bets_core.evidence.portfolio import (
    KellyPolicy,
    Opportunity,
    allocate_portfolio,
)
from oracle_bets_core.evidence.settlement import (
    PositionTerms,
    SettlementResult,
    settlement_pnl,
)

_RESULTS = frozenset({"win", "loss", "push", "void"})
_MODES = frozenset({"paper", "real"})
_MAX_CURRENCY_LENGTH = 12
_PERCENT = Decimal(100)
_MIN_COHORT_REVIEW_COVERAGE = 0.9
_COHORT_STAGES = frozenset(
    {
        "no_market",
        "reviewed",
        "recommended",
        "rejected",
        "opened",
        "shadow_result",
        "settled",
    }
)
_OPPORTUNITY_STAGES = frozenset(
    {"recommended", "rejected", "opened", "shadow_result", "settled"}
)


class BetEvidenceError(ValueError):
    """Raised when owner-entered bet evidence is invalid or conflicting."""


def enroll_strategy_cohort(
    store: EvidenceStore,
    *,
    league: str,
    week_start: str,
    fixture_ids: tuple[str, ...] | list[str],
    cells: tuple[str, ...] | list[str],
    policy_version: str,
) -> str:
    """Preregister one complete league-week before forecasts or prices exist."""
    normalized_league = league.strip()
    normalized_policy = policy_version.strip()
    normalized_fixtures = tuple(dict.fromkeys(value.strip() for value in fixture_ids))
    normalized_cells = tuple(dict.fromkeys(value.strip() for value in cells))
    try:
        week = date.fromisoformat(week_start)
    except ValueError as error:
        raise BetEvidenceError("Cohort week-start must use YYYY-MM-DD.") from error
    if week.weekday() != 0:
        raise BetEvidenceError("Cohort week-start must be a Monday.")
    if not normalized_league or not normalized_policy:
        raise BetEvidenceError("Cohort league and policy version are required.")
    if not normalized_fixtures or any(not value for value in normalized_fixtures):
        raise BetEvidenceError("Cohort fixture IDs must be non-empty.")
    if not normalized_cells or any(not value for value in normalized_cells):
        raise BetEvidenceError("Cohort cells must be predeclared and non-empty.")

    fixtures = {
        str(row["id"]): row
        for row in store.get_many(EvidenceTable.FIXTURES, normalized_fixtures)
    }
    if set(fixtures) != set(normalized_fixtures):
        raise BetEvidenceError("Every cohort fixture must exist before enrollment.")
    if any(
        str(row["competition_id"]) != normalized_league for row in fixtures.values()
    ):
        raise BetEvidenceError("Every cohort fixture must belong to the cohort league.")
    enrolled_at = _utc_now()
    week_end = week + timedelta(days=7)
    if any(
        not week <= (start := _as_utc(_datetime(row["start_time"]))).date() < week_end
        or start <= enrolled_at
        for row in fixtures.values()
    ):
        raise BetEvidenceError(
            "Every cohort fixture must start in the declared week after enrollment."
        )
    placeholders = ", ".join("?" for _ in normalized_fixtures)
    with store.connection(read_only=True) as connection:
        evidence_exists = any(
            connection.execute(
                f"SELECT 1 FROM {table} WHERE fixture_id IN ({placeholders}) LIMIT 1",  # noqa: S608
                normalized_fixtures,
            ).fetchone()
            for table in ("predictions", "forecasts", "market_candidates")
        )
    if evidence_exists:
        raise BetEvidenceError(
            "Strategy cohorts must be enrolled before forecasts or prices are seen."
        )

    identity = "|".join(
        (
            normalized_league,
            week.isoformat(),
            normalized_policy,
            *normalized_fixtures,
            *normalized_cells,
        )
    )
    cohort_id = _id("strategy-cohort", identity)
    store.append(
        EvidenceTable.RUNS,
        {
            "id": cohort_id,
            "run_type": "strategy_cohort",
            "started_at": enrolled_at,
            "status": "active",
            "idempotency_key": cohort_id,
            "payload_json": {
                "league": normalized_league,
                "week_start": week.isoformat(),
                "fixture_ids": normalized_fixtures,
                "cells": normalized_cells,
                "policy_version": normalized_policy,
            },
        },
    )
    return cohort_id


def record_cohort_stage(
    store: EvidenceStore,
    cohort_id: str,
    fixture_id: str,
    stage: str,
    *,
    opportunity_id: str | None = None,
    payload: dict[str, Any] | None = None,
    recorded_at: datetime | None = None,
) -> str:
    """Append one idempotent intention-to-treat cohort transition."""
    cohort = store.get(EvidenceTable.RUNS, cohort_id)
    if cohort is None or cohort["run_type"] != "strategy_cohort":
        raise BetEvidenceError(f"Unknown strategy cohort: {cohort_id}")
    cohort_payload = _payload(cohort["payload_json"])
    if fixture_id not in cohort_payload.get("fixture_ids", []):
        raise BetEvidenceError("Fixture is not enrolled in this strategy cohort.")
    normalized_stage = stage.strip().casefold()
    if normalized_stage not in _COHORT_STAGES:
        raise BetEvidenceError(f"Unsupported cohort stage: {stage}")
    normalized_opportunity = opportunity_id.strip() if opportunity_id else None
    if normalized_stage in _OPPORTUNITY_STAGES and not normalized_opportunity:
        raise BetEvidenceError(f"{normalized_stage} requires an opportunity ID.")
    details = dict(payload or {})
    if normalized_stage in _OPPORTUNITY_STAGES:
        assert normalized_opportunity is not None
        candidate = store.get(EvidenceTable.MARKET_CANDIDATES, normalized_opportunity)
        if candidate is None or candidate["fixture_id"] != fixture_id:
            raise BetEvidenceError(
                "Cohort opportunity must be a market for this fixture."
            )
        if (
            normalized_stage == "recommended"
            and _payload(candidate["payload_json"]).get("classification")
            != "recommended"
        ):
            raise BetEvidenceError(
                "Cohort recommendation requires a recommended market."
            )
    if normalized_stage == "shadow_result" and (
        details.get("result") not in _RESULTS
        or not str(details.get("source_reference") or "").strip()
    ):
        raise BetEvidenceError("Shadow result requires a result and source reference.")
    if normalized_stage in {"reviewed", "no_market"} and not _cohort_review_valid(
        store, fixture_id, details
    ):
        raise BetEvidenceError("Cohort review must link to a completed fixture review.")
    if normalized_stage == "no_market" and not str(details.get("reason") or "").strip():
        raise BetEvidenceError("No-market review requires a reason.")
    identity = "|".join(
        (cohort_id, fixture_id, normalized_stage, normalized_opportunity or "")
    )
    event_id = _id("cohort-event", identity)
    event_payload = details | {
        "fixture_id": fixture_id,
        "opportunity_id": normalized_opportunity,
    }
    existing = store.get(EvidenceTable.RUN_EVENTS, event_id)
    if existing is not None:
        if _payload(existing["payload_json"]) == event_payload:
            return event_id
        raise BetEvidenceError("This cohort stage already stores different evidence.")
    store.append(
        EvidenceTable.RUN_EVENTS,
        {
            "id": event_id,
            "run_id": cohort_id,
            "event_at": _as_utc(recorded_at or datetime.now(UTC)),
            "event_type": f"cohort_{normalized_stage}",
            "status": "recorded",
            "idempotency_key": event_id,
            "payload_json": event_payload,
        },
    )
    return event_id


def cohort_coverage(store: EvidenceStore, cohort_id: str) -> dict[str, Any]:
    """Calculate the fixed coverage gates for one preregistered cohort."""
    cohort = store.get(EvidenceTable.RUNS, cohort_id)
    if cohort is None or cohort["run_type"] != "strategy_cohort":
        raise BetEvidenceError(f"Unknown strategy cohort: {cohort_id}")
    fixtures = set(_payload(cohort["payload_json"]).get("fixture_ids", []))
    reviewed: set[str] = set()
    recommendations: set[tuple[str, str]] = set()
    results: set[tuple[str, str]] = set()
    candidates = {
        str(row["id"]): row for row in store.list(EvidenceTable.MARKET_CANDIDATES)
    }
    for event in store.list(EvidenceTable.RUN_EVENTS):
        if event["run_id"] != cohort_id:
            continue
        stage = str(event["event_type"]).removeprefix("cohort_")
        details = _payload(event["payload_json"])
        fixture_id = str(details.get("fixture_id") or "")
        opportunity_id = str(details.get("opportunity_id") or "")
        if fixture_id not in fixtures:
            continue
        candidate = candidates.get(opportunity_id)
        linked = candidate is not None and candidate["fixture_id"] == fixture_id
        if (
            stage in {"reviewed", "no_market"}
            and _cohort_review_valid(store, fixture_id, details)
            and (stage != "no_market" or str(details.get("reason") or "").strip())
        ):
            reviewed.add(fixture_id)
        if (
            stage == "recommended"
            and linked
            and _payload(candidate["payload_json"]).get("classification")
            == "recommended"
        ):
            recommendations.add((fixture_id, opportunity_id))
        elif (
            stage == "shadow_result"
            and linked
            and details.get("result") in _RESULTS
            and str(details.get("source_reference") or "").strip()
        ):
            results.add((fixture_id, opportunity_id))
    fixture_fraction = len(reviewed & fixtures) / len(fixtures) if fixtures else 0.0
    captured = recommendations & results
    result_fraction = len(captured) / len(recommendations) if recommendations else None
    return {
        "cohort_id": cohort_id,
        "enrolled_fixtures": len(fixtures),
        "reviewed_or_no_market_fixtures": len(reviewed & fixtures),
        "fixture_review_fraction": fixture_fraction,
        "recommendation_opportunities": len(recommendations),
        "recommendation_results": len(captured),
        "recommendation_result_fraction": result_fraction,
        "activation_evidence_complete": bool(recommendations)
        and fixture_fraction >= _MIN_COHORT_REVIEW_COVERAGE
        and result_fraction == 1.0,
    }


def _cohort_review_valid(
    store: EvidenceStore, fixture_id: str, details: dict[str, Any]
) -> bool:
    review_id = str(details.get("review_id") or "")
    review = store.get(EvidenceTable.RUNS, review_id) if review_id else None
    if (
        review is None
        or review["run_type"]
        not in {
            "market_review",
            "manual_lol_market_review",
        }
        or review["status"] not in {"complete", "completed"}
    ):
        return False
    if fixture_id in _payload(review["payload_json"]).get("fixture_ids", []):
        return True
    return any(
        row["run_id"] == review_id and row["fixture_id"] == fixture_id
        for row in store.list(EvidenceTable.MARKET_CANDIDATES)
    )


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
    accepted_snapshot_id: str | None = None,
    idempotency_key: str | None = None,
) -> str:
    """Record a bet the owner already chose and placed outside Oracle Bets."""
    with store.transaction() as transaction:
        return _record_bet(
            transaction,
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
            accepted_snapshot_id=accepted_snapshot_id,
            idempotency_key=idempotency_key,
        )


def _record_bet(store: EvidenceStore, **values: Any) -> str:
    """Validate and append while the caller holds the ledger write transaction."""
    idempotency_key = values.get("idempotency_key")
    if idempotency_key:
        existing = store.get(EvidenceTable.BETS, _id("bet", idempotency_key.strip()))
        if existing is not None:
            retry = {
                key: value for key, value in values.items() if key != "idempotency_key"
            }
            _validate_bet_retry(existing, **retry)
            return str(existing["id"])
    entry = prepare_bet(store, **values)
    store.append(EvidenceTable.BETS, entry)
    return str(entry["id"])


def review_state(store: EvidenceStore, review_id: str) -> str:
    """Return the latest append-only state for one market review."""
    run = store.get(EvidenceTable.RUNS, review_id)
    if run is None:
        raise BetEvidenceError(f"Unknown review: {review_id}")
    if _is_superseded(store, ("runs", review_id)):
        return "invalidated"
    states = [
        row
        for row in store.list(EvidenceTable.RUN_EVENTS)
        if row["run_id"] == review_id and row["event_type"] == "review_state"
    ]
    return str(states[-1]["status"] if states else run["status"])


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
    accepted_snapshot_id: str | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Validate and build an immutable bet entry without writing it."""
    if mode not in _MODES:
        raise BetEvidenceError("Bet mode must be paper or real.")
    recorded_at = _utc_now()
    if mode == "paper" and opened_at is not None:
        raise BetEvidenceError(
            "Paper entries cannot be backdated; confirm at the current time."
        )
    opened = _as_utc(opened_at) if opened_at is not None else recorded_at
    if opened > recorded_at:
        raise BetEvidenceError("A bet cannot be recorded with a future opening time.")
    candidate, payload, lane, opened = _bet_context(
        store,
        review_id=review_id,
        market_id=market_id,
        opened_at=opened,
        prospective=mode == "paper",
    )
    fixture = store.get(EvidenceTable.FIXTURES, str(candidate["fixture_id"]))
    horizon_hours = (
        (_as_utc(_datetime(fixture["start_time"])) - opened).total_seconds() / 3600
        if fixture
        else None
    )
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

    snapshot = None
    if mode == "paper":
        snapshot = _accepted_quote(
            store,
            candidate,
            payload,
            accepted_snapshot_id=accepted_snapshot_id,
            confirmed_at=recorded_at,
            odds=odds,
            amount=amount,
            currency=normalized_currency,
        )
    elif accepted_snapshot_id is not None:
        snapshot = _matching_quote(store, candidate, payload, accepted_snapshot_id)
    probability = _optional_decimal(payload.get("probability"))
    point_ev = probability * odds - 1 if probability is not None else None
    classification = (
        "model_unavailable"
        if probability is None
        else "model_positive_ev"
        if point_ev is not None and point_ev > 0
        else "model_non_positive_ev"
    )
    policy_evidence = _stake_limit(
        store,
        candidate=candidate,
        payload=payload,
        fixture=fixture,
        currency=normalized_currency,
        bankroll=bankroll,
        odds=odds,
        mode=mode,
        now=recorded_at,
    )
    compliant = amount <= policy_evidence["allowed_stake_amount"]
    if mode == "paper" and not compliant:
        raise BetEvidenceError(
            f"Kelly limit is {policy_evidence['allowed_stake_amount']} "
            f"{normalized_currency}; reduce stake and reprice before confirmation."
        )
    policy_evidence["chosen_stake_compliant"] = compliant
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
        "accepted_snapshot_id": snapshot["id"] if snapshot else None,
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
        "evidence_classification": classification
        if mode == "paper"
        else "retrospective_real",
        "actor_id": actor_id,
        "idempotency_key": retry_key or bet_id,
        "payload_json": {
            "recorded_at": recorded_at.isoformat(),
            "entry_timing": "prospective" if mode == "paper" else "retrospective",
            "quote_validation": "priced_stake_verified"
            if mode == "paper"
            else "owner_recorded",
            "model_classification": classification,
            "kelly_policy": policy_evidence,
            "reason": rationale,
            "lane": lane,
            "readiness": payload.get("readiness"),
            "policy_version": payload.get("policy_version"),
            "strategy_version": payload.get("strategy_version"),
            "probability_source": payload.get("probability_source"),
            "semantic_key": payload.get("semantic_key"),
            "semantic_fingerprint": payload.get("semantic_fingerprint"),
            "sizing": payload.get("sizing"),
            "reason_codes": payload.get("reason_codes") or [],
            "correlation_group": policy_evidence["correlation_group"],
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
            "horizon_hours": horizon_hours,
            "horizon_bucket": _horizon_bucket(horizon_hours),
        },
    }


def paper_stake_limit(
    store: EvidenceStore,
    *,
    market_id: str,
    currency: str,
    bankroll_before: Decimal | str,
    accepted_odds: Decimal | str,
) -> dict[str, Any]:
    """Return a provisional paper limit; acceptance recalculates it atomically."""
    candidate = store.get(EvidenceTable.MARKET_CANDIDATES, market_id)
    if candidate is None:
        raise BetEvidenceError(f"Unknown market: {market_id}")
    return _stake_limit(
        store,
        candidate=candidate,
        payload=_payload(candidate["payload_json"]),
        fixture=store.get(EvidenceTable.FIXTURES, str(candidate["fixture_id"])),
        currency=currency.strip().upper(),
        bankroll=_positive_decimal(bankroll_before, "bankroll-before"),
        odds=_positive_decimal(accepted_odds, "accepted-odds"),
        mode="paper",
        now=_utc_now(),
    )


def _fixture_group(fixture: dict[str, Any] | None, payload: dict[str, Any]) -> str:
    if fixture:
        source_key = _payload(fixture.get("payload_json")).get("source_match_key")
        if source_key:
            return f"{fixture['sport']}:{source_key}"
    return str(
        payload.get("correlation_group") or (fixture or {}).get("id") or "unknown"
    )


def _stake_limit(
    store: EvidenceStore,
    *,
    candidate: dict[str, Any],
    payload: dict[str, Any],
    fixture: dict[str, Any] | None,
    currency: str,
    bankroll: Decimal,
    odds: Decimal,
    mode: str,
    now: datetime,
) -> dict[str, Any]:
    group = _fixture_group(fixture, payload)
    exposure: dict[str, float] = {}
    for bet in store.list(EvidenceTable.BETS):
        if bet["mode"] != mode or bet["currency"] != currency:
            continue
        if _is_superseded(store, ("bets", str(bet["id"]))):
            continue
        settlement = _settlement_event(store, str(bet["id"]))
        if settlement and _as_utc(_datetime(settlement["event_at"])) <= now:
            continue
        bet_group = _fixture_group(
            store.get(EvidenceTable.FIXTURES, str(bet["fixture_id"])),
            _payload(bet["payload_json"]),
        )
        exposure[bet_group] = exposure.get(bet_group, 0.0) + float(bet["stake_amount"])
    policy = KellyPolicy()
    probability = _optional_decimal(payload.get("probability"))
    allowed = 0.0
    if (
        probability is not None
        and 0 <= probability <= 1
        and odds > 1
        and sum(exposure.values()) <= float(bankroll)
    ):
        ticket = Opportunity(
            str(candidate["id"]),
            group,
            now,
            None,
            float(probability),
            float(odds),
            None,
        )
        allowed = allocate_portfolio(
            (ticket,),
            policy,
            equity=float(bankroll),
            open_stakes=exposure,
        )[ticket.ticket_id]
    amount = Decimal(str(allowed)).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    return {
        "name": policy.name,
        "status": "provisional_unvalidated",
        "fraction": policy.fraction,
        "ticket_cap": policy.ticket_cap,
        "fixture_cap": policy.fixture_cap,
        "total_cap": policy.total_cap,
        "correlation_group": group,
        "mode": mode,
        "currency": currency,
        "assessed_at": now.isoformat(),
        "open_stakes": exposure,
        "allowed_stake_amount": amount,
        "allowed_stake_percent": amount / bankroll * _PERCENT,
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
    closing_snapshot_id: str | None = None,
) -> str:
    """Append one owner-verified settlement; repeated identical calls are safe."""
    bet = store.get(EvidenceTable.BETS, bet_id)
    if bet is None or _is_superseded(store, ("bets", bet_id)):
        raise BetEvidenceError(f"Unknown or superseded bet: {bet_id}")
    payload = _settlement_payload(
        store,
        bet,
        result=result,
        source_reference=source_reference,
        note=note,
        closing_odds=closing_odds,
        closing_snapshot_id=closing_snapshot_id,
    )
    existing = _settlement_event(store, bet_id)
    if existing is not None:
        existing_payload = _payload(existing["payload_json"])
        if (
            all(existing_payload.get(key) == payload.get(key) for key in payload)
            and str(existing["actor_id"]) == actor_id
        ):
            return str(existing["id"])
        raise BetEvidenceError("This bet already has a conflicting settlement.")

    settled = _as_utc(settled_at or datetime.now(UTC))
    event_id = _id("bet-event", f"{bet_id}|settlement")
    store.append(
        EvidenceTable.BET_EVENTS,
        {
            "id": event_id,
            "bet_id": bet_id,
            "event_at": settled,
            "event_type": "settlement",
            "closing_snapshot_id": payload.get("closing_snapshot_id"),
            "actor_id": actor_id,
            "idempotency_key": event_id,
            "payload_json": payload,
        },
    )
    return event_id


def replace_bet_settlement(
    store: EvidenceStore,
    *,
    bet_id: str,
    result: str,
    source_reference: str,
    correction_reason: str,
    actor_id: str = "owner-cli",
    note: str | None = None,
    corrected_at: datetime | None = None,
    closing_odds: Decimal | str | None = None,
    closing_snapshot_id: str | None = None,
) -> str:
    """Supersede one mistaken settlement with a new append-only fact."""
    bet = store.get(EvidenceTable.BETS, bet_id)
    if bet is None or _is_superseded(store, ("bets", bet_id)):
        raise BetEvidenceError(f"Unknown or superseded bet: {bet_id}")
    active = _settlement_event(store, bet_id)
    if active is None:
        raise BetEvidenceError("This bet has no settlement to correct.")
    rationale = correction_reason.strip()
    if not rationale:
        raise BetEvidenceError("A settlement correction reason is required.")
    payload = _settlement_payload(
        store,
        bet,
        result=result,
        source_reference=source_reference,
        note=note,
        closing_odds=closing_odds,
        closing_snapshot_id=closing_snapshot_id,
    )
    active_payload = _payload(active["payload_json"])
    if (
        active["event_type"] == "settlement_correction"
        and all(active_payload.get(key) == payload.get(key) for key in payload)
        and active_payload.get("correction_reason") == rationale
        and active["actor_id"] == actor_id
    ):
        return str(active["id"])
    payload["correction_reason"] = rationale
    payload["corrects_event_id"] = str(active["id"])
    identity = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    replacement_id = _id("bet-event", f"{bet_id}|settlement-correction|{identity}")
    correction_id = _id(
        "correction",
        f"bet_events|{active['id']}|{replacement_id}|{rationale}|{actor_id}",
    )
    timestamp = _as_utc(corrected_at or datetime.now(UTC))
    store.append_transaction(
        (
            (
                EvidenceTable.BET_EVENTS,
                {
                    "id": replacement_id,
                    "bet_id": bet_id,
                    "event_at": timestamp,
                    "event_type": "settlement_correction",
                    "closing_snapshot_id": payload.get("closing_snapshot_id"),
                    "actor_id": actor_id,
                    "idempotency_key": replacement_id,
                    "payload_json": payload,
                },
            ),
            (
                EvidenceTable.CORRECTIONS,
                {
                    "id": correction_id,
                    "target_table": "bet_events",
                    "target_id": active["id"],
                    "created_at": timestamp,
                    "reason": rationale,
                    "replacement_id": replacement_id,
                    "idempotency_key": correction_id,
                    "payload_json": {"actor_id": actor_id},
                },
            ),
        )
    )
    return replacement_id


def record_bet_result(
    store: EvidenceStore,
    *,
    bet_id: str,
    source_reference: str,
    winning_selection: str | None = None,
    observed_value: Decimal | str | None = None,
    actor_id: str = "owner-cli",
    recorded_at: datetime | None = None,
) -> str:
    """Append one owner-sourced result fact without settling automatically."""
    bet = store.get(EvidenceTable.BETS, bet_id)
    if bet is None or _is_superseded(store, ("bets", bet_id)):
        raise BetEvidenceError(f"Unknown or superseded bet: {bet_id}")
    reference = source_reference.strip()
    if not reference:
        raise BetEvidenceError("Result source reference is required.")
    result = _derive_result(
        bet,
        winning_selection=winning_selection,
        observed_value=observed_value,
    )
    existing = _result_event(store, bet_id)
    normalized_winner = winning_selection.strip() if winning_selection else None
    normalized_value = str(observed_value) if observed_value is not None else None
    if existing is not None:
        payload = _payload(existing["payload_json"])
        if (
            payload.get("derived_result") == result.value
            and payload.get("winning_selection") == normalized_winner
            and payload.get("observed_value") == normalized_value
            and payload.get("source_reference") == reference
            and str(existing["actor_id"]) == actor_id
        ):
            return str(existing["id"])
        raise BetEvidenceError("This bet already has a conflicting result fact.")
    event_id = _id("bet-event", f"{bet_id}|result")
    store.append(
        EvidenceTable.BET_EVENTS,
        {
            "id": event_id,
            "bet_id": bet_id,
            "event_at": _as_utc(recorded_at or datetime.now(UTC)),
            "event_type": "result",
            "actor_id": actor_id,
            "idempotency_key": event_id,
            "payload_json": {
                "derived_result": result.value,
                "winning_selection": normalized_winner,
                "observed_value": normalized_value,
                "source_reference": reference,
                "owner_verified": True,
            },
        },
    )
    return event_id


def preview_bet_result(
    store: EvidenceStore,
    *,
    bet_id: str,
    winning_selection: str | None = None,
    observed_value: Decimal | str | None = None,
) -> str:
    """Derive a result for owner confirmation without writing evidence."""
    bet = store.get(EvidenceTable.BETS, bet_id)
    if bet is None or _is_superseded(store, ("bets", bet_id)):
        raise BetEvidenceError(f"Unknown or superseded bet: {bet_id}")
    return _derive_result(
        bet,
        winning_selection=winning_selection,
        observed_value=observed_value,
    ).value


def record_fixture_results(
    store: EvidenceStore,
    resolved: list[dict[str, Any]],
    *,
    source_reference: str,
    actor_id: str,
) -> list[str]:
    """Record and settle a fixture preview as one atomic ledger operation."""
    event_ids: list[str] = []
    with store.transaction() as transaction:
        for item in resolved:
            record_bet_result(
                transaction,
                bet_id=str(item["bet_id"]),
                source_reference=source_reference,
                winning_selection=item.get("winning_selection"),
                observed_value=item.get("observed_value"),
                actor_id=actor_id,
            )
            event_ids.append(
                settle_bet(
                    transaction,
                    bet_id=str(item["bet_id"]),
                    result=str(item["result"]),
                    source_reference=source_reference,
                    actor_id=actor_id,
                )
            )
    return event_ids


def supersede_evidence(
    store: EvidenceStore,
    *,
    target_table: str,
    target_id: str,
    reason: str,
    replacement_id: str | None = None,
    actor_id: str = "owner-cli",
    created_at: datetime | None = None,
) -> str:
    """Append one validated supersession without changing the original fact."""
    try:
        table = EvidenceTable(target_table)
    except ValueError as error:
        raise BetEvidenceError(
            f"Unknown correction target table: {target_table}"
        ) from error
    if table is EvidenceTable.CORRECTIONS:
        raise BetEvidenceError("Corrections cannot supersede other corrections.")
    if store.get(table, target_id) is None:
        raise BetEvidenceError(f"Unknown correction target: {target_table}/{target_id}")
    if replacement_id is not None and store.get(table, replacement_id) is None:
        raise BetEvidenceError(
            f"Unknown correction replacement: {target_table}/{replacement_id}"
        )
    rationale = reason.strip()
    if not rationale:
        raise BetEvidenceError("A supersession reason is required.")
    existing = next(
        (
            row
            for row in store.list(EvidenceTable.CORRECTIONS)
            if row["target_table"] == table.value and row["target_id"] == target_id
        ),
        None,
    )
    if existing is not None:
        payload = _payload(existing["payload_json"])
        if (
            existing["reason"] == rationale
            and existing["replacement_id"] == replacement_id
            and payload.get("actor_id") == actor_id
        ):
            return str(existing["id"])
        raise BetEvidenceError(
            "This evidence fact already has a conflicting supersession."
        )
    correction_id = _id(
        "correction",
        f"{table.value}|{target_id}|{replacement_id or ''}|{rationale}|{actor_id}",
    )
    store.append(
        EvidenceTable.CORRECTIONS,
        {
            "id": correction_id,
            "target_table": table.value,
            "target_id": target_id,
            "created_at": _as_utc(created_at or datetime.now(UTC)),
            "reason": rationale,
            "replacement_id": replacement_id,
            "idempotency_key": correction_id,
            "payload_json": {"actor_id": actor_id},
        },
    )
    return correction_id


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
    corrections = {
        (str(row["target_table"]), str(row["target_id"]))
        for row in store.list(EvidenceTable.CORRECTIONS)
    }
    settlements = {
        str(row["bet_id"]): row
        for row in store.list(EvidenceTable.BET_EVENTS)
        if row["event_type"] in {"settlement", "settlement_correction"}
        and ("bet_events", str(row["id"])) not in corrections
    }
    output: list[dict[str, Any]] = []
    for raw in reversed(store.list(EvidenceTable.BETS)):
        if ("bets", str(raw["id"])) in corrections:
            continue
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
    lane: str | None = None,
    target: str | None = None,
    provider: str | None = None,
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
    if lane:
        rows = [row for row in rows if row["payload"].get("lane") == lane]
    if target:
        rows = [row for row in rows if row["target"] == target]
    if provider:
        rows = [row for row in rows if row["provider"] == provider]
    return rows


def summarize_bets(rows: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    """Summarize already-loaded settled bets without another database read."""
    return summarize_unified_bets(rows, mode=mode)


def count_open_bets(store: EvidenceStore) -> int:
    return len(list_bets(store, state="open"))


def _bet_context(
    store: EvidenceStore,
    *,
    review_id: str,
    market_id: str,
    opened_at: datetime,
    prospective: bool,
) -> tuple[dict[str, Any], dict[str, Any], str, datetime]:
    candidate = store.get(EvidenceTable.MARKET_CANDIDATES, market_id)
    if candidate is None:
        raise BetEvidenceError(f"Unknown reviewed market: {market_id}")
    if str(candidate["run_id"]) != review_id:
        raise BetEvidenceError("The market does not belong to the supplied review.")
    review = store.get(EvidenceTable.RUNS, review_id)
    if review is None or review_state(store, review_id) not in {
        "complete",
        "completed",
    }:
        raise BetEvidenceError("The review is not completed and cannot record bets.")
    fixture = store.get(EvidenceTable.FIXTURES, str(candidate["fixture_id"]))
    if fixture is None:
        raise BetEvidenceError("The reviewed fixture is unavailable.")
    if _is_superseded(
        store,
        ("runs", review_id),
        ("fixtures", str(fixture["id"])),
        ("market_candidates", market_id),
    ):
        raise BetEvidenceError("The review or market was superseded.")
    opened = _as_utc(opened_at)
    if prospective and opened >= _as_utc(_datetime(fixture["start_time"])):
        raise BetEvidenceError("The fixture already started; the review is expired.")
    payload = _payload(candidate.get("payload_json"))
    lane = str(payload.get("classification") or "")
    if lane not in {"recommended", "exploration"}:
        raise BetEvidenceError("The reviewed market is not recordable.")
    return candidate, payload, lane, opened


def _settlement_event(store: EvidenceStore, bet_id: str) -> dict[str, Any] | None:
    corrections = {
        (str(row["target_table"]), str(row["target_id"])): row["replacement_id"]
        for row in store.list(EvidenceTable.CORRECTIONS)
    }
    event = store.get(
        EvidenceTable.BET_EVENTS, _id("bet-event", f"{bet_id}|settlement")
    )
    while event is not None:
        key = ("bet_events", str(event["id"]))
        if key not in corrections:
            break
        replacement = corrections[key]
        event = (
            store.get(EvidenceTable.BET_EVENTS, str(replacement))
            if replacement
            else None
        )
    return event


def _settlement_payload(
    store: EvidenceStore,
    bet: dict[str, Any],
    *,
    result: str,
    source_reference: str,
    note: str | None,
    closing_odds: Decimal | str | None,
    closing_snapshot_id: str | None,
) -> dict[str, Any]:
    normalized_result = result.strip().casefold()
    if normalized_result not in _RESULTS:
        raise BetEvidenceError("Result must be win, loss, push, or void.")
    reference = source_reference.strip()
    if not reference:
        raise BetEvidenceError("Settlement source reference is required.")
    entered_closing_odds = None
    if closing_odds is not None:
        entered_closing_odds = _positive_decimal(closing_odds, "closing-odds")
        if entered_closing_odds <= 1:
            raise BetEvidenceError("Closing decimal odds must be greater than 1.")
    closing = _closing_evidence(store, bet, closing_snapshot_id)
    if (
        entered_closing_odds is not None
        and closing
        and entered_closing_odds != Decimal(closing["closing_odds"])
    ):
        raise BetEvidenceError("Closing odds differ from the captured closing quote.")
    settlement_result = SettlementResult(normalized_result)
    pnl = settlement_pnl(
        PositionTerms(
            str(bet["id"]),
            Decimal(str(bet["stake_amount"])),
            Decimal(str(bet["accepted_odds"])),
        ),
        settlement_result,
    )
    return {
        "result": normalized_result,
        "source_reference": reference,
        "note": note.strip() if note else None,
        "pnl_amount": str(pnl),
        "counterfactual_pnl": _counterfactual_pnl(bet, result=settlement_result),
        "currency": bet["currency"],
        "closing_odds": closing.get("closing_odds"),
        "closing_snapshot_id": closing.get("closing_snapshot_id"),
        "clv_verified": bool(closing),
        "unverified_closing_odds": str(entered_closing_odds)
        if entered_closing_odds is not None and not closing
        else None,
        "owner_verified": True,
    }


def _result_event(store: EvidenceStore, bet_id: str) -> dict[str, Any] | None:
    event = store.get(EvidenceTable.BET_EVENTS, _id("bet-event", f"{bet_id}|result"))
    return (
        None
        if event is not None and _is_superseded(store, ("bet_events", str(event["id"])))
        else event
    )


def _derive_result(
    bet: dict[str, Any],
    *,
    winning_selection: str | None,
    observed_value: Decimal | str | None,
) -> SettlementResult:
    payload = _payload(bet.get("payload_json"))
    semantic = payload.get("semantic_key")
    target = str(bet.get("target") or "")
    if target in {"series_winner", "map_winner"}:
        if not winning_selection or not winning_selection.strip():
            raise BetEvidenceError("Winner markets require the winning selection.")
        won = winning_selection.strip().casefold() == str(bet["selection"]).casefold()
        return SettlementResult.WIN if won else SettlementResult.LOSS
    if not isinstance(semantic, dict) or semantic.get("line") is None:
        raise BetEvidenceError("This market has no controlled result semantics.")
    if observed_value is None:
        raise BetEvidenceError("This market requires the observed numeric result.")
    value = _finite_decimal(observed_value, "observed-value")
    line = Decimal(str(semantic["line"]))
    selection = str(bet["selection"]).casefold()
    if target == "series_handicap":
        margin = value + line
        if margin == 0:
            return SettlementResult.PUSH
        return SettlementResult.WIN if margin > 0 else SettlementResult.LOSS
    if value == line:
        return SettlementResult.PUSH
    if selection == "over":
        return SettlementResult.WIN if value > line else SettlementResult.LOSS
    if selection == "under":
        return SettlementResult.WIN if value < line else SettlementResult.LOSS
    raise BetEvidenceError("Numeric result selection must be Over or Under.")


def _is_superseded(store: EvidenceStore, *targets: tuple[str, str]) -> bool:
    wanted = set(targets)
    return any(
        (str(row["target_table"]), str(row["target_id"])) in wanted
        for row in store.list(EvidenceTable.CORRECTIONS)
    )


def _counterfactual_pnl(
    bet: dict[str, Any], *, result: SettlementResult
) -> dict[str, str]:
    payload = _payload(bet.get("payload_json"))
    sizing = payload.get("sizing")
    fractions = sizing.get("bankroll_fractions") if isinstance(sizing, dict) else None
    if not isinstance(fractions, dict):
        return {}
    bankroll = Decimal(str(bet["bankroll_before"]))
    odds = Decimal(str(bet["accepted_odds"]))
    output: dict[str, str] = {}
    for name in ("flat_1u", "full_kelly", "half_kelly", "quarter_kelly"):
        fraction = Decimal(str(fractions.get(name, 0)))
        stake = bankroll * fraction
        if result is SettlementResult.WIN:
            pnl = stake * (odds - 1)
        elif result is SettlementResult.LOSS:
            pnl = -stake
        else:
            pnl = Decimal(0)
        output[name] = format(pnl.normalize(), "f")
    return output


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


def _finite_decimal(value: Any, name: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise BetEvidenceError(f"{name} must be numeric.") from exc
    if not number.is_finite():
        raise BetEvidenceError(f"{name} must be finite.")
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
    try:
        return (
            value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        )
    except ValueError as error:
        raise BetEvidenceError(
            "Evidence timestamp must be a valid ISO datetime."
        ) from error


def _id(prefix: str, identity: str) -> str:
    return f"{prefix}-{hashlib.sha256(identity.encode()).hexdigest()[:24]}"


def _horizon_bucket(hours: float | None) -> str | None:
    if hours is None:
        return None
    if hours < 24:  # noqa: PLR2004
        return "under_24h"
    if hours < 72:  # noqa: PLR2004
        return "24h_72h"
    if hours < 168:  # noqa: PLR2004
        return "3d_7d"
    return "7d_plus"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _validate_bet_retry(existing: dict[str, Any], **values: Any) -> None:
    """Retry the immutable fact even after expiry, but never accept changed terms."""
    amount = (
        _positive_decimal(values["stake_amount"], "stake-amount")
        if values["stake_amount"] is not None
        else _positive_decimal(values["bankroll_before"], "bankroll-before")
        * _positive_decimal(values["stake_percent"], "stake-percent")
        / _PERCENT
    )
    text_fields = {
        "review_id": values["review_id"],
        "market_candidate_id": values["market_id"],
        "mode": values["mode"],
        "currency": values["currency"].strip().upper(),
        "actor_id": values["actor_id"],
    }
    numeric_fields = {
        "bankroll_before": values["bankroll_before"],
        "stake_percent": values["stake_percent"],
        "stake_amount": amount,
        "accepted_odds": values["accepted_odds"],
    }
    matches = (
        all(str(existing[key]) == str(value) for key, value in text_fields.items())
        and all(
            Decimal(str(existing[key])) == _positive_decimal(value, key)
            for key, value in numeric_fields.items()
        )
        and _payload(existing["payload_json"]).get("reason") == values["reason"].strip()
        and (
            values["accepted_snapshot_id"] is None
            or existing["accepted_snapshot_id"] == values["accepted_snapshot_id"]
        )
        and (
            values["opened_at"] is None
            or _as_utc(_datetime(existing["opened_at"])) == _as_utc(values["opened_at"])
        )
    )
    if not matches:
        raise BetEvidenceError(
            "This confirmation already stores different immutable bet terms."
        )


def _matching_quote(
    store: EvidenceStore,
    candidate: dict[str, Any],
    payload: dict[str, Any],
    snapshot_id: str,
) -> dict[str, Any]:
    snapshot = store.get(EvidenceTable.MARKET_SNAPSHOTS, snapshot_id)
    if snapshot is None or snapshot["market_candidate_id"] != candidate["id"]:
        raise BetEvidenceError(
            "The quote belongs to another fixture, selection, or contract."
        )
    details = _payload(snapshot["payload_json"])
    if (
        _is_superseded(store, ("market_snapshots", snapshot_id))
        or not payload.get("semantic_fingerprint")
        or details.get("semantic_fingerprint") != payload["semantic_fingerprint"]
        or details.get("terms_verified") is not True
        or details.get("complete") is not True
    ):
        raise BetEvidenceError(
            "Quote terms or completeness are unverified; capture a verified quote."
        )
    return snapshot


def _accepted_quote(
    store: EvidenceStore,
    candidate: dict[str, Any],
    payload: dict[str, Any],
    *,
    accepted_snapshot_id: str | None,
    confirmed_at: datetime,
    odds: Decimal,
    amount: Decimal,
    currency: str,
) -> dict[str, Any]:
    snapshots = [
        row
        for row in store.list(EvidenceTable.MARKET_SNAPSHOTS)
        if row["market_candidate_id"] == candidate["id"]
    ]
    if not snapshots:
        raise BetEvidenceError(
            "A captured quote is required; capture and reprice the proposal."
        )
    latest = max(
        snapshots, key=lambda row: (row["observed_at"], row["sequence_number"])
    )
    snapshot_id = accepted_snapshot_id or str(latest["id"])
    if accepted_snapshot_id is not None and snapshot_id != latest["id"]:
        # Validate identity first so a foreign selection never looks like a refresh.
        _matching_quote(store, candidate, payload, snapshot_id)
        raise BetEvidenceError(
            "The quote was replaced; capture and reprice the proposal."
        )
    snapshot = _matching_quote(store, candidate, payload, snapshot_id)
    details = _payload(snapshot["payload_json"])
    if not details.get("quote_basis"):
        raise BetEvidenceError(
            "The quote price basis is missing; recapture the proposal."
        )
    timestamps = [snapshot["observed_at"]]
    if details.get("provider_timestamp"):
        timestamps.append(details["provider_timestamp"])
    for timestamp in timestamps:
        age = (confirmed_at - _as_utc(_datetime(timestamp))).total_seconds()
        if not 0 <= age <= QUOTE_TTL_SECONDS:
            raise BetEvidenceError(
                "The quote expired; capture a new quote and reprice the proposal."
            )
    if odds != _positive_decimal(snapshot["expected_decimal_odds"], "quoted-odds"):
        raise BetEvidenceError(
            "Accepted odds differ from the captured quote; reprice the proposal."
        )
    if details.get("stake_currency") != currency or details.get("stake_amount") is None:
        raise BetEvidenceError(
            "The quote has no priced stake in this currency; recapture for this stake."
        )
    quoted_amount = _positive_decimal(details["stake_amount"], "quote-stake-amount")
    if abs(quoted_amount - amount) > Decimal("0.01"):
        raise BetEvidenceError(
            "The stake differs from the priced quote; recapture for this stake."
        )
    return snapshot


def record_closing_observation(
    store: EvidenceStore,
    *,
    bet_id: str,
    snapshot_id: str,
    source_reference: str,
    actor_id: str = "owner-cli",
) -> str:
    """Link an independently captured pre-start quote, without settling the bet."""
    bet = store.get(EvidenceTable.BETS, bet_id)
    if bet is None or _is_superseded(store, ("bets", bet_id)):
        raise BetEvidenceError(f"Unknown or superseded bet: {bet_id}")
    reference = source_reference.strip()
    if not reference:
        raise BetEvidenceError("Closing observation source reference is required.")
    details = _closing_evidence(store, bet, snapshot_id)
    event_id = _id("bet-event", f"{bet_id}|closing|{snapshot_id}")
    payload = details | {"source_reference": reference}
    existing = store.get(EvidenceTable.BET_EVENTS, event_id)
    if existing:
        if (
            _payload(existing["payload_json"]) == payload
            and existing["actor_id"] == actor_id
        ):
            return event_id
        raise BetEvidenceError(
            "This closing observation already stores different evidence."
        )
    store.append(
        EvidenceTable.BET_EVENTS,
        {
            "id": event_id,
            "bet_id": bet_id,
            "event_at": _utc_now(),
            "event_type": "closing_observation",
            "closing_snapshot_id": snapshot_id,
            "actor_id": actor_id,
            "idempotency_key": event_id,
            "payload_json": payload,
        },
    )
    return event_id


def _closing_evidence(
    store: EvidenceStore,
    bet: dict[str, Any],
    snapshot_id: str | None,
) -> dict[str, Any]:
    if snapshot_id is None:
        observations = [
            event
            for event in store.list(EvidenceTable.BET_EVENTS)
            if event["bet_id"] == bet["id"]
            and event["event_type"] == "closing_observation"
            and not _is_superseded(store, ("bet_events", str(event["id"])))
        ]
        if not observations:
            return {}
        snapshot_id = str(
            max(observations, key=lambda event: event["event_at"])[
                "closing_snapshot_id"
            ]
        )
    candidate = store.get(
        EvidenceTable.MARKET_CANDIDATES, str(bet["market_candidate_id"])
    )
    if candidate is None:
        raise BetEvidenceError("The closing market is unavailable.")
    snapshot = _matching_quote(
        store, candidate, _payload(bet["payload_json"]), snapshot_id
    )
    fixture = store.get(EvidenceTable.FIXTURES, str(bet["fixture_id"]))
    if fixture is None:
        raise BetEvidenceError("The closing fixture is unavailable.")
    observed = _as_utc(_datetime(snapshot["observed_at"]))
    opened = _as_utc(_datetime(bet["opened_at"]))
    if (
        not opened < observed < _as_utc(_datetime(fixture["start_time"]))
        or observed > _utc_now()
    ):
        raise BetEvidenceError(
            "Closing quotes must be observed after acceptance and before fixture start."
        )
    provider_timestamp = _payload(snapshot["payload_json"]).get("provider_timestamp")
    if provider_timestamp is not None:
        provider_time = _as_utc(_datetime(provider_timestamp))
        if (
            not opened < provider_time <= observed
            or (observed - provider_time).total_seconds() > QUOTE_TTL_SECONDS
        ):
            raise BetEvidenceError(
                "Closing provider timestamp is stale or outside the closing window."
            )
    odds = _positive_decimal(snapshot["expected_decimal_odds"], "closing-odds")
    if odds <= 1:
        raise BetEvidenceError("Closing decimal odds must be greater than 1.")
    # Real or legacy tickets without accepted quote proof cannot establish comparable CLV.
    if not bet.get("accepted_snapshot_id"):
        raise BetEvidenceError(
            "Comparable closing evidence requires an accepted quote reference."
        )
    accepted = _matching_quote(
        store,
        candidate,
        _payload(bet["payload_json"]),
        str(bet["accepted_snapshot_id"]),
    )
    accepted_details = _payload(accepted["payload_json"])
    closing_details = _payload(snapshot["payload_json"])
    comparable = all(
        accepted_details.get(field)
        and accepted_details[field] == closing_details.get(field)
        for field in ("quote_basis", "stake_currency")
    )
    comparable = comparable and (
        accepted_details.get("stake_amount") is not None
        and closing_details.get("stake_amount") is not None
        and _positive_decimal(accepted_details["stake_amount"], "accepted quote stake")
        == _positive_decimal(closing_details["stake_amount"], "closing quote stake")
        == _positive_decimal(bet["stake_amount"], "accepted bet stake")
        and accepted_details["stake_currency"] == bet["currency"]
        and _positive_decimal(accepted["expected_decimal_odds"], "accepted quote odds")
        == _positive_decimal(bet["accepted_odds"], "accepted bet odds")
    )
    if not comparable:
        raise BetEvidenceError(
            "Closing and accepted quotes have mismatched price bases."
        )
    return {"closing_snapshot_id": snapshot_id, "closing_odds": str(odds)}
