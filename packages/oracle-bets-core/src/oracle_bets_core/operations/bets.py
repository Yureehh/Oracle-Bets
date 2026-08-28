"""Owner-recorded paper and real bets; never bookmaker transactions."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.evidence.performance import summarize_unified_bets
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
    enrolled_at: datetime | None = None,
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
            "started_at": _as_utc(enrolled_at or datetime.now(UTC)),
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
    for event in store.list(EvidenceTable.RUN_EVENTS):
        if event["run_id"] != cohort_id:
            continue
        stage = str(event["event_type"]).removeprefix("cohort_")
        details = _payload(event["payload_json"])
        fixture_id = str(details.get("fixture_id") or "")
        opportunity_id = str(details.get("opportunity_id") or "")
        if stage in {"reviewed", "no_market"}:
            reviewed.add(fixture_id)
        if stage == "recommended":
            recommendations.add((fixture_id, opportunity_id))
        elif stage == "shadow_result":
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
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Validate and build an immutable bet entry without writing it."""
    candidate, payload, lane, opened = _bet_context(
        store,
        review_id=review_id,
        market_id=market_id,
        opened_at=opened_at,
    )
    fixture = store.get(EvidenceTable.FIXTURES, str(candidate["fixture_id"]))
    horizon_hours = (
        (_as_utc(_datetime(fixture["start_time"])) - opened).total_seconds() / 3600
        if fixture
        else None
    )
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

    probability = _optional_decimal(payload.get("probability"))
    point_ev = probability * odds - 1 if probability is not None else None
    classification = (
        "model_unavailable"
        if probability is None
        else "model_positive_ev"
        if point_ev is not None and point_ev > 0
        else "model_non_positive_ev"
    )
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
            "lane": lane,
            "readiness": payload.get("readiness"),
            "policy_version": payload.get("policy_version"),
            "strategy_version": payload.get("strategy_version"),
            "probability_source": payload.get("probability_source"),
            "semantic_key": payload.get("semantic_key"),
            "semantic_fingerprint": payload.get("semantic_fingerprint"),
            "sizing": payload.get("sizing"),
            "reason_codes": payload.get("reason_codes") or [],
            "correlation_group": payload.get("correlation_group")
            or candidate["fixture_id"],
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
    if bet is None or _is_superseded(store, ("bets", bet_id)):
        raise BetEvidenceError(f"Unknown or superseded bet: {bet_id}")
    payload = _settlement_payload(
        bet,
        result=result,
        source_reference=source_reference,
        note=note,
        closing_odds=closing_odds,
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
        bet,
        result=result,
        source_reference=source_reference,
        note=note,
        closing_odds=closing_odds,
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
    opened_at: datetime | None,
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
    opened = _as_utc(opened_at or datetime.now(UTC))
    if opened >= _as_utc(_datetime(fixture["start_time"])):
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
    bet: dict[str, Any],
    *,
    result: str,
    source_reference: str,
    note: str | None,
    closing_odds: Decimal | str | None,
) -> dict[str, Any]:
    normalized_result = result.strip().casefold()
    if normalized_result not in _RESULTS:
        raise BetEvidenceError("Result must be win, loss, push, or void.")
    reference = source_reference.strip()
    if not reference:
        raise BetEvidenceError("Settlement source reference is required.")
    normalized_closing_odds = (
        str(_positive_decimal(closing_odds, "closing-odds"))
        if closing_odds is not None
        else None
    )
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
        "closing_odds": normalized_closing_odds,
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
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))


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
