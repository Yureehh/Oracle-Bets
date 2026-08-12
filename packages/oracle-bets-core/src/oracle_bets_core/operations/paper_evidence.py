"""Append-only paper proposal, decision, settlement, and reporting services."""

from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.evidence.contracts import DecisionMode
from oracle_bets_core.evidence.performance import (
    SettledPerformanceRow,
    aggregate_performance,
    bootstrap_roi,
    group_performance,
    prediction_quality,
)
from oracle_bets_core.evidence.settlement import (
    MarketCloseSnapshot,
    PositionTerms,
    SettlementResult,
    calculate_clv,
    reconcile_settlement,
    select_fixed_close,
)
from oracle_bets_core.markets import walk_buy_book
from oracle_bets_core.operations.paper import (
    RESEARCH_PROP_UNITS,
    ActionGateInput,
    ActionState,
    apply_action_gate,
)

if TYPE_CHECKING:
    from oracle_bets_core.markets import OrderBookClient


class PaperEvidenceError(ValueError):
    """Raised when paper evidence is missing, ambiguous, or inconsistent."""


def paper_rows(
    store: EvidenceStore,
    *,
    state: str | None = None,
    target: str | None = None,
    league: str | None = None,
) -> list[dict[str, Any]]:
    """Return proposal/position/settlement rows with derived lifecycle state."""
    with store.connection(read_only=True) as conn:
        rows = conn.execute(
            """
            SELECT p.id AS proposal_id, p.run_id, p.state AS gate_state, p.created_at,
                   p.strategy, p.stake_units, p.payload_json,
                   f.competition_id AS league, f.start_time,
                   f.payload_json AS fixture_payload_json,
                   team_a.canonical_name AS team_a,
                   team_b.canonical_name AS team_b,
                   pr.fixture_id, pr.selection_id, pr.probability_point,
                   pr.probability_lower, pr.probability_upper,
                   pr.payload_json AS prediction_payload_json,
                   mv.target AS model_target, mv.id AS model_version_id,
                   mc.id AS market_candidate_id, mc.provider, mc.provider_market_id,
                   mc.provider_selection_id,
                   a.id AS approval_id, a.decision AS approval_decision,
                   pp.id AS position_id, pp.opened_at, pp.decimal_odds,
                   s.id AS settlement_id, s.settled_at, s.result, s.pnl_units,
                   s.payload_json AS settlement_payload_json
            FROM proposals p
            JOIN predictions pr ON pr.id = p.prediction_id
            JOIN fixtures f ON f.id = pr.fixture_id
            JOIN identities team_a ON team_a.id = f.team_a_identity_id
            JOIN identities team_b ON team_b.id = f.team_b_identity_id
            JOIN model_versions mv ON mv.id = pr.model_version_id
            LEFT JOIN approvals a ON a.proposal_id = p.id
            LEFT JOIN paper_positions pp ON pp.proposal_id = p.id
            LEFT JOIN settlements s ON s.paper_position_id = pp.id
            LEFT JOIN market_snapshots ms ON ms.id = p.market_snapshot_id
            LEFT JOIN market_candidates mc ON mc.id = ms.market_candidate_id
            WHERE (
                ? IS NULL
                OR (? = 'settled' AND s.id IS NOT NULL)
                OR (? = 'open' AND pp.id IS NOT NULL AND s.id IS NULL)
                OR (
                    ? = 'pending'
                    AND pp.id IS NULL
                    AND (a.decision IS NULL OR a.decision != 'reject')
                    AND p.state = 'paper_actionable'
                    AND NOT EXISTS (
                        SELECT 1 FROM corrections c
                        WHERE c.target_table = 'proposals' AND c.target_id = p.id
                    )
                )
            )
            AND (? IS NULL OR f.competition_id = ?)
            ORDER BY p.created_at DESC, p.id
            """,
            (state, state, state, state, league, league),
        ).fetchall()
    output: list[dict[str, Any]] = []
    for item in rows:
        row = dict(item)
        row["state"] = _paper_state(row)
        row["payload"] = json.loads(row.pop("payload_json"))
        for source, destination in (
            ("fixture_payload_json", "fixture_payload"),
            ("prediction_payload_json", "prediction_payload"),
            ("settlement_payload_json", "settlement_payload"),
        ):
            raw = row.pop(source)
            row[destination] = json.loads(raw) if raw else None
        model_target = row.pop("model_target")
        row["target"] = str(
            (row.get("prediction_payload") or {}).get("target") or model_target
        )
        if target and row["target"] != target:
            continue
        if state == "pending" and _proposal_time_expired(row):
            continue
        output.append(row)
    return output


def _proposal_time_expired(row: dict[str, Any], *, at: datetime | None = None) -> bool:
    now = at or datetime.now(UTC)
    try:
        if datetime.fromisoformat(str(row["start_time"])) <= now:
            return True
        expires_at = (row.get("payload") or {}).get("expires_at")
        return bool(expires_at and datetime.fromisoformat(str(expires_at)) <= now)
    except (TypeError, ValueError):
        return True


def paper_show(store: EvidenceStore, record_id: str) -> dict[str, Any]:
    matches = [
        row
        for row in paper_rows(store)
        if record_id in {row["proposal_id"], row["position_id"]}
    ]
    if len(matches) != 1:
        raise PaperEvidenceError(f"Unknown paper proposal or position: {record_id}")
    return matches[0]


def count_open_positions(store: EvidenceStore) -> int:
    """Count unresolved positions without materializing the evidence graph."""
    with store.connection(read_only=True) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM paper_positions pp
            LEFT JOIN settlements s ON s.paper_position_id = pp.id
            WHERE s.id IS NULL
            """
        ).fetchone()
    return int(row[0])


def quote_prop(
    store: EvidenceStore,
    *,
    forecast_id: str,
    line: float,
    over_odds: float,
    under_odds: float,
    source: str,
    created_at: datetime | None = None,
) -> str:
    """Price a manual prop line with the forecast's exact stored calibrator."""
    forecast = store.get(EvidenceTable.FORECASTS, forecast_id)
    if forecast is None:
        raise PaperEvidenceError(f"Unknown forecast: {forecast_id}")
    payload = json.loads(forecast["payload_json"])
    calibrator_uri = payload.get("calibrator_uri")
    if not calibrator_uri or not Path(calibrator_uri).is_file():
        raise PaperEvidenceError("Forecast's registered calibrator artifact is missing")
    with Path(calibrator_uri).open("rb") as handle:
        calibrator = pickle.load(handle)  # noqa: S301
    metadata = payload.get("calibration_metadata") or {}
    signal = calibrator.price(
        mean=float(forecast["point_value"]),
        line=line,
        metadata=metadata,
        over_odds=over_odds,
        under_odds=under_odds,
    )
    candidates = (
        ("over", float(signal.over_probability), over_odds, signal.over_edge),
        ("under", float(signal.under_probability), under_odds, signal.under_edge),
    )
    side, probability, odds, edge = max(candidates, key=lambda item: item[3])
    now = created_at or datetime.now(UTC)
    identity = f"{forecast_id}|{line}|{over_odds}|{under_odds}|{source}"
    prediction_id = _id("prediction", identity)
    records: list[tuple[EvidenceTable, dict[str, Any]]] = [
        (
            EvidenceTable.PREDICTIONS,
            {
                "id": prediction_id,
                "run_id": forecast["run_id"],
                "fixture_id": forecast["fixture_id"],
                "model_version_id": forecast["model_version_id"],
                "selection_id": f"{side}:{line:g}",
                "mode": "prematch",
                "created_at": now,
                "probability_point": str(probability),
                "probability_lower": str(probability),
                "probability_upper": str(probability),
                "warnings_json": ["research_only_prop", "manual_bookmaker_line"],
                "idempotency_key": prediction_id,
                "payload_json": {
                    "forecast_id": forecast_id,
                    "line": line,
                    "side": side,
                    "game_number": 1,
                },
            },
        )
    ]
    market_id = _id("market", identity)
    records.append(
        (
            EvidenceTable.MARKET_CANDIDATES,
            {
                "id": market_id,
                "run_id": forecast["run_id"],
                "fixture_id": forecast["fixture_id"],
                "provider": source,
                "provider_market_id": identity,
                "provider_selection_id": side,
                "discovered_at": now,
                "match_status": "manual_prop_line",
                "rejection_reason": None,
                "idempotency_key": market_id,
                "payload_json": {
                    "line": line,
                    "over_odds": over_odds,
                    "under_odds": under_odds,
                },
            },
        )
    )
    snapshot_id = _id("snapshot", identity)
    records.append(
        (
            EvidenceTable.MARKET_SNAPSHOTS,
            {
                "id": snapshot_id,
                "market_candidate_id": market_id,
                "observed_at": now,
                "sequence_number": 1,
                "intended_stake_units": str(RESEARCH_PROP_UNITS),
                "expected_decimal_odds": str(odds),
                "available_stake_units": str(RESEARCH_PROP_UNITS),
                "book_json": {"manual": True},
                "idempotency_key": snapshot_id,
                "payload_json": {"source": source},
            },
        )
    )
    proposal_id = _id("proposal", identity)
    records.append(
        (
            EvidenceTable.PROPOSALS,
            {
                "id": proposal_id,
                "run_id": forecast["run_id"],
                "prediction_id": prediction_id,
                "market_snapshot_id": snapshot_id,
                "created_at": now,
                "state": ActionState.RESEARCH_ONLY,
                "rejection_reason": None,
                "ruleset_version": "paper-gates-v1",
                "strategy": "fixed_research_prop",
                "stake_units": str(RESEARCH_PROP_UNITS),
                "idempotency_key": proposal_id,
                "payload_json": {
                    "edge": edge,
                    "odds": odds,
                    "forecast_id": forecast_id,
                },
            },
        )
    )
    store.append_transaction(records)
    return proposal_id


def decide_paper(
    store: EvidenceStore,
    *,
    proposal_id: str,
    decision: str,
    reason: str | None,
    actor_id: str,
    created_at: datetime | None = None,
    requote_token: str | None = None,
) -> str | None:
    """Append one idempotent owner decision and, on acceptance, one position."""
    if decision not in {"accept", "reject"}:
        raise PaperEvidenceError("decision must be accept or reject")
    proposal = store.get(EvidenceTable.PROPOSALS, proposal_id)
    if proposal is None:
        raise PaperEvidenceError(f"Unknown proposal: {proposal_id}")
    approval_id = _id("approval", proposal_id)
    prior = store.get(EvidenceTable.APPROVALS, approval_id)
    if prior:
        prior_decision = prior["decision"]
        if prior_decision != decision:
            raise PaperEvidenceError(f"Proposal already decided: {prior_decision}")
        position = store.get(
            EvidenceTable.PAPER_POSITIONS, _id("position", proposal_id)
        )
        return str(position["id"]) if position else None
    payload = json.loads(proposal["payload_json"])
    now = created_at or datetime.now(UTC)
    if decision == "accept":
        if proposal["state"] != ActionState.PAPER_ACTIONABLE:
            raise PaperEvidenceError(
                f"Proposal state cannot be accepted: {proposal['state']}"
            )
        if any(
            row["target_table"] == EvidenceTable.PROPOSALS.value
            and row["target_id"] == proposal_id
            for row in store.list(EvidenceTable.CORRECTIONS)
        ):
            raise PaperEvidenceError("Proposal has expired or been corrected")
        proposal_view = paper_show(store, proposal_id)
        if _proposal_time_expired(proposal_view, at=now):
            raise PaperEvidenceError("Proposal has expired or the fixture has started")
        if payload.get("odds") is None or proposal["stake_units"] is None:
            raise PaperEvidenceError("Proposal is missing immutable odds or stake")
        if proposal["ruleset_version"] == "independent-winner-v2":
            confirmed = _confirmed_requote(
                store, proposal_id=proposal_id, token=requote_token, now=now
            )
            payload = payload | {
                "odds": confirmed["decimal_odds"],
                "requote_token": requote_token,
            }
    records: list[tuple[EvidenceTable, dict[str, Any]]] = [
        (
            EvidenceTable.APPROVALS,
            {
                "id": approval_id,
                "proposal_id": proposal_id,
                "created_at": now,
                "actor_id": actor_id,
                "decision": decision,
                "idempotency_key": approval_id,
                "payload_json": {"reason": reason},
            },
        )
    ]
    if decision == "reject":
        store.append_transaction(records)
        return None
    odds = payload.get("odds")
    position_id = _id("position", proposal_id)
    records.append(
        (
            EvidenceTable.PAPER_POSITIONS,
            {
                "id": position_id,
                "proposal_id": proposal_id,
                "opened_at": now,
                "strategy": proposal["strategy"],
                "stake_units": proposal["stake_units"],
                "decimal_odds": str(odds),
                "state": "open",
                "idempotency_key": position_id,
                "payload_json": {
                    "approval_id": approval_id,
                    "requote_token": requote_token,
                },
            },
        )
    )
    store.append_transaction(records)
    return position_id


def requote_paper(
    store: EvidenceStore,
    *,
    proposal_id: str,
    client: OrderBookClient,
    model_healthy: bool,
    now: datetime | None = None,
    interval_seconds: int = 45,
    sleeper: Any = time.sleep,
) -> dict[str, Any]:
    """Fetch a fresh two-observation quote and rerun all deterministic gates."""
    from oracle_bets_core.markets import (
        capture_minimum_order_book_batch,
        confirmed_executable_fill,
    )

    current = now or datetime.now(UTC)
    row = paper_show(store, proposal_id)
    proposal = store.get(EvidenceTable.PROPOSALS, proposal_id)
    if proposal is None or proposal["state"] != ActionState.PAPER_ACTIONABLE:
        state = proposal["state"] if proposal is not None else "missing"
        raise PaperEvidenceError(f"Proposal state cannot be requoted: {state}")
    if any(
        correction["target_table"] == EvidenceTable.PROPOSALS.value
        and correction["target_id"] == proposal_id
        for correction in store.list(EvidenceTable.CORRECTIONS)
    ):
        raise PaperEvidenceError("Proposal has expired or been corrected")
    if _proposal_time_expired(row, at=current):
        raise PaperEvidenceError("Proposal has expired or the fixture has started")
    token_id = str(row.get("provider_selection_id") or "")
    if not token_id:
        raise PaperEvidenceError("Proposal has no verified Polymarket token")
    batch = capture_minimum_order_book_batch(
        client,
        token_ids=(token_id,),
        interval_seconds=interval_seconds,
        sleeper=sleeper,
        clock=lambda: current,
    )
    failure = batch.failures.get(token_id)
    observations = batch.observations.get(token_id)
    if failure is not None or observations is None:
        reason = failure.reason if failure is not None else "book_unavailable"
        raise PaperEvidenceError(f"Fresh quote failed: {reason}")
    fill = confirmed_executable_fill(observations)
    odds = float(fill.decimal_odds or 0.0)
    payload = row.get("payload") or {}
    probability = float(row["probability_point"])
    lower = float(row.get("probability_lower") or probability)
    start = datetime.fromisoformat(str(row["start_time"]))
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    decision = apply_action_gate(
        ActionGateInput(
            proposal_id=proposal_id,
            fixture_id=str(row.get("fixture_id") or proposal_id),
            target=str(row["target"]),
            league=str(row["league"]),
            probability=probability,
            probability_lower=lower,
            decimal_odds=odds,
            model_healthy=model_healthy,
            roster_ready=payload.get("roster_ready") is True,
            uncertainty_available=payload.get("uncertainty_available") is True,
            market_supported=True,
            quote_valid=fill.complete,
            is_model_favorite=payload.get("is_model_favorite") is True,
            hours_to_start=(start - current).total_seconds() / 3600,
            market_probability=1.0 / odds if odds > 1.0 else None,
            rating_baseline_probability=float(
                payload.get("rating_baseline_probability") or 0.5
            ),
            attribution_stable=payload.get("attribution_stable") is True,
        )
    )
    if decision.state is not ActionState.PAPER_ACTIONABLE:
        raise PaperEvidenceError(f"Fresh quote blocked: {decision.reason}")
    token = _id(
        "requote",
        f"{proposal_id}|{current.isoformat()}|{fill.total_cost}|{odds}",
    )
    event_id = _id("event", token)
    store.append(
        EvidenceTable.RUN_EVENTS,
        {
            "id": event_id,
            "run_id": row["run_id"],
            "event_at": current,
            "event_type": "paper_requote_confirmable",
            "status": "pending_owner_confirmation",
            "idempotency_key": event_id,
            "payload_json": {
                "proposal_id": proposal_id,
                "requote_token": token,
                "decimal_odds": odds,
                "conservative_edge": decision.conservative_edge,
                "expires_at": (current + timedelta(seconds=120)).isoformat(),
                "token_id": token_id,
                "read_only": True,
            },
        },
    )
    return {
        "requote_token": token,
        "decimal_odds": odds,
        "conservative_edge": decision.conservative_edge,
        "expires_at": (current + timedelta(seconds=120)).isoformat(),
    }


def _confirmed_requote(
    store: EvidenceStore,
    *,
    proposal_id: str,
    token: str | None,
    now: datetime,
) -> dict[str, Any]:
    if not token:
        raise PaperEvidenceError("Accept requires a fresh requote confirmation token")
    for event in store.list(EvidenceTable.RUN_EVENTS):
        if event["event_type"] != "paper_requote_confirmable":
            continue
        payload = json.loads(event["payload_json"])
        if payload.get("requote_token") != token:
            continue
        if payload.get("proposal_id") != proposal_id:
            raise PaperEvidenceError("Requote token belongs to another proposal")
        if datetime.fromisoformat(payload["expires_at"]) <= now:
            raise PaperEvidenceError("Requote confirmation expired")
        return payload
    raise PaperEvidenceError("Unknown requote confirmation token")


def record_map_state(
    store: EvidenceStore,
    *,
    series_id: str,
    map_number: int,
    winner: str,
    source_reference: str,
    observed_at: datetime | None = None,
) -> str:
    """Append owner-sourced completed-map state for reactive shadow research."""
    series_id = series_id.strip()
    winner = winner.strip()
    source_reference = source_reference.strip()
    if not series_id or not winner or not source_reference:
        raise PaperEvidenceError(
            "series id, winner, and source reference must be non-empty"
        )
    if map_number <= 0:
        raise PaperEvidenceError("map number must be positive")
    now = observed_at or datetime.now(UTC)
    prior_states: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for snapshot in store.list(EvidenceTable.SOURCE_SNAPSHOTS):
        if snapshot["source_type"] != "completed_map_state":
            continue
        payload = json.loads(snapshot["payload_json"])
        if payload.get("series_id") == series_id:
            prior_states.append((snapshot, payload))
    for snapshot, payload in prior_states:
        if int(payload["map_number"]) != map_number:
            continue
        if (
            payload.get("winner") == winner
            and payload.get("source_reference") == source_reference
        ):
            return str(snapshot["id"])
        raise PaperEvidenceError("completed map already has conflicting owner evidence")
    expected_map = 1 + max(
        (int(payload["map_number"]) for _, payload in prior_states), default=0
    )
    if map_number != expected_map:
        raise PaperEvidenceError(
            f"completed maps must be recorded sequentially; expected map {expected_map}"
        )
    winners = [
        str(payload["winner"])
        for _, payload in sorted(
            prior_states, key=lambda item: int(item[1]["map_number"])
        )
    ]
    winners.append(winner)
    score = {team: winners.count(team) for team in sorted(set(winners))}
    identity = f"{series_id}|{map_number}|{winner}|{source_reference}"
    run_id = _id("map-state-run", identity)
    snapshot_id = _id("map-state", identity)
    store.append_transaction(
        [
            (
                EvidenceTable.RUNS,
                {
                    "id": run_id,
                    "run_type": "manual_map_state",
                    "started_at": now,
                    "status": "completed",
                    "idempotency_key": run_id,
                    "payload_json": {"reactive_mode": "shadow_only"},
                },
            ),
            (
                EvidenceTable.SOURCE_SNAPSHOTS,
                {
                    "id": snapshot_id,
                    "run_id": run_id,
                    "provider": "owner",
                    "source_type": "completed_map_state",
                    "observed_at": now,
                    "source_uri": source_reference,
                    "schema_fingerprint": "manual-map-state-v1",
                    "idempotency_key": snapshot_id,
                    "payload_json": {
                        "series_id": series_id,
                        "map_number": map_number,
                        "winner": winner,
                        "source_reference": source_reference,
                        "score_after_map": score,
                        "next_map_number": map_number + 1,
                        "reactive_mode": "shadow_only",
                    },
                },
            ),
        ]
    )
    return snapshot_id


def expire_proposals(
    store: EvidenceStore,
    *,
    strategy_version: str,
    reason: str,
    actor_id: str = "owner-cli",
    created_at: datetime | None = None,
) -> list[str]:
    """Append invalidation corrections for matching undecided proposals."""
    strategy_version = strategy_version.strip()
    reason = reason.strip()
    actor_id = actor_id.strip()
    if not strategy_version:
        raise PaperEvidenceError("strategy version cannot be empty")
    if not reason:
        raise PaperEvidenceError("expiry reason cannot be empty")
    if not actor_id:
        raise PaperEvidenceError("expiry actor identity cannot be empty")
    approved = {str(row["proposal_id"]) for row in store.list(EvidenceTable.APPROVALS)}
    corrected = {
        str(row["target_id"])
        for row in store.list(EvidenceTable.CORRECTIONS)
        if row["target_table"] == EvidenceTable.PROPOSALS.value
    }
    now = created_at or datetime.now(UTC)
    records: list[tuple[EvidenceTable, dict[str, Any]]] = []
    expired: list[str] = []
    for proposal in store.list(EvidenceTable.PROPOSALS):
        proposal_id = str(proposal["id"])
        payload = json.loads(proposal["payload_json"])
        candidate_version = str(
            payload.get("strategy_version") or proposal.get("ruleset_version") or ""
        )
        if (
            proposal_id in approved
            or proposal_id in corrected
            or strategy_version not in {"all", candidate_version}
        ):
            continue
        correction_id = _id(
            "correction", f"proposal|{proposal_id}|expired|{strategy_version}|{reason}"
        )
        records.append(
            (
                EvidenceTable.CORRECTIONS,
                {
                    "id": correction_id,
                    "target_table": EvidenceTable.PROPOSALS.value,
                    "target_id": proposal_id,
                    "created_at": now,
                    "reason": reason,
                    "replacement_id": None,
                    "idempotency_key": correction_id,
                    "payload_json": {
                        "actor_id": actor_id,
                        "strategy_version": candidate_version,
                    },
                },
            )
        )
        expired.append(proposal_id)
    if records:
        store.append_transaction(records)
    return expired


def settle_paper(
    store: EvidenceStore,
    *,
    position_id: str,
    result: SettlementResult,
    source_reference: str,
    settled_at: datetime | None = None,
    actor_id: str = "owner-cli",
    note: str | None = None,
) -> str:
    """Append one owner-verified settlement using immutable position terms."""
    source_reference = source_reference.strip()
    actor_id = actor_id.strip()
    if not source_reference:
        raise PaperEvidenceError("settlement source reference cannot be empty")
    if not actor_id:
        raise PaperEvidenceError("settlement owner identity cannot be empty")
    position = store.get(EvidenceTable.PAPER_POSITIONS, position_id)
    if position is None:
        raise PaperEvidenceError(f"Unknown position: {position_id}")
    existing = [
        row
        for row in store.list(EvidenceTable.SETTLEMENTS)
        if row["paper_position_id"] == position_id
    ]
    if existing:
        prior = existing[0]
        payload = json.loads(prior["payload_json"])
        if (
            str(prior["result"]) == result.value
            and payload.get("source_reference") == source_reference
        ):
            return str(prior["id"])
        raise PaperEvidenceError(
            "Position is already settled with a conflicting result or source reference"
        )
    record = reconcile_settlement(
        PositionTerms(position_id, position["stake_units"], position["decimal_odds"]),
        settled_at=settled_at or datetime.now(UTC),
        provider_result=None,
        owner_result=result,
        owner_reference=source_reference,
    )
    clv_payload = _clv_payload(store, position)
    values = {
        "id": record.settlement_id,
        "paper_position_id": position_id,
        "settled_at": record.settled_at,
        "result": record.result,
        "result_source": record.source,
        "pnl_units": record.pnl_units,
        "idempotency_key": record.settlement_id,
        "payload_json": {
            "source_reference": source_reference,
            "actor_id": actor_id,
            "note": note.strip() if note and note.strip() else None,
            "entry_stake_units": str(position["stake_units"]),
            "entry_decimal_odds": str(position["decimal_odds"]),
            **clv_payload,
        },
    }
    try:
        store.append(EvidenceTable.SETTLEMENTS, values)
    except sqlite3.IntegrityError as error:
        concurrent = [
            row
            for row in store.list(EvidenceTable.SETTLEMENTS)
            if row["paper_position_id"] == position_id
        ]
        if concurrent:
            prior = concurrent[0]
            payload = json.loads(prior["payload_json"])
            if (
                str(prior["result"]) == result.value
                and payload.get("source_reference") == source_reference
            ):
                return str(prior["id"])
            raise PaperEvidenceError(
                "Position is already settled with a conflicting result or source reference"
            ) from error
        raise
    return record.settlement_id


def capture_closing_snapshots(
    store: EvidenceStore,
    *,
    client: OrderBookClient,
    now: datetime | None = None,
    window_minutes: int = 15,
    maximum_book_age_seconds: int = 120,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Capture fresh, size-aware public books shortly before fixture start."""
    observed_at = now or datetime.now(UTC)
    if observed_at.tzinfo is None or observed_at.utcoffset() != timedelta(0):
        raise PaperEvidenceError("closing snapshot clock must be timezone-aware UTC")
    if window_minutes <= 0 or maximum_book_age_seconds <= 0:
        raise PaperEvidenceError("closing window and maximum book age must be positive")

    pending: dict[str, dict[str, Any]] = {}
    for row in paper_rows(store, state="open"):
        if row.get("provider") != "polymarket":
            continue
        candidate_id = str(row.get("market_candidate_id") or "").strip()
        token_id = str(row.get("provider_selection_id") or "").strip()
        if not candidate_id or not token_id:
            continue
        start_time = _as_utc_datetime(row["start_time"])
        seconds_to_start = (start_time - observed_at).total_seconds()
        if not 0 < seconds_to_start <= window_minutes * 60:
            continue
        stake = Decimal(str(row["stake_units"]))
        item = pending.setdefault(
            candidate_id,
            {
                "candidate_id": candidate_id,
                "token_id": token_id,
                "start_time": start_time,
                "stake_units": stake,
                "position_ids": [],
            },
        )
        item["stake_units"] = max(item["stake_units"], stake)
        item["position_ids"].append(str(row["position_id"]))

    if not pending:
        return {"eligible": 0, "captured": 0, "would_capture": 0, "details": []}
    candidate_ids = tuple(pending)
    placeholders = ", ".join("?" for _ in candidate_ids)
    with store.connection(read_only=True) as conn:
        sequence_rows = conn.execute(
            f"""
            SELECT market_candidate_id, MAX(sequence_number)
            FROM market_snapshots
            WHERE market_candidate_id IN ({placeholders})
            GROUP BY market_candidate_id
            """,  # noqa: S608 - placeholders contain only generated question marks
            candidate_ids,
        ).fetchall()
    max_sequence_by_candidate = {str(row[0]): int(row[1]) for row in sequence_rows}
    details: list[dict[str, Any]] = []
    captured = 0
    for item in pending.values():
        candidate_id = item["candidate_id"]
        try:
            book, fill = _validated_closing_fill(
                client,
                item=item,
                observed_at=observed_at,
                maximum_book_age_seconds=maximum_book_age_seconds,
            )
            sequence = max_sequence_by_candidate.get(candidate_id, 0) + 1
            snapshot_id = _id(
                "snapshot",
                f"{candidate_id}|closing|{observed_at.isoformat()}",
            )
            if not dry_run:
                store.append(
                    EvidenceTable.MARKET_SNAPSHOTS,
                    {
                        "id": snapshot_id,
                        "market_candidate_id": candidate_id,
                        "observed_at": observed_at,
                        "sequence_number": sequence,
                        "intended_stake_units": str(item["stake_units"]),
                        "expected_decimal_odds": str(fill.decimal_odds),
                        # Paper units are normalized, not USDC. A complete
                        # minimum-share quote covers the recorded paper stake;
                        # the actual hypothetical cost lives in payload_json.
                        "available_stake_units": str(item["stake_units"]),
                        "book_json": {
                            "bids": [asdict(level) for level in book.bids],
                            "asks": [asdict(level) for level in book.asks],
                        },
                        "idempotency_key": snapshot_id,
                        "payload_json": {
                            "snapshot_role": "closing",
                            "book_hash": book.book_hash,
                            "complete": True,
                            "quote_basis": "minimum_order_shares",
                            "minimum_order_size": str(book.minimum_order_size),
                            "requested_shares": str(fill.requested_shares),
                            "hypothetical_cost": str(fill.total_cost),
                            "position_ids": item["position_ids"],
                            "read_only": True,
                        },
                    },
                )
            captured += 1
            details.append(
                {
                    "candidate_id": candidate_id,
                    "snapshot_id": snapshot_id,
                    "decimal_odds": fill.decimal_odds,
                    "requested_shares": str(fill.requested_shares),
                    "hypothetical_cost": str(fill.total_cost),
                    "status": "would_capture" if dry_run else "captured",
                }
            )
        except Exception as error:
            details.append(
                {
                    "candidate_id": candidate_id,
                    "status": "skipped",
                    "reason": f"{type(error).__name__}:{error}",
                }
            )
    return {
        "eligible": len(pending),
        "captured": 0 if dry_run else captured,
        "would_capture": captured if dry_run else 0,
        "details": details,
    }


def _validated_closing_fill(
    client: OrderBookClient,
    *,
    item: dict[str, Any],
    observed_at: datetime,
    maximum_book_age_seconds: int,
) -> tuple[Any, Any]:
    book = client.get_order_book(item["token_id"])
    if (
        book.timestamp is None
        or abs((observed_at - book.timestamp).total_seconds())
        > maximum_book_age_seconds
    ):
        raise PaperEvidenceError("CLOB order book is stale or lacks a timestamp")
    if observed_at >= item["start_time"]:
        raise PaperEvidenceError("fixture started before closing capture completed")
    fill = walk_buy_book(book, book.minimum_order_size)
    if not fill.complete or fill.decimal_odds is None:
        raise PaperEvidenceError("closing book lacks complete intended depth")
    return book, fill


def _clv_payload(
    store: EvidenceStore,
    position: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the latest complete pre-start closing quote for one position."""
    proposal = store.get(EvidenceTable.PROPOSALS, str(position["proposal_id"]))
    entry_snapshot = (
        store.get(EvidenceTable.MARKET_SNAPSHOTS, proposal["market_snapshot_id"])
        if proposal and proposal.get("market_snapshot_id")
        else None
    )
    prediction = (
        store.get(EvidenceTable.PREDICTIONS, str(proposal["prediction_id"]))
        if proposal
        else None
    )
    fixture = (
        store.get(EvidenceTable.FIXTURES, str(prediction["fixture_id"]))
        if prediction
        else None
    )
    candidate = (
        store.get(
            EvidenceTable.MARKET_CANDIDATES,
            str(entry_snapshot["market_candidate_id"]),
        )
        if entry_snapshot
        else None
    )
    closes: list[MarketCloseSnapshot] = []
    if entry_snapshot and candidate:
        for snapshot in store.list(EvidenceTable.MARKET_SNAPSHOTS):
            if snapshot["market_candidate_id"] != entry_snapshot["market_candidate_id"]:
                continue
            payload = json.loads(snapshot["payload_json"])
            if (
                payload.get("snapshot_role") != "closing"
                or payload.get("complete") is not True
                or snapshot.get("expected_decimal_odds") is None
            ):
                continue
            closes.append(
                MarketCloseSnapshot(
                    snapshot_id=str(snapshot["id"]),
                    observed_at=_as_utc_datetime(snapshot["observed_at"]),
                    decimal_odds=snapshot["expected_decimal_odds"],
                    available_stake_units=snapshot["available_stake_units"] or "0",
                    source=str(candidate["provider"]),
                    warnings=tuple(payload.get("warnings") or ()),
                )
            )
    close = (
        select_fixed_close(
            closes,
            event_start=_as_utc_datetime(fixture["start_time"]),
            intended_stake_units=position["stake_units"],
        )
        if fixture
        else None
    )
    value = calculate_clv(entry_decimal_odds=position["decimal_odds"], close=close)
    return {
        "closing_line": asdict(value),
        "probability_clv": value.probability_clv,
        "odds_ratio_clv": value.odds_ratio_clv,
    }


def _as_utc_datetime(value: Any) -> datetime:
    parsed = (
        value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    )
    if parsed.tzinfo is None:
        raise PaperEvidenceError("evidence timestamp must include a timezone")
    return parsed.astimezone(UTC)


def performance_summary(
    store: EvidenceStore,
    *,
    since: datetime | None = None,
    target: str | None = None,
    league: str | None = None,
) -> dict[str, Any]:
    if since is not None and since.tzinfo is None:
        raise PaperEvidenceError("--since must include a timezone")
    rows = []
    evidence_rows = paper_rows(
        store,
        state="settled",
        target=target,
        league=league,
    )
    if since is not None:
        evidence_rows = [
            item
            for item in evidence_rows
            if datetime.fromisoformat(str(item["settled_at"])) >= since
        ]
    actuals: list[int] = []
    probabilities: list[float] = []
    for index, item in enumerate(reversed(evidence_rows)):
        settlement_payload = item.get("settlement_payload") or {}
        probability_clv = settlement_payload.get("probability_clv")
        rows.append(
            SettledPerformanceRow(
                position_id=item["position_id"],
                settled_order=index,
                stake_units=Decimal(item["stake_units"]),
                pnl_units=Decimal(item["pnl_units"]),
                result=SettlementResult(item["result"]),
                league=item["league"],
                market=item["target"],
                strategy=item["strategy"],
                model_version=item["model_version_id"],
                mode=DecisionMode.PREMATCH,
                edge_band="recorded",
                probability_clv=(
                    float(probability_clv) if probability_clv is not None else None
                ),
            )
        )
        if item["result"] in {SettlementResult.WIN, SettlementResult.LOSS}:
            actuals.append(int(item["result"] == SettlementResult.WIN))
            probabilities.append(float(item["probability_point"]))
    metrics = aggregate_performance(rows)
    interval = bootstrap_roi(rows) if rows else None
    quality = None
    if actuals:
        quality = asdict(prediction_quality(actuals, probabilities))
    return {
        **asdict(metrics),
        "roi_interval": asdict(interval) if interval else None,
        "prediction_quality": quality,
        "by_target": {
            key: asdict(value)
            for key, value in group_performance(
                rows, key=lambda row: row.market
            ).items()
        },
        "by_league": {
            key: asdict(value)
            for key, value in group_performance(
                rows, key=lambda row: row.league
            ).items()
        },
    }


def _id(prefix: str, identity: str) -> str:
    return f"{prefix}-{hashlib.sha256(identity.encode()).hexdigest()[:24]}"


def _paper_state(row: dict[str, Any]) -> str:
    if row["settlement_id"]:
        return "settled"
    if row["position_id"]:
        return "open"
    if row["approval_decision"] == "reject":
        return "rejected"
    return "pending"
