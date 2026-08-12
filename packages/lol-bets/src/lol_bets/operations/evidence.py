"""Persist daily LoL outputs into the canonical evidence graph."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.evidence.identity import EntityType, canonical_identity_id
from oracle_bets_core.operations import daily_run_key
from oracle_bets_core.operations.paper import RESEARCH_PROP_TARGETS
from oracle_bets_core.paths import (
    GAMELENGTH_PREDICTION_MODEL_PATH,
    GAMELENGTH_PREDICTION_PROP_CALIBRATOR,
    MODEL_REGISTRY_DIR,
    MODELS_DIR,
    OUTCOME_PREDICTION_MODEL_PATH,
    SERIES_WINNER_MODEL_PATH,
    SERIES_WINNER_PROBABILITY_CALIBRATOR,
    TOTAL_KILLS_PREDICTION_MODEL_PATH,
    TOTAL_KILLS_PREDICTION_PROP_CALIBRATOR,
    TOTAL_TOWERS_PREDICTION_MODEL_PATH,
    TOTAL_TOWERS_PREDICTION_PROP_CALIBRATOR,
)
from oracle_bets_core.pd import pd

from lol_bets.operations.identity import sync_schedule_identity_graph
from lol_bets.operations.models import ModelRegistry

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def record_daily_evidence(
    *,
    store: EvidenceStore,
    scheduled_for: datetime,
    effective_config: dict[str, Any],
    schedule: pd.DataFrame,
    snapshot_rows: Sequence[dict[str, Any]],
    steps: Sequence[Any],
    market_actions: Sequence[dict[str, Any]] | None = None,
) -> str:
    """Record predictions and honest no-bet outcomes without upgrading prices."""
    store.initialize_schema()
    run_key = daily_run_key(
        "lol",
        scheduled_for=scheduled_for,
        config=effective_config,
    )
    run_id = _id("run", run_key)
    _append_if_missing(
        store,
        EvidenceTable.RUNS,
        {
            "id": run_id,
            "run_type": "daily_lol",
            "started_at": scheduled_for,
            "status": "completed" if all(step.ok for step in steps) else "partial",
            "idempotency_key": run_key,
            "payload_json": {
                "run_key": run_key,
                "steps": [
                    {"name": step.name, "ok": step.ok, "detail": step.detail}
                    for step in steps
                ],
            },
        },
    )
    _record_daily_step_events(
        store,
        run_id=run_id,
        scheduled_for=scheduled_for,
        steps=steps,
    )
    sync_schedule_identity_graph(
        store,
        schedule,
        observed_at=scheduled_for,
    )
    _record_schedule_source(store, run_id, scheduled_for, schedule)
    fixtures, fixture_supersessions = _record_fixtures(
        store,
        run_id,
        schedule,
        observed_at=scheduled_for,
    )
    model_ids = {
        target: _record_model(store, scheduled_for, target=target)
        for target in (
            "map_win",
            "series_winner",
            "gamelength",
            "total_kills",
            "total_towers",
        )
    }
    for row in snapshot_rows:
        if row.get("market") not in {"winner", "series_winner"}:
            _record_scalar_forecast(
                store,
                run_id=run_id,
                fixture_id=fixtures.get(_fixture_lookup_key(row)),
                model_ids=model_ids,
                row=row,
            )
            continue
        fixture_id = fixtures.get(_fixture_lookup_key(row))
        if fixture_id is None:
            continue
        selection_id = _identity_id(str(row["selection"]))
        point = float(row["model_value"])
        lower = float(row.get("probability_lower", point))
        upper = float(row.get("probability_upper", point))
        prediction_id = _id(
            "prediction",
            f"{run_id}|{fixture_id}|{selection_id}|prematch",
        )
        prediction_warnings: list[str] = []
        if not row.get("uncertainty_method") or (lower == point and upper == point):
            prediction_warnings.append("uncertainty_interval_unavailable")
        if not row.get("drivers"):
            prediction_warnings.append("local_drivers_unavailable")
        if row.get("lineup_ready") is not True:
            prediction_warnings.append("roster_unknown")
        _append_if_missing(
            store,
            EvidenceTable.PREDICTIONS,
            {
                "id": prediction_id,
                "run_id": run_id,
                "fixture_id": fixture_id,
                "model_version_id": model_ids[
                    "series_winner"
                    if row.get("market") == "series_winner"
                    else "map_win"
                ],
                "selection_id": selection_id,
                "mode": "prematch",
                "created_at": _as_utc(row["run_ts"]),
                "probability_point": str(point),
                "probability_lower": str(lower),
                "probability_upper": str(upper),
                "warnings_json": prediction_warnings,
                "idempotency_key": prediction_id,
                "payload_json": {
                    "probability_source": row.get("probability_source"),
                    "uncertainty_method": row.get("uncertainty_method"),
                    "uncertainty_confidence": row.get("uncertainty_confidence"),
                    "uncertainty_sample_count": row.get("uncertainty_sample_count"),
                    "drivers": list(row.get("drivers") or []),
                    "paired_distribution": row.get("paired_distribution"),
                    "display_market_price": row.get("poly_price"),
                    "lineup_ready": row.get("lineup_ready") is True,
                    "roster_ready": row.get("roster_ready") is True,
                    "team_a_roster_state": row.get("team_a_roster_state"),
                    "team_b_roster_state": row.get("team_b_roster_state"),
                    "team_a_roster_version": row.get("team_a_roster_version"),
                    "team_b_roster_version": row.get("team_b_roster_version"),
                    "team_a_completed_roster_series": row.get(
                        "team_a_completed_roster_series"
                    ),
                    "team_b_completed_roster_series": row.get(
                        "team_b_completed_roster_series"
                    ),
                },
            },
        )
        if market_actions is not None:
            continue
        market_candidate_id = _record_display_market(
            store,
            run_id=run_id,
            fixture_id=fixture_id,
            row=row,
        )
        if row.get("lineup_ready") is not True:
            reason = "roster_unknown"
        elif row.get("roster_ready") is not True:
            reason = "roster_unstable"
        elif market_candidate_id:
            reason = "executable_order_book_unavailable"
        else:
            reason = "market_not_found"
        proposal_id = _id("proposal", prediction_id)
        _append_if_missing(
            store,
            EvidenceTable.PROPOSALS,
            {
                "id": proposal_id,
                "run_id": run_id,
                "prediction_id": prediction_id,
                "market_snapshot_id": None,
                "created_at": _as_utc(row["run_ts"]),
                "state": "rejected",
                "rejection_reason": reason,
                "ruleset_version": "legacy-daily-bridge-v1",
                "strategy": None,
                "stake_units": None,
                "idempotency_key": proposal_id,
                "payload_json": {
                    "market_candidate_id": market_candidate_id,
                    "reason": (
                        "Gamma display prices are research context, not an "
                        "executable two-observation CLOB price."
                    ),
                },
            },
        )
    if market_actions is not None:
        _record_typed_market_actions(
            store,
            run_id=run_id,
            fixtures=fixtures,
            schedule=schedule,
            model_id=model_ids["series_winner"],
            actions=market_actions,
            created_at=scheduled_for,
        )
    _supersede_fixture_dependents(
        store,
        fixture_supersessions,
        created_at=scheduled_for,
    )
    return run_id


def _record_typed_market_actions(
    store: EvidenceStore,
    *,
    run_id: str,
    fixtures: dict[str, str],
    schedule: pd.DataFrame,
    model_id: str,
    actions: Sequence[dict[str, Any]],
    created_at: datetime,
) -> None:
    fixture_ids = {
        _optional_text(row.get("match_key")): fixtures.get(_fixture_lookup_key(row))
        for _, row in schedule.iterrows()
    }
    fixture_starts = {
        _optional_text(row.get("match_key")): row.get("start_utc")
        for _, row in schedule.iterrows()
    }
    for action in actions:
        if not action.get("token_id"):
            continue
        fixture_id = fixture_ids.get(_optional_text(action.get("fixture_key")))
        if fixture_id is None:
            continue
        proposal_id = str(action["proposal_id"])
        prediction_id = _id("prediction", proposal_id)
        point = float(action["probability"])
        lower = float(action["probability_lower"])
        records: list[tuple[EvidenceTable, dict[str, Any]]] = [
            (
                EvidenceTable.PREDICTIONS,
                {
                    "id": prediction_id,
                    "run_id": run_id,
                    "fixture_id": fixture_id,
                    "model_version_id": model_id,
                    "selection_id": (
                        f"{action['target']}:{action.get('game_number') or ''}:"
                        f"{action.get('total_line') or ''}:{action['selection']}"
                    ),
                    "mode": "prematch",
                    "created_at": created_at,
                    "probability_point": str(point),
                    "probability_lower": str(lower),
                    "probability_upper": str(point),
                    "warnings_json": list(action.get("warnings") or []),
                    "idempotency_key": prediction_id,
                    "payload_json": {
                        "target": action["target"],
                        "game_number": action.get("game_number"),
                        "total_line": action.get("total_line"),
                        "selection": action["selection"],
                        "source": "direct_series_winner_v2",
                    },
                },
            )
        ]
        candidate_id = _id(
            "market",
            f"{run_id}|{action['market_id']}|{action['token_id']}",
        )
        records.append(
            (
                EvidenceTable.MARKET_CANDIDATES,
                {
                    "id": candidate_id,
                    "run_id": run_id,
                    "fixture_id": fixture_id,
                    "provider": "polymarket",
                    "provider_market_id": action["market_id"],
                    "provider_selection_id": action["token_id"],
                    "discovered_at": created_at,
                    "match_status": "typed_exact",
                    "rejection_reason": None,
                    "idempotency_key": candidate_id,
                    "payload_json": {
                        "target": action["target"],
                        "selection": action["selection"],
                        "url": action.get("market_url"),
                        "resolution_source": action.get("resolution_source"),
                    },
                },
            )
        )
        snapshot_ids: list[tuple[str, float]] = []
        for observation in action["observations"]:
            snapshot_id = _id(
                "snapshot",
                f"{candidate_id}|{observation['sequence_number']}|"
                f"{observation['observed_at']}|{observation['book_hash']}",
            )
            observation_odds = float(observation["decimal_odds"] or 0.0)
            snapshot_ids.append((snapshot_id, observation_odds))
            records.append(
                (
                    EvidenceTable.MARKET_SNAPSHOTS,
                    {
                        "id": snapshot_id,
                        "market_candidate_id": candidate_id,
                        "observed_at": _as_utc(observation["observed_at"]),
                        "sequence_number": int(observation["sequence_number"]),
                        "intended_stake_units": str(action.get("stake_units") or 0.0),
                        "expected_decimal_odds": str(observation_odds),
                        # Paper units are normalized, not USDC; exact quote cost
                        # remains in the snapshot payload.
                        "available_stake_units": str(action.get("stake_units") or 0.0),
                        "book_json": observation["book"],
                        "idempotency_key": snapshot_id,
                        "payload_json": {
                            "book_hash": observation["book_hash"],
                            "complete": observation["complete"],
                            "quote_basis": "minimum_order_shares",
                            "minimum_order_size": observation.get("minimum_order_size"),
                            "requested_shares": observation.get("requested_shares"),
                            "filled_shares": observation.get("filled_shares"),
                            "hypothetical_cost": observation.get("hypothetical_cost"),
                        },
                    },
                )
            )
        worse_snapshot_id = (
            min(snapshot_ids, key=lambda item: item[1])[0] if snapshot_ids else None
        )
        state = str(action["state"])
        if state == "paper_actionable":
            records.append(
                (
                    EvidenceTable.PROPOSALS,
                    {
                        "id": proposal_id,
                        "run_id": run_id,
                        "prediction_id": prediction_id,
                        "market_snapshot_id": worse_snapshot_id,
                        "created_at": created_at,
                        "state": state,
                        "rejection_reason": action.get("reason"),
                        "ruleset_version": "independent-winner-v2",
                        "strategy": "flat_one_unit_series_favorite",
                        "stake_units": "1.0",
                        "idempotency_key": proposal_id,
                        "payload_json": {
                            "odds": action.get("decimal_odds"),
                            "conservative_edge": action.get("conservative_edge"),
                            "strategy_version": "independent-winner-v2",
                            "rating_baseline_probability": action.get(
                                "rating_baseline_probability"
                            ),
                            "full_model_probability": action.get(
                                "full_model_probability"
                            ),
                            "roster_ready": action.get("roster_ready") is True,
                            "uncertainty_available": (
                                action.get("uncertainty_available") is True
                            ),
                            "attribution_stable": (
                                action.get("attribution_stable") is True
                            ),
                            "is_model_favorite": (
                                action.get("is_model_favorite") is True
                            ),
                            "counterfactual_quarter_kelly_units": action.get(
                                "counterfactual_quarter_kelly_units"
                            ),
                            "expires_at": (
                                _as_utc(
                                    action.get("start_time")
                                    or fixture_starts.get(
                                        _optional_text(action.get("fixture_key"))
                                    )
                                )
                                - timedelta(hours=24)
                            ).isoformat(),
                            "read_only": True,
                            "resolution_source": action.get("resolution_source"),
                            "provider_selection_id": action.get("token_id"),
                        },
                    },
                )
            )
        # Predictions, candidates, and proposals are the canonical owner-decision
        # graph for this daily run. A same-day rerun may observe different books;
        # retain the original decision terms while appending its new snapshots.
        new_records = [
            (table, values)
            for table, values in records
            if table is EvidenceTable.MARKET_SNAPSHOTS
            or store.get(table, str(values["id"])) is None
        ]
        store.append_transaction(new_records)


def _record_daily_step_events(
    store: EvidenceStore,
    *,
    run_id: str,
    scheduled_for: datetime,
    steps: Sequence[Any],
) -> None:
    for step in steps:
        detail = str(step.detail)
        if not step.ok:
            status = "failed"
        elif detail.startswith("skipped"):
            status = "skipped"
        else:
            status = "completed"
        identity = f"{run_id}|{step.name}|{status}|{detail}"
        event_id = _id("run-event", identity)
        _append_if_missing(
            store,
            EvidenceTable.RUN_EVENTS,
            {
                "id": event_id,
                "run_id": run_id,
                "event_at": scheduled_for,
                "event_type": "daily_step",
                "status": status,
                "idempotency_key": event_id,
                "payload_json": {
                    "step": step.name,
                    "detail": detail,
                },
            },
        )


def _record_schedule_source(
    store: EvidenceStore,
    run_id: str,
    observed_at: datetime,
    schedule: pd.DataFrame,
) -> None:
    columns = sorted(str(column) for column in schedule.columns)
    fingerprint = hashlib.sha256("|".join(columns).encode()).hexdigest()
    fixture_observations = [
        {
            "source_match_key": _optional_text(row.get("match_key")),
            "fixture_version": _optional_text(row.get("fixture_version")),
            "lineup_source": _optional_text(row.get("lineup_source")),
            "lineup_observed_at": _optional_text(row.get("lineup_observed_at")),
            "lineup_refresh_error": _optional_text(row.get("lineup_refresh_error")),
        }
        for _, row in schedule.iterrows()
    ]
    content_fingerprint = hashlib.sha256(
        json.dumps(
            fixture_observations,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    source_id = _id("source", f"{run_id}|{content_fingerprint}")
    _append_if_missing(
        store,
        EvidenceTable.SOURCE_SNAPSHOTS,
        {
            "id": source_id,
            "run_id": run_id,
            "provider": "pandascore",
            "source_type": "future_fixtures",
            "observed_at": observed_at,
            "source_uri": "pandascore://lol/matches",
            "schema_fingerprint": fingerprint,
            "idempotency_key": source_id,
            "payload_json": {
                "row_count": len(schedule),
                "columns": columns,
                "content_fingerprint": content_fingerprint,
                "fixtures": fixture_observations,
            },
        },
    )


def _record_fixtures(
    store: EvidenceStore,
    run_id: str,
    schedule: pd.DataFrame,
    *,
    observed_at: datetime,
) -> tuple[dict[str, str], dict[str, str]]:
    fixtures: dict[str, str] = {}
    supersessions: dict[str, str] = {}
    prior_fixtures = list(store.list(EvidenceTable.FIXTURES))
    corrected_fixture_ids = _corrected_target_ids(store, EvidenceTable.FIXTURES)
    for _, row in schedule.iterrows():
        team_a = str(row.get("team_a") or "").strip()
        team_b = str(row.get("team_b") or "").strip()
        if not team_a or not team_b:
            continue
        for name in (team_a, team_b):
            identity_id = _identity_id(name)
            _append_if_missing(
                store,
                EvidenceTable.IDENTITIES,
                {
                    "id": identity_id,
                    "entity_type": "team",
                    "canonical_name": name,
                    "created_at": _as_utc(row["start_utc"]),
                    "idempotency_key": f"lol-team:{name.casefold()}",
                    "payload_json": {},
                },
            )
        lookup = _fixture_lookup_key(row)
        source_match_key = _optional_text(row.get("match_key"))
        version = _optional_text(row.get("fixture_version")) or _id(
            "fixture-version", lookup
        )
        fixture_id = _id("fixture", f"{lookup}|{version}")
        _append_if_missing(
            store,
            EvidenceTable.FIXTURES,
            {
                "id": fixture_id,
                "run_id": run_id,
                "sport": "lol",
                "competition_id": str(row.get("league") or "unknown"),
                "team_a_identity_id": _identity_id(team_a),
                "team_b_identity_id": _identity_id(team_b),
                "start_time": _as_utc(row["start_utc"]),
                "best_of": int(row.get("best_of") or 1),
                "status": str(row.get("status") or "scheduled"),
                "idempotency_key": f"lol:{lookup}:{version}",
                "payload_json": {
                    "source_match_key": source_match_key,
                    "provider_match_id": _optional_text(row.get("provider_match_id")),
                    "team_a_provider_id": _optional_text(row.get("team_a_id")),
                    "team_b_provider_id": _optional_text(row.get("team_b_id")),
                    "series_identity_id": _schedule_series_identity_id(row),
                    "fixture_version": version,
                    "lineup_source": _optional_text(row.get("lineup_source")),
                    "lineup_observed_at": _optional_text(row.get("lineup_observed_at")),
                    "lineup_refresh_error": _optional_text(
                        row.get("lineup_refresh_error")
                    ),
                    "team_a_lineup_json": _json_or_text(row.get("team_a_lineup_json")),
                    "team_b_lineup_json": _json_or_text(row.get("team_b_lineup_json")),
                },
            },
        )
        fixtures[lookup] = fixture_id
        if source_match_key:
            for prior in prior_fixtures:
                prior_id = str(prior["id"])
                if (
                    prior_id == fixture_id
                    or prior_id in corrected_fixture_ids
                    or _fixture_source_key(prior) != source_match_key
                ):
                    continue
                supersessions[prior_id] = fixture_id
                _append_correction(
                    store,
                    target_table=EvidenceTable.FIXTURES,
                    target_id=prior_id,
                    replacement_id=fixture_id,
                    created_at=observed_at,
                    reason="provider_fixture_changed",
                    payload={
                        "source_match_key": source_match_key,
                        "replacement_fixture_version": version,
                    },
                )
    return fixtures, supersessions


def _supersede_fixture_dependents(
    store: EvidenceStore,
    fixture_supersessions: dict[str, str],
    *,
    created_at: datetime,
) -> None:
    if not fixture_supersessions:
        return
    predictions = list(store.list(EvidenceTable.PREDICTIONS))
    proposals = list(store.list(EvidenceTable.PROPOSALS))
    candidates = list(store.list(EvidenceTable.MARKET_CANDIDATES))
    approvals = list(store.list(EvidenceTable.APPROVALS))
    positions = list(store.list(EvidenceTable.PAPER_POSITIONS))
    corrected = {
        table: _corrected_target_ids(store, table)
        for table in (
            EvidenceTable.PREDICTIONS,
            EvidenceTable.MARKET_CANDIDATES,
            EvidenceTable.PROPOSALS,
            EvidenceTable.APPROVALS,
            EvidenceTable.PAPER_POSITIONS,
        )
    }
    for old_fixture_id, new_fixture_id in fixture_supersessions.items():
        prediction_replacements = _supersede_predictions(
            store,
            predictions,
            old_fixture_id=old_fixture_id,
            new_fixture_id=new_fixture_id,
            created_at=created_at,
            corrected_ids=corrected[EvidenceTable.PREDICTIONS],
        )
        _supersede_market_candidates(
            store,
            candidates,
            old_fixture_id=old_fixture_id,
            new_fixture_id=new_fixture_id,
            created_at=created_at,
            corrected_ids=corrected[EvidenceTable.MARKET_CANDIDATES],
        )
        proposal_replacements = _supersede_proposals(
            store,
            proposals,
            prediction_replacements=prediction_replacements,
            created_at=created_at,
            corrected_ids=corrected[EvidenceTable.PROPOSALS],
        )
        _supersede_approval_and_position_records(
            store,
            approvals=approvals,
            positions=positions,
            proposal_replacements=proposal_replacements,
            created_at=created_at,
            corrected_approval_ids=corrected[EvidenceTable.APPROVALS],
            corrected_position_ids=corrected[EvidenceTable.PAPER_POSITIONS],
        )


def _supersede_predictions(
    store: EvidenceStore,
    predictions: list[dict[str, Any]],
    *,
    old_fixture_id: str,
    new_fixture_id: str,
    created_at: datetime,
    corrected_ids: set[str],
) -> dict[str, str | None]:
    new_by_key = {
        (str(row["selection_id"]), str(row["mode"])): str(row["id"])
        for row in predictions
        if row["fixture_id"] == new_fixture_id
    }
    replacements: dict[str, str | None] = {}
    for row in predictions:
        old_id = str(row["id"])
        if row["fixture_id"] != old_fixture_id or old_id in corrected_ids:
            continue
        replacement = new_by_key.get((str(row["selection_id"]), str(row["mode"])))
        replacements[old_id] = replacement
        _append_correction(
            store,
            target_table=EvidenceTable.PREDICTIONS,
            target_id=old_id,
            replacement_id=replacement,
            created_at=created_at,
            reason="fixture_change_invalidated_prediction",
            payload={
                "old_fixture_id": old_fixture_id,
                "new_fixture_id": new_fixture_id,
            },
        )
    return replacements


def _supersede_market_candidates(
    store: EvidenceStore,
    candidates: list[dict[str, Any]],
    *,
    old_fixture_id: str,
    new_fixture_id: str,
    created_at: datetime,
    corrected_ids: set[str],
) -> None:
    new_by_key = {
        (
            str(row["provider"]),
            str(row["provider_market_id"]),
            str(row["provider_selection_id"]),
        ): str(row["id"])
        for row in candidates
        if row["fixture_id"] == new_fixture_id
    }
    for row in candidates:
        old_id = str(row["id"])
        if row["fixture_id"] != old_fixture_id or old_id in corrected_ids:
            continue
        replacement = new_by_key.get(
            (
                str(row["provider"]),
                str(row["provider_market_id"]),
                str(row["provider_selection_id"]),
            )
        )
        _append_correction(
            store,
            target_table=EvidenceTable.MARKET_CANDIDATES,
            target_id=old_id,
            replacement_id=replacement,
            created_at=created_at,
            reason="fixture_change_invalidated_market_match",
            payload={
                "old_fixture_id": old_fixture_id,
                "new_fixture_id": new_fixture_id,
            },
        )


def _supersede_proposals(
    store: EvidenceStore,
    proposals: list[dict[str, Any]],
    *,
    prediction_replacements: dict[str, str | None],
    created_at: datetime,
    corrected_ids: set[str],
) -> dict[str, str | None]:
    new_by_prediction = {str(row["prediction_id"]): str(row["id"]) for row in proposals}
    replacements: dict[str, str | None] = {}
    for row in proposals:
        old_id = str(row["id"])
        old_prediction_id = str(row["prediction_id"])
        if old_prediction_id not in prediction_replacements or old_id in corrected_ids:
            continue
        new_prediction_id = prediction_replacements[old_prediction_id]
        replacement = (
            new_by_prediction.get(new_prediction_id)
            if new_prediction_id is not None
            else None
        )
        replacements[old_id] = replacement
        _append_correction(
            store,
            target_table=EvidenceTable.PROPOSALS,
            target_id=old_id,
            replacement_id=replacement,
            created_at=created_at,
            reason="fixture_change_expired_proposal",
            payload={"replacement_prediction_id": new_prediction_id},
        )
    return replacements


def _supersede_approval_and_position_records(
    store: EvidenceStore,
    *,
    approvals: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    proposal_replacements: dict[str, str | None],
    created_at: datetime,
    corrected_approval_ids: set[str],
    corrected_position_ids: set[str],
) -> None:
    for row in approvals:
        record_id = str(row["id"])
        proposal_id = str(row["proposal_id"])
        if (
            proposal_id in proposal_replacements
            and record_id not in corrected_approval_ids
        ):
            _append_correction(
                store,
                target_table=EvidenceTable.APPROVALS,
                target_id=record_id,
                replacement_id=None,
                created_at=created_at,
                reason="approved_proposal_expired_after_fixture_change",
                payload={"replacement_proposal_id": proposal_replacements[proposal_id]},
            )
    for row in positions:
        record_id = str(row["id"])
        proposal_id = str(row["proposal_id"])
        if (
            proposal_id in proposal_replacements
            and record_id not in corrected_position_ids
        ):
            _append_correction(
                store,
                target_table=EvidenceTable.PAPER_POSITIONS,
                target_id=record_id,
                replacement_id=None,
                created_at=created_at,
                reason="paper_position_fixture_change_requires_review",
                payload={"replacement_proposal_id": proposal_replacements[proposal_id]},
            )


def _append_correction(
    store: EvidenceStore,
    *,
    target_table: EvidenceTable,
    target_id: str,
    replacement_id: str | None,
    created_at: datetime,
    reason: str,
    payload: dict[str, Any],
) -> None:
    identity = "|".join(
        (target_table.value, target_id, replacement_id or "none", reason)
    )
    correction_id = _id("correction", identity)
    _append_if_missing(
        store,
        EvidenceTable.CORRECTIONS,
        {
            "id": correction_id,
            "target_table": target_table.value,
            "target_id": target_id,
            "created_at": created_at,
            "reason": reason,
            "replacement_id": replacement_id,
            "idempotency_key": correction_id,
            "payload_json": payload,
        },
    )


def _corrected_target_ids(
    store: EvidenceStore,
    table: EvidenceTable,
) -> set[str]:
    return {
        str(row["target_id"])
        for row in store.list(EvidenceTable.CORRECTIONS)
        if row["target_table"] == table.value
    }


def _fixture_source_key(fixture: dict[str, Any]) -> str:
    try:
        payload = json.loads(str(fixture["payload_json"]))
    except (json.JSONDecodeError, KeyError, TypeError):
        return ""
    return _optional_text(payload.get("source_match_key"))


def _schedule_series_identity_id(row: Any) -> str | None:
    series_id = _optional_text(row.get("serie_id"))
    if not series_id:
        return None
    return canonical_identity_id(
        "lol",
        EntityType.SERIES,
        "pandascore",
        series_id,
    )


def _json_or_text(value: Any) -> Any:
    text = _optional_text(value)
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _record_model(
    store: EvidenceStore,
    created_at: datetime,
    *,
    target: str,
) -> str:
    registry = ModelRegistry(MODEL_REGISTRY_DIR)
    champion = registry.champion_id()
    model_id = champion or "legacy-current"
    if champion:
        artifact_uri = str(registry.candidates / champion)
        checksum = _path_hash(registry.candidates / champion / "manifest.json")
    else:
        artifact_uri = str(OUTCOME_PREDICTION_MODEL_PATH)
        checksum = _path_hash(OUTCOME_PREDICTION_MODEL_PATH)
    target_paths = {
        "map_win": (OUTCOME_PREDICTION_MODEL_PATH, None),
        "series_winner": (
            SERIES_WINNER_MODEL_PATH,
            SERIES_WINNER_PROBABILITY_CALIBRATOR,
        ),
        "gamelength": (
            GAMELENGTH_PREDICTION_MODEL_PATH,
            GAMELENGTH_PREDICTION_PROP_CALIBRATOR,
        ),
        "total_kills": (
            TOTAL_KILLS_PREDICTION_MODEL_PATH,
            TOTAL_KILLS_PREDICTION_PROP_CALIBRATOR,
        ),
        "total_towers": (
            TOTAL_TOWERS_PREDICTION_MODEL_PATH,
            TOTAL_TOWERS_PREDICTION_PROP_CALIBRATOR,
        ),
    }
    legacy_artifact, calibrator = target_paths[target]
    if champion and calibrator is not None:
        calibrator = registry.artifact_path(
            calibrator.relative_to(MODELS_DIR).as_posix(),
            model_id=champion,
        )
    if not champion:
        artifact_uri = str(legacy_artifact)
        checksum = _path_hash(legacy_artifact)
    evidence_id = _id("model", f"{model_id}|{target}")
    _append_if_missing(
        store,
        EvidenceTable.MODEL_VERSIONS,
        {
            "id": evidence_id,
            "sport": "lol",
            "target": target,
            "created_at": created_at,
            "artifact_uri": artifact_uri,
            "artifact_checksum": checksum,
            "idempotency_key": f"lol-model:{model_id}:{target}",
            "payload_json": {
                "registry_model_id": champion,
                "bootstrap_legacy": champion is None,
                "calibrator_uri": str(calibrator) if calibrator else None,
            },
        },
    )
    return evidence_id


def _record_scalar_forecast(
    store: EvidenceStore,
    *,
    run_id: str,
    fixture_id: str | None,
    model_ids: dict[str, str],
    row: dict[str, Any],
) -> None:
    if fixture_id is None:
        return
    target = str(row.get("market", "")).removesuffix("_mean")
    if target not in RESEARCH_PROP_TARGETS:
        return
    model = store.get(EvidenceTable.MODEL_VERSIONS, model_ids[target])
    assert model is not None
    model_payload = json.loads(model["payload_json"])
    forecast_id = _id("forecast", f"{run_id}|{fixture_id}|{target}")
    evidence_status = (
        "below_constant_baseline" if target == "total_towers" else "weak_signal"
    )
    _append_if_missing(
        store,
        EvidenceTable.FORECASTS,
        {
            "id": forecast_id,
            "run_id": run_id,
            "fixture_id": fixture_id,
            "model_version_id": model_ids[target],
            "target": target,
            "created_at": _as_utc(row["run_ts"]),
            "point_value": str(float(row["model_value"])),
            "uncertainty_json": {
                "method": row.get("uncertainty_method"),
                "confidence": row.get("uncertainty_confidence"),
            },
            "evidence_status": evidence_status,
            "idempotency_key": forecast_id,
            "payload_json": {
                "calibrator_uri": model_payload.get("calibrator_uri"),
                "calibration_metadata": {
                    "league": row.get("league"),
                    "bo_format": row.get("match_type"),
                },
                "lineup_ready": row.get("lineup_ready") is True,
                "roster_ready": row.get("roster_ready") is True,
            },
        },
    )


def _record_display_market(
    store: EvidenceStore,
    *,
    run_id: str,
    fixture_id: str,
    row: dict[str, Any],
) -> str | None:
    provider_market_id = row.get("poly_market_id")
    if not provider_market_id:
        return None
    candidate_id = _id(
        "market",
        f"{run_id}|{fixture_id}|{provider_market_id}|{row['selection']}",
    )
    _append_if_missing(
        store,
        EvidenceTable.MARKET_CANDIDATES,
        {
            "id": candidate_id,
            "run_id": run_id,
            "fixture_id": fixture_id,
            "provider": "polymarket",
            "provider_market_id": str(provider_market_id),
            "provider_selection_id": str(row["selection"]),
            "discovered_at": _as_utc(row["run_ts"]),
            "match_status": "display_only",
            "rejection_reason": "executable_order_book_unavailable",
            "idempotency_key": candidate_id,
            "payload_json": {
                "question": row.get("poly_question"),
                "url": row.get("poly_url"),
                "display_probability": row.get("poly_price"),
            },
        },
    )
    return candidate_id


def _fixture_lookup_key(row: Any) -> str:
    return "|".join(
        (
            str(row.get("league") or "unknown"),
            str(row.get("team_a") or "").casefold(),
            str(row.get("team_b") or "").casefold(),
            _as_utc(row.get("start_utc")).isoformat(),
        )
    )


def _identity_id(name: str) -> str:
    return _id("team", name.casefold())


def _id(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode()).hexdigest()[:24]}"


def _as_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp is not timezone aware: {value}")
    return parsed.astimezone(UTC)


def _path_hash(path: Path) -> str:
    if not path.is_file():
        return "unavailable"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append_if_missing(
    store: EvidenceStore,
    table: EvidenceTable,
    values: dict[str, Any],
) -> None:
    if store.get(table, str(values["id"])) is None:
        store.append(table, values)
