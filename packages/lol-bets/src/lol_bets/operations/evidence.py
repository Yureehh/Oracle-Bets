"""Persist daily LoL outputs into the canonical evidence graph."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.evidence.identity import EntityType, canonical_identity_id
from oracle_bets_core.operations import daily_run_key
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

from lol_bets.inference.team_resolver import canonical_team_name
from lol_bets.operations.identity import (
    canonical_team_identity_id,
    sync_schedule_identity_graph,
)
from lol_bets.operations.models import ModelRegistry

RESEARCH_PROP_TARGETS = frozenset({"gamelength", "total_kills", "total_towers"})

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
    market_actions: Sequence[dict[str, Any]] = (),
    run_type: str = "daily_lol",
    run_key: str | None = None,
) -> str:
    """Record predictions and honest no-bet outcomes without upgrading prices."""
    store.initialize_schema()
    evidence_key = run_key or daily_run_key(
        "lol", scheduled_for=scheduled_for, config=effective_config
    )
    run_id = _id("run", evidence_key)
    _append_if_missing(
        store,
        EvidenceTable.RUNS,
        {
            "id": run_id,
            "run_type": run_type,
            "started_at": scheduled_for,
            "status": "completed" if all(step.ok for step in steps) else "partial",
            "idempotency_key": evidence_key,
            "payload_json": {
                "run_key": evidence_key,
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
        if row.get("market") not in {"winner", "map_winner", "series_winner"}:
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
        selection_id = _selection_id(str(row["selection"]), str(row.get("market")))
        point = float(row["model_value"])
        lower = float(row.get("probability_lower", point))
        upper = float(row.get("probability_upper", point))
        prediction_id = _id(
            "prediction",
            f"{run_id}|{fixture_id}|{row.get('market')}|{selection_id}|prematch",
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
                    "target": row.get("market"),
                    "probability_source": row.get("probability_source"),
                    "uncertainty_method": row.get("uncertainty_method"),
                    "uncertainty_confidence": row.get("uncertainty_confidence"),
                    "uncertainty_sample_count": row.get("uncertainty_sample_count"),
                    "drivers": list(row.get("drivers") or []),
                    "paired_distribution": row.get("paired_distribution"),
                    "rating_baseline_probability": row.get(
                        "rating_baseline_probability"
                    ),
                    "full_model_probability": row.get("full_model_probability"),
                    "blend_weight": row.get("blend_weight"),
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
    if market_actions:
        _record_typed_market_actions(
            store,
            run_id=run_id,
            fixtures=fixtures,
            schedule=schedule,
            actions=market_actions,
            created_at=scheduled_for,
            model_ids=model_ids,
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
    actions: Sequence[dict[str, Any]],
    created_at: datetime,
    model_ids: dict[str, str],
) -> None:
    fixture_ids = {
        _optional_text(row.get("match_key")): fixtures.get(_fixture_lookup_key(row))
        for _, row in schedule.iterrows()
    }
    predictions_by_selection = {}
    existing_prediction_ids: set[str] = set()
    for row in store.list(EvidenceTable.PREDICTIONS):
        existing_prediction_ids.add(str(row["id"]))
        if str(row["run_id"]) != run_id:
            continue
        payload = json.loads(row["payload_json"])
        predictions_by_selection[
            (
                str(row["fixture_id"]),
                str(payload.get("target") or "winner"),
                str(row["selection_id"]),
            )
        ] = str(row["id"])
    existing_candidate_ids = {
        str(row["id"]) for row in store.list(EvidenceTable.MARKET_CANDIDATES)
    }
    all_records: list[tuple[EvidenceTable, dict[str, Any]]] = []
    for action in actions:
        if not action.get("token_id"):
            continue
        fixture_id = fixture_ids.get(_optional_text(action.get("fixture_key")))
        if fixture_id is None:
            continue
        target = str(action.get("target") or "unknown")
        prediction_id = predictions_by_selection.get(
            (fixture_id, target, _selection_id(str(action["selection"]), target))
        )
        if prediction_id is None and action.get("probability") is not None:
            prediction = _action_prediction_record(
                run_id=run_id,
                fixture_id=fixture_id,
                model_ids=model_ids,
                action=action,
                created_at=created_at,
            )
            if str(prediction["id"]) not in existing_prediction_ids:
                all_records.append((EvidenceTable.PREDICTIONS, prediction))
                existing_prediction_ids.add(str(prediction["id"]))
        candidate_id = _id(
            "market",
            f"{run_id}|{action['market_id']}|{action['token_id']}",
        )
        if candidate_id not in existing_candidate_ids:
            all_records.append(
                (
                    EvidenceTable.MARKET_CANDIDATES,
                    {
                        "id": candidate_id,
                        "run_id": run_id,
                        "fixture_id": fixture_id,
                        "provider": action.get("provider") or "polymarket",
                        "provider_market_id": action["market_id"],
                        "provider_selection_id": action["token_id"],
                        "discovered_at": created_at,
                        "match_status": (
                            "owner_entered"
                            if action.get("quote_basis") == "owner_entered"
                            else "not_comparable"
                            if action.get("hard_blocks")
                            else "typed_exact"
                        ),
                        "rejection_reason": ",".join(action.get("hard_blocks") or [])
                        or None,
                        "idempotency_key": candidate_id,
                        "payload_json": {
                            "target": action["target"],
                            "selection": action["selection"],
                            "url": action.get("market_url"),
                            "resolution_source": action.get("resolution_source"),
                            "gate_state": action.get("state"),
                            "gate_reason": action.get("reason"),
                            "hard_blocks": list(action.get("hard_blocks") or []),
                            "warnings": list(action.get("warnings") or []),
                            "probability": action.get("probability"),
                            "probability_lower": action.get("probability_lower"),
                            "decimal_odds": action.get("decimal_odds"),
                            "strategy_version": action.get("strategy_version"),
                            "probability_source": action.get("probability_source"),
                            "correlation_rank": action.get("correlation_rank"),
                            "line": action.get("line"),
                            "game_number": action.get("game_number"),
                            "quote_basis": action.get("quote_basis") or "public_book",
                        },
                    },
                )
            )
            existing_candidate_ids.add(candidate_id)
        for observation in action["observations"]:
            snapshot_id = _id(
                "snapshot",
                f"{candidate_id}|{observation['sequence_number']}|"
                f"{observation['observed_at']}|{observation['book_hash']}",
            )
            observation_odds = float(observation["decimal_odds"] or 0.0)
            all_records.append(
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
                        "book_json": observation.get("depth")
                        or observation.get("book")
                        or {},
                        "idempotency_key": snapshot_id,
                        "payload_json": {
                            "book_hash": observation["book_hash"],
                            "complete": observation["complete"],
                            "quote_basis": action.get("quote_basis")
                            or "minimum_order_shares",
                            "minimum_order_size": observation.get("minimum_order_size"),
                            "requested_shares": observation.get("requested_shares"),
                            "filled_shares": observation.get("filled_shares"),
                            "hypothetical_cost": observation.get("hypothetical_cost"),
                        },
                    },
                )
            )
    # One dependency-ordered transaction avoids one SQLite round-trip per outcome.
    store.append_transaction(all_records)


def _action_prediction_record(
    *,
    run_id: str,
    fixture_id: str,
    model_ids: dict[str, str],
    action: dict[str, Any],
    created_at: datetime,
) -> dict[str, Any]:
    """Build one derived/experimental probability used by a reviewed contract."""
    target = str(action["target"])
    selection_id = _selection_id(str(action["selection"]), target)
    game_number = action.get("game_number")
    line = action.get("line")
    prediction_id = _id(
        "prediction",
        f"{run_id}|{fixture_id}|{target}|{game_number}|{line}|{selection_id}|prematch",
    )
    point = float(action["probability"])
    lower = float(action.get("probability_lower", point))
    return {
        "id": prediction_id,
        "run_id": run_id,
        "fixture_id": fixture_id,
        "model_version_id": model_ids[_action_model_target(target)],
        "selection_id": selection_id,
        "mode": "prematch",
        "created_at": created_at,
        "probability_point": str(point),
        "probability_lower": str(lower),
        "probability_upper": str(point),
        "warnings_json": list(action.get("warnings") or []),
        "idempotency_key": prediction_id,
        "payload_json": {
            "target": target,
            "strategy_version": action.get("strategy_version"),
            "probability_source": action.get("probability_source"),
            "game_number": game_number,
            "line": line,
        },
    }


def _action_model_target(target: str) -> str:
    return {
        "series_winner": "series_winner",
        "gamelength_mean": "gamelength",
        "total_kills_mean": "total_kills",
        "total_towers_mean": "total_towers",
    }.get(target, "map_win")


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
        team_a = canonical_team_name(str(row.get("team_a") or ""))
        team_b = canonical_team_name(str(row.get("team_b") or ""))
        if not team_a or not team_b:
            continue
        for name in (team_a, team_b):
            identity_id = canonical_team_identity_id(name)
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
                "team_a_identity_id": canonical_team_identity_id(team_a),
                "team_b_identity_id": canonical_team_identity_id(team_b),
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
    candidates = list(store.list(EvidenceTable.MARKET_CANDIDATES))
    corrected = {
        table: _corrected_target_ids(store, table)
        for table in (
            EvidenceTable.PREDICTIONS,
            EvidenceTable.MARKET_CANDIDATES,
        )
    }
    for old_fixture_id, new_fixture_id in fixture_supersessions.items():
        _supersede_predictions(
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


def _supersede_predictions(
    store: EvidenceStore,
    predictions: list[dict[str, Any]],
    *,
    old_fixture_id: str,
    new_fixture_id: str,
    created_at: datetime,
    corrected_ids: set[str],
) -> None:
    new_by_key = {
        (str(row["selection_id"]), str(row["mode"])): str(row["id"])
        for row in predictions
        if row["fixture_id"] == new_fixture_id
    }
    for row in predictions:
        old_id = str(row["id"])
        if row["fixture_id"] != old_fixture_id or old_id in corrected_ids:
            continue
        replacement = new_by_key.get((str(row["selection_id"]), str(row["mode"])))
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


def _fixture_lookup_key(row: Any) -> str:
    return "|".join(
        (
            str(row.get("league") or "unknown"),
            canonical_team_name(str(row.get("team_a") or "")).casefold(),
            canonical_team_name(str(row.get("team_b") or "")).casefold(),
            _as_utc(row.get("start_utc")).isoformat(),
        )
    )


def _selection_id(name: str, target: str) -> str:
    if target in {"winner", "map_winner", "series_winner"}:
        return canonical_team_identity_id(name)
    return _id("selection", name.casefold())


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
