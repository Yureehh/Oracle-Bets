"""Immutable outcome evidence for every prediction, including no-bet shadows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from oracle_bets_core.evidence import EvidenceStore, EvidenceTable

if TYPE_CHECKING:
    from collections.abc import Mapping


class PredictionOutcomeError(ValueError):
    """Raised when prediction outcome evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class PredictionOutcomeRecord:
    """One append-only binary result used to score a stored prediction."""

    snapshot_id: str
    prediction_id: str
    fixture_id: str
    actual: int
    observed_at: datetime
    provider: str
    source_reference: str
    superseded_snapshot_id: str | None


def record_prediction_outcome(
    store: EvidenceStore,
    *,
    prediction_id: str,
    actual: int,
    observed_at: datetime,
    provider: str,
    source_reference: str,
    winning_selection_id: str | None = None,
    payload: Mapping[str, object] | None = None,
) -> PredictionOutcomeRecord:
    """Record or correct the result of one prediction without mutating history."""
    if not prediction_id.strip():
        raise PredictionOutcomeError("prediction_id cannot be empty")
    if isinstance(actual, bool) or actual not in {0, 1}:
        raise PredictionOutcomeError("actual must be exactly 0 or 1")
    _require_utc(observed_at)
    if not provider.strip() or not source_reference.strip():
        raise PredictionOutcomeError("provider and source_reference cannot be empty")

    prediction = store.get(EvidenceTable.PREDICTIONS, prediction_id)
    if prediction is None:
        raise PredictionOutcomeError(f"unknown prediction: {prediction_id}")
    fixture_id = str(prediction["fixture_id"])
    fixture = store.get(EvidenceTable.FIXTURES, fixture_id)
    if fixture is None:
        raise PredictionOutcomeError(f"prediction fixture is missing: {fixture_id}")
    if winning_selection_id is not None:
        allowed = {
            str(fixture["team_a_identity_id"]),
            str(fixture["team_b_identity_id"]),
        }
        if winning_selection_id not in allowed:
            raise PredictionOutcomeError(
                "winning_selection_id must be one of the fixture teams"
            )
        expected_actual = int(str(prediction["selection_id"]) == winning_selection_id)
        if actual != expected_actual:
            raise PredictionOutcomeError(
                "actual conflicts with prediction selection and winning selection"
            )

    previous = _latest_active_outcome(store, prediction_id)
    outcome_payload: dict[str, object] = {
        "schema_version": 1,
        "prediction_id": prediction_id,
        "fixture_id": fixture_id,
        "selection_id": str(prediction["selection_id"]),
        "actual": actual,
        "winning_selection_id": winning_selection_id,
        "source_reference": source_reference,
        "details": dict(payload or {}),
    }
    identity = "|".join(
        (
            prediction_id,
            str(actual),
            observed_at.isoformat(),
            provider,
            source_reference,
            winning_selection_id or "",
        )
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
    snapshot_id = f"prediction-outcome-{digest}"
    store.append(
        EvidenceTable.SOURCE_SNAPSHOTS,
        {
            "id": snapshot_id,
            "run_id": str(prediction["run_id"]),
            "provider": provider,
            "source_type": "prediction_outcome",
            "observed_at": observed_at,
            "source_uri": source_reference,
            "schema_fingerprint": "oracle-bets:prediction-outcome:v1",
            "idempotency_key": snapshot_id,
            "payload_json": outcome_payload,
        },
    )

    superseded_id = None
    if previous is not None and str(previous["id"]) != snapshot_id:
        superseded_id = str(previous["id"])
        correction_id = f"correction-{hashlib.sha256(f'{superseded_id}|{snapshot_id}'.encode()).hexdigest()[:24]}"
        store.append(
            EvidenceTable.CORRECTIONS,
            {
                "id": correction_id,
                "target_table": EvidenceTable.SOURCE_SNAPSHOTS.value,
                "target_id": superseded_id,
                "created_at": observed_at,
                "reason": "prediction_outcome_revised",
                "replacement_id": snapshot_id,
                "idempotency_key": correction_id,
                "payload_json": {
                    "prediction_id": prediction_id,
                    "old_actual": _outcome_actual(previous),
                    "new_actual": actual,
                    "source_reference": source_reference,
                },
            },
        )
    return PredictionOutcomeRecord(
        snapshot_id=snapshot_id,
        prediction_id=prediction_id,
        fixture_id=fixture_id,
        actual=actual,
        observed_at=observed_at,
        provider=provider,
        source_reference=source_reference,
        superseded_snapshot_id=superseded_id,
    )


def _latest_active_outcome(
    store: EvidenceStore,
    prediction_id: str,
) -> dict[str, object] | None:
    corrected = {
        str(row["target_id"])
        for row in store.list(EvidenceTable.CORRECTIONS)
        if row["target_table"] == EvidenceTable.SOURCE_SNAPSHOTS.value
    }
    for row in reversed(store.list(EvidenceTable.SOURCE_SNAPSHOTS)):
        if row["source_type"] != "prediction_outcome" or row["id"] in corrected:
            continue
        try:
            decoded = json.loads(str(row["payload_json"]))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(decoded, dict) and decoded.get("prediction_id") == prediction_id:
            return dict(row)
    return None


def _outcome_actual(row: Mapping[str, object]) -> int | None:
    try:
        decoded = json.loads(str(row["payload_json"]))
    except (json.JSONDecodeError, TypeError):
        return None
    value = decoded.get("actual") if isinstance(decoded, dict) else None
    return int(value) if value in {0, 1} else None


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise PredictionOutcomeError("observed_at must be timezone-aware UTC")
