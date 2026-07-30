from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.evidence.results import (
    PredictionOutcomeError,
    record_prediction_outcome,
)

NOW = datetime(2026, 7, 26, 12, tzinfo=UTC)
EXPECTED_OUTCOME_SNAPSHOTS = 2


@pytest.fixture
def result_store(tmp_path):
    store = EvidenceStore(tmp_path / "results.db")
    store.initialize_schema()
    store.append(
        EvidenceTable.RUNS,
        {
            "id": "run",
            "run_type": "test",
            "started_at": NOW,
            "status": "completed",
            "idempotency_key": "run",
            "payload_json": {},
        },
    )
    for team in ("a", "b"):
        store.append(
            EvidenceTable.IDENTITIES,
            {
                "id": team,
                "entity_type": "team",
                "canonical_name": team.upper(),
                "created_at": NOW,
                "idempotency_key": team,
                "payload_json": {},
            },
        )
    store.append(
        EvidenceTable.FIXTURES,
        {
            "id": "fixture",
            "run_id": "run",
            "sport": "lol",
            "competition_id": "LCK",
            "team_a_identity_id": "a",
            "team_b_identity_id": "b",
            "start_time": NOW,
            "best_of": 3,
            "status": "finished",
            "idempotency_key": "fixture",
            "payload_json": {},
        },
    )
    store.append(
        EvidenceTable.MODEL_VERSIONS,
        {
            "id": "model",
            "sport": "lol",
            "target": "series_win",
            "created_at": NOW,
            "artifact_uri": "models/model",
            "artifact_checksum": "checksum",
            "idempotency_key": "model",
            "payload_json": {},
        },
    )
    store.append(
        EvidenceTable.PREDICTIONS,
        {
            "id": "prediction",
            "run_id": "run",
            "fixture_id": "fixture",
            "model_version_id": "model",
            "selection_id": "a",
            "mode": "prematch",
            "created_at": NOW,
            "probability_point": "0.6",
            "probability_lower": "0.5",
            "probability_upper": "0.7",
            "warnings_json": [],
            "idempotency_key": "prediction",
            "payload_json": {},
        },
    )
    return store


def test_prediction_outcome_is_idempotent_and_corrections_are_append_only(
    result_store,
):
    first = record_prediction_outcome(
        result_store,
        prediction_id="prediction",
        actual=1,
        observed_at=NOW,
        provider="provider",
        source_reference="result:1",
        winning_selection_id="a",
    )
    duplicate = record_prediction_outcome(
        result_store,
        prediction_id="prediction",
        actual=1,
        observed_at=NOW,
        provider="provider",
        source_reference="result:1",
        winning_selection_id="a",
    )
    revised = record_prediction_outcome(
        result_store,
        prediction_id="prediction",
        actual=0,
        observed_at=NOW + timedelta(minutes=1),
        provider="provider",
        source_reference="result:1-correction",
        winning_selection_id="b",
    )

    assert duplicate.snapshot_id == first.snapshot_id
    assert revised.superseded_snapshot_id == first.snapshot_id
    assert (
        result_store.count(EvidenceTable.SOURCE_SNAPSHOTS) == EXPECTED_OUTCOME_SNAPSHOTS
    )
    correction = result_store.list(EvidenceTable.CORRECTIONS)[0]
    assert correction["target_id"] == first.snapshot_id
    assert correction["replacement_id"] == revised.snapshot_id


def test_prediction_outcome_rejects_conflicting_or_untrusted_facts(result_store):
    with pytest.raises(PredictionOutcomeError, match="conflicts"):
        record_prediction_outcome(
            result_store,
            prediction_id="prediction",
            actual=0,
            observed_at=NOW,
            provider="provider",
            source_reference="result:1",
            winning_selection_id="a",
        )
    with pytest.raises(PredictionOutcomeError, match="timezone-aware UTC"):
        record_prediction_outcome(
            result_store,
            prediction_id="prediction",
            actual=1,
            observed_at=NOW.replace(tzinfo=None),
            provider="provider",
            source_reference="result:1",
        )
