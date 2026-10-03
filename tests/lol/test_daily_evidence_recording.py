from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from lol_bets.daily import DailyStepResult
from lol_bets.operations import evidence as evidence_module
from lol_bets.operations.evidence import record_daily_evidence
from lol_bets.operations.identity import canonical_team_identity_id
from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.pd import pd

NOW = datetime(2026, 8, 24, 8, tzinfo=UTC)
START = NOW + timedelta(days=2)
EXPECTED_ODDS = 1.8
TWO_ROWS = 2
HORIZON_HOURS = 72


def test_daily_cohort_enrolls_before_forecasts_and_survives_reschedule(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    options = {
        "store": store,
        "effective_config": {"horizon_hours": HORIZON_HOURS},
        "snapshot_rows": (),
        "steps": [DailyStepResult("schedule", True, "complete")],
    }
    record_daily_evidence(
        **options,
        scheduled_for=NOW,
        observed_at=NOW,
        schedule=_schedule(),
    )
    record_daily_evidence(
        **options,
        scheduled_for=NOW + timedelta(hours=1),
        observed_at=NOW + timedelta(hours=1),
        schedule=_schedule(version="v2", start=START + timedelta(hours=2)),
    )
    enrollments = [
        row
        for row in store.list(EvidenceTable.RUN_EVENTS)
        if row["event_type"] == "cohort_enrollment"
    ]
    assert len(enrollments) == 1
    payload = json.loads(enrollments[0]["payload_json"])
    assert payload["sporting_event_key"] == "pandascore:1"
    assert payload["first_start_utc"] == START.isoformat()
    assert payload["horizon_hours"] == HORIZON_HOURS


def test_daily_cohort_refuses_enrollment_after_existing_forecast(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    options = {
        "store": store,
        "effective_config": {},
        "schedule": _schedule(),
        "steps": [DailyStepResult("schedule", True, "complete")],
    }
    record_daily_evidence(
        **options, scheduled_for=NOW, observed_at=NOW, snapshot_rows=[_snapshot()]
    )
    with pytest.raises(ValueError, match="after its first forecast"):
        record_daily_evidence(
            **options,
            scheduled_for=NOW + timedelta(hours=1),
            observed_at=NOW + timedelta(hours=1),
            snapshot_rows=(),
        )


@pytest.fixture(autouse=True)
def _isolate_model_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(
        evidence_module,
        "MODEL_REGISTRY_DIR",
        tmp_path / "model-registry",
    )


def _schedule(*, version: str = "v1", start: datetime = START):
    return pd.DataFrame(
        [
            {
                "provider_match_id": "1",
                "match_key": "pandascore:1",
                "fixture_version": version,
                "league": "LCK",
                "team_a": "T1",
                "team_b": "Gen.G",
                "start_utc": start.isoformat(),
                "best_of": 3,
                "status": "scheduled",
            }
        ]
    )


def _snapshot(*, start: datetime = START):
    return {
        "run_ts": NOW,
        "source_match_key": "pandascore:1",
        "league": "LCK",
        "team_a": "T1",
        "team_b": "Gen.G",
        "start_utc": start.isoformat(),
        "market": "series_winner",
        "selection": "T1",
        "model_value": 0.61,
        "probability_lower": 0.56,
        "probability_upper": 0.66,
        "probability_source": "calibrated",
        "lineup_ready": True,
        "roster_ready": True,
    }


def _market_action(observed_at: datetime = NOW):
    return {
        "comparison_id": "comparison-1",
        "fixture_key": "pandascore:1",
        "target": "series_winner",
        "selection": "T1",
        "probability": 0.61,
        "probability_lower": 0.56,
        "market_id": "market-1",
        "token_id": "token-1",
        "market_url": "https://polymarket.com/event/test",
        "warnings": [],
        "hard_blocks": [],
        "state": "recommended",
        "classification": "recommended",
        "readiness": "recommendation_active",
        "reason": "all_recommendation_gates_passed",
        "reason_codes": ["all_recommendation_gates_passed"],
        "policy_version": "lol-market-policy-v1",
        "strategy_version": "series_direct_v2",
        "probability_source": "direct_series_model",
        "semantic_key": {
            "version": 1,
            "target": "series_winner",
            "period": "series",
            "selection": "t1",
            "line": None,
        },
        "semantic_fingerprint": "semantic-1",
        "sizing": {
            "bankroll_fractions": {
                "flat_1u": 0.01,
                "full_kelly": 0.11,
                "half_kelly": 0.055,
                "quarter_kelly": 0.0275,
            },
            "stake_units": {
                "flat_1u": 1.0,
                "full_kelly": 11.0,
                "half_kelly": 5.5,
                "quarter_kelly": 2.75,
            },
            "selected_path": "full_kelly",
        },
        "correlation_group": "pandascore:1",
        "point_edge": 0.098,
        "stake_units": 0.0,
        "decimal_odds": 1.8,
        "observations": [
            {
                "sequence_number": 1,
                "observed_at": observed_at.isoformat(),
                "book_hash": f"book-{observed_at.minute}",
                "token_id": "token-1",
                "complete": True,
                "average_price": "0.5555555556",
                "decimal_odds": 1.8,
                "depth": {"bids": [], "asks": []},
            }
        ],
    }


def test_forecast_timestamps_use_decision_time_instead_of_scheduled_run(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    decision = NOW + timedelta(hours=3)
    winner = _snapshot() | {"decision_at": decision.isoformat()}
    prop = winner | {"market": "total_kills_mean", "model_value": 25.0}
    record_daily_evidence(
        store=store,
        scheduled_for=NOW,
        observed_at=decision,
        effective_config={},
        schedule=_schedule(),
        snapshot_rows=[winner, prop],
        steps=[DailyStepResult("review", True, "complete")],
    )
    for table in (EvidenceTable.PREDICTIONS, EvidenceTable.FORECASTS):
        rows = store.list(table)
        assert len(rows) == 1
        assert datetime.fromisoformat(rows[0]["created_at"]) == decision


def test_review_evidence_records_forecasts_quotes_but_never_legacy_proposals(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    run_id = record_daily_evidence(
        store=store,
        scheduled_for=NOW,
        effective_config={"source": "owner-links"},
        schedule=_schedule(),
        snapshot_rows=[_snapshot()],
        steps=[DailyStepResult("review", True, "complete")],
        market_actions=[_market_action()],
        run_type="manual_lol_market_review",
    )

    assert store.get(EvidenceTable.RUNS, run_id)["status"] == "completed"
    assert store.count(EvidenceTable.PREDICTIONS) == 1
    assert store.count(EvidenceTable.MARKET_CANDIDATES) == 1
    assert store.count(EvidenceTable.MARKET_SNAPSHOTS) == 1
    assert store.count(EvidenceTable.PROPOSALS) == 0
    fixture = store.list(EvidenceTable.FIXTURES)[0]
    assert fixture["team_a_identity_id"] == canonical_team_identity_id("T1")
    assert fixture["team_b_identity_id"] == canonical_team_identity_id("Gen.G")
    candidate = store.list(EvidenceTable.MARKET_CANDIDATES)[0]
    payload = json.loads(candidate["payload_json"])
    assert payload["decimal_odds"] == EXPECTED_ODDS
    assert payload["classification"] == "recommended"
    assert payload["semantic_fingerprint"] == "semantic-1"
    assert payload["sizing"]["selected_path"] == "full_kelly"
    assert candidate["prediction_id"] == store.list(EvidenceTable.PREDICTIONS)[0]["id"]
    quote = json.loads(store.list(EvidenceTable.MARKET_SNAPSHOTS)[0]["payload_json"])
    assert quote["semantic_fingerprint"] == "semantic-1"
    assert quote["terms_verified"] is False
    assert quote["stake_amount"] is None
    assert quote["stake_currency"] is None


def test_same_review_is_idempotent_and_new_book_is_append_only(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    arguments = {
        "store": store,
        "scheduled_for": NOW,
        "effective_config": {"source": "owner-links"},
        "schedule": _schedule(),
        "snapshot_rows": [_snapshot()],
        "steps": [DailyStepResult("review", True, "complete")],
        "run_type": "manual_lol_market_review",
        "run_key": "manual-review-1",
    }
    record_daily_evidence(**arguments, market_actions=[_market_action()])
    record_daily_evidence(
        **arguments,
        market_actions=[_market_action(NOW + timedelta(minutes=5))],
    )

    assert store.count(EvidenceTable.RUNS) == 1
    assert store.count(EvidenceTable.MARKET_CANDIDATES) == 1
    assert store.count(EvidenceTable.MARKET_SNAPSHOTS) == TWO_ROWS
    assert store.count(EvidenceTable.PROPOSALS) == 0


def test_fixture_change_supersedes_prediction_without_creating_proposal(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    arguments = {
        "store": store,
        "effective_config": {},
        "steps": [DailyStepResult("schedule", True, "complete")],
    }
    record_daily_evidence(
        **arguments,
        scheduled_for=NOW,
        schedule=_schedule(),
        snapshot_rows=[_snapshot()],
    )
    moved = START + timedelta(hours=2)
    record_daily_evidence(
        **arguments,
        scheduled_for=NOW + timedelta(days=1),
        schedule=_schedule(version="v2", start=moved),
        snapshot_rows=[_snapshot(start=moved)],
    )

    assert store.count(EvidenceTable.FIXTURES) == TWO_ROWS
    assert store.count(EvidenceTable.PREDICTIONS) == TWO_ROWS
    assert store.count(EvidenceTable.PROPOSALS) == 0
    assert {row["target_table"] for row in store.list(EvidenceTable.CORRECTIONS)} == {
        "fixtures",
        "predictions",
    }


def test_schedule_capture_time_is_separate_from_scheduled_run_identity(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    captured = NOW + timedelta(hours=8)
    run_id = record_daily_evidence(
        store=store,
        scheduled_for=NOW,
        observed_at=captured,
        effective_config={},
        schedule=_schedule(),
        snapshot_rows=[],
        steps=[DailyStepResult("refresh", True, "complete")],
    )
    source = store.list(EvidenceTable.SOURCE_SNAPSHOTS)[0]
    assert datetime.fromisoformat(source["observed_at"]) == captured
    run = store.get(EvidenceTable.RUNS, run_id)
    assert json.loads(run["payload_json"])["scheduled_for"] == NOW.isoformat()


def test_market_snapshot_keeps_provider_timestamp(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    action = _market_action()
    provider_time = (NOW - timedelta(seconds=3)).isoformat()
    action["observations"][0]["provider_timestamp"] = provider_time
    record_daily_evidence(
        store=store,
        scheduled_for=NOW,
        effective_config={},
        schedule=_schedule(),
        snapshot_rows=[_snapshot()],
        steps=[DailyStepResult("review", True, "complete")],
        market_actions=[action],
    )
    snapshot = store.list(EvidenceTable.MARKET_SNAPSHOTS)[0]
    assert json.loads(snapshot["payload_json"])["provider_timestamp"] == provider_time
