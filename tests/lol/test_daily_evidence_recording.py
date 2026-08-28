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
