from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from lol_bets.daily import DailyStepResult
from lol_bets.operations.evidence import record_daily_evidence
from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.pd import pd

NOW = datetime(2026, 7, 27, 8, tzinfo=UTC)
TWO_VERSIONS = 2


def test_daily_bridge_records_complete_chain_and_honest_no_bet(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    schedule = pd.DataFrame(
        [
            {
                "league": "LCK",
                "team_a": "T1",
                "team_b": "Gen.G",
                "start_utc": "2026-07-27T10:00:00Z",
                "best_of": 3,
                "status": "scheduled",
                "match_key": "pandascore-1",
            }
        ]
    )
    snapshots = [
        {
            "run_ts": NOW,
            "league": "LCK",
            "team_a": "T1",
            "team_b": "Gen.G",
            "start_utc": "2026-07-27T10:00:00Z",
            "market": "winner",
            "selection": "T1",
            "model_value": 0.61,
            "probability_lower": 0.56,
            "probability_upper": 0.66,
            "probability_source": "calibrated",
            "uncertainty_method": "held_out_calibration_residual_mean",
            "uncertainty_confidence": 0.9,
            "uncertainty_sample_count": 80,
            "drivers": ["Team rating strength pushed the model toward T1"],
            "lineup_ready": True,
            "roster_ready": True,
            "poly_market_id": "market-1",
            "poly_price": 0.54,
            "poly_question": "Will T1 beat Gen.G?",
            "poly_url": "https://polymarket.com/event/test",
        }
    ]

    run_id = record_daily_evidence(
        store=store,
        scheduled_for=NOW,
        effective_config={"horizon_hours": 36},
        schedule=schedule,
        snapshot_rows=snapshots,
        steps=[DailyStepResult("schedule", True, "one fixture")],
    )

    assert store.get(EvidenceTable.RUNS, run_id)["status"] == "completed"
    assert store.count(EvidenceTable.FIXTURES) == 1
    assert store.count(EvidenceTable.PREDICTIONS) == 1
    prediction = store.list(EvidenceTable.PREDICTIONS)[0]
    assert prediction["probability_lower"] == "0.56"
    assert prediction["probability_upper"] == "0.66"
    assert json.loads(prediction["warnings_json"]) == []
    assert json.loads(prediction["payload_json"])["drivers"] == [
        "Team rating strength pushed the model toward T1"
    ]
    market = store.list(EvidenceTable.MARKET_CANDIDATES)[0]
    assert market["match_status"] == "display_only"
    proposal = store.list(EvidenceTable.PROPOSALS)[0]
    assert proposal["state"] == "rejected"
    assert proposal["rejection_reason"] == "executable_order_book_unavailable"
    assert proposal["market_snapshot_id"] is None
    assert store.count(EvidenceTable.PAPER_POSITIONS) == 0


def test_daily_bridge_is_idempotent_for_same_run(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    schedule = pd.DataFrame(
        [
            {
                "league": "LCK",
                "team_a": "T1",
                "team_b": "Gen.G",
                "start_utc": "2026-07-27T10:00:00Z",
                "best_of": 3,
            }
        ]
    )
    snapshot = {
        "run_ts": NOW,
        "league": "LCK",
        "team_a": "T1",
        "team_b": "Gen.G",
        "start_utc": "2026-07-27T10:00:00Z",
        "market": "winner",
        "selection": "T1",
        "model_value": 0.61,
        "probability_source": "calibrated",
        "poly_market_id": None,
    }
    arguments = {
        "store": store,
        "scheduled_for": NOW,
        "effective_config": {"horizon_hours": 36},
        "schedule": schedule,
        "snapshot_rows": [snapshot],
        "steps": [DailyStepResult("schedule", True, "one fixture")],
    }

    assert record_daily_evidence(**arguments) == record_daily_evidence(**arguments)
    assert store.count(EvidenceTable.RUNS) == 1
    assert store.count(EvidenceTable.PREDICTIONS) == 1
    assert store.count(EvidenceTable.PROPOSALS) == 1


def test_fixture_change_appends_replacements_and_superseding_corrections(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")

    def schedule(version: str, start: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "provider_match_id": "77",
                    "match_key": "pandascore:77",
                    "fixture_version": version,
                    "league": "LCK",
                    "team_a": "T1",
                    "team_b": "Gen.G",
                    "start_utc": start,
                    "best_of": 3,
                    "status": "scheduled",
                    "lineup_source": "pandascore_match_detail",
                    "lineup_observed_at": NOW,
                }
            ]
        )

    def snapshot(start: str, probability: float) -> dict:
        return {
            "run_ts": NOW,
            "league": "LCK",
            "team_a": "T1",
            "team_b": "Gen.G",
            "start_utc": start,
            "market": "winner",
            "selection": "T1",
            "model_value": probability,
            "probability_source": "calibrated",
            "lineup_ready": True,
            "roster_ready": True,
            "poly_market_id": None,
        }

    first_start = "2026-07-27T10:00:00Z"
    second_start = "2026-07-27T11:00:00Z"
    record_daily_evidence(
        store=store,
        scheduled_for=NOW,
        effective_config={"horizon_hours": 36},
        schedule=schedule("fixture-v1", first_start),
        snapshot_rows=[snapshot(first_start, 0.61)],
        steps=[DailyStepResult("schedule", True, "one fixture")],
    )
    record_daily_evidence(
        store=store,
        scheduled_for=NOW + timedelta(days=1),
        effective_config={"horizon_hours": 36},
        schedule=schedule("fixture-v2", second_start),
        snapshot_rows=[snapshot(second_start, 0.58)],
        steps=[DailyStepResult("schedule", True, "one moved fixture")],
    )

    assert store.count(EvidenceTable.FIXTURES) == TWO_VERSIONS
    assert store.count(EvidenceTable.PREDICTIONS) == TWO_VERSIONS
    assert store.count(EvidenceTable.PROPOSALS) == TWO_VERSIONS
    corrections = store.list(EvidenceTable.CORRECTIONS)
    assert {row["target_table"] for row in corrections} == {
        "fixtures",
        "predictions",
        "proposals",
    }
    assert all(row["replacement_id"] for row in corrections)
    fixture_correction = next(
        row for row in corrections if row["target_table"] == "fixtures"
    )
    replacement = store.get(
        EvidenceTable.FIXTURES,
        fixture_correction["replacement_id"],
    )
    assert json.loads(replacement["payload_json"])["fixture_version"] == "fixture-v2"


def test_restricted_fixture_expires_old_decisions_without_fabricating_replacements(
    tmp_path,
):
    store = EvidenceStore(tmp_path / "evidence.db")
    base = {
        "provider_match_id": "77",
        "match_key": "pandascore:77",
        "league": "LCK",
        "team_a": "T1",
        "team_b": "Gen.G",
        "start_utc": "2026-07-27T10:00:00Z",
        "best_of": 3,
    }
    snapshot = {
        "run_ts": NOW,
        "league": "LCK",
        "team_a": "T1",
        "team_b": "Gen.G",
        "start_utc": base["start_utc"],
        "market": "winner",
        "selection": "T1",
        "model_value": 0.61,
        "probability_source": "calibrated",
        "lineup_ready": True,
        "roster_ready": True,
        "poly_market_id": None,
    }
    record_daily_evidence(
        store=store,
        scheduled_for=NOW,
        effective_config={"horizon_hours": 36},
        schedule=pd.DataFrame(
            [{**base, "fixture_version": "v1", "status": "scheduled"}]
        ),
        snapshot_rows=[snapshot],
        steps=[DailyStepResult("schedule", True, "scheduled")],
    )
    record_daily_evidence(
        store=store,
        scheduled_for=NOW + timedelta(days=1),
        effective_config={"horizon_hours": 36},
        schedule=pd.DataFrame(
            [{**base, "fixture_version": "v2", "status": "cancelled"}]
        ),
        snapshot_rows=[],
        steps=[DailyStepResult("schedule", True, "cancelled")],
    )

    corrections = store.list(EvidenceTable.CORRECTIONS)
    prediction_correction = next(
        row for row in corrections if row["target_table"] == "predictions"
    )
    proposal_correction = next(
        row for row in corrections if row["target_table"] == "proposals"
    )
    assert prediction_correction["replacement_id"] is None
    assert proposal_correction["replacement_id"] is None
