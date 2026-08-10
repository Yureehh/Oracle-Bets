from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from lol_bets.daily import DailyStepResult
from lol_bets.operations import evidence as evidence_module
from lol_bets.operations.evidence import record_daily_evidence
from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.evidence.settlement import SettlementResult
from oracle_bets_core.markets import OrderBook, OrderLevel
from oracle_bets_core.operations.paper_evidence import (
    PaperEvidenceError,
    capture_closing_snapshots,
    decide_paper,
    performance_summary,
    settle_paper,
)
from oracle_bets_core.pd import pd

NOW = datetime(2026, 7, 27, 8, tzinfo=UTC)
TWO_VERSIONS = 2
EXPECTED_RERUN_SNAPSHOTS = 4
INITIAL_ODDS = 2.0


@pytest.fixture(autouse=True)
def _isolate_model_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(
        evidence_module,
        "MODEL_REGISTRY_DIR",
        tmp_path / "model-registry",
    )


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


def test_typed_executable_market_chain_is_recorded_atomically(tmp_path):
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
                "match_key": "pandascore:1",
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
        "probability_lower": 0.56,
        "probability_upper": 0.66,
        "probability_source": "calibrated",
        "uncertainty_method": "held_out",
        "lineup_ready": True,
        "roster_ready": True,
    }
    observations = [
        {
            "sequence_number": sequence,
            "observed_at": (NOW + timedelta(seconds=45 * (sequence - 1))).isoformat(),
            "book_hash": f"book-{sequence}",
            "token_id": "token-t1",
            "complete": True,
            "average_price": "0.50",
            "decimal_odds": 2.0,
            "filled_risk": "1.0",
            "book": {"bids": [], "asks": []},
        }
        for sequence in (1, 2)
    ]
    action = {
        "proposal_id": "proposal-typed-1",
        "fixture_key": "pandascore:1",
        "target": "series_winner",
        "game_number": None,
        "total_line": None,
        "selection": "T1",
        "probability": 0.70,
        "probability_lower": 0.65,
        "market_id": "market-typed-1",
        "token_id": "token-t1",
        "market_url": "https://polymarket.com/event/test",
        "warnings": [],
        "state": "paper_actionable",
        "reason": None,
        "conservative_edge": 0.30,
        "stake_units": 1.0,
        "decimal_odds": 2.0,
        "observations": observations,
    }
    failed_action = {
        **action,
        "proposal_id": "proposal-typed-failed",
        "selection": "Gen.G",
        "token_id": "token-geng",
        "state": "blocked",
        "reason": "book_unavailable",
        "conservative_edge": None,
        "stake_units": 0.0,
        "decimal_odds": None,
        "observations": [],
    }

    record_daily_evidence(
        store=store,
        scheduled_for=NOW,
        effective_config={"horizon_hours": 36},
        schedule=schedule,
        snapshot_rows=[snapshot],
        steps=[DailyStepResult("schedule", True, "one fixture")],
        market_actions=[action, failed_action],
    )

    assert store.count(EvidenceTable.PREDICTIONS) == TWO_VERSIONS + 1
    assert store.count(EvidenceTable.MARKET_CANDIDATES) == TWO_VERSIONS
    assert store.count(EvidenceTable.MARKET_SNAPSHOTS) == TWO_VERSIONS
    proposal = store.get(EvidenceTable.PROPOSALS, "proposal-typed-1")
    assert proposal["state"] == "paper_actionable"
    assert json.loads(proposal["payload_json"])["read_only"] is True
    failed = store.get(EvidenceTable.PROPOSALS, "proposal-typed-failed")
    assert failed["state"] == "blocked"
    assert failed["market_snapshot_id"] is None
    assert failed["rejection_reason"] == "book_unavailable"

    position_id = decide_paper(
        store,
        proposal_id="proposal-typed-1",
        decision="accept",
        reason="paper test",
        actor_id="owner",
        created_at=NOW,
    )
    close_time = datetime(2026, 7, 27, 9, 55, tzinfo=UTC)

    class ClosingClient:
        def get_order_book(self, token_id):
            return OrderBook(
                condition_id="market-typed-1",
                token_id=token_id,
                timestamp=close_time,
                book_hash="closing-book",
                bids=(OrderLevel(price=Decimal("0.54"), size=Decimal(10)),),
                asks=(OrderLevel(price=Decimal("0.55"), size=Decimal(10)),),
                minimum_order_size=Decimal("0.1"),
                tick_size=Decimal("0.01"),
                negative_risk=False,
                last_trade_price=Decimal("0.54"),
            )

    capture = capture_closing_snapshots(
        store,
        client=ClosingClient(),
        now=close_time,
    )
    assert capture["captured"] == 1, capture
    original_append = store.append
    settlement_barrier = threading.Barrier(2)

    def racing_append(table, values):
        if table is EvidenceTable.SETTLEMENTS:
            settlement_barrier.wait()
        return original_append(table, values)

    store.append = racing_append
    with ThreadPoolExecutor(max_workers=2) as executor:
        settlement_ids = list(
            executor.map(
                lambda _: settle_paper(
                    store,
                    position_id=position_id,
                    result=SettlementResult.WIN,
                    source_reference="pandascore:match:1",
                    actor_id="owner",
                    note="verified result",
                ),
                range(2),
            )
        )
    store.append = original_append
    settlement_id = settlement_ids[0]
    assert settlement_ids == [settlement_id, settlement_id]
    assert (
        settle_paper(
            store,
            position_id=position_id,
            result=SettlementResult.WIN,
            source_reference="pandascore:match:1",
            actor_id="owner",
        )
        == settlement_id
    )
    settlement = store.get(EvidenceTable.SETTLEMENTS, settlement_id)
    assert settlement["result_source"] == "owner_verified"
    settlement_payload = json.loads(settlement["payload_json"])
    assert settlement_payload["actor_id"] == "owner"
    assert settlement_payload["entry_stake_units"] == "1.0"
    with pytest.raises(PaperEvidenceError, match="conflicting"):
        settle_paper(
            store,
            position_id=position_id,
            result=SettlementResult.LOSS,
            source_reference="pandascore:match:1",
            actor_id="owner",
        )
    with pytest.raises(PaperEvidenceError, match="conflicting"):
        settle_paper(
            store,
            position_id=position_id,
            result=SettlementResult.WIN,
            source_reference="pandascore:match:other",
            actor_id="owner",
        )
    summary = performance_summary(
        store,
        since=NOW - timedelta(minutes=1),
        target="series_winner",
        league="LCK",
    )

    assert summary["settled_count"] == 1
    assert summary["prediction_quality"]["count"] == 1
    assert summary["by_target"]["series_winner"]["wins"] == 1
    assert summary["mean_probability_clv"] == pytest.approx(0.05)


def test_same_day_market_rerun_appends_books_without_replacing_proposal(tmp_path):
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
                "match_key": "pandascore:1",
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
        "lineup_ready": True,
        "roster_ready": True,
    }

    def action(observed_at: datetime, odds: float) -> dict:
        return {
            "proposal_id": "proposal-stable",
            "fixture_key": "pandascore:1",
            "target": "series_winner",
            "game_number": None,
            "total_line": None,
            "selection": "T1",
            "probability": 0.70,
            "probability_lower": 0.65,
            "market_id": "market-1",
            "token_id": "token-t1",
            "market_url": "https://polymarket.com/event/test",
            "warnings": [],
            "state": "paper_actionable",
            "reason": None,
            "conservative_edge": 0.10,
            "stake_units": 1.0,
            "decimal_odds": odds,
            "observations": [
                {
                    "sequence_number": sequence,
                    "observed_at": (
                        observed_at + timedelta(seconds=45 * (sequence - 1))
                    ).isoformat(),
                    "book_hash": f"book-{observed_at.minute}-{sequence}",
                    "token_id": "token-t1",
                    "complete": True,
                    "average_price": str(1 / odds),
                    "decimal_odds": odds,
                    "filled_risk": "1.0",
                    "book": {"bids": [], "asks": []},
                }
                for sequence in (1, 2)
            ],
        }

    arguments = {
        "store": store,
        "scheduled_for": NOW,
        "effective_config": {"horizon_hours": 36},
        "schedule": schedule,
        "snapshot_rows": [snapshot],
        "steps": [DailyStepResult("schedule", True, "one fixture")],
    }
    record_daily_evidence(**arguments, market_actions=[action(NOW, INITIAL_ODDS)])
    record_daily_evidence(
        **arguments,
        market_actions=[action(NOW + timedelta(minutes=5), 1.9)],
    )

    assert store.count(EvidenceTable.PROPOSALS) == 1
    assert store.count(EvidenceTable.MARKET_CANDIDATES) == 1
    assert store.count(EvidenceTable.MARKET_SNAPSHOTS) == EXPECTED_RERUN_SNAPSHOTS
    proposal = store.get(EvidenceTable.PROPOSALS, "proposal-stable")
    assert json.loads(proposal["payload_json"])["odds"] == INITIAL_ODDS


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
