from datetime import UTC, datetime

import pytest
from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.operations.bets import (
    BetEvidenceError,
    cohort_coverage,
    enroll_strategy_cohort,
    record_cohort_stage,
)

ENROLLED_AT = datetime(2026, 8, 24, tzinfo=UTC)


def _cohort_store(tmp_path, fixture_count=10):
    store = EvidenceStore(tmp_path / "cohort.db")
    store.initialize_schema()
    for suffix in ("a", "b"):
        store.append(
            EvidenceTable.IDENTITIES,
            {
                "id": f"team-{suffix}",
                "entity_type": "team",
                "canonical_name": f"Team {suffix.upper()}",
                "created_at": ENROLLED_AT,
                "idempotency_key": f"team-{suffix}",
                "payload_json": {},
            },
        )
    for number in range(fixture_count):
        fixture_id = f"fixture-{number}"
        store.append(
            EvidenceTable.FIXTURES,
            {
                "id": fixture_id,
                "run_id": None,
                "sport": "lol",
                "competition_id": "LPL",
                "team_a_identity_id": "team-a",
                "team_b_identity_id": "team-b",
                "start_time": datetime(2026, 8, 25, number, tzinfo=UTC),
                "best_of": 3,
                "status": "scheduled",
                "idempotency_key": fixture_id,
                "payload_json": {},
            },
        )
    return store


def test_cohort_enrollment_precedes_forecasts_and_tracks_intention_to_treat(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "oracle_bets_core.operations.bets._utc_now", lambda: ENROLLED_AT
    )
    store = _cohort_store(tmp_path)
    fixtures = tuple(f"fixture-{number}" for number in range(10))
    cohort_id = enroll_strategy_cohort(
        store,
        league="LPL",
        week_start="2026-08-24",
        fixture_ids=fixtures,
        cells=("series_winner|prematch|0.50-0.60",),
        policy_version="lol-market-policy-v1",
    )
    store.append(
        EvidenceTable.RUNS,
        {
            "id": "review-1",
            "run_type": "market_review",
            "started_at": ENROLLED_AT,
            "status": "completed",
            "idempotency_key": "review-1",
            "payload_json": {"fixture_ids": fixtures},
        },
    )
    store.append(
        EvidenceTable.MARKET_CANDIDATES,
        {
            "id": "series-a",
            "run_id": "review-1",
            "fixture_id": "fixture-0",
            "provider": "thunderpick",
            "provider_market_id": "series-a",
            "provider_selection_id": "team-a",
            "discovered_at": ENROLLED_AT,
            "match_status": "matched",
            "rejection_reason": None,
            "idempotency_key": "series-a",
            "payload_json": {"classification": "recommended"},
        },
    )
    for fixture_id in fixtures[:9]:
        record_cohort_stage(
            store, cohort_id, fixture_id, "reviewed", payload={"review_id": "review-1"}
        )
    record_cohort_stage(
        store,
        cohort_id,
        "fixture-0",
        "recommended",
        opportunity_id="series-a",
    )
    record_cohort_stage(
        store,
        cohort_id,
        "fixture-0",
        "rejected",
        opportunity_id="series-a",
    )
    record_cohort_stage(
        store,
        cohort_id,
        "fixture-0",
        "shadow_result",
        opportunity_id="series-a",
        payload={"result": "win", "source_reference": "official result"},
    )

    coverage = cohort_coverage(store, cohort_id)

    assert coverage == {
        "cohort_id": cohort_id,
        "enrolled_fixtures": 10,
        "reviewed_or_no_market_fixtures": 9,
        "fixture_review_fraction": 0.9,
        "recommendation_opportunities": 1,
        "recommendation_results": 1,
        "recommendation_result_fraction": 1.0,
        "activation_evidence_complete": True,
    }
    assert record_cohort_stage(
        store,
        cohort_id,
        "fixture-0",
        "shadow_result",
        opportunity_id="series-a",
        payload={"result": "win", "source_reference": "official result"},
    )


def test_cohort_enrollment_fails_after_any_prediction_or_price_evidence(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "oracle_bets_core.operations.bets._utc_now", lambda: ENROLLED_AT
    )
    store = _cohort_store(tmp_path, fixture_count=1)
    store.append(
        EvidenceTable.RUNS,
        {
            "id": "review-1",
            "run_type": "market_review",
            "started_at": ENROLLED_AT,
            "status": "completed",
            "idempotency_key": "review-1",
            "payload_json": {},
        },
    )
    store.append(
        EvidenceTable.MARKET_CANDIDATES,
        {
            "id": "market-1",
            "run_id": "review-1",
            "fixture_id": "fixture-0",
            "provider": "polymarket",
            "provider_market_id": "market-1",
            "provider_selection_id": "team-a",
            "discovered_at": ENROLLED_AT,
            "match_status": "matched",
            "rejection_reason": None,
            "idempotency_key": "market-1",
            "payload_json": {},
        },
    )

    with pytest.raises(BetEvidenceError, match="before forecasts or prices"):
        enroll_strategy_cohort(
            store,
            league="LPL",
            week_start="2026-08-24",
            fixture_ids=("fixture-0",),
            cells=("series_winner|prematch|0.50-0.60",),
            policy_version="lol-market-policy-v1",
        )


def test_cohort_rejects_wrong_week_or_past_fixture(tmp_path, monkeypatch):
    store = _cohort_store(tmp_path, fixture_count=1)
    monkeypatch.setattr(
        "oracle_bets_core.operations.bets._utc_now", lambda: ENROLLED_AT
    )
    with pytest.raises(BetEvidenceError, match="declared week"):
        enroll_strategy_cohort(
            store,
            league="LPL",
            week_start="2026-08-31",
            fixture_ids=("fixture-0",),
            cells=("series_winner",),
            policy_version="v1",
        )
    monkeypatch.setattr(
        "oracle_bets_core.operations.bets._utc_now",
        lambda: datetime(2026, 8, 26, tzinfo=UTC),
    )
    with pytest.raises(BetEvidenceError, match="after enrollment"):
        enroll_strategy_cohort(
            store,
            league="LPL",
            week_start="2026-08-24",
            fixture_ids=("fixture-0",),
            cells=("series_winner",),
            policy_version="v1",
        )


def test_cohort_cannot_claim_unlinked_recommendation_result(tmp_path, monkeypatch):
    store = _cohort_store(tmp_path, fixture_count=1)
    monkeypatch.setattr(
        "oracle_bets_core.operations.bets._utc_now", lambda: ENROLLED_AT
    )
    cohort_id = enroll_strategy_cohort(
        store,
        league="LPL",
        week_start="2026-08-24",
        fixture_ids=("fixture-0",),
        cells=("series_winner",),
        policy_version="v1",
    )
    with pytest.raises(BetEvidenceError, match="completed fixture review"):
        record_cohort_stage(store, cohort_id, "fixture-0", "reviewed")
    with pytest.raises(BetEvidenceError, match="market for this fixture"):
        record_cohort_stage(
            store, cohort_id, "fixture-0", "recommended", opportunity_id="invented"
        )
    assert cohort_coverage(store, cohort_id)["activation_evidence_complete"] is False
