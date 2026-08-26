from datetime import UTC, datetime

import pytest
from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.operations.bets import (
    BetEvidenceError,
    list_bets,
    performance_summary,
    record_bet,
    settle_bet,
)

TWO_BETS = 2


def _store(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    store.initialize_schema()
    store.append(
        EvidenceTable.RUNS,
        {
            "id": "review-1",
            "run_type": "manual_lol_market_review",
            "started_at": datetime(2026, 8, 24, tzinfo=UTC),
            "status": "complete",
            "idempotency_key": "review-1",
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
                "created_at": datetime(2026, 8, 24, tzinfo=UTC),
                "idempotency_key": team,
                "payload_json": {},
            },
        )
    store.append(
        EvidenceTable.FIXTURES,
        {
            "id": "fixture-1",
            "run_id": "review-1",
            "sport": "lol",
            "competition_id": "LPL",
            "team_a_identity_id": "a",
            "team_b_identity_id": "b",
            "start_time": datetime(2026, 8, 28, tzinfo=UTC),
            "best_of": 3,
            "status": "not_started",
            "idempotency_key": "fixture-1",
            "payload_json": {},
        },
    )
    store.append(
        EvidenceTable.MARKET_CANDIDATES,
        {
            "id": "market-1",
            "run_id": "review-1",
            "fixture_id": "fixture-1",
            "provider": "polymarket",
            "provider_market_id": "provider-1",
            "provider_selection_id": "token-1",
            "discovered_at": datetime(2026, 8, 24, tzinfo=UTC),
            "match_status": "matched",
            "rejection_reason": None,
            "idempotency_key": "market-1",
            "payload_json": {
                "target": "series_winner",
                "selection": "A",
                "probability": 0.6,
                "decimal_odds": 1.8,
                "url": "https://polymarket.com/event/test",
            },
        },
    )
    return store


def test_unified_bet_record_and_owner_settlement_are_append_only(tmp_path):
    store = _store(tmp_path)
    bet_id = record_bet(
        store,
        review_id="review-1",
        market_id="market-1",
        mode="paper",
        currency="eur",
        bankroll_before="1000",
        stake_percent="1",
        accepted_odds="2.0",
        reason="Owner selected it",
        opened_at=datetime(2026, 8, 24, 12, tzinfo=UTC),
    )

    row = list_bets(store, state="open", mode="paper")[0]
    assert row["id"] == bet_id
    assert row["stake_amount"] == "10"
    assert row["currency"] == "EUR"
    assert row["evidence_classification"] == "model_positive_ev"

    event_id = settle_bet(
        store,
        bet_id=bet_id,
        result="win",
        source_reference="match-result-1",
        settled_at=datetime(2026, 8, 29, tzinfo=UTC),
    )
    assert (
        settle_bet(
            store,
            bet_id=bet_id,
            result="win",
            source_reference="match-result-1",
            settled_at=datetime(2026, 8, 30, tzinfo=UTC),
        )
        == event_id
    )
    with pytest.raises(BetEvidenceError, match="conflicting"):
        settle_bet(
            store,
            bet_id=bet_id,
            result="loss",
            source_reference="match-result-2",
        )
    assert list_bets(store, state="open") == []
    assert list_bets(store, state="settled")[0]["settlement"]["pnl_amount"] == "10.0"


def test_performance_never_combines_modes_or_currencies(tmp_path):
    store = _store(tmp_path)
    for mode, currency, percent in (
        ("paper", "EUR", "1"),
        ("paper", "USD", "2"),
        ("real", "USD", "3"),
    ):
        bet_id = record_bet(
            store,
            review_id="review-1",
            market_id="market-1",
            mode=mode,
            currency=currency,
            bankroll_before="1000",
            stake_percent=percent,
            accepted_odds="2",
            reason=f"{mode} record",
            opened_at=datetime(2026, 8, 24, 12 + int(percent), tzinfo=UTC),
        )
        settle_bet(
            store,
            bet_id=bet_id,
            result="loss",
            source_reference=f"result-{mode}",
        )

    paper = performance_summary(store, mode="paper")["currencies"]
    assert paper["EUR"] == {
        "settled": 1,
        "graded": 1,
        "turnover": "10",
        "pnl": "-10",
        "roi": -1.0,
    }
    assert paper["USD"]["turnover"] == "20"
    assert paper["USD"]["pnl"] == "-20"
    assert set(performance_summary(store, mode="real")["currencies"]) == {"USD"}


def test_invalid_bet_inputs_do_not_write(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(BetEvidenceError, match="greater than 1"):
        record_bet(
            store,
            review_id="review-1",
            market_id="market-1",
            mode="paper",
            currency="EUR",
            bankroll_before=100,
            stake_percent=1,
            accepted_odds=1,
            reason="invalid",
        )
    assert store.count(EvidenceTable.BETS) == 0


def test_duplicate_record_is_idempotent_with_explicit_retry_key(tmp_path):
    store = _store(tmp_path)
    values = {
        "review_id": "review-1",
        "market_id": "market-1",
        "mode": "paper",
        "currency": "EUR",
        "bankroll_before": "1000",
        "stake_percent": "1",
        "accepted_odds": "1.9",
        "reason": "same owner action",
    }
    first = record_bet(
        store,
        **values,
        opened_at=datetime(2026, 8, 24, 12, tzinfo=UTC),
        idempotency_key="discord-interaction-1",
    )
    second = record_bet(
        store,
        **values,
        opened_at=datetime(2026, 8, 24, 12, tzinfo=UTC),
        idempotency_key="discord-interaction-1",
    )

    assert first == second
    assert store.count(EvidenceTable.BETS) == 1


def test_separate_same_terms_bets_are_distinct_owner_actions(tmp_path):
    store = _store(tmp_path)
    values = {
        "review_id": "review-1",
        "market_id": "market-1",
        "mode": "paper",
        "currency": "EUR",
        "bankroll_before": "1000",
        "stake_percent": "1",
        "accepted_odds": "1.9",
        "reason": "same terms, separate action",
    }

    first = record_bet(store, **values, opened_at=datetime(2026, 8, 24, 12, tzinfo=UTC))
    second = record_bet(
        store, **values, opened_at=datetime(2026, 8, 24, 12, 1, tzinfo=UTC)
    )

    assert first != second
    assert store.count(EvidenceTable.BETS) == TWO_BETS


def test_distinct_bankroll_evidence_never_collapses_to_one_bet(tmp_path):
    store = _store(tmp_path)
    common = {
        "review_id": "review-1",
        "market_id": "market-1",
        "mode": "paper",
        "currency": "EUR",
        "accepted_odds": "1.9",
        "reason": "same market, different owner terms",
    }
    first = record_bet(store, **common, bankroll_before="1000", stake_percent="1")
    second = record_bet(store, **common, bankroll_before="500", stake_percent="2")

    assert first != second
    assert store.count(EvidenceTable.BETS) == TWO_BETS


def test_settlement_retry_rejects_changed_audit_fields(tmp_path):
    store = _store(tmp_path)
    bet_id = record_bet(
        store,
        review_id="review-1",
        market_id="market-1",
        mode="paper",
        currency="EUR",
        bankroll_before="1000",
        stake_percent="1",
        accepted_odds="1.9",
        reason="owner action",
    )
    settle_bet(
        store,
        bet_id=bet_id,
        result="win",
        source_reference="result-1",
        note="verified",
    )

    with pytest.raises(BetEvidenceError, match="conflicting"):
        settle_bet(
            store,
            bet_id=bet_id,
            result="win",
            source_reference="result-1",
            note="changed after settlement",
        )


def test_explicit_stake_amount_must_match_bankroll_percentage(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(BetEvidenceError, match="must match"):
        record_bet(
            store,
            review_id="review-1",
            market_id="market-1",
            mode="real",
            currency="EUR",
            bankroll_before="1000",
            stake_percent="1",
            stake_amount="50",
            accepted_odds="1.9",
            reason="inconsistent",
        )
