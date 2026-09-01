import json
import sqlite3
from datetime import UTC, datetime

import pytest
from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.operations.bets import (
    BetEvidenceError,
    list_bets,
    performance_summary,
    prepare_bet,
    record_bet,
    record_bet_result,
    replace_bet_settlement,
    settle_bet,
    supersede_evidence,
)
from oracle_bets_discord.ui.bets import (
    bet_confirmation_text,
    parse_result_facts,
    preview_fixture_results,
)
from oracle_bets_discord.ui.presentation import market_options

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
            "start_time": datetime(2099, 1, 1, tzinfo=UTC),
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
                "probability_lower": 0.56,
                "decimal_odds": 1.8,
                "url": "https://polymarket.com/event/test",
                "classification": "recommended",
                "readiness": "recommendation_active",
                "policy_version": "lol-market-policy-v1",
                "strategy_version": "series_direct_v2",
                "probability_source": "direct_series_model",
                "semantic_key": {
                    "version": 1,
                    "target": "series_winner",
                    "period": "series",
                    "selection": "a",
                    "line": None,
                },
                "semantic_fingerprint": "semantic-1",
                "sizing": {
                    "bankroll_fractions": {
                        "flat_1u": 0.01,
                        "full_kelly": 0.2,
                        "half_kelly": 0.1,
                        "quarter_kelly": 0.05,
                    },
                    "stake_units": {
                        "flat_1u": 1.0,
                        "full_kelly": 20.0,
                        "half_kelly": 10.0,
                        "quarter_kelly": 5.0,
                    },
                    "selected_path": "full_kelly",
                },
                "reason_codes": ["all_recommendation_gates_passed"],
                "correlation_group": "fixture-1",
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
    assert row["payload"]["lane"] == "recommended"
    assert row["payload"]["semantic_fingerprint"] == "semantic-1"
    assert row["payload"]["sizing"]["selected_path"] == "full_kelly"

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
    settlement = list_bets(store, state="settled")[0]["settlement"]
    assert settlement["pnl_amount"] == "10.0"
    assert settlement["counterfactual_pnl"] == {
        "flat_1u": "10",
        "full_kelly": "200",
        "half_kelly": "100",
        "quarter_kelly": "50",
    }
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        store.append(
            EvidenceTable.BET_EVENTS,
            {
                "id": "duplicate-settlement",
                "bet_id": bet_id,
                "event_at": datetime(2026, 8, 30, tzinfo=UTC),
                "event_type": "settlement",
                "actor_id": "owner-cli",
                "idempotency_key": "duplicate-settlement",
                "payload_json": {"result": "win"},
            },
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("started", "already started"),
        ("invalidated", "not completed"),
        ("corrected", "superseded"),
        ("not_comparable", "not recordable"),
    ],
)
def test_invalid_or_superseded_review_facts_cannot_open_bets(
    tmp_path, mutation, message
):
    store = _store(tmp_path)
    if mutation == "started":
        with store.connection() as connection:
            connection.execute("DROP TRIGGER prevent_fixtures_update")
            connection.execute(
                "UPDATE fixtures SET start_time = ? WHERE id = 'fixture-1'",
                ("2026-08-23T00:00:00Z",),
            )
    elif mutation == "invalidated":
        with store.connection() as connection:
            connection.execute("DROP TRIGGER prevent_runs_update")
            connection.execute(
                "UPDATE runs SET status = 'invalidated' WHERE id = 'review-1'"
            )
    elif mutation == "corrected":
        store.append(
            EvidenceTable.CORRECTIONS,
            {
                "id": "correction-market-1",
                "target_table": "market_candidates",
                "target_id": "market-1",
                "created_at": datetime(2026, 8, 24, 1, tzinfo=UTC),
                "reason": "Owner invalidated this match.",
                "replacement_id": None,
                "idempotency_key": "correction-market-1",
                "payload_json": {},
            },
        )
    else:
        with store.connection() as connection:
            connection.execute("DROP TRIGGER prevent_market_candidates_update")
            payload = json.loads(
                connection.execute(
                    "SELECT payload_json FROM market_candidates WHERE id = 'market-1'"
                ).fetchone()[0]
            )
            payload["classification"] = "not_comparable"
            connection.execute(
                "UPDATE market_candidates SET payload_json = ? WHERE id = 'market-1'",
                (json.dumps(payload),),
            )

    with pytest.raises(BetEvidenceError, match=message):
        record_bet(
            store,
            review_id="review-1",
            market_id="market-1",
            mode="paper",
            currency="EUR",
            bankroll_before="1000",
            stake_percent="1",
            accepted_odds="2",
            reason="must fail",
            opened_at=datetime(2026, 8, 24, 12, tzinfo=UTC),
        )


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
    assert {
        key: paper["EUR"][key]
        for key in ("settled", "graded", "turnover", "pnl", "roi")
    } == {
        "settled": 1,
        "graded": 1,
        "turnover": "10",
        "pnl": "-10",
        "roi": -1.0,
    }
    assert paper["USD"]["turnover"] == "20"
    assert paper["USD"]["pnl"] == "-20"
    assert set(performance_summary(store, mode="real")["currencies"]) == {"USD"}


def test_performance_reports_prediction_quality_clv_drawdown_and_fixture_clusters(
    tmp_path,
):
    store = _store(tmp_path)
    for index, result in enumerate(("win", "loss")):
        bet_id = record_bet(
            store,
            review_id="review-1",
            market_id="market-1",
            mode="paper",
            currency="EUR",
            bankroll_before="1000",
            stake_percent="1",
            accepted_odds="2",
            reason=f"owner action {index}",
            opened_at=datetime(2026, 8, 24, 12 + index, tzinfo=UTC),
        )
        settle_bet(
            store,
            bet_id=bet_id,
            result=result,
            source_reference=f"result-{index}",
            closing_odds="1.8",
            settled_at=datetime(2026, 8, 29, index, tzinfo=UTC),
        )

    summary = performance_summary(store, mode="paper")
    eur = summary["currencies"]["EUR"]

    assert eur["settled"] == TWO_BETS
    assert eur["fixture_clusters"] == 1
    assert eur["prediction_quality"]["count"] == TWO_BETS
    assert eur["prediction_quality"]["brier"] == pytest.approx(0.26)
    assert eur["clv"]["complete"] == TWO_BETS
    assert eur["clv"]["mean_probability"] == pytest.approx(1 / 1.8 - 1 / 2)
    assert eur["maximum_drawdown"] == "10"
    assert eur["fixture_clustered_roi_95"]["point"] == pytest.approx(0)
    assert set(eur["counterfactual_paths"]) == {
        "flat_1u",
        "full_kelly",
        "half_kelly",
        "quarter_kelly",
    }
    assert summary["cohorts"]["lane"]["recommended"]["tickets"] == TWO_BETS


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


def test_owner_result_fact_is_idempotent_and_conflicts_are_rejected(tmp_path):
    store = _store(tmp_path)
    bet_id = record_bet(
        store,
        review_id="review-1",
        market_id="market-1",
        mode="paper",
        currency="EUR",
        bankroll_before="1000",
        stake_percent="1",
        accepted_odds="2",
        reason="owner action",
        opened_at=datetime(2026, 8, 24, 12, tzinfo=UTC),
    )

    first = record_bet_result(
        store,
        bet_id=bet_id,
        winning_selection="A",
        source_reference="official-series-result",
        recorded_at=datetime(2026, 9, 1, 12, tzinfo=UTC),
    )
    repeated = record_bet_result(
        store,
        bet_id=bet_id,
        winning_selection="A",
        source_reference="official-series-result",
        recorded_at=datetime(2026, 9, 1, 13, tzinfo=UTC),
    )

    assert first == repeated
    event = store.get(EvidenceTable.BET_EVENTS, first)
    assert json.loads(event["payload_json"])["derived_result"] == "win"
    with pytest.raises(BetEvidenceError, match="conflicting"):
        record_bet_result(
            store,
            bet_id=bet_id,
            winning_selection="B",
            source_reference="different-result",
        )


def test_superseded_bet_or_settlement_is_excluded_without_mutation(tmp_path):
    store = _store(tmp_path)
    bet_id = record_bet(
        store,
        review_id="review-1",
        market_id="market-1",
        mode="paper",
        currency="EUR",
        bankroll_before="1000",
        stake_percent="1",
        accepted_odds="2",
        reason="owner action",
        opened_at=datetime(2026, 8, 24, 12, tzinfo=UTC),
    )
    settlement_id = settle_bet(
        store,
        bet_id=bet_id,
        result="win",
        source_reference="result-1",
    )
    correction_id = supersede_evidence(
        store,
        target_table="bet_events",
        target_id=settlement_id,
        reason="Owner selected the wrong result; replacement pending.",
    )

    assert store.get(EvidenceTable.BET_EVENTS, settlement_id) is not None
    assert store.get(EvidenceTable.CORRECTIONS, correction_id) is not None
    assert list_bets(store, state="settled") == []
    assert list_bets(store, state="open")[0]["id"] == bet_id

    supersede_evidence(
        store,
        target_table="bets",
        target_id=bet_id,
        reason="Entry terms were recorded incorrectly.",
    )
    assert list_bets(store) == []


def test_mistaken_settlement_is_replaced_append_only(tmp_path):
    store = _store(tmp_path)
    bet_id = record_bet(
        store,
        review_id="review-1",
        market_id="market-1",
        mode="paper",
        currency="EUR",
        bankroll_before="1000",
        stake_percent="1",
        accepted_odds="2",
        reason="owner action",
        opened_at=datetime(2026, 8, 24, 12, tzinfo=UTC),
    )
    original_id = settle_bet(
        store,
        bet_id=bet_id,
        result="loss",
        source_reference="mistyped-result",
    )

    replacement_id = replace_bet_settlement(
        store,
        bet_id=bet_id,
        result="win",
        source_reference="official-result",
        correction_reason="Owner selected Loss instead of Win.",
    )

    assert replacement_id != original_id
    assert (
        replace_bet_settlement(
            store,
            bet_id=bet_id,
            result="win",
            source_reference="official-result",
            correction_reason="Owner selected Loss instead of Win.",
        )
        == replacement_id
    )
    settled = list_bets(store, state="settled")[0]
    assert settled["settlement"]["result"] == "win"
    assert settled["settlement"]["pnl_amount"] == "10"
    assert store.get(EvidenceTable.BET_EVENTS, original_id) is not None
    correction = next(
        row
        for row in store.list(EvidenceTable.CORRECTIONS)
        if row["target_id"] == original_id
    )
    assert correction["replacement_id"] == replacement_id


def test_discord_uses_only_current_completed_review_and_previews_fixture_result(
    tmp_path,
):
    store = _store(tmp_path)
    store.append(
        EvidenceTable.RUNS,
        {
            "id": "review-new-partial",
            "run_type": "manual_lol_market_review",
            "started_at": datetime(2026, 8, 25, tzinfo=UTC),
            "status": "partial",
            "idempotency_key": "review-new-partial",
            "payload_json": {},
        },
    )

    options = market_options(store)
    bet_id = record_bet(
        store,
        review_id="review-1",
        market_id=options[0]["market_id"],
        mode="paper",
        currency="EUR",
        bankroll_before="1000",
        stake_percent=options[0]["stake_percent"],
        accepted_odds="2",
        reason="owner action",
        opened_at=datetime(2026, 8, 24, 12, tzinfo=UTC),
    )
    rows = list_bets(store, state="open")
    facts = parse_result_facts("series_winner=A")
    resolved, unresolved = preview_fixture_results(store, rows, facts)

    assert options[0]["review_id"] == "review-1"
    assert options[0]["stake_percent"] == "20.0"
    assert resolved[0]["bet_id"] == bet_id
    assert resolved[0]["result"] == "win"
    assert unresolved == []
    entry = prepare_bet(
        store,
        review_id="review-1",
        market_id="market-1",
        mode="paper",
        currency="EUR",
        bankroll_before="1000",
        stake_percent="20",
        accepted_odds="2",
        reason="owner action",
        opened_at=datetime(2026, 8, 24, 14, tzinfo=UTC),
    )
    confirmation = bet_confirmation_text(entry)
    assert "recommended" in confirmation
    assert "Full Kelly" in confirmation
    assert "Quarter Kelly" in confirmation
