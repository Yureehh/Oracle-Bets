import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from threading import Barrier

import pytest
from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.operations.bets import (
    BetEvidenceError,
    list_bets,
    paper_stake_limit,
    performance_summary,
    prepare_bet,
    record_bet,
    record_bet_result,
    record_closing_observation,
    record_fixture_results,
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
START_DELAY_SECONDS = 30
NOW = datetime(2026, 8, 24, 12, tzinfo=UTC)


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    from oracle_bets_core.operations import bets

    current = [NOW]
    monkeypatch.setattr(bets, "_utc_now", lambda: current[0])
    return current


def _quote(
    store,
    *,
    amount="10",
    currency="EUR",
    odds="2",
    observed_at=NOW,
    candidate_id="market-1",
    **details,
):
    sequence = store.count(EvidenceTable.MARKET_SNAPSHOTS) + 1
    snapshot_id = f"quote-{sequence}"
    store.append(
        EvidenceTable.MARKET_SNAPSHOTS,
        {
            "id": snapshot_id,
            "market_candidate_id": candidate_id,
            "observed_at": observed_at,
            "sequence_number": sequence,
            "intended_stake_units": "1",
            "expected_decimal_odds": odds,
            "available_stake_units": "1",
            "book_json": {},
            "idempotency_key": snapshot_id,
            "payload_json": {
                "complete": True,
                "semantic_fingerprint": "semantic-1",
                "terms_verified": True,
                "stake_currency": currency,
                "stake_amount": amount,
                "quote_basis": "stake_budget_vwap",
                **details,
            },
        },
    )
    return snapshot_id


def _store(tmp_path, *, with_quote=True, source_match_key=None):
    store = EvidenceStore(tmp_path / "evidence.db", lock_writes=False)
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
            "payload_json": {"source_match_key": source_match_key},
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
                        "balanced_kelly": 0.02,
                        "flat_1u": 0.01,
                        "full_kelly": 0.2,
                        "half_kelly": 0.1,
                        "quarter_kelly": 0.05,
                    },
                    "stake_units": {
                        "balanced_kelly": 2.0,
                        "flat_1u": 1.0,
                        "full_kelly": 20.0,
                        "half_kelly": 10.0,
                        "quarter_kelly": 5.0,
                    },
                    "selected_path": "balanced_kelly",
                },
                "reason_codes": ["all_recommendation_gates_passed"],
                "correlation_group": "fixture-1",
            },
        },
    )
    if with_quote:
        _quote(store)
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
    )

    row = list_bets(store, state="open", mode="paper")[0]
    assert row["id"] == bet_id
    assert row["stake_amount"] == "10"
    assert row["currency"] == "EUR"
    assert row["evidence_classification"] == "model_positive_ev"
    assert row["payload"]["lane"] == "recommended"
    assert row["payload"]["semantic_fingerprint"] == "semantic-1"
    assert row["payload"]["sizing"]["selected_path"] == "balanced_kelly"

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
        )


def test_performance_never_combines_modes_or_currencies(tmp_path):
    store = _store(tmp_path)
    for mode, currency, percent in (
        ("paper", "EUR", "1"),
        ("paper", "USD", "2"),
        ("real", "USD", "3"),
    ):
        _quote(store, amount=str(1000 * int(percent) // 100), currency=currency)
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
    clock,
):
    store = _store(tmp_path)
    bet_ids = []
    for index in range(2):
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
        )
        bet_ids.append(bet_id)
    clock[0] += timedelta(seconds=30)
    closing_id = _quote(store, odds="1.8", observed_at=clock[0])
    for index, (bet_id, result) in enumerate(
        zip(bet_ids, ("win", "loss"), strict=True)
    ):
        record_closing_observation(
            store,
            bet_id=bet_id,
            snapshot_id=closing_id,
            source_reference="provider-close-1",
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
        "accepted_odds": "2",
        "reason": "same owner action",
    }
    first = record_bet(
        store,
        **values,
        idempotency_key="discord-interaction-1",
    )
    second = record_bet(
        store,
        **values,
        idempotency_key="discord-interaction-1",
    )

    assert first == second
    assert store.count(EvidenceTable.BETS) == 1


def test_separate_same_terms_bets_are_distinct_owner_actions(tmp_path, clock):
    store = _store(tmp_path)
    values = {
        "review_id": "review-1",
        "market_id": "market-1",
        "mode": "paper",
        "currency": "EUR",
        "bankroll_before": "1000",
        "stake_percent": "1",
        "accepted_odds": "2",
        "reason": "same terms, separate action",
    }

    first = record_bet(store, **values)
    clock[0] += timedelta(seconds=60)
    second = record_bet(store, **values)

    assert first != second
    assert store.count(EvidenceTable.BETS) == TWO_BETS


def test_distinct_bankroll_evidence_never_collapses_to_one_bet(tmp_path):
    store = _store(tmp_path)
    common = {
        "review_id": "review-1",
        "market_id": "market-1",
        "mode": "paper",
        "currency": "EUR",
        "accepted_odds": "2",
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
        accepted_odds="2",
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
            accepted_odds="2",
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
    _quote(store, amount="20")
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
    )
    rows = list_bets(store, state="open")
    facts = parse_result_facts("series_winner=A")
    resolved, unresolved = preview_fixture_results(store, rows, facts)

    assert options[0]["review_id"] == "review-1"
    assert options[0]["stake_percent"] == "2.0"
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
        stake_percent="2",
        accepted_odds="2",
        reason="owner action",
    )
    confirmation = bet_confirmation_text(entry)
    assert "recommended" in confirmation
    assert "Full Kelly" in confirmation
    assert "Quarter Kelly" in confirmation


def test_fixture_result_batch_rolls_back_if_any_ticket_fails(tmp_path):
    store = _store(tmp_path)
    _quote(store, amount="20")
    bet_id = record_bet(
        store,
        review_id="review-1",
        market_id="market-1",
        mode="paper",
        currency="EUR",
        bankroll_before="1000",
        stake_percent="2",
        accepted_odds="2",
        reason="owner action",
    )
    with pytest.raises(BetEvidenceError, match="Unknown or superseded bet"):
        record_fixture_results(
            store,
            [
                {"bet_id": bet_id, "winning_selection": "A", "result": "win"},
                {"bet_id": "missing", "winning_selection": "A", "result": "win"},
            ],
            source_reference="official result",
            actor_id="owner",
        )
    assert store.count(EvidenceTable.BET_EVENTS) == 0


def test_discord_handicap_margin_is_oriented_to_each_selection(tmp_path, monkeypatch):
    from oracle_bets_core.operations.bets import _derive_result
    from oracle_bets_discord.ui import bets as discord_bets

    store = _store(tmp_path, with_quote=False)
    monkeypatch.setattr(
        discord_bets,
        "preview_bet_result",
        lambda _store, *, bet_id, **facts: (
            _derive_result(
                {
                    "target": "series_handicap",
                    "selection": bet_id,
                    "payload_json": json.dumps(
                        {"semantic_key": {"line": "-1.5" if bet_id == "A" else "1.5"}}
                    ),
                },
                **facts,
            ).value
        ),
    )
    rows = [
        {
            "id": team,
            "fixture_id": "fixture-1",
            "target": "series_handicap",
            "selection": team,
        }
        for team in ("A", "B")
    ]
    resolved, unresolved = preview_fixture_results(
        store, rows, {"series_winner": "A", "series_maps": "2"}
    )
    assert unresolved == []
    assert [(item["observed_value"], item["result"]) for item in resolved] == [
        ("2", "win"),
        ("-2", "loss"),
    ]
    with pytest.raises(BetEvidenceError, match="fixture teams"):
        preview_fixture_results(
            store, rows, {"series_winner": "Unknown", "series_maps": "2"}
        )


def test_paper_rejects_backdating_at_service_boundary(tmp_path, monkeypatch):
    from oracle_bets_core.operations import bets

    store = _store(tmp_path)
    monkeypatch.setattr(
        bets,
        "_utc_now",
        lambda: datetime(2026, 8, 24, 12, 3, tzinfo=UTC),
        raising=False,
    )
    with pytest.raises(BetEvidenceError, match="backdat"):
        record_bet(
            store,
            review_id="review-1",
            market_id="market-1",
            mode="paper",
            currency="EUR",
            bankroll_before="1000",
            stake_percent="1",
            accepted_odds="2",
            reason="too late",
            opened_at=datetime(2026, 8, 24, 12, tzinfo=UTC),
        )
    assert store.count(EvidenceTable.BETS) == 0


def test_paper_requires_captured_quote(tmp_path):
    store = _store(tmp_path, with_quote=False)
    with pytest.raises(BetEvidenceError, match="quote"):
        record_bet(
            store,
            review_id="review-1",
            market_id="market-1",
            mode="paper",
            currency="EUR",
            bankroll_before="1000",
            stake_percent="1",
            accepted_odds="2",
            reason="no quote",
        )


def _record(store, **overrides):
    values = {
        "review_id": "review-1",
        "market_id": "market-1",
        "mode": "paper",
        "currency": "EUR",
        "bankroll_before": "1000",
        "stake_percent": "1",
        "accepted_odds": "2",
        "reason": "owner action",
    }
    return record_bet(store, **(values | overrides))


@pytest.mark.parametrize(
    ("details", "message"),
    [
        ({"observed_at": NOW - timedelta(seconds=121)}, "expired"),
        ({"observed_at": NOW + timedelta(seconds=1)}, "expired"),
        ({"provider_timestamp": (NOW - timedelta(seconds=121)).isoformat()}, "expired"),
        ({"provider_timestamp": "not-a-time"}, "valid ISO datetime"),
        ({"provider_timestamp": "2026-08-24T12:00:00"}, "timezone-aware"),
        ({"terms_verified": False}, "unverified"),
        ({"semantic_fingerprint": "different-contract"}, "unverified"),
        ({"complete": False}, "completeness"),
        ({"odds": "1.9"}, "Accepted odds"),
        ({"currency": "USD"}, "currency"),
        ({"amount": None}, "priced stake"),
        ({"amount": "20"}, "stake differs"),
        ({"quote_basis": None}, "basis"),
    ],
)
def test_paper_refuses_invalid_quote_evidence(tmp_path, details, message):
    store = _store(tmp_path, with_quote=False)
    snapshot_id = _quote(store, **details)
    with pytest.raises(BetEvidenceError, match=message):
        _record(store, accepted_snapshot_id=snapshot_id)
    assert store.count(EvidenceTable.BETS) == 0


@pytest.mark.parametrize("change", ["selection", "fixture", "contract"])
def test_quote_must_belong_to_exact_candidate(tmp_path, change):
    store = _store(tmp_path)
    candidate = store.get(EvidenceTable.MARKET_CANDIDATES, "market-1")
    candidate.pop("content_hash")
    candidate.update(id="market-other", idempotency_key="market-other")
    if change == "selection":
        candidate["provider_selection_id"] = "token-other"
    elif change == "contract":
        candidate["provider_market_id"] = "contract-other"
    else:
        fixture = store.get(EvidenceTable.FIXTURES, "fixture-1")
        fixture.pop("content_hash")
        fixture.update(id="fixture-other", idempotency_key="fixture-other")
        store.append(EvidenceTable.FIXTURES, fixture)
        candidate["fixture_id"] = "fixture-other"
    store.append(EvidenceTable.MARKET_CANDIDATES, candidate)
    wrong_quote = _quote(store, candidate_id="market-other")
    with pytest.raises(
        BetEvidenceError, match="another fixture, selection, or contract"
    ):
        _record(store, accepted_snapshot_id=wrong_quote)


def test_paper_records_quote_and_actual_confirmation_time(tmp_path, clock):
    store = _store(tmp_path)
    clock[0] += timedelta(seconds=60)
    bet_id = _record(store)
    row = store.get(EvidenceTable.BETS, bet_id)
    assert datetime.fromisoformat(row["opened_at"]) == clock[0]
    assert row["accepted_snapshot_id"] == "quote-1"
    assert json.loads(row["payload_json"])["entry_timing"] == "prospective"


def test_changed_quote_requires_new_proposal(tmp_path):
    store = _store(tmp_path)
    _quote(store, odds="1.9")
    with pytest.raises(BetEvidenceError, match="replaced"):
        _record(store, accepted_snapshot_id="quote-1")


def test_retry_after_expiry_keeps_original_fact_and_rejects_changed_terms(
    tmp_path, clock
):
    store = _store(tmp_path)
    bet_id = _record(store, idempotency_key="confirmation-1")
    before = store.get(EvidenceTable.BETS, bet_id)
    clock[0] += timedelta(days=1)
    assert _record(store, idempotency_key="confirmation-1") == bet_id
    assert store.get(EvidenceTable.BETS, bet_id) == before
    with pytest.raises(BetEvidenceError, match="different immutable"):
        _record(store, idempotency_key="confirmation-1", reason="changed reason")
    assert store.count(EvidenceTable.BETS) == 1


def test_retrospective_real_entry_is_explicit_and_never_paper_evidence(tmp_path, clock):
    store = _store(tmp_path, with_quote=False)
    clock[0] = datetime(2100, 1, 1, tzinfo=UTC)
    bet_id = _record(store, mode="real", opened_at=NOW)
    row = store.get(EvidenceTable.BETS, bet_id)
    assert row["evidence_classification"] == "retrospective_real"
    assert row["accepted_snapshot_id"] is None
    payload = json.loads(row["payload_json"])
    assert payload["entry_timing"] == "retrospective"
    assert datetime.fromisoformat(payload["recorded_at"]) == clock[0]
    assert list_bets(store, mode="paper") == []
    with pytest.raises(BetEvidenceError, match="already started"):
        _record(store)


@pytest.mark.parametrize("odds", ["0.8", "1"])
def test_closing_odds_at_or_below_one_are_rejected(tmp_path, odds):
    store = _store(tmp_path)
    bet_id = _record(store)
    with pytest.raises(BetEvidenceError, match="greater than 1"):
        settle_bet(
            store,
            bet_id=bet_id,
            result="win",
            source_reference="result",
            closing_odds=odds,
        )
    assert store.count(EvidenceTable.BET_EVENTS) == 0


def test_bare_closing_odds_remain_unverified_and_excluded_from_clv(tmp_path):
    store = _store(tmp_path)
    bet_id = _record(store)
    settle_bet(
        store,
        bet_id=bet_id,
        result="win",
        source_reference="result",
        closing_odds="1.8",
    )
    row = list_bets(store, state="settled")[0]
    assert row["settlement"]["unverified_closing_odds"] == "1.8"
    assert row["settlement"]["closing_odds"] is None
    clv = performance_summary(store, mode="paper")["currencies"]["EUR"]["clv"]
    assert clv["complete"] == 0
    assert clv["missing"] == 1


@pytest.mark.parametrize(
    ("details", "message"),
    [
        ({"observed_at": NOW}, "after acceptance"),
        ({"observed_at": datetime(2100, 1, 1, tzinfo=UTC)}, "before fixture start"),
        ({"odds": "1"}, "greater than 1"),
        ({"semantic_fingerprint": "other-terms"}, "unverified"),
        ({"currency": "USD"}, "price bases"),
        ({"amount": "20"}, "price bases"),
        ({"quote_basis": None}, "price bases"),
        ({"quote_basis": "minimum_order_shares"}, "price bases"),
        ({"provider_timestamp": NOW.isoformat()}, "provider timestamp"),
    ],
)
def test_noncomparable_closing_quote_cannot_generate_clv(
    tmp_path, clock, details, message
):
    store = _store(tmp_path)
    bet_id = _record(store)
    clock[0] += timedelta(seconds=60)
    quote_id = _quote(store, **({"observed_at": clock[0], "odds": "1.8"} | details))
    with pytest.raises(BetEvidenceError, match=message):
        record_closing_observation(
            store, bet_id=bet_id, snapshot_id=quote_id, source_reference="close"
        )
    assert store.count(EvidenceTable.BET_EVENTS) == 0


def test_closing_capture_is_separate_idempotent_and_linked_through_settlement(
    tmp_path, clock
):
    store = _store(tmp_path)
    bet_id = _record(store)
    clock[0] += timedelta(seconds=60)
    snapshot_id = _quote(store, observed_at=clock[0], odds="1.8")
    event_id = record_closing_observation(
        store, bet_id=bet_id, snapshot_id=snapshot_id, source_reference="provider-close"
    )
    assert list_bets(store, state="open")[0]["id"] == bet_id
    assert (
        record_closing_observation(
            store,
            bet_id=bet_id,
            snapshot_id=snapshot_id,
            source_reference="provider-close",
        )
        == event_id
    )
    settled_id = settle_bet(
        store, bet_id=bet_id, result="win", source_reference="result"
    )
    assert (
        store.get(EvidenceTable.BET_EVENTS, settled_id)["closing_snapshot_id"]
        == snapshot_id
    )
    assert (
        performance_summary(store, mode="paper")["currencies"]["EUR"]["clv"]["complete"]
        == 1
    )


@pytest.mark.parametrize("delay_seconds", [30, 121])
def test_discord_confirm_uses_callback_clock_not_modal_clock(
    tmp_path, clock, delay_seconds, monkeypatch
):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from oracle_bets_discord.ui import bets as bet_ui
    from oracle_bets_discord.ui.bets import build_bet_views
    from oracle_bets_discord.ui.common import build_common_views

    discord = pytest.importorskip("discord")
    monkeypatch.setattr(
        bet_ui,
        "capture_paper_quote",
        lambda *_args, **_kwargs: {"id": "quote-1", "expected_decimal_odds": "2"},
    )
    store = _store(tmp_path)
    with store.connection() as connection:
        connection.execute("DROP TRIGGER prevent_fixtures_update")
        connection.execute(
            "UPDATE fixtures SET start_time = ?",
            (
                (
                    NOW
                    + timedelta(
                        seconds=30 if delay_seconds == START_DELAY_SECONDS else 300
                    )
                ).isoformat(),
            ),
        )

    async def scenario():
        owner_view, page_view = build_common_views(discord, 1)
        views = build_bet_views(
            discord, store=store, owner_id=1, OwnerView=owner_view, PageView=page_view
        )
        view = views.BetOptionsView(
            [
                {
                    "market_id": "market-1",
                    "review_id": "review-1",
                    "label": "A",
                    "description": "Series winner",
                    "stake_percent": "1",
                }
            ]
        )
        view.market_id = "market-1"
        interaction = SimpleNamespace(
            user=SimpleNamespace(id=1),
            id=123,
            response=SimpleNamespace(send_modal=AsyncMock(), defer=AsyncMock()),
            edit_original_response=AsyncMock(),
        )
        await next(
            item for item in view.children if getattr(item, "label", "") == "Fill bet"
        ).callback(interaction)
        modal = interaction.response.send_modal.await_args.args[0]
        modal.accepted_odds._value = "2"
        modal.bankroll._value = "1000"
        modal.stake_amount._value = "10"
        modal.attestation._value = "VERIFIED"
        modal.reason._value = "actual confirmation"
        await modal.on_submit(interaction)
        confirmation = interaction.edit_original_response.await_args.kwargs["view"]
        assert "opened_at" not in confirmation.entry
        assert confirmation.entry["accepted_snapshot_id"] == "quote-1"
        clock[0] += timedelta(seconds=delay_seconds)
        await next(
            item for item in confirmation.children if item.label == "Confirm record"
        ).callback(interaction)
        expected = (
            "already started" if delay_seconds == START_DELAY_SECONDS else "expired"
        )
        assert (
            expected in interaction.edit_original_response.await_args.kwargs["content"]
        )
        assert store.count(EvidenceTable.BETS) == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "overrides", [{"stake_percent": "2"}, {"accepted_odds": "1.9"}]
)
def test_retrospective_real_quote_reference_cannot_fabricate_comparable_clv(
    tmp_path, clock, overrides
):
    store = _store(tmp_path)
    bet_id = _record(store, mode="real", accepted_snapshot_id="quote-1", **overrides)
    clock[0] += timedelta(seconds=60)
    closing_id = _quote(store, observed_at=clock[0], odds="1.8")
    with pytest.raises(BetEvidenceError, match="price bases"):
        record_closing_observation(
            store, bet_id=bet_id, snapshot_id=closing_id, source_reference="close"
        )


def test_paper_kelly_ticket_limit_rejects_oversize_before_record(tmp_path):
    store = _store(tmp_path)
    _quote(store, amount="30")
    with pytest.raises(BetEvidenceError, match=r"Kelly.*reprice"):
        record_bet(
            store,
            review_id="review-1",
            market_id="market-1",
            mode="paper",
            currency="EUR",
            bankroll_before="1000",
            stake_percent="3",
            accepted_odds="2",
            reason="too large",
        )
    assert store.count(EvidenceTable.BETS) == 0


def test_paper_kelly_no_positive_edge_rejects(tmp_path):
    store = _store(tmp_path)
    _quote(store, odds="1.5")
    with pytest.raises(BetEvidenceError, match=r"Kelly.*reprice"):
        record_bet(
            store,
            review_id="review-1",
            market_id="market-1",
            mode="paper",
            currency="EUR",
            bankroll_before="1000",
            stake_percent="1",
            accepted_odds="1.5",
            reason="no edge",
        )


def _kelly_record(
    store,
    *,
    key,
    percent="2",
    mode="paper",
    currency="EUR",
    market_id="market-1",
    snapshot_id=None,
):
    return record_bet(
        store,
        review_id="review-1",
        market_id=market_id,
        mode=mode,
        currency=currency,
        bankroll_before="1000",
        stake_percent=percent,
        accepted_odds="2",
        reason="portfolio test",
        idempotency_key=key,
        accepted_snapshot_id=snapshot_id,
    )


def _kelly_limit(store, *, market_id="market-1", currency="EUR"):
    return paper_stake_limit(
        store,
        market_id=market_id,
        currency=currency,
        bankroll_before="1000",
        accepted_odds="2",
    )


def _kelly_other_fixture(store, suffix, *, source_match_key=None):
    fixture = dict(store.get(EvidenceTable.FIXTURES, "fixture-1"))
    fixture.pop("content_hash")
    fixture.update(
        id=f"fixture-{suffix}",
        idempotency_key=f"fixture-{suffix}",
        payload_json={"source_match_key": source_match_key},
    )
    store.append(EvidenceTable.FIXTURES, fixture)
    candidate = dict(store.get(EvidenceTable.MARKET_CANDIDATES, "market-1"))
    candidate.pop("content_hash")
    payload = json.loads(candidate["payload_json"])
    payload["correlation_group"] = fixture["id"]
    candidate.update(
        id=f"market-{suffix}",
        idempotency_key=f"market-{suffix}",
        fixture_id=fixture["id"],
        payload_json=payload,
    )
    store.append(EvidenceTable.MARKET_CANDIDATES, candidate)
    return candidate["id"]


def test_kelly_reserves_fixture_cap_and_allows_smaller_positive_stake(tmp_path):
    store = _store(tmp_path)
    _quote(store, amount="20")
    _kelly_record(store, key="one")
    _kelly_record(store, key="two")
    limit = _kelly_limit(store)
    assert limit["allowed_stake_amount"] == Decimal(10)
    assert limit["allowed_stake_percent"] == Decimal(1)
    _quote(store, amount="5")
    bet = _kelly_record(store, key="three", percent="0.5")
    evidence = json.loads(store.get(EvidenceTable.BETS, bet)["payload_json"])
    assert evidence["kelly_policy"]["status"] == "provisional_unvalidated"
    assert evidence["kelly_policy"]["chosen_stake_compliant"] is True
    assert evidence["kelly_policy"]["open_stakes"] == {"fixture-1": 40.0}
    assert _kelly_limit(store)["allowed_stake_amount"] == Decimal(5)


def test_kelly_releases_only_settlements_effective_by_confirmation(tmp_path, clock):
    store = _store(tmp_path)
    _quote(store, amount="20")
    first = _kelly_record(store, key="one")
    _kelly_record(store, key="two")
    settle_bet(
        store,
        bet_id=first,
        result="win",
        source_reference="fixture result",
        settled_at=NOW + timedelta(seconds=30),
    )
    assert _kelly_limit(store)["allowed_stake_amount"] == Decimal(10)
    clock[0] = NOW + timedelta(seconds=30)
    assert _kelly_limit(store)["allowed_stake_amount"] == Decimal(20)


def test_kelly_mode_currency_isolation_and_real_noncompliance(tmp_path):
    store = _store(tmp_path)
    real = _kelly_record(store, key="real", mode="real", percent="30")
    real_evidence = json.loads(store.get(EvidenceTable.BETS, real)["payload_json"])
    assert real_evidence["kelly_policy"]["chosen_stake_compliant"] is False
    _quote(store, amount="20", currency="USD")
    _kelly_record(store, key="dollars", currency="USD")
    assert _kelly_limit(store)["open_stakes"] == {}
    assert _kelly_limit(store, currency="USD")["open_stakes"] == {"fixture-1": 20.0}


def test_kelly_sporting_fixture_cap_survives_review_versions(tmp_path):
    store = _store(tmp_path, source_match_key="panda:123")
    _quote(store, amount="20")
    _kelly_record(store, key="one")
    _kelly_record(store, key="two")
    another = _kelly_other_fixture(store, "version-2", source_match_key="panda:123")
    limit = _kelly_limit(store, market_id=another)
    assert limit["correlation_group"] == "lol:panda:123"
    assert limit["allowed_stake_amount"] == Decimal(10)


def test_kelly_total_cap_across_independent_fixtures(tmp_path):
    store = _store(tmp_path)
    for index in range(10):
        market = _kelly_other_fixture(store, str(index + 2))
        quote = _quote(store, amount="20", candidate_id=market)
        _kelly_record(store, key=str(index), market_id=market, snapshot_id=quote)
    assert _kelly_limit(store)["allowed_stake_amount"] == 0
    with pytest.raises(BetEvidenceError, match=r"Kelly.*reprice"):
        _kelly_record(store, key="overflow", percent="1")


def test_concurrent_kelly_confirmations_serialize_exposure_check(tmp_path):
    store = _store(tmp_path)
    _quote(store, amount="20")
    _kelly_record(store, key="one")
    _kelly_record(store, key="two")
    _quote(store, amount="10")
    barrier = Barrier(2)

    def confirm(key):
        barrier.wait()
        try:
            return _kelly_record(store, key=key, percent="1")
        except BetEvidenceError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(confirm, ("race-one", "race-two")))
    assert len([result for result in results if isinstance(result, str)]) == 1
    assert (
        len([result for result in results if isinstance(result, BetEvidenceError)]) == 1
    )
    assert sum(
        Decimal(row["stake_amount"]) for row in store.list(EvidenceTable.BETS)
    ) == Decimal(50)


def test_kelly_idempotent_retry_survives_filled_cap_and_expired_quote(tmp_path, clock):
    store = _store(tmp_path)
    _quote(store, amount="20")
    first = _kelly_record(store, key="one")
    _kelly_record(store, key="two")
    _quote(store, amount="10")
    _kelly_record(store, key="three", percent="1")
    clock[0] = NOW + timedelta(minutes=10)
    assert _kelly_record(store, key="one") == first
    assert store.count(EvidenceTable.BETS) == TWO_BETS + 1


def test_evidence_validation_transaction_rolls_back_nested_append(tmp_path):
    store = _store(tmp_path)

    def aborted_write():
        with store.transaction() as transaction:
            entry = prepare_bet(
                transaction,
                review_id="review-1",
                market_id="market-1",
                mode="paper",
                currency="EUR",
                bankroll_before="1000",
                stake_percent="1",
                accepted_odds="2",
                reason="rollback test",
            )
            transaction.append(EvidenceTable.BETS, entry)
            assert transaction.count(EvidenceTable.BETS) == 1
            raise RuntimeError("abort")

    with pytest.raises(RuntimeError, match="abort"):
        aborted_write()
    assert store.count(EvidenceTable.BETS) == 0
