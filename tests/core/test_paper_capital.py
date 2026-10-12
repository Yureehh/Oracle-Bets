import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from oracle_bets_core.evidence import EvidenceTable
from oracle_bets_core.operations import bets
from oracle_bets_core.operations.capital import load_paper_capital, paper_capital_status

from tests.core.test_bet_evidence import NOW, _quote, _store


def _config(tmp_path, **conversion):
    default = load_paper_capital()
    from oracle_bets_core.paths import SUITE_ROOT

    payload = json.loads(
        (SUITE_ROOT / "config/research/lol-paper-capital.json").read_text()
    )
    payload["venues"]["polymarket"].update(conversion)
    path = tmp_path / "capital.json"
    path.write_text(json.dumps(payload))
    return path, default


def _verified(tmp_path):
    path, _ = _config(
        tmp_path,
        native_units_per_eur="1.1",
        conversion_cost_native="2",
        costs_verified=True,
        source_reference="synthetic EUR/PUSD conversion",
        verified_at="2026-01-01T00:00:00+00:00",
    )
    return load_paper_capital(path)


def test_default_capital_allocation_blocks_unsourced_pusd():
    status = paper_capital_status()
    assert status["total_allocation_eur"] == "500"
    assert status["venues"]["thunderpick"]["initial_native_bankroll"] == "250"
    assert status["venues"]["polymarket"]["entry_ready"] is False
    with pytest.raises(ValueError, match="owner-verified"):
        load_paper_capital().venue("polymarket", "PUSD").initial_bankroll()


@pytest.mark.parametrize(
    "conversion",
    [
        {"native_units_per_eur": "NaN"},
        {"native_units_per_eur": "1.1"},
        {"allocation_eur": "300"},
        {"native_currency": "USD"},
        {"verified_at": datetime.now(UTC).replace(tzinfo=None).isoformat()},
        {"verified_at": (datetime.now(UTC) + timedelta(days=1)).isoformat()},
    ],
)
def test_invalid_conversion_is_rejected(tmp_path, conversion):
    path, _ = _config(tmp_path, **conversion)
    with pytest.raises(ValueError, match=r"Paper|conversion|Conversion"):
        load_paper_capital(path)


def test_current_bankroll_uses_active_corrected_pnl_and_provider_scope(
    tmp_path, monkeypatch
):
    capital = _verified(tmp_path)
    monkeypatch.setattr(bets, "load_paper_capital", lambda: capital)
    monkeypatch.setattr(bets, "_utc_now", lambda: NOW)
    store = _store(tmp_path)
    _quote(store, amount="2.73", currency="PUSD")
    bet = bets.record_bet(
        store,
        review_id="review-1",
        market_id="market-1",
        mode="paper",
        currency="PUSD",
        bankroll_before="273",
        stake_percent="1",
        accepted_odds="2",
        reason="test",
    )
    row = bets.list_bets(store)[0]
    assert row["payload"]["capital"]["config_hash"] == capital.config_hash
    bets.settle_bet(
        store, bet_id=bet, result="win", source_reference="result", settled_at=NOW
    )
    with pytest.raises(bets.BetEvidenceError, match="bankroll-before"):
        bets.paper_stake_limit(
            store,
            market_id="market-1",
            currency="PUSD",
            bankroll_before="273",
            accepted_odds="2",
        )
    limit = bets.paper_stake_limit(
        store,
        market_id="market-1",
        currency="PUSD",
        bankroll_before="275.73",
        accepted_odds="2",
    )
    assert limit["capital"]["settled_pnl"] == "2.73"
    bets.replace_bet_settlement(
        store,
        bet_id=bet,
        result="loss",
        source_reference="correct",
        correction_reason="mistake",
    )
    limit = bets.paper_stake_limit(
        store,
        market_id="market-1",
        currency="PUSD",
        bankroll_before="270.27",
        accepted_odds="2",
    )
    assert limit["capital"]["settled_pnl"] == "-2.73"
    assert limit["allowed_stake_amount"] == Decimal("5.40")


def test_capital_hash_is_locked_after_entry(tmp_path, monkeypatch):
    from dataclasses import replace

    capital = _verified(tmp_path)
    monkeypatch.setattr(bets, "load_paper_capital", lambda: capital)
    monkeypatch.setattr(bets, "_utc_now", lambda: NOW)
    store = _store(tmp_path)
    _quote(store, amount="2.73", currency="PUSD")
    bets.record_bet(
        store,
        review_id="review-1",
        market_id="market-1",
        mode="paper",
        currency="PUSD",
        bankroll_before="273",
        stake_percent="1",
        accepted_odds="2",
        reason="test",
    )
    changed = replace(capital, config_hash="changed")
    monkeypatch.setattr(bets, "load_paper_capital", lambda: changed)
    with pytest.raises(bets.BetEvidenceError, match="locked"):
        bets.paper_stake_limit(
            store,
            market_id="market-1",
            currency="PUSD",
            bankroll_before="273",
            accepted_odds="2",
        )
    assert store.count(EvidenceTable.BETS) == 1
