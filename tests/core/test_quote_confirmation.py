import json
from datetime import timedelta

import pytest
from oracle_bets_core.evidence import EvidenceTable
from oracle_bets_core.evidence.repository import TABLE_COLUMNS
from oracle_bets_core.markets import OrderBook
from oracle_bets_core.operations.bets import BetEvidenceError, prepare_bet
from oracle_bets_core.operations.quotes import capture_paper_quote

from tests.core.test_bet_evidence import NOW, _store

TWO_QUOTES = 2


def _candidate(store, provider="thunderpick"):
    row = store.get(EvidenceTable.MARKET_CANDIDATES, "market-1")
    row = {
        key: value
        for key, value in row.items()
        if key in TABLE_COLUMNS[EvidenceTable.MARKET_CANDIDATES]
    }
    payload = json.loads(row["payload_json"])
    payload["semantic_key"]["resolution_terms"] = "Unplayed maps void."
    payload["url"] = f"https://{provider}.com/event/test"
    row.update(id="capture-market", idempotency_key="capture-market", provider=provider)
    row["payload_json"] = payload
    store.append(EvidenceTable.MARKET_CANDIDATES, row)


def _capture(store, **overrides):
    return capture_paper_quote(
        store,
        **(
            {
                "review_id": "review-1",
                "market_id": "capture-market",
                "stake_amount": "10",
                "currency": "EUR",
                "net_decimal_odds": "2",
                "terms_attested": True,
                "actor_id": "owner-1",
                "clock": lambda: NOW,
            }
            | overrides
        ),
    )


def test_owner_capture_creates_sized_quote_accepted_by_paper_service(
    tmp_path, monkeypatch
):
    from oracle_bets_core.operations import bets

    monkeypatch.setattr(bets, "_utc_now", lambda: NOW)
    store = _store(tmp_path, with_quote=False)
    _candidate(store)
    quote = _capture(store)
    entry = prepare_bet(
        store,
        review_id="review-1",
        market_id="capture-market",
        mode="paper",
        currency="EUR",
        bankroll_before="1000",
        stake_percent="1",
        accepted_odds=quote["expected_decimal_odds"],
        accepted_snapshot_id=quote["id"],
        reason="Fresh owner capture",
    )
    assert entry["accepted_snapshot_id"] == quote["id"]
    details = quote["payload_json"]
    assert details["stake_amount"] == "10"
    assert details["terms_attested_by"] == "owner-1"
    assert details["resolution_terms"] == "Unplayed maps void."
    assert details["quote_basis"] == "owner_verified_net_odds"
    assert store.count(EvidenceTable.BETS) == 0


def test_recapture_preserves_candidate_and_does_not_reuse_timestamp(tmp_path):
    store = _store(tmp_path, with_quote=False)
    _candidate(store)
    first = _capture(store)
    second = _capture(
        store, clock=lambda: NOW + timedelta(seconds=121), net_decimal_odds="1.9"
    )
    assert first["id"] != second["id"]
    assert first["market_candidate_id"] == second["market_candidate_id"]
    assert second["expected_decimal_odds"] == "1.9"
    assert store.count(EvidenceTable.MARKET_SNAPSHOTS) == TWO_QUOTES


@pytest.mark.parametrize(
    "changes",
    [
        {"terms_attested": False},
        {"stake_amount": "NaN"},
        {"stake_amount": "0"},
        {"net_decimal_odds": "1"},
        {"review_id": "wrong"},
        {"actor_id": ""},
    ],
)
def test_capture_rejects_incomplete_owner_evidence(tmp_path, changes):
    store = _store(tmp_path, with_quote=False)
    _candidate(store)
    with pytest.raises(BetEvidenceError):
        _capture(store, **changes)
    assert store.count(EvidenceTable.MARKET_SNAPSHOTS) == 0


class BookClient:
    def get_order_book(self, token_id):
        return OrderBook.from_payload(
            {
                "market": "condition",
                "asset_id": token_id,
                "timestamp": NOW.isoformat(),
                "hash": "current-book",
                "bids": [],
                "asks": [{"price": "0.5", "size": "100"}],
                "min_order_size": "5",
                "tick_size": "0.01",
            },
            expected_token_id=token_id,
        )


def test_poly_capture_preserves_owner_net_cost_basis_and_book_depth(tmp_path):
    store = _store(tmp_path, with_quote=False)
    _candidate(store, "polymarket")
    quote = _capture(
        store, currency="PUSD", net_decimal_odds="1.95", clob_client=BookClient()
    )
    assert quote["expected_decimal_odds"] == "1.95"
    assert quote["payload_json"]["gross_book_odds"] == "2.0"
    assert quote["payload_json"]["stake_currency"] == "PUSD"
    assert quote["book_json"]["asks"][0]["price"] == "0.5"


@pytest.mark.parametrize(
    "changes",
    [
        {"currency": "EUR"},
        {"net_decimal_odds": "2.01"},
        {"clock": lambda: NOW + timedelta(seconds=121)},
    ],
)
def test_poly_capture_refuses_wrong_currency_optimistic_price_or_old_book(
    tmp_path, changes
):
    store = _store(tmp_path, with_quote=False)
    _candidate(store, "polymarket")
    with pytest.raises(BetEvidenceError):
        _capture(store, **({"currency": "PUSD", "clob_client": BookClient()} | changes))
    assert store.count(EvidenceTable.MARKET_SNAPSHOTS) == 0


@pytest.mark.parametrize("verified", [True, False])
def test_discord_paper_capture_refresh_and_confirm(tmp_path, monkeypatch, verified):  # noqa: PLR0915
    import asyncio
    from functools import partial
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from oracle_bets_core.operations import bets
    from oracle_bets_discord.ui import bets as ui
    from oracle_bets_discord.ui.common import build_common_views

    discord = pytest.importorskip("discord")
    current = [NOW]
    monkeypatch.setattr(bets, "_utc_now", lambda: current[0])
    monkeypatch.setattr(
        ui,
        "capture_paper_quote",
        partial(capture_paper_quote, clock=lambda: current[0]),
    )
    store = _store(tmp_path, with_quote=False)
    _candidate(store)

    async def scenario():  # noqa: PLR0915
        owner_view, page_view = build_common_views(discord, 1)
        views = ui.build_bet_views(
            discord,
            store=store,
            owner_id=1,
            OwnerView=owner_view,
            PageView=page_view,
        )
        view = views.BetOptionsView(
            [
                {
                    "market_id": "capture-market",
                    "review_id": "review-1",
                    "label": "A",
                    "description": "Winner",
                    "stake_percent": "1",
                }
            ]
        )
        view.market_id = "capture-market"
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

        async def fill(form, odds="2"):
            form.bankroll._value = "1000"
            form.stake_amount._value = "10"
            form.accepted_odds._value = odds
            form.reason._value = "Verified provider quote"
            form.attestation._value = "VERIFIED" if verified else ""
            await form.on_submit(interaction)
            return interaction.edit_original_response.await_args.kwargs["view"]

        confirmation = await fill(modal)
        confirm = next(
            item for item in confirmation.children if item.label == "Confirm record"
        )
        if not verified:
            assert confirm.disabled
            assert store.count(EvidenceTable.MARKET_SNAPSHOTS) == 0
            return
        original_quote = confirmation.entry["accepted_snapshot_id"]
        current[0] += timedelta(seconds=121)
        await confirm.callback(interaction)
        assert (
            "expired" in interaction.edit_original_response.await_args.kwargs["content"]
        )
        assert store.count(EvidenceTable.BETS) == 0
        await next(
            item for item in confirmation.children if item.label == "Refresh quote"
        ).callback(interaction)
        refreshed_modal = interaction.response.send_modal.await_args.args[0]
        assert refreshed_modal.market_id == "capture-market"
        interaction.id = 124
        refreshed = await fill(refreshed_modal, "1.9")
        assert refreshed.entry["accepted_snapshot_id"] != original_quote
        assert store.count(EvidenceTable.BETS) == 0
        await next(
            item for item in refreshed.children if item.label == "Confirm record"
        ).callback(interaction)
        row = bets.list_bets(store)[0]
        assert row["market_candidate_id"] == "capture-market"
        assert row["accepted_odds"] == "1.9"
        assert row["payload"]["entry_timing"] == "prospective"
        settlement_view = interaction.edit_original_response.await_args.kwargs["view"]
        current[0] += timedelta(seconds=60)
        await next(
            item for item in settlement_view.children if item.label == "Capture close"
        ).callback(interaction)
        close_modal = interaction.response.send_modal.await_args.args[0]
        close_modal.odds._value = "1.8"
        close_modal.attestation._value = "VERIFIED"
        close_modal.source._value = "same provider at original stake"
        await close_modal.on_submit(interaction)
        assert (
            "Closing quote captured"
            in interaction.edit_original_response.await_args.kwargs["content"]
        )
        bets.settle_bet(
            store,
            bet_id=row["id"],
            result="win",
            source_reference="official-result",
            settled_at=current[0] + timedelta(hours=1),
        )
        summary = bets.performance_summary(store, mode="paper")
        assert summary["currencies"]["EUR"]["pnl"] == "9.0"
        settlement = bets.list_bets(store)[0]["settlement"]
        assert settlement["clv_verified"] is True
        assert settlement["closing_odds"] == "1.8"

    asyncio.run(scenario())
