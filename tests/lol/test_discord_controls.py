from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

SELECT_LIMIT = 25
OVER_LIMIT = 26


def test_market_selector_reaches_second_page():
    discord = pytest.importorskip("discord")
    from oracle_bets_discord.ui.bets import build_bet_views
    from oracle_bets_discord.ui.common import build_common_views

    OwnerView, PageView = build_common_views(discord, 1)
    views = build_bet_views(
        discord, store=None, owner_id=1, OwnerView=OwnerView, PageView=PageView
    )
    rows = [
        {
            "market_id": str(i),
            "review_id": "r",
            "label": str(i),
            "description": "market",
            "stake_percent": "1",
        }
        for i in range(26)
    ]
    view = views.BetOptionsView(rows)
    selects = [item for item in view.children if isinstance(item, discord.ui.Select)]
    assert len(selects[0].options) == SELECT_LIMIT
    next_button = next(
        item for item in view.children if getattr(item, "label", None) == "Next markets"
    )
    interaction = SimpleNamespace(response=SimpleNamespace(edit_message=AsyncMock()))
    import asyncio

    asyncio.run(next_button.callback(interaction))
    assert selects[0].options[0].value == "25"


def test_open_tickets_reopen_settlement_and_reach_all_fixtures():
    discord = pytest.importorskip("discord")
    from oracle_bets_discord.ui.bets import build_bet_views
    from oracle_bets_discord.ui.common import build_common_views

    OwnerView, PageView = build_common_views(discord, 1)
    views = build_bet_views(
        discord, store=None, owner_id=1, OwnerView=OwnerView, PageView=PageView
    )
    rows = [
        {
            "id": str(i),
            "fixture_id": f"fixture{i}",
            "selection": "A",
            "provider": "thunderpick",
            "target": "series_winner",
            "accepted_odds": "2",
            "stake_percent": "1",
            "stake_amount": "1",
            "currency": "EUR",
        }
        for i in range(26)
    ]
    view = views.OpenBetsView(rows)
    select = next(
        item
        for item in view.children
        if getattr(item, "placeholder", "") == "Choose an open ticket"
    )
    next_button = next(
        item for item in view.children if getattr(item, "label", None) == "Next tickets"
    )
    interaction = SimpleNamespace(
        response=SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock())
    )
    import asyncio

    asyncio.run(next_button.callback(interaction))
    assert select.options[0].value == "25"
    select._values = ["25"]
    asyncio.run(select.callback(interaction))
    settlement = interaction.response.send_message.call_args.kwargs["view"]
    assert isinstance(settlement, views.SettlementView)
    assert {item.label for item in settlement.children} == {
        "Win",
        "Loss",
        "Push",
        "Void",
        "Capture close",
    }
    next_fixture = next(
        item
        for item in view.children
        if getattr(item, "label", None) == "Next fixtures"
    )
    asyncio.run(next_fixture.callback(interaction))
    fixture_select = next(
        item
        for item in view.children
        if getattr(item, "placeholder", "") == "Choose a fixture to record results"
    )
    assert fixture_select.options[0].value == "fixture25"


def test_performance_chart_orders_by_settlement_without_mutation(monkeypatch):
    import matplotlib.axes
    from oracle_bets_discord.ui.charts import performance_png

    rows = [
        {
            "currency": "EUR",
            "target": "series_winner",
            "provider": "thunderpick",
            "stake_amount": "1",
            "settlement": {"event_at": date, "pnl_amount": amount, "result": "win"},
            "evidence_classification": "model_unavailable",
            "payload": {},
        }
        for date, amount in [
            ("2026-10-12T00:00:00+00:00", "5"),
            ("2026-10-10T00:00:00+00:00", "2"),
            ("2026-10-11T00:00:00+00:00", "-1"),
        ]
    ]
    original = deepcopy(rows)
    monkeypatch.setattr(
        "oracle_bets_core.operations.bets.summarize_bets",
        lambda _rows, mode: {},  # noqa: ARG005
    )
    plotted = []
    original_plot = matplotlib.axes.Axes.plot

    def capture(self, *args, **kwargs):
        if kwargs.get("label") == "EUR":
            plotted.append(args[1])
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", capture)
    performance_png(rows, mode="paper")
    assert rows == original
    assert plotted == [[2.0, 1.0, 6.0]]


def test_paper_health_exposes_scope_and_readiness(monkeypatch):
    from oracle_bets_discord.ui.presentation import health_message

    monkeypatch.setattr(
        "lol_bets.module.LoLBetsModule.artifact_health",
        lambda _self: SimpleNamespace(ok=True),
    )
    monkeypatch.setattr(
        "oracle_bets_core.markets.PolymarketGammaAdapter.search_markets",
        lambda *_args, **_kwargs: [object()],
    )
    monkeypatch.setattr(
        "oracle_bets_core.operations.bets.count_open_bets", lambda _store: 0
    )
    registry = SimpleNamespace(
        champion_id=lambda: "paper",
        is_actionable=lambda _model: True,
        actionability=lambda _model: {"scope": "paper_research_only"},
        strategy_readiness=lambda: {
            "cells": [
                {
                    "state": "exploration_only",
                    "reasons": ["conservative_probability_coverage_failed"],
                }
            ]
        },
    )
    store = SimpleNamespace(integrity_check=lambda: "ok", list=lambda _table: [])
    message = health_message(store, registry)
    assert "paper_research_only" in message
    assert "exploration_only" in message
    assert "conservative_probability_coverage_failed" in message
    assert "All systems operational" not in message


def test_market_options_does_not_discard_26th_candidate(monkeypatch):
    import json

    from oracle_bets_core.evidence import EvidenceTable
    from oracle_bets_discord.ui.presentation import market_options

    monkeypatch.setattr(
        "oracle_bets_core.operations.bets.review_state", lambda _store, _run: "complete"
    )
    candidates = [
        {
            "id": str(i),
            "run_id": "review",
            "provider": "thunderpick",
            "payload_json": json.dumps(
                {
                    "classification": "exploration",
                    "target": "series_winner",
                    "selection": "A",
                    "decimal_odds": 2,
                    "sizing": {
                        "selected_path": "kelly",
                        "bankroll_fractions": {"kelly": 0.01},
                    },
                }
            ),
        }
        for i in range(26)
    ]
    store = SimpleNamespace(
        list=lambda _table: [] if _table == EvidenceTable.CORRECTIONS else None,
        get_many=lambda _table, _ids: candidates,
    )
    assert len(market_options(store, [str(i) for i in range(OVER_LIMIT)])) == OVER_LIMIT


def test_thunderpick_fixture_selector_reaches_all_candidates(monkeypatch):
    import asyncio

    discord = pytest.importorskip("discord")
    from oracle_bets_discord.ui.common import build_common_views
    from oracle_bets_discord.ui.review import build_review_views

    fixtures = [
        {
            "match_key": str(i),
            "team_a": "A",
            "team_b": "B",
            "start_utc": "2026-10-12T00:00:00Z",
            "league": "LCK",
            "best_of": 3,
        }
        for i in range(OVER_LIMIT)
    ]
    monkeypatch.setattr(
        "oracle_bets_discord.ui.review.thunderpick_fixture_options",
        lambda _url: fixtures,
    )
    OwnerView, _ = build_common_views(discord, 1)
    views = build_review_views(
        discord,
        store=None,
        owner_id=1,
        logger=SimpleNamespace(exception=lambda *_args: None),
        OwnerView=OwnerView,
        BetOptionsView=OwnerView,
    )
    modal = views.ReviewLinksModal()
    modal.links._value = "https://thunderpick.io/esports/league-of-legends"
    interaction = SimpleNamespace(
        user=SimpleNamespace(id=1),
        response=SimpleNamespace(defer=AsyncMock()),
        edit_original_response=AsyncMock(),
    )
    asyncio.run(modal.on_submit(interaction))
    setup = interaction.edit_original_response.call_args.kwargs["view"]
    select = next(
        item for item in setup.children if isinstance(item, discord.ui.Select)
    )
    next_button = next(
        item
        for item in setup.children
        if getattr(item, "label", None) == "Next fixtures"
    )
    interaction.response.edit_message = AsyncMock()
    asyncio.run(next_button.callback(interaction))
    assert select.options[0].value == "25"
    select._values = ["25"]
    asyncio.run(select.callback(interaction))
    assert setup.fixture_key == "25"


def test_duplicate_gateway_error_names_installed_launch_agent(tmp_path):
    from oracle_bets_discord.bot import _acquire_instance_lock

    lock = _acquire_instance_lock(tmp_path / "gateway.lock")
    try:
        with pytest.raises(RuntimeError, match=r"com\.oracle-bets\.discord-bot\.plist"):
            _acquire_instance_lock(tmp_path / "gateway.lock")
    finally:
        lock.close()
