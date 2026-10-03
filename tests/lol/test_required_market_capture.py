from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from lol_bets.daily import DailyStepResult
from lol_bets.operations import manual_market
from lol_bets.operations.evidence import record_daily_evidence
from oracle_bets_core.evidence import EvidenceStore
from oracle_bets_core.pd import pd

REQUIRED_CELLS = 14
MODAL_FIELDS = 5
KILLS_LINE = 25.5


def test_required_capture_keeps_both_providers_and_unquoted_families():
    from lol_bets.operations.market_capture import required_market_capture

    coverage = required_market_capture(
        "fixture-1",
        actions=[
            {
                "provider": "thunderpick",
                "target": "series_winner",
                "decimal_odds": 1.9,
                "semantic_key": {"period": "series"},
            },
            {
                "provider": "thunderpick",
                "target": "map_winner",
                "decimal_odds": 1.9,
                "hard_blocks": ["semantic_contract:game number required"],
            },
        ],
    )
    assert len(coverage) == REQUIRED_CELLS
    cells = {(row["provider"], row["target"]): row for row in coverage}
    assert cells["thunderpick", "series_winner"]["state"] == "captured"
    assert cells["thunderpick", "map_winner"]["state"] == "unsupported"
    assert "game number required" in cells["thunderpick", "map_winner"]["reason"]
    assert cells["thunderpick", "series_handicap"]["state"] == "missing"
    assert all(
        row["state"] == "missing" for row in coverage if row["provider"] == "polymarket"
    )
    assert all(row["fixture_key"] == "fixture-1" and row["reason"] for row in coverage)


def test_inference_failure_retains_no_market_fixture_denominator(tmp_path, monkeypatch):
    fixture = {
        "match_key": "pandascore:approved",
        "provider": "pandascore",
        "provider_match_id": "approved",
        "league": "LPL",
        "team_a": "Team WE",
        "team_b": "Top Esports",
        "start_utc": "2026-08-22T20:00:00Z",
        "best_of": 3,
        "status": "not_started",
    }
    monkeypatch.setattr(
        manual_market, "_load_stored_schedule", lambda: pd.DataFrame([fixture])
    )
    monkeypatch.setattr(
        manual_market,
        "_build_prediction_snapshots",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("inference unavailable")),
    )
    store = EvidenceStore(tmp_path / "evidence.db")
    record_daily_evidence(
        store=store,
        scheduled_for=datetime(2026, 8, 21, 7, tzinfo=UTC),
        observed_at=datetime(2026, 8, 21, 7, tzinfo=UTC),
        effective_config={"horizon_hours": 48},
        schedule=pd.DataFrame([fixture]),
        snapshot_rows=(),
        steps=(DailyStepResult("schedule", True, "ready"),),
    )
    result = manual_market.review_polymarket_events(
        ["https://thunderpick.io/esports/league-of-legends/team-we-vs-top-esports"],
        fixture_key="pandascore:approved",
        store=store,
        report_dir=tmp_path / "reports",
        now=datetime(2026, 8, 21, 8, tzinfo=UTC),
    )
    payload = json.loads(result.report_paths[0].read_text())
    assert result.fixtures == 1
    assert result.predictions == 0
    assert payload["capture_fixture_count"] == 1
    assert len(payload["required_market_capture"]) == REQUIRED_CELLS
    assert all(row["state"] == "missing" for row in payload["required_market_capture"])
    assert (
        result.to_dict()["required_market_capture"]
        == payload["required_market_capture"]
    )
    assert "series_handicap" in result.report_paths[1].read_text()
    assert "Required market capture" in result.discord_message
    assert "Handicap" in result.discord_message
    assert result.failures[0]["reason"] == "inference_failed"
    assert result.cohort_status == "enrolled_before_review"
    assert payload["cohort_status"] == "enrolled_before_review"


def test_discord_capture_checklist_survives_line_and_batch_callbacks(monkeypatch):
    discord = pytest.importorskip("discord")
    from oracle_bets_discord.ui import review
    from oracle_bets_discord.ui.common import build_common_views

    async def exercise():
        owner_view, _ = build_common_views(discord, 1)
        views = review.build_review_views(
            discord,
            store=None,
            owner_id=1,
            logger=SimpleNamespace(exception=lambda *_a: None),
            OwnerView=owner_view,
            BetOptionsView=owner_view,
        )
        monkeypatch.setattr(
            review,
            "thunderpick_fixture_options",
            lambda _url: [
                {
                    "match_key": "fixture-1",
                    "team_a": "A",
                    "team_b": "B",
                    "start_utc": "2026-10-03T12:00:00Z",
                    "league": "LPL",
                    "best_of": 3,
                    "_url_match": True,
                }
            ],
        )
        interaction = SimpleNamespace(
            user=SimpleNamespace(id=1),
            id=123,
            response=SimpleNamespace(
                defer=AsyncMock(),
                edit_message=AsyncMock(),
                send_modal=AsyncMock(),
                send_message=AsyncMock(),
            ),
            edit_original_response=AsyncMock(),
        )
        modal = views.ReviewLinksModal()
        modal.links._value = "https://thunderpick.io/esports/league-of-legends/a-vs-b"
        await modal.on_submit(interaction)
        setup = interaction.edit_original_response.call_args.kwargs["view"]
        assert (
            "Required market capture"
            in interaction.edit_original_response.call_args.kwargs["content"]
        )
        await next(
            item
            for item in setup.children
            if getattr(item, "label", "") == "Add Thunderpick line"
        ).callback(interaction)
        line_modal = interaction.response.send_modal.call_args.args[0]
        assert len(line_modal.children) == MODAL_FIELDS
        for field, value in {
            "target": "total_kills_mean",
            "selection": "Over",
            "odds": "1.9",
            "line_game": "25.5 | 1",
            "terms": "Includes overtime; void on cancellation",
        }.items():
            getattr(line_modal, field)._value = value
        await line_modal.on_submit(interaction)
        assert (
            setup.manual_lines[0]["terms"] == "Includes overtime; void on cancellation"
        )
        assert setup.manual_lines[0]["game_number"] == 1
        assert setup.manual_lines[0]["line"] == KILLS_LINE
        assert "terms_verified" not in setup.manual_lines[0]
        assert (
            "Required market capture"
            in interaction.response.edit_message.call_args.kwargs["content"]
        )
        await next(
            item
            for item in setup.children
            if getattr(item, "label", "") == "Paste lines"
        ).callback(interaction)
        batch_modal = interaction.response.send_modal.call_args.args[0]
        batch_modal.lines._value = (
            "series_winner | A | 1.9 | | | | Void on cancellation"
        )
        await batch_modal.on_submit(interaction)
        assert (
            "Handicap" in interaction.response.edit_message.call_args.kwargs["content"]
        )
        assert (
            "Required market capture"
            in interaction.response.edit_message.call_args.kwargs["content"]
        )

    asyncio.run(exercise())
