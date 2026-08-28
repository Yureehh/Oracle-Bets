import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from oracle_bets_core.evidence import EvidenceStore
from oracle_bets_discord.bot import (
    HUB_BUTTON_LAYOUT,
    _acquire_instance_lock,
    _is_owner,
    _performance_result,
)
from oracle_bets_discord.delivery import (
    DiscordDeliveryMode,
    check_gateway_access,
    resolve_delivery_mode,
)
from oracle_bets_discord.formatting import sanitize_discord_text
from oracle_bets_discord.ui.charts import performance_png
from oracle_bets_discord.ui.common import require_owner
from oracle_bets_discord.ui.presentation import health_message, schedule_pages

EXPECTED_READ_CALLS = 2


def test_delivery_mode_is_explicit_and_gateway_prevents_webhook_duplication():
    environment = {
        "DISCORD_TOKEN": "token",
        "DISCORD_CHANNEL_ID": "123",
        "DISCORD_OWNER_USER_ID": "42",
        "DISCORD_WEBHOOK_URL": "https://discord.invalid/webhook",
    }

    assert resolve_delivery_mode("gateway", environment) is DiscordDeliveryMode.GATEWAY
    with pytest.raises(ValueError, match="gateway or off"):
        resolve_delivery_mode("webhook", environment)
    assert resolve_delivery_mode("off", environment) is DiscordDeliveryMode.OFF
    assert resolve_delivery_mode(None, environment) is DiscordDeliveryMode.GATEWAY


def test_delivery_mode_rejects_invalid_value():
    with pytest.raises(ValueError, match="gateway or off"):
        resolve_delivery_mode("both", {})


def test_explicit_empty_environment_does_not_fall_back_to_process_environment(
    monkeypatch,
):
    monkeypatch.setenv("DISCORD_DELIVERY_MODE", "gateway")
    assert resolve_delivery_mode(None, {}) is DiscordDeliveryMode.OFF


def test_discord_controls_reject_non_owner_identity():
    assert _is_owner(42, 42)
    assert not _is_owner(41, 42)


def test_modal_owner_guard_and_external_text_disable_mentions():
    class Response:
        def __init__(self):
            self.messages = []

        @staticmethod
        def is_done():
            return False

        async def send_message(self, content, **kwargs):
            self.messages.append((content, kwargs))

    interaction = SimpleNamespace(
        user=SimpleNamespace(id=41),
        response=Response(),
    )

    assert not asyncio.run(require_owner(interaction, 42))
    assert interaction.response.messages[0][1]["ephemeral"] is True
    assert sanitize_discord_text("@everyone\x00") == "@\u200beveryone"


def test_gateway_process_lock_rejects_second_instance(tmp_path):
    lock_path = tmp_path / "discord.lock"
    first = _acquire_instance_lock(lock_path)
    try:
        assert lock_path.read_text().strip().isdigit()
        with pytest.raises(RuntimeError, match="launchctl bootout"):
            _acquire_instance_lock(lock_path)
    finally:
        first.close()


def test_owner_console_button_rows_match_manual_workflow():
    assert HUB_BUTTON_LAYOUT == (
        ("Review Markets", "Record Bet", "Open Bets", "Closed Bets"),
        ("Schedule", "Performance", "Health"),
    )


def test_schedule_pages_use_short_utc_time_then_league_bo_and_teams():
    pages = schedule_pages(
        [
            {
                "start_utc": "2026-08-22 11:00:00+00:00",
                "league": "LCK",
                "best_of": 3,
                "team_a": "DN SOOPers",
                "team_b": "Kiwoom DRX",
            }
        ]
    )
    assert pages == [
        "**Upcoming actionable fixtures (UTC)**\n"
        "2026-08-22:11:00 · LCK · BO3 · **DN SOOPers vs Kiwoom DRX**"
    ]


def test_health_message_is_sectioned_and_human_readable(monkeypatch):
    monkeypatch.setattr(
        "lol_bets.module.LoLBetsModule.artifact_health",
        lambda _self: SimpleNamespace(ok=True),
    )
    monkeypatch.setattr(
        "oracle_bets_core.markets.PolymarketGammaAdapter.search_markets",
        lambda _self, _query, limit: [object()] if limit == 1 else [],
    )

    class Store:
        @staticmethod
        def integrity_check():
            return "ok"

        @staticmethod
        def list(_table):
            return []

    class Registry:
        @staticmethod
        def champion_id():
            return "lol-champion"

        @staticmethod
        def is_actionable(_champion):
            return True

    monkeypatch.setattr(
        "oracle_bets_core.operations.bets.count_open_bets", lambda _store: 0
    )
    message = health_message(Store(), Registry())

    assert "**Oracle Bets · System Health**" in message
    assert "🟢 **All systems operational**" in message
    assert "**Serving model**" in message
    assert "**Evidence**" in message
    assert "Thunderpick: 📝 Manual lines only" in message


def test_live_doctor_uses_read_only_discord_requests():
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Session:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            if url.endswith("/users/@me"):
                return Response({"username": "Oracle"})
            return Response({"name": "paper-bets"})

    session = Session()
    result = check_gateway_access("token", "123", session=session)

    assert result["bot"] == "Oracle"
    assert result["channel"] == "paper-bets"
    assert len(session.calls) == EXPECTED_READ_CALLS


def test_performance_returns_immediately_without_settled_bets(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    store.initialize_schema()

    image, summary = performance_png([], mode="paper")

    assert image == b""
    assert summary["settled_bets"] == 0
    assert not list(tmp_path.glob("*.png"))


def test_performance_failure_becomes_visible_message(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("renderer exploded")

    monkeypatch.setattr("oracle_bets_discord.ui.performance.performance_png", fail)
    monkeypatch.setattr(
        "oracle_bets_discord.ui.performance.performance_rows",
        lambda *_args, **_kwargs: [object()],
    )
    monkeypatch.setattr(
        "oracle_bets_discord.ui.performance.summarize_bets",
        lambda *_args, **_kwargs: {"settled": 1},
    )
    monkeypatch.setattr(
        "oracle_bets_discord.bot.logger.exception", lambda *_args, **_kwargs: None
    )

    image, message = asyncio.run(
        _performance_result(object(), mode="paper", since=None)
    )

    assert image is None
    assert "Performance chart failed" in message
    assert "renderer exploded" not in message


def test_discord_runtime_has_no_background_publisher_or_markers():
    from oracle_bets_discord import bot

    assert not hasattr(bot, "_publication_marker")
    assert not hasattr(bot, "_send_or_recover")
    assert "oracle-ref" not in Path(bot.__file__).read_text()
