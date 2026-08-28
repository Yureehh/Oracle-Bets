"""Discord Gateway lifecycle and owner-console dependency wiring."""

from __future__ import annotations

import fcntl
import os
from typing import TYPE_CHECKING, Any

from lol_bets.operations.models import ModelRegistry
from oracle_bets_core.evidence import EvidenceStore
from oracle_bets_core.logger import LOG_TOPIC, instantiate_logger
from oracle_bets_core.paths import MODEL_REGISTRY_DIR, PRODUCT_STATE_DIR

from oracle_bets_discord.ui.bets import build_bet_views
from oracle_bets_discord.ui.common import build_common_views
from oracle_bets_discord.ui.hub import HUB_BUTTON_LAYOUT as _HUB_BUTTON_LAYOUT
from oracle_bets_discord.ui.hub import build_hub_view
from oracle_bets_discord.ui.performance import (
    build_performance_view,
    performance_result,
)
from oracle_bets_discord.ui.review import build_review_views

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path
    from typing import TextIO

logger = instantiate_logger(LOG_TOPIC.DISCORD)
HUB_BUTTON_LAYOUT = _HUB_BUTTON_LAYOUT


async def _performance_result(
    store: Any, *, mode: str, since: datetime | None
) -> tuple[bytes | None, str]:
    """Compatibility wrapper around the focused performance UI service."""
    return await performance_result(store, mode=mode, since=since, logger=logger)


def run_bot() -> None:
    """Run the owner console; never expose execution or trading tools."""
    import discord

    token = _required("DISCORD_TOKEN")
    owner_id = int(_required("DISCORD_OWNER_USER_ID"))
    instance_lock = _acquire_instance_lock(PRODUCT_STATE_DIR / "discord-bot.lock")
    store = EvidenceStore()
    store.initialize_schema()
    registry = ModelRegistry(MODEL_REGISTRY_DIR)
    client = discord.Client(intents=discord.Intents.none())
    tree = discord.app_commands.CommandTree(client)

    OwnerView, PageView = build_common_views(discord, owner_id)
    bet_views = build_bet_views(
        discord,
        store=store,
        owner_id=owner_id,
        OwnerView=OwnerView,
        PageView=PageView,
    )
    review_views = build_review_views(
        discord,
        store=store,
        owner_id=owner_id,
        logger=logger,
        OwnerView=OwnerView,
        BetOptionsView=bet_views.BetOptionsView,
    )
    PerformanceView = build_performance_view(
        discord, store=store, logger=logger, OwnerView=OwnerView
    )
    OracleHub = build_hub_view(
        discord,
        store=store,
        registry=registry,
        logger=logger,
        OwnerView=OwnerView,
        PageView=PageView,
        ReviewLinksModal=review_views.ReviewLinksModal,
        BetOptionsView=bet_views.BetOptionsView,
        OpenBetsView=bet_views.OpenBetsView,
        PerformanceView=PerformanceView,
        RecentReviewsView=review_views.RecentReviewsView,
    )

    @tree.command(name="oracle", description="Open the Oracle Bets owner console")
    async def oracle(interaction: discord.Interaction) -> None:
        if not _is_owner(interaction.user.id, owner_id):
            await interaction.response.send_message(
                "Owner-only control.", ephemeral=True
            )
            return
        await interaction.response.send_message(
            "**Oracle Bets owner console**\nIndependent predictions and manual bet tracking.",
            view=OracleHub(),
            ephemeral=True,
        )

    ready_once = False

    @client.event
    async def on_ready() -> None:
        nonlocal ready_once
        if ready_once:
            return
        try:
            await tree.sync()
        except Exception:
            logger.exception("Discord command sync failed")
            return
        ready_once = True
        logger.info("Discord owner console ready")

    try:
        client.run(token)
    finally:
        instance_lock.close()


def _acquire_instance_lock(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            "Discord Gateway bot is already running. Stop its launchd job with "
            "`launchctl bootout gui/$(id -u) "
            "~/Library/LaunchAgents/com.oracle-bets.discord.plist` "
            "or terminate the existing `oracle-bets discord run` process."
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for the Discord Gateway bot")
    return value


def _is_owner(user_id: int, owner_id: int) -> bool:
    return int(user_id) == int(owner_id)
