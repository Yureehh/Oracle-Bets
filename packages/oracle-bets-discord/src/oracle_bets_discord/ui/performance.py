"""Bounded Discord performance rendering for the unified ledger."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from io import BytesIO
from typing import Any

from oracle_bets_core.operations.bets import performance_rows, summarize_bets

from oracle_bets_discord.ui.charts import performance_png

PERFORMANCE_TIMEOUT_SECONDS = 20
_render_inflight: asyncio.Task[tuple[bytes, dict[str, Any]]] | None = None


async def performance_result(
    store: Any,
    *,
    mode: str,
    since: datetime | None,
    logger: Any,
) -> tuple[bytes | None, str]:
    """Return immediately when no evidence exists; bound expensive rendering."""
    global _render_inflight  # noqa: PLW0603
    try:
        rows = await asyncio.to_thread(performance_rows, store, mode=mode, since=since)
        summary = summarize_bets(rows, mode=mode)
    except Exception:
        logger.exception("Discord performance evidence read failed")
        return (
            None,
            "Performance evidence could not be read. Check `logs/lol/discord.log`.",
        )
    settled = int(summary.get("settled") or 0)
    if not settled:
        return None, f"No settled `{mode}` bets exist for this timeframe."
    if _render_inflight is not None and not _render_inflight.done():
        return None, "A performance chart is already rendering. Try again shortly."
    _render_inflight = asyncio.create_task(
        asyncio.to_thread(performance_png, rows, mode=mode)
    )
    active = _render_inflight
    try:
        image, _ = await asyncio.wait_for(
            asyncio.shield(active),
            timeout=PERFORMANCE_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        logger.exception("Discord performance chart timed out")
        return None, "Performance chart timed out. Check `logs/lol/discord.log`."
    except Exception:
        logger.exception("Discord performance chart failed")
        return None, "Performance chart failed. Check `logs/lol/discord.log`."
    finally:
        if active.done() and _render_inflight is active:
            _render_inflight = None
    return image, f"**{mode.title()} performance** · {settled} settled bet(s)."


def build_performance_view(
    discord: Any,
    *,
    store: Any,
    logger: Any,
    OwnerView: type[Any],
) -> type[Any]:
    class PerformanceView(OwnerView):
        def __init__(self) -> None:
            super().__init__(timeout=900)
            self.mode = "paper"
            self.days: int | None = None
            mode = discord.ui.Select(
                placeholder="Mode",
                options=[
                    discord.SelectOption(label="Paper", value="paper", default=True),
                    discord.SelectOption(label="Real", value="real"),
                ],
            )

            async def choose_mode(interaction: Any) -> None:
                self.mode = mode.values[0]
                await interaction.response.defer()

            mode.callback = choose_mode
            self.add_item(mode)
            timeframe = discord.ui.Select(
                placeholder="Timeframe",
                options=[
                    discord.SelectOption(label="All time", value="all", default=True),
                    discord.SelectOption(label="30 days", value="30"),
                    discord.SelectOption(label="90 days", value="90"),
                ],
            )

            async def choose_timeframe(interaction: Any) -> None:
                self.days = (
                    None if timeframe.values[0] == "all" else int(timeframe.values[0])
                )
                await interaction.response.defer()

            timeframe.callback = choose_timeframe
            self.add_item(timeframe)

        @discord.ui.button(label="Render chart", style=discord.ButtonStyle.primary)
        async def render(self, interaction: Any, _button: Any) -> None:
            await interaction.response.defer(ephemeral=True, thinking=True)
            since = datetime.now(UTC) - timedelta(days=self.days) if self.days else None
            image, message = await performance_result(
                store, mode=self.mode, since=since, logger=logger
            )
            await interaction.edit_original_response(
                content=message,
                attachments=[
                    discord.File(BytesIO(image), filename="bet-performance.png")
                ]
                if image
                else [],
            )

    return PerformanceView
