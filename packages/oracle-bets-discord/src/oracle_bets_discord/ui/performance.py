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
    lane: str | None = None,
    target: str | None = None,
    provider: str | None = None,
) -> tuple[bytes | None, str]:
    """Return immediately when no evidence exists; bound expensive rendering."""
    global _render_inflight  # noqa: PLW0603
    try:
        rows = await asyncio.to_thread(
            performance_rows,
            store,
            mode=mode,
            since=since,
            lane=lane,
            target=target,
            provider=provider,
        )
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
    filters = " · ".join(
        value
        for value in (
            mode,
            lane or "all lanes",
            target or "all targets",
            provider or "all providers",
        )
    )
    currency_lines = [
        f"• **{currency}:** {bucket['settled']} tickets · "
        f"{bucket['fixture_clusters']} fixtures · ROI {float(bucket['roi'] or 0):+.1%} · "
        f"PnL {bucket['pnl']} · CLV {bucket['clv']['complete']}/{bucket['settled']}"
        for currency, bucket in summary["currencies"].items()
    ]
    return image, "\n".join(
        (f"**{mode.title()} performance**", f"Filters: `{filters}`", *currency_lines)
    )


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
            self.lane: str | None = None
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
            lane = discord.ui.Select(
                placeholder="Decision lane",
                options=[
                    discord.SelectOption(label="All lanes", value="all", default=True),
                    discord.SelectOption(label="Recommended", value="recommended"),
                    discord.SelectOption(label="Exploration", value="exploration"),
                ],
            )

            async def choose_lane(interaction: Any) -> None:
                self.lane = None if lane.values[0] == "all" else lane.values[0]
                await interaction.response.defer()

            lane.callback = choose_lane
            self.add_item(lane)

        @discord.ui.button(label="Continue", style=discord.ButtonStyle.primary)
        async def render(self, interaction: Any, _button: Any) -> None:
            await interaction.response.edit_message(
                content="Choose optional target/provider filters, then render.",
                view=PerformanceDetailView(self.mode, self.days, self.lane),
            )

    class PerformanceDetailView(OwnerView):
        def __init__(self, mode: str, days: int | None, lane: str | None) -> None:
            super().__init__(timeout=900)
            self.mode = mode
            self.days = days
            self.lane = lane
            self.target: str | None = None
            self.provider: str | None = None
            target = discord.ui.Select(
                placeholder="Target",
                options=[
                    discord.SelectOption(
                        label="All targets", value="all", default=True
                    ),
                    *[
                        discord.SelectOption(
                            label=value.replace("_", " ").title(), value=value
                        )
                        for value in (
                            "series_winner",
                            "map_winner",
                            "series_total_maps",
                            "series_handicap",
                            "gamelength_mean",
                            "total_kills_mean",
                            "total_towers_mean",
                        )
                    ],
                ],
            )

            async def choose_target(interaction: Any) -> None:
                self.target = None if target.values[0] == "all" else target.values[0]
                await interaction.response.defer()

            target.callback = choose_target
            self.add_item(target)
            provider = discord.ui.Select(
                placeholder="Provider",
                options=[
                    discord.SelectOption(
                        label="All providers", value="all", default=True
                    ),
                    discord.SelectOption(label="Polymarket", value="polymarket"),
                    discord.SelectOption(label="Thunderpick", value="thunderpick"),
                ],
            )

            async def choose_provider(interaction: Any) -> None:
                self.provider = (
                    None if provider.values[0] == "all" else provider.values[0]
                )
                await interaction.response.defer()

            provider.callback = choose_provider
            self.add_item(provider)

        @discord.ui.button(label="Render chart", style=discord.ButtonStyle.primary)
        async def render(self, interaction: Any, _button: Any) -> None:
            await interaction.response.defer(ephemeral=True, thinking=True)
            since = datetime.now(UTC) - timedelta(days=self.days) if self.days else None
            image, message = await performance_result(
                store,
                mode=self.mode,
                since=since,
                logger=logger,
                lane=self.lane,
                target=self.target,
                provider=self.provider,
            )
            await interaction.edit_original_response(
                content=message,
                attachments=[
                    discord.File(
                        BytesIO(image),
                        filename="bet-performance.png",
                        description="Oracle Bets filtered performance chart; metrics are summarized in the message.",
                    )
                ]
                if image
                else [],
            )

    return PerformanceView
