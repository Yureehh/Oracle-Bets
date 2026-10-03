"""Composition of the Discord owner-console hub."""

from __future__ import annotations

import asyncio
from typing import Any

from lol_bets.data_generation.ingestion.schedule import fetch_and_store_schedule
from oracle_bets_core.league_selection import actionable_leagues
from oracle_bets_core.operations.bets import list_bets

from oracle_bets_discord.ui.presentation import (
    bet_pages,
    health_message,
    market_options,
    schedule_pages,
)

HUB_BUTTON_LAYOUT = (
    ("Review Markets", "Record Bet", "Open Bets", "Closed Bets"),
    ("Schedule", "Performance", "Health"),
)


def build_hub_view(
    discord: Any,
    *,
    store: Any,
    registry: Any,
    logger: Any,
    OwnerView: type[Any],
    PageView: type[Any],
    ReviewLinksModal: type[Any],
    BetOptionsView: type[Any],
    OpenBetsView: type[Any],
    PerformanceView: type[Any],
    RecentReviewsView: type[Any],
) -> type[Any]:
    """Build the seven-button owner hub from focused UI components."""

    class OracleHub(OwnerView):
        def __init__(self) -> None:
            super().__init__(timeout=900)
            callbacks = {
                "Review Markets": self.review,
                "Record Bet": self.record,
                "Open Bets": self.open_bets,
                "Closed Bets": self.closed_bets,
                "Schedule": self.schedule,
                "Performance": self.performance,
                "Health": self.health,
            }
            for row_number, labels in enumerate(HUB_BUTTON_LAYOUT):
                for label in labels:
                    button = discord.ui.Button(
                        label=label,
                        style=(
                            discord.ButtonStyle.primary
                            if label == "Review Markets"
                            else discord.ButtonStyle.secondary
                        ),
                        row=row_number,
                    )
                    button.callback = callbacks[label]
                    self.add_item(button)

        async def review(self, interaction: Any) -> None:
            await interaction.response.send_modal(ReviewLinksModal())

        async def record(self, interaction: Any) -> None:
            await interaction.response.defer(ephemeral=True, thinking=True)
            rows = await asyncio.to_thread(market_options, store)
            if not rows:
                await interaction.edit_original_response(
                    content="No reviewed markets are available. Review links first."
                )
                return
            await interaction.edit_original_response(
                content="Choose a reviewed outcome and how you already placed it.",
                view=BetOptionsView(rows),
            )

        async def open_bets(self, interaction: Any) -> None:
            await interaction.response.defer(ephemeral=True, thinking=True)
            rows = await asyncio.to_thread(list_bets, store, state="open")
            view = OpenBetsView(rows)
            await interaction.edit_original_response(
                content=f"{view.pages[0]}\n\nPage 1/{len(view.pages)}",
                view=view,
            )

        async def closed_bets(self, interaction: Any) -> None:
            await interaction.response.defer(ephemeral=True, thinking=True)
            rows = await asyncio.to_thread(list_bets, store, state="settled")
            pages = bet_pages(rows, title="Closed bets")
            view = PageView(pages)
            await interaction.edit_original_response(
                content=f"{pages[0]}\n\nPage 1/{len(pages)}", view=view
            )

        async def schedule(self, interaction: Any) -> None:
            await interaction.response.defer(ephemeral=True, thinking=True)
            try:
                frame = await asyncio.to_thread(
                    fetch_and_store_schedule,
                    window_days=14,
                    leagues=",".join(actionable_leagues()),
                )
                pages = schedule_pages(frame.to_dict(orient="records"))
            except Exception as error:
                logger.exception("Discord schedule failed")
                await interaction.edit_original_response(
                    content=f"Schedule failed: {type(error).__name__}. Check the private log."
                )
                return
            view = PageView(pages)
            await interaction.edit_original_response(
                content=f"{pages[0]}\n\nPage 1/{len(pages)}", view=view
            )

        async def performance(self, interaction: Any) -> None:
            await interaction.response.send_message(
                "Choose paper or real evidence and a timeframe.",
                view=PerformanceView(),
                ephemeral=True,
            )

        async def health(self, interaction: Any) -> None:
            await interaction.response.defer(ephemeral=True, thinking=True)
            text = await asyncio.to_thread(health_message, store, registry)
            await interaction.edit_original_response(
                content=text, view=RecentReviewsView()
            )

    return OracleHub
