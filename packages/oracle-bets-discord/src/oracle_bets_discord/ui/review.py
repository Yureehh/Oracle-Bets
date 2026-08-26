"""Discord market-review controls with direct interaction completion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from lol_bets.operations.manual_market import (
    normalize_market_urls,
    parse_manual_lines_text,
    review_polymarket_events,
    thunderpick_fixture_options,
)

from oracle_bets_discord.formatting import DELIVERY_TARGET
from oracle_bets_discord.ui.presentation import market_options


@dataclass(frozen=True)
class ReviewViews:
    ReviewLinksModal: type[Any]


def build_review_views(
    discord: Any,
    *,
    store: Any,
    logger: Any,
    OwnerView: type[Any],
    BetOptionsView: type[Any],
) -> ReviewViews:
    """Build exact-link review controls bound to shared services."""

    async def run_review(
        interaction: Any,
        urls: tuple[str, ...],
        manual_lines: tuple[dict[str, Any], ...] = (),
        fixture_key: str | None = None,
    ) -> None:
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            result = await asyncio.to_thread(
                review_polymarket_events,
                urls,
                manual_lines=manual_lines,
                fixture_key=fixture_key,
            )
        except Exception as error:
            logger.exception("Owner market review failed")
            await interaction.edit_original_response(
                content=f"Review failed: {type(error).__name__}: {str(error)[:300]}"
            )
            return
        rows = await asyncio.to_thread(market_options, store, result.market_ids)
        attachment = discord.File(
            result.report_paths[1], filename="lol-market-review.md"
        )
        await interaction.edit_original_response(
            content=result.discord_message[:DELIVERY_TARGET],
            attachments=[attachment],
            view=BetOptionsView(rows) if rows else None,
        )

    class ThunderpickLineModal(discord.ui.Modal, title="Add Thunderpick line"):
        target = discord.ui.TextInput(
            label="Target", placeholder="series_winner or total_kills_mean"
        )
        selection = discord.ui.TextInput(
            label="Selection", placeholder="Team or Over/Under"
        )
        odds = discord.ui.TextInput(label="Decimal odds", placeholder="1.90")
        line = discord.ui.TextInput(label="Line (optional)", required=False)
        game = discord.ui.TextInput(label="Game number (optional)", required=False)

        def __init__(self, setup: Any) -> None:
            super().__init__()
            self.setup = setup

        async def on_submit(self, interaction: Any) -> None:
            try:
                line = parse_manual_lines_text(
                    " | ".join(
                        (
                            str(self.target),
                            str(self.selection),
                            str(self.odds),
                            str(self.line),
                            str(self.game),
                        )
                    )
                )[0]
            except Exception as error:
                await interaction.response.send_message(str(error), ephemeral=True)
                return
            self.setup.manual_lines.append(line)
            await interaction.response.edit_message(
                content=(
                    "Thunderpick is manual-only. "
                    f"**{len(self.setup.manual_lines)} line(s)** added. "
                    "Add another or finish the review."
                ),
                view=self.setup,
            )

    class ThunderpickSetupView(OwnerView):
        def __init__(
            self, urls: tuple[str, ...], fixtures: list[dict[str, Any]]
        ) -> None:
            super().__init__(timeout=900)
            self.urls = urls
            self.fixture_key: str | None = None
            self.manual_lines: list[dict[str, Any]] = []
            matching = [row for row in fixtures if row.get("_url_match")]
            if len(matching) == 1:
                self.fixture_key = str(matching[0]["match_key"])
            if fixtures:
                select = discord.ui.Select(
                    placeholder="Fixture (required if URL is ambiguous)",
                    options=[
                        discord.SelectOption(
                            label=f"{row['team_a']} vs {row['team_b']}"[:100],
                            value=str(row["match_key"]),
                            description=(
                                f"{str(row['start_utc'])[:16]} · "
                                f"{row['league']} · BO{row['best_of']}"
                            )[:100],
                            default=str(row["match_key"]) == self.fixture_key,
                        )
                        for row in fixtures[:25]
                    ],
                    row=0,
                )

                async def choose_fixture(interaction: Any) -> None:
                    self.fixture_key = select.values[0]
                    await interaction.response.defer()

                select.callback = choose_fixture
                self.add_item(select)

        @discord.ui.button(
            label="Add Thunderpick line", style=discord.ButtonStyle.primary
        )
        async def lines(self, interaction: Any, _button: Any) -> None:
            if self.fixture_key is None and len(self.urls) == 1:
                await interaction.response.send_message(
                    "Choose the fixture first.", ephemeral=True
                )
                return
            await interaction.response.send_modal(ThunderpickLineModal(self))

        @discord.ui.button(label="Finish Review")
        async def finish(self, interaction: Any, _button: Any) -> None:
            if self.fixture_key is None and len(self.urls) == 1:
                await interaction.response.send_message(
                    "Choose the fixture first.", ephemeral=True
                )
                return
            await run_review(
                interaction,
                self.urls,
                tuple(self.manual_lines),
                fixture_key=self.fixture_key,
            )

    class ReviewLinksModal(discord.ui.Modal, title="Review LoL market links"):
        links = discord.ui.TextInput(
            label="One or two links",
            style=discord.TextStyle.paragraph,
            placeholder="Polymarket and/or Thunderpick URLs",
        )

        async def on_submit(self, interaction: Any) -> None:
            try:
                urls = normalize_market_urls((str(self.links),))
            except Exception as error:
                await interaction.response.send_message(str(error), ephemeral=True)
                return
            if any("thunderpick.io" in url for url in urls):
                await interaction.response.defer(ephemeral=True, thinking=True)
                try:
                    thunderpick_url = next(
                        url for url in urls if "thunderpick.io" in url
                    )
                    fixtures = await asyncio.to_thread(
                        thunderpick_fixture_options, thunderpick_url
                    )
                    await interaction.edit_original_response(
                        content=(
                            "Thunderpick is manual-only. Select the fixture, add "
                            "visible lines, then finish the review."
                        ),
                        view=ThunderpickSetupView(urls, fixtures),
                    )
                except Exception:
                    logger.exception("Thunderpick fixture selection failed")
                    await interaction.edit_original_response(
                        content=(
                            "Thunderpick fixture lookup failed. Refresh the saved "
                            "schedule and try again."
                        ),
                        view=None,
                    )
                return
            await run_review(interaction, urls)

    return ReviewViews(ReviewLinksModal)
