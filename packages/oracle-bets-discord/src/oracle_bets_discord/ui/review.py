"""Discord market-review controls with direct interaction completion."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from lol_bets.operations.manual_market import (
    normalize_market_urls,
    parse_manual_lines_batch,
    parse_manual_lines_text,
    review_polymarket_events,
    thunderpick_fixture_options,
)
from lol_bets.operations.market_capture import (
    capture_checklist,
    required_market_capture,
)

from oracle_bets_discord.formatting import DELIVERY_TARGET, sanitize_discord_text
from oracle_bets_discord.ui.common import require_owner
from oracle_bets_discord.ui.presentation import market_options


@dataclass(frozen=True)
class ReviewViews:
    ReviewLinksModal: type[Any]
    RecentReviewsView: type[Any]


def build_review_views(
    discord: Any,
    *,
    store: Any,
    owner_id: int,
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
        review_key: str | None = None,
    ) -> None:
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            result = await asyncio.to_thread(
                review_polymarket_events,
                urls,
                manual_lines=manual_lines,
                fixture_key=fixture_key,
                store=store,
                review_key=review_key or f"discord-review:{interaction.id}",
            )
        except Exception as error:
            logger.exception("Owner market review failed")
            await interaction.edit_original_response(
                content=f"Review failed: {type(error).__name__}. Check the private log."
            )
            return
        rows = await asyncio.to_thread(market_options, store, result.market_ids)
        attachment = discord.File(
            result.report_paths[1], filename="lol-market-review.md"
        )
        await interaction.edit_original_response(
            content=sanitize_discord_text(result.discord_message)[:DELIVERY_TARGET],
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
        line_game = discord.ui.TextInput(
            label="Line | game number (optional)",
            placeholder="25.5 | 1",
            required=False,
        )
        terms = discord.ui.TextInput(
            label="Settlement terms",
            style=discord.TextStyle.paragraph,
            placeholder="Copy the market's settlement, void and overtime rules",
        )

        def __init__(self, setup: Any) -> None:
            super().__init__()
            self.setup = setup

        async def on_submit(self, interaction: Any) -> None:
            if not await require_owner(interaction, owner_id):
                return
            line_game = str(self.line_game).split("|")
            if len(line_game) > 2:  # noqa: PLR2004
                await interaction.response.send_message(
                    "Enter line | game number, for example 25.5 | 1.", ephemeral=True
                )
                return
            line_game += [""] * (2 - len(line_game))
            try:
                line = parse_manual_lines_text(
                    " | ".join(
                        (
                            str(self.target),
                            str(self.selection),
                            str(self.odds),
                            *line_game,
                            "",
                            str(self.terms),
                        )
                    )
                )[0]
            except Exception as error:
                await interaction.response.send_message(str(error), ephemeral=True)
                return
            self.setup.manual_lines.append(line)
            await interaction.response.edit_message(
                content=self.setup.summary(),
                view=self.setup,
            )

    class ThunderpickSetupView(OwnerView):
        def __init__(
            self,
            urls: tuple[str, ...],
            fixtures: list[dict[str, Any]],
            review_key: str | None = None,
        ) -> None:
            super().__init__(timeout=900)
            self.urls = urls
            self.fixture_key: str | None = None
            self.manual_lines: list[dict[str, Any]] = []
            self.review_key = review_key
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
                    row=1,
                )

                async def choose_fixture(interaction: Any) -> None:
                    self.fixture_key = select.values[0]
                    await interaction.response.edit_message(
                        content=self.summary(), view=self
                    )

                select.callback = choose_fixture
                self.add_item(select)

        def summary(self) -> str:
            coverage = required_market_capture(
                self.fixture_key or "unselected",
                actions=[
                    row | {"provider": "thunderpick"} for row in self.manual_lines
                ],
            )
            return (
                "Thunderpick is manual-only. Select the fixture, add visible lines and settlement "
                "terms, then finish the review. Missing targets remain recorded.\n"
                f"**{len(self.manual_lines)} line(s)** entered; review pending.\n\n"
                + capture_checklist(coverage)
            )

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

        @discord.ui.button(label="Paste lines", style=discord.ButtonStyle.secondary)
        async def paste(self, interaction: Any, _button: Any) -> None:
            await interaction.response.send_modal(ThunderpickBatchModal(self))

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
                review_key=self.review_key,
            )

    class ReviewLinksModal(discord.ui.Modal, title="Review LoL market links"):
        links = discord.ui.TextInput(
            label="One or two links",
            style=discord.TextStyle.paragraph,
            placeholder="Polymarket and/or Thunderpick URLs",
        )

        def __init__(
            self, *, default_links: str = "", review_key: str | None = None
        ) -> None:
            super().__init__()
            self.review_key = review_key
            if default_links:
                self.links.default = default_links

        async def on_submit(self, interaction: Any) -> None:
            if not await require_owner(interaction, owner_id):
                return
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
                    setup = ThunderpickSetupView(urls, fixtures, self.review_key)
                    await interaction.edit_original_response(
                        content=setup.summary(),
                        view=setup,
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
            await run_review(interaction, urls, review_key=self.review_key)

    class ThunderpickBatchModal(discord.ui.Modal, title="Paste Thunderpick lines"):
        lines = discord.ui.TextInput(
            label="One line per market",
            style=discord.TextStyle.paragraph,
            placeholder="target | selection | odds | line | game | observed_at | terms",
        )

        def __init__(self, setup: Any) -> None:
            super().__init__()
            self.setup = setup

        async def on_submit(self, interaction: Any) -> None:
            if not await require_owner(interaction, owner_id):
                return
            try:
                parsed, failures = parse_manual_lines_batch(str(self.lines))
            except Exception as error:
                await interaction.response.send_message(str(error), ephemeral=True)
                return
            self.setup.manual_lines.extend(parsed)
            failure_text = (
                "\n"
                + "\n".join(
                    f"• Row {item['row']}: {item['reason']}" for item in failures[:5]
                )
                if failures
                else ""
            )
            await interaction.response.edit_message(
                content=(
                    f"Added **{len(parsed)}** line(s); "
                    f"**{len(failures)}** row(s) rejected.{failure_text}\n\n"
                    + self.setup.summary()
                ),
                view=self.setup,
            )

    class RecentReviewsView(OwnerView):
        def __init__(self) -> None:
            super().__init__(timeout=900)
            from oracle_bets_core.evidence import EvidenceTable
            from oracle_bets_core.operations.bets import review_state

            rows = [
                row
                for row in store.list(EvidenceTable.RUNS)
                if row["run_type"] == "manual_lol_market_review"
            ][-25:]
            self.rows = {str(row["id"]): row for row in rows}
            self.review_id: str | None = None
            if rows:
                select = discord.ui.Select(
                    placeholder="Recent review run",
                    options=[
                        discord.SelectOption(
                            label=(
                                f"{review_state(store, str(row['id']))} · "
                                f"{str(row['started_at'])[:16]}"
                            )[:100],
                            value=str(row["id"]),
                            description=str(row["id"])[-24:],
                        )
                        for row in reversed(rows)
                    ],
                )

                async def choose(interaction: Any) -> None:
                    self.review_id = select.values[0]
                    await interaction.response.defer()

                select.callback = choose
                self.add_item(select)

        @discord.ui.button(label="Resume", style=discord.ButtonStyle.primary)
        async def resume(self, interaction: Any, _button: Any) -> None:
            if self.review_id is None:
                await interaction.response.send_message(
                    "Choose a review first.", ephemeral=True
                )
                return
            row = self.rows[self.review_id]
            from oracle_bets_core.operations.bets import review_state

            if review_state(store, self.review_id) not in {
                "queued",
                "running",
                "partial",
            }:
                await interaction.response.send_message(
                    "Only queued, running, or partial reviews can resume.",
                    ephemeral=True,
                )
                return
            payload = json.loads(str(row["payload_json"]))
            urls = "\n".join(payload.get("input_links") or [])
            await interaction.response.send_modal(
                ReviewLinksModal(
                    default_links=urls,
                    review_key=str(payload.get("run_key") or ""),
                )
            )

        @discord.ui.button(label="Invalidate", style=discord.ButtonStyle.danger)
        async def invalidate(self, interaction: Any, _button: Any) -> None:
            if self.review_id is None:
                await interaction.response.send_message(
                    "Choose a review first.", ephemeral=True
                )
                return
            from oracle_bets_core.operations.bets import supersede_evidence

            await asyncio.to_thread(
                supersede_evidence,
                store,
                target_table="runs",
                target_id=self.review_id,
                reason="Owner invalidated the Discord market review.",
                actor_id=str(interaction.user.id),
            )
            await interaction.response.edit_message(
                content=f"Invalidated review `{self.review_id}`.", view=None
            )

    return ReviewViews(ReviewLinksModal, RecentReviewsView)
