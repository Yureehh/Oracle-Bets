"""Owner-only Discord controls for the unified manual bet ledger."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from oracle_bets_core.operations.bets import (
    BetEvidenceError,
    prepare_bet,
    record_bet,
    settle_bet,
)

from oracle_bets_discord.ui.presentation import bet_pages


@dataclass(frozen=True)
class BetViews:
    BetOptionsView: type[Any]
    OpenBetsView: type[Any]
    SettlementView: type[Any]


def build_bet_views(
    discord: Any,
    *,
    store: Any,
    OwnerView: type[Any],
    PageView: type[Any],
) -> BetViews:
    """Build ledger views bound to one evidence store."""

    class SettlementModal(discord.ui.Modal, title="Settle tracked bet"):
        source = discord.ui.TextInput(label="Result source reference", required=True)
        closing = discord.ui.TextInput(
            label="Closing decimal odds (optional)", required=False
        )
        note = discord.ui.TextInput(label="Note (optional)", required=False)

        def __init__(self, bet_id: str, result: str) -> None:
            super().__init__()
            self.bet_id = bet_id
            self.result = result

        async def on_submit(self, interaction: Any) -> None:
            await interaction.response.defer(ephemeral=True, thinking=True)
            try:
                event_id = await asyncio.to_thread(
                    settle_bet,
                    store,
                    bet_id=self.bet_id,
                    result=self.result,
                    source_reference=str(self.source),
                    actor_id=str(interaction.user.id),
                    note=str(self.note) or None,
                    closing_odds=str(self.closing) or None,
                )
            except BetEvidenceError as error:
                await interaction.edit_original_response(content=str(error))
                return
            await interaction.edit_original_response(
                content=f"Recorded `{self.result}` settlement `{event_id}`."
            )

    class SettlementView(OwnerView):
        def __init__(self, bet_id: str) -> None:
            super().__init__(timeout=900)
            self.bet_id = bet_id
            for result, style in (
                ("win", discord.ButtonStyle.success),
                ("loss", discord.ButtonStyle.danger),
                ("push", discord.ButtonStyle.secondary),
                ("void", discord.ButtonStyle.secondary),
            ):
                button = discord.ui.Button(label=result.title(), style=style)
                button.callback = self._callback(result)
                self.add_item(button)

        def _callback(self, result: str) -> Any:
            async def callback(interaction: Any) -> None:
                await interaction.response.send_modal(
                    SettlementModal(self.bet_id, result)
                )

            return callback

    class BetConfirmationView(OwnerView):
        def __init__(self, **entry: Any) -> None:
            super().__init__(timeout=120)
            self.entry = entry

        @discord.ui.button(label="Confirm record", style=discord.ButtonStyle.success)
        async def confirm(self, interaction: Any, _button: Any) -> None:
            await interaction.response.defer(ephemeral=True, thinking=True)
            try:
                bet_id = await asyncio.to_thread(
                    record_bet,
                    store,
                    **self.entry,
                )
            except BetEvidenceError as error:
                await interaction.edit_original_response(content=str(error), view=None)
                return
            mode = str(self.entry["mode"])
            warning = (
                "\nThis records a bet you already placed manually; "
                "Oracle Bets did not place it."
                if mode == "real"
                else ""
            )
            await interaction.edit_original_response(
                content=f"Recorded `{mode}` bet `{bet_id}`.{warning}",
                view=SettlementView(bet_id),
            )

        @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
        async def cancel(self, interaction: Any, _button: Any) -> None:
            await interaction.response.edit_message(
                content="Bet record cancelled.", view=None
            )

    class BetRecordModal(discord.ui.Modal, title="Record manually placed bet"):
        accepted_odds = discord.ui.TextInput(label="Accepted decimal odds")
        bankroll = discord.ui.TextInput(label="Bankroll before bet")
        stake_percent = discord.ui.TextInput(label="Stake percentage")
        stake_amount = discord.ui.TextInput(
            label="Actual stake amount (optional)", required=False
        )
        reason = discord.ui.TextInput(
            label="Rationale", style=discord.TextStyle.paragraph
        )

        def __init__(
            self,
            *,
            review_id: str,
            market_id: str,
            mode: str,
            currency: str,
        ) -> None:
            super().__init__()
            self.review_id = review_id
            self.market_id = market_id
            self.mode = mode
            self.currency = currency

        async def on_submit(self, interaction: Any) -> None:
            await interaction.response.defer(ephemeral=True, thinking=True)
            opened_at = datetime.now(UTC)
            arguments = {
                "review_id": self.review_id,
                "market_id": self.market_id,
                "mode": self.mode,
                "currency": self.currency,
                "bankroll_before": str(self.bankroll),
                "stake_percent": str(self.stake_percent),
                "stake_amount": str(self.stake_amount) or None,
                "accepted_odds": str(self.accepted_odds),
                "reason": str(self.reason),
                "actor_id": str(interaction.user.id),
                "opened_at": opened_at,
                "idempotency_key": f"discord-bet:{interaction.id}",
            }
            try:
                entry = await asyncio.to_thread(prepare_bet, store, **arguments)
            except BetEvidenceError as error:
                await interaction.edit_original_response(content=str(error))
                return
            await interaction.edit_original_response(
                content=bet_confirmation_text(entry),
                view=BetConfirmationView(**arguments),
            )

    class BetOptionsView(OwnerView):
        def __init__(self, rows: list[dict[str, str]]) -> None:
            super().__init__(timeout=900)
            self.rows = {row["market_id"]: row for row in rows}
            self.market_id: str | None = None
            self.mode = "paper"
            self.currency = "EUR"
            market_select = discord.ui.Select(
                placeholder="Market outcome",
                options=[
                    discord.SelectOption(
                        label=row["label"],
                        value=row["market_id"],
                        description=row["description"],
                    )
                    for row in rows
                ],
                row=0,
            )

            async def choose_market(interaction: Any) -> None:
                self.market_id = market_select.values[0]
                await interaction.response.defer()

            market_select.callback = choose_market
            self.add_item(market_select)
            mode_select = discord.ui.Select(
                placeholder="Evidence mode",
                options=[
                    discord.SelectOption(label="Paper", value="paper", default=True),
                    discord.SelectOption(label="Real", value="real"),
                ],
                row=1,
            )

            async def choose_mode(interaction: Any) -> None:
                self.mode = mode_select.values[0]
                await interaction.response.defer()

            mode_select.callback = choose_mode
            self.add_item(mode_select)
            currency_select = discord.ui.Select(
                placeholder="Currency",
                options=[
                    discord.SelectOption(
                        label=value, value=value, default=value == "EUR"
                    )
                    for value in ("EUR", "USD", "USDT", "UNIT")
                ],
                row=2,
            )

            async def choose_currency(interaction: Any) -> None:
                self.currency = currency_select.values[0]
                await interaction.response.defer()

            currency_select.callback = choose_currency
            self.add_item(currency_select)

        @discord.ui.button(label="Fill bet", style=discord.ButtonStyle.primary, row=3)
        async def fill(self, interaction: Any, _button: Any) -> None:
            if self.market_id is None:
                await interaction.response.send_message(
                    "Choose a market outcome first.", ephemeral=True
                )
                return
            row = self.rows[self.market_id]
            await interaction.response.send_modal(
                BetRecordModal(
                    review_id=row["review_id"],
                    market_id=self.market_id,
                    mode=self.mode,
                    currency=self.currency,
                )
            )

    class OpenBetsView(PageView):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            super().__init__(bet_pages(rows, title="Open bets"))
            if not rows:
                return
            selection = discord.ui.Select(
                placeholder="Choose a bet to settle",
                options=[
                    discord.SelectOption(
                        label=f"{row['selection']} · {row['target']}"[:100],
                        value=str(row["id"]),
                        description=(
                            f"{row['mode']} · {row['stake_amount']} {row['currency']}"
                        )[:100],
                    )
                    for row in rows[:25]
                ],
                row=1,
            )

            async def select_bet(interaction: Any) -> None:
                bet_id = selection.values[0]
                await interaction.response.send_message(
                    f"Settle `{bet_id}`:",
                    view=SettlementView(bet_id),
                    ephemeral=True,
                )

            selection.callback = select_bet
            self.add_item(selection)

    return BetViews(BetOptionsView, OpenBetsView, SettlementView)


def bet_confirmation_text(entry: dict[str, Any]) -> str:
    payload = entry.get("payload_json") or {}
    real_notice = (
        "\n\n⚠️ This only records a bet you already placed manually. "
        "Oracle Bets will not place it."
        if entry.get("mode") == "real"
        else ""
    )
    return (
        "**Confirm bet record**\n"
        f"• Mode: **{str(entry['mode']).title()}** · {entry['provider']}\n"
        f"• {entry['target']} · **{entry['selection']}**\n"
        f"• Odds: **{entry['accepted_odds']}**\n"
        f"• Stake: **{entry['stake_amount']} {entry['currency']}** "
        f"({entry['stake_percent']}% of {entry['bankroll_before']})\n"
        f"• Evidence: `{entry['evidence_classification']}`\n"
        f"• Note: {payload.get('reason')}"
        f"{real_notice}"
    )
