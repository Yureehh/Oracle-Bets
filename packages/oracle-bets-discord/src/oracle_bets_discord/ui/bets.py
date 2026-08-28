"""Owner-only Discord controls for the unified manual bet ledger."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from oracle_bets_core.operations.bets import (
    BetEvidenceError,
    prepare_bet,
    preview_bet_result,
    record_bet,
    record_bet_result,
    settle_bet,
)

from oracle_bets_discord.ui.common import require_owner
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
    owner_id: int,
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
            if not await require_owner(interaction, owner_id):
                return
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
        stake_amount = discord.ui.TextInput(
            label="Stake amount (optional; must match strategy)", required=False
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
            stake_percent: str,
        ) -> None:
            super().__init__()
            self.review_id = review_id
            self.market_id = market_id
            self.mode = mode
            self.currency = currency
            self.stake_percent = stake_percent

        async def on_submit(self, interaction: Any) -> None:
            if not await require_owner(interaction, owner_id):
                return
            await interaction.response.defer(ephemeral=True, thinking=True)
            opened_at = datetime.now(UTC)
            arguments = {
                "review_id": self.review_id,
                "market_id": self.market_id,
                "mode": self.mode,
                "currency": self.currency,
                "bankroll_before": str(self.bankroll),
                "stake_percent": self.stake_percent,
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
                    stake_percent=row["stake_percent"],
                )
            )

    class OpenBetsView(PageView):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            super().__init__(bet_pages(rows, title="Open bets"))
            if not rows:
                return
            fixture_rows: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                fixture_rows.setdefault(str(row["fixture_id"]), []).append(row)
            selection = discord.ui.Select(
                placeholder="Choose a fixture to record results",
                options=[
                    discord.SelectOption(
                        label=f"{fixture_id[-12:]} · {len(group)} open bet(s)"[:100],
                        value=fixture_id,
                        description="Enter winner and/or named map statistics",
                    )
                    for fixture_id, group in list(fixture_rows.items())[:25]
                ],
                row=1,
            )

            async def select_bet(interaction: Any) -> None:
                fixture_id = selection.values[0]
                await interaction.response.send_modal(
                    FixtureResultModal(fixture_rows[fixture_id])
                )

            selection.callback = select_bet
            self.add_item(selection)

    class FixtureResultConfirmationView(OwnerView):
        def __init__(self, resolved: list[dict[str, Any]], source: str) -> None:
            super().__init__(timeout=120)
            self.resolved = resolved
            self.source = source

        @discord.ui.button(label="Confirm results", style=discord.ButtonStyle.success)
        async def confirm(self, interaction: Any, _button: Any) -> None:
            await interaction.response.defer(ephemeral=True, thinking=True)
            event_ids = []
            try:
                for item in self.resolved:
                    await asyncio.to_thread(
                        record_bet_result,
                        store,
                        bet_id=item["bet_id"],
                        source_reference=self.source,
                        winning_selection=item.get("winning_selection"),
                        observed_value=item.get("observed_value"),
                        actor_id=str(interaction.user.id),
                    )
                    event_ids.append(
                        await asyncio.to_thread(
                            settle_bet,
                            store,
                            bet_id=item["bet_id"],
                            result=item["result"],
                            source_reference=self.source,
                            actor_id=str(interaction.user.id),
                        )
                    )
            except BetEvidenceError as error:
                await interaction.edit_original_response(content=str(error), view=None)
                return
            await interaction.edit_original_response(
                content=f"Recorded {len(event_ids)} owner-confirmed settlement(s).",
                view=None,
            )

    class FixtureResultModal(discord.ui.Modal, title="Record fixture results"):
        facts = discord.ui.TextInput(
            label="Result facts",
            style=discord.TextStyle.paragraph,
            placeholder="series_winner=Team A\nmap1_kills=28",
        )
        source = discord.ui.TextInput(label="Result source reference")

        def __init__(self, rows: list[dict[str, Any]]) -> None:
            super().__init__()
            self.rows = rows

        async def on_submit(self, interaction: Any) -> None:
            if not await require_owner(interaction, owner_id):
                return
            await interaction.response.defer(ephemeral=True, thinking=True)
            try:
                facts = parse_result_facts(str(self.facts))
                resolved, unresolved = await asyncio.to_thread(
                    preview_fixture_results, store, self.rows, facts
                )
            except BetEvidenceError as error:
                await interaction.edit_original_response(content=str(error))
                return
            lines = [
                f"• `{item['bet_id'][-8:]}` {item['target']} → **{item['result']}**"
                for item in resolved
            ] + [
                f"• `{row['id'][-8:]}` {row['target']} → unresolved"
                for row in unresolved
            ]
            await interaction.edit_original_response(
                content="**Confirm fixture result preview**\n" + "\n".join(lines),
                view=(
                    FixtureResultConfirmationView(resolved, str(self.source))
                    if resolved
                    else None
                ),
            )

    return BetViews(BetOptionsView, OpenBetsView, SettlementView)


def parse_result_facts(value: str) -> dict[str, str]:
    """Parse controlled `name=value` result facts from Discord or tests."""
    output: dict[str, str] = {}
    for raw in value.splitlines():
        if not raw.strip():
            continue
        name, separator, fact = raw.partition("=")
        key = name.strip().casefold()
        if not separator or not key or not fact.strip():
            raise BetEvidenceError("Result facts must use one name=value per line.")
        if key in output:
            raise BetEvidenceError(f"Duplicate result fact: {key}")
        output[key] = fact.strip()
    if not output:
        raise BetEvidenceError("At least one result fact is required.")
    return output


def preview_fixture_results(
    store: Any,
    rows: list[dict[str, Any]],
    facts: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Preview every open ticket from named fixture facts without writing."""
    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload") or {}
        game = int(payload.get("game_number") or 1)
        target = str(row["target"])
        key = {
            "series_winner": "series_winner",
            "map_winner": f"map{game}_winner",
            "series_total_maps": "series_maps",
            "series_handicap": "series_margin",
            "gamelength_mean": f"map{game}_length",
            "total_kills_mean": f"map{game}_kills",
            "total_towers_mean": f"map{game}_towers",
        }.get(target)
        if key is None or key not in facts:
            unresolved.append(row)
            continue
        winner_target = target in {"series_winner", "map_winner"}
        arguments = {
            "winning_selection": facts[key] if winner_target else None,
            "observed_value": None if winner_target else facts[key],
        }
        result = preview_bet_result(store, bet_id=str(row["id"]), **arguments)
        resolved.append(
            {
                "bet_id": str(row["id"]),
                "target": target,
                "result": result,
                **arguments,
            }
        )
    return resolved, unresolved


def bet_confirmation_text(entry: dict[str, Any]) -> str:
    payload = entry.get("payload_json") or {}
    real_notice = (
        "\n\n⚠️ This only records a bet you already placed manually. "
        "Oracle Bets will not place it."
        if entry.get("mode") == "real"
        else ""
    )
    sizing = payload.get("sizing") or {}
    stake_units = sizing.get("stake_units") or {}
    sizing_lines = "\n".join(
        f"> {label}: **{float(stake_units.get(key, 0)):.2f}u**"
        for key, label in (
            ("full_kelly", "Full Kelly"),
            ("half_kelly", "Half Kelly"),
            ("quarter_kelly", "Quarter Kelly"),
            ("flat_1u", "Flat"),
        )
    )
    probability = payload.get("model_probability")
    point_ev = payload.get("point_ev")
    probability_line = (
        f" · Model **{float(probability):.1%}** · EV **{float(point_ev):+.1%}**"
        if probability is not None and point_ev is not None
        else " · No model probability"
    )
    return (
        "**Confirm bet record**\n"
        f"• **{str(entry['mode']).title()} · {payload.get('lane')}** · {entry['provider']}\n"
        f"• {entry['target']} · **{entry['selection']}**\n"
        f"• Accepted odds: **{entry['accepted_odds']}**{probability_line}\n"
        f"• Stake: **{entry['stake_amount']} {entry['currency']}** "
        f"({entry['stake_percent']}% of {entry['bankroll_before']})\n"
        f"• Strategy sizing (`1u = 1%`):\n{sizing_lines}\n"
        f"• Evidence: `{entry['evidence_classification']}`\n"
        f"• Note: {payload.get('reason')}"
        f"{real_notice}"
    )
