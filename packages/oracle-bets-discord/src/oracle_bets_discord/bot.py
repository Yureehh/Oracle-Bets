"""Owner-only persistent Discord Gateway controls for paper decisions."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TextIO

from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.evidence.settlement import SettlementResult
from oracle_bets_core.logger import logger
from oracle_bets_core.markets import PolymarketClobClient
from oracle_bets_core.operations.paper_evidence import (
    PaperEvidenceError,
    capture_closing_snapshots,
    decide_paper,
    paper_rows,
    paper_show,
    quote_prop,
    settle_paper,
)
from oracle_bets_core.paths import PRODUCT_STATE_DIR, REPORTS_DIR

from oracle_bets_discord.formatting import DELIVERY_TARGET

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path


def run_bot() -> None:  # noqa: PLR0915
    """Run review controls; never expose execution or trading tools."""
    import discord
    from discord.ext import tasks

    token = _required("DISCORD_TOKEN")
    owner_id = int(_required("DISCORD_OWNER_USER_ID"))
    channel_id = int(_required("DISCORD_CHANNEL_ID"))
    instance_lock = _acquire_instance_lock(PRODUCT_STATE_DIR / "discord-bot.lock")
    store = EvidenceStore()
    store.initialize_schema()
    clob = PolymarketClobClient()
    client = discord.Client(intents=discord.Intents.none())
    tree = discord.app_commands.CommandTree(client)

    class OwnerView(discord.ui.View):
        async def interaction_check(self, interaction: discord.Interaction) -> bool:
            if _is_owner(interaction.user.id, owner_id):
                return True
            await interaction.response.send_message(
                "Owner-only control.",
                ephemeral=True,
                allowed_mentions=discord.AllowedMentions.none(),
            )
            return False

    class PaperSettlementView(OwnerView):
        def __init__(self, position_id: str, *, disabled: bool = False) -> None:
            super().__init__(timeout=None)
            self.position_id = position_id
            for result, style in (
                (SettlementResult.WIN, discord.ButtonStyle.success),
                (SettlementResult.LOSS, discord.ButtonStyle.danger),
                (SettlementResult.PUSH, discord.ButtonStyle.secondary),
                (SettlementResult.VOID, discord.ButtonStyle.secondary),
            ):
                button = discord.ui.Button(
                    label=result.value.title(),
                    style=style,
                    custom_id=f"oracle:settle:{result.value}:{position_id}",
                    disabled=disabled,
                )
                button.callback = self._callback(result)
                self.add_item(button)

        def _callback(self, result: SettlementResult):
            async def callback(interaction: discord.Interaction) -> None:
                await interaction.response.send_modal(
                    SettlementConfirmationModal(self.position_id, result)
                )

            return callback

    class SettlementConfirmationModal(
        discord.ui.Modal,
        title="Confirm paper settlement",
    ):
        source_reference = discord.ui.TextInput(
            label="Result URL or source ID",
            required=True,
            max_length=300,
        )
        note = discord.ui.TextInput(
            label="Optional note",
            required=False,
            style=discord.TextStyle.paragraph,
            max_length=300,
        )

        def __init__(self, position_id: str, result: SettlementResult) -> None:
            super().__init__()
            self.position_id = position_id
            self.result = result

        async def on_submit(self, interaction: discord.Interaction) -> None:
            if not _is_owner(interaction.user.id, owner_id):
                await interaction.response.send_message(
                    "Owner-only control.", ephemeral=True
                )
                return
            try:
                _settle_owner_position(
                    store,
                    position_id=self.position_id,
                    result=self.result,
                    source_reference=self.source_reference.value,
                    actor_id=str(owner_id),
                    note=self.note.value,
                )
            except PaperEvidenceError as error:
                await interaction.response.send_message(str(error), ephemeral=True)
                return
            await interaction.response.edit_message(
                content=_settlement_message(paper_show(store, self.position_id)),
                view=PaperSettlementView(self.position_id, disabled=True),
            )

    class PaperDecisionView(OwnerView):
        def __init__(self, proposal_id: str) -> None:
            super().__init__(timeout=None)
            self.proposal_id = proposal_id
            for decision, style in (
                ("accept", discord.ButtonStyle.success),
                ("reject", discord.ButtonStyle.danger),
            ):
                button = discord.ui.Button(
                    label=f"{decision.title()} Paper",
                    style=style,
                    custom_id=f"oracle:paper:{decision}:{proposal_id}",
                )
                button.callback = self._callback(decision)
                self.add_item(button)

        def _callback(self, decision: str):
            async def callback(interaction: discord.Interaction) -> None:
                try:
                    position = decide_paper(
                        store,
                        proposal_id=self.proposal_id,
                        decision=decision,
                        reason="discord_owner_decision",
                        actor_id=str(owner_id),
                    )
                except PaperEvidenceError as error:
                    await interaction.response.send_message(str(error), ephemeral=True)
                    return
                for child in self.children:
                    if isinstance(child, discord.ui.Button):
                        child.disabled = True
                text = f"Decision recorded: {decision}."
                if position:
                    row = paper_show(store, position)
                    intent = _publication_intent(store, row, kind="position")
                    if interaction.message is not None:
                        message_id = int(interaction.message.id)
                        await interaction.response.edit_message(
                            content=_settlement_message(
                                row,
                                marker=str(intent["marker"]),
                            ),
                            view=PaperSettlementView(position),
                        )
                        _record_position_publication(
                            store,
                            row,
                            message_id,
                        )
                        published_positions[position] = message_id
                    else:
                        await interaction.response.send_message(
                            f"Position opened: `{position}`.",
                            ephemeral=True,
                            allowed_mentions=discord.AllowedMentions.none(),
                        )
                    published.pop(self.proposal_id, None)
                    return
                await interaction.response.edit_message(content=text, view=self)
                published.pop(self.proposal_id, None)

            return callback

    class PropQuoteModal(discord.ui.Modal, title="Record bookmaker prop line"):
        line = discord.ui.TextInput(label="Line", max_length=20)
        over_odds = discord.ui.TextInput(label="Over decimal odds", max_length=20)
        under_odds = discord.ui.TextInput(label="Under decimal odds", max_length=20)
        source = discord.ui.TextInput(label="Source", max_length=80)

        def __init__(self, forecast_id: str) -> None:
            super().__init__()
            self.forecast_id = forecast_id

        async def on_submit(self, interaction: discord.Interaction) -> None:
            if not _is_owner(interaction.user.id, owner_id):
                await interaction.response.send_message(
                    "Owner-only control.", ephemeral=True
                )
                return
            try:
                proposal_id = quote_prop(
                    store,
                    forecast_id=self.forecast_id,
                    line=float(self.line.value),
                    over_odds=float(self.over_odds.value),
                    under_odds=float(self.under_odds.value),
                    source=self.source.value,
                )
            except (PaperEvidenceError, ValueError) as error:
                await interaction.response.send_message(str(error), ephemeral=True)
                return
            await interaction.response.send_message(
                f"Proposal recorded: `{proposal_id}`. The Gateway publisher will "
                "post the owner decision card.",
                ephemeral=True,
                allowed_mentions=discord.AllowedMentions.none(),
            )

    @tree.command(name="paper_decide", description="Accept or reject a paper proposal")
    async def paper_decide(
        interaction: discord.Interaction,
        proposal_id: str,
        decision: str,
        reason: str = "",
    ) -> None:
        if not _is_owner(interaction.user.id, owner_id):
            await interaction.response.send_message(
                "Owner-only control.", ephemeral=True
            )
            return
        try:
            position = decide_paper(
                store,
                proposal_id=proposal_id,
                decision=decision,
                reason=reason or None,
                actor_id=str(owner_id),
            )
        except PaperEvidenceError as error:
            await interaction.response.send_message(str(error), ephemeral=True)
            return
        await interaction.response.send_message(
            f"Decision recorded: {decision}."
            + (f" Position: `{position}`." if position else ""),
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    @tree.command(
        name="paper_prop", description="Enter a line and odds for a prop forecast"
    )
    async def paper_prop(interaction: discord.Interaction, forecast_id: str) -> None:
        if not _is_owner(interaction.user.id, owner_id):
            await interaction.response.send_message(
                "Owner-only control.", ephemeral=True
            )
            return
        await interaction.response.send_modal(PropQuoteModal(forecast_id))

    published = _published_proposals(store)
    published_positions = _published_positions(store)
    report_cache: dict[str, Path | None] = {}
    pending_ids = {
        str(row["proposal_id"]) for row in paper_rows(store, state="pending")
    }
    for proposal_id, message_id in published.items():
        if proposal_id in pending_ids:
            client.add_view(PaperDecisionView(proposal_id), message_id=message_id)
    open_ids = {str(row["position_id"]) for row in paper_rows(store, state="open")}
    for position_id, message_id in published_positions.items():
        if position_id in open_ids:
            client.add_view(PaperSettlementView(position_id), message_id=message_id)

    async def refresh_controls(channel, pending_rows, open_rows) -> None:
        pending_ids = {str(row["proposal_id"]) for row in pending_rows}
        open_ids = {str(row["position_id"]) for row in open_rows}
        for proposal_id, message_id in tuple(published.items()):
            if proposal_id in pending_ids:
                continue
            try:
                row = await asyncio.to_thread(paper_show, store, proposal_id)
                message = await channel.fetch_message(message_id)
                position_id = row.get("position_id")
                if row["state"] == "open" and position_id:
                    await message.edit(
                        content=_settlement_message(row),
                        view=PaperSettlementView(str(position_id)),
                    )
                    published_positions[str(position_id)] = message_id
                    _record_position_publication(store, row, message_id)
                else:
                    view = PaperDecisionView(proposal_id)
                    for child in view.children:
                        if isinstance(child, discord.ui.Button):
                            child.disabled = True
                    await message.edit(
                        content=f"Decision recorded: {row.get('approval_decision')}.",
                        view=view,
                    )
                published.pop(proposal_id, None)
            except Exception as error:
                logger.warning(
                    "Discord proposal control refresh failed: %s",
                    type(error).__name__,
                )
        for position_id, message_id in tuple(published_positions.items()):
            if position_id in open_ids:
                continue
            try:
                row = await asyncio.to_thread(paper_show, store, position_id)
                message = await channel.fetch_message(message_id)
                await message.edit(
                    content=_settlement_message(row),
                    view=PaperSettlementView(position_id, disabled=True),
                )
                published_positions.pop(position_id, None)
            except Exception as error:
                logger.warning(
                    "Discord settlement control refresh failed: %s",
                    type(error).__name__,
                )

    async def publish_pending_once() -> None:
        await asyncio.to_thread(capture_closing_snapshots, store, client=clob)
        channel = client.get_channel(channel_id) or await client.fetch_channel(
            channel_id
        )
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            raise TypeError("DISCORD_CHANNEL_ID must identify a text channel or thread")
        pending_rows, open_rows = await asyncio.gather(
            asyncio.to_thread(paper_rows, store, state="pending"),
            asyncio.to_thread(paper_rows, store, state="open"),
        )
        await refresh_controls(channel, pending_rows, open_rows)
        for row in _unpublished_actionable_rows(pending_rows, published):
            proposal_id = str(row["proposal_id"])
            intent = await asyncio.to_thread(
                _publication_intent, store, row, kind="proposal"
            )
            view = PaperDecisionView(proposal_id)
            run_id = str(row["run_id"])
            if run_id not in report_cache:
                report_cache[run_id] = _report_for_run(run_id)
            report = report_cache[run_id]
            attachment = {"file": discord.File(report)} if report else {}

            async def send_proposal(
                proposal_row=row,
                marker=str(intent["marker"]),
                proposal_view=view,
                files=attachment,
            ):
                return await channel.send(
                    _proposal_message(proposal_row, marker=marker),
                    view=proposal_view,
                    allowed_mentions=discord.AllowedMentions.none(),
                    **files,
                )

            message_id, _ = await _send_or_recover(
                channel,
                marker=str(intent["marker"]),
                after=str(intent["intent_at"]),
                author_id=int(client.user.id),
                send=send_proposal,
            )
            published[proposal_id] = message_id
            _record_publication(store, row, message_id)
        for row in _unpublished_open_position_rows(open_rows, published_positions):
            position_id = str(row["position_id"])
            intent = await asyncio.to_thread(
                _publication_intent, store, row, kind="position"
            )

            async def send_position(
                position_row=row,
                marker=str(intent["marker"]),
                published_position_id=position_id,
            ):
                return await channel.send(
                    _settlement_message(position_row, marker=marker),
                    view=PaperSettlementView(published_position_id),
                    allowed_mentions=discord.AllowedMentions.none(),
                )

            message_id, _ = await _send_or_recover(
                channel,
                marker=str(intent["marker"]),
                after=str(intent["intent_at"]),
                author_id=int(client.user.id),
                send=send_position,
            )
            published_positions[position_id] = message_id
            _record_position_publication(store, row, message_id)

    @tasks.loop(seconds=60)
    async def publish_pending() -> None:
        try:
            await publish_pending_once()
        except Exception:
            logger.exception("Discord publication iteration failed; retrying next tick")

    @publish_pending.before_loop
    async def wait_until_ready() -> None:
        await client.wait_until_ready()

    ready_once = False

    @client.event
    async def on_ready() -> None:
        nonlocal ready_once
        if not publish_pending.is_running():
            publish_pending.start()
        if ready_once:
            return
        try:
            await tree.sync()
        except Exception:
            logger.exception("Discord command sync failed; retrying on reconnect")
            return
        ready_once = True

    try:
        client.run(token)
    finally:
        instance_lock.close()


def _proposal_message(row: dict[str, Any], *, marker: str | None = None) -> str:
    payload = row.get("payload") or {}
    edge = payload.get("conservative_edge")
    odds = payload.get("odds") or row.get("decimal_odds")
    parts = [
        "**Paper proposal**",
        f"{row.get('league')} · {row.get('team_a')} vs {row.get('team_b')}",
        f"Target: `{row.get('target')}` · selection: `{row.get('selection_id')}`",
        f"Probability: {float(row.get('probability_point') or 0):.1%}",
    ]
    if odds:
        parts.append(f"Executable odds: {float(odds):.3f}")
    if edge is not None:
        parts.append(f"Conservative edge: {float(edge):.1%}")
    parts.append(f"Stake: {float(row.get('stake_units') or 0):.2f}u")
    parts.append(f"Evidence run: `{row.get('run_id')}`")
    return _bounded_message(parts, marker=marker)


def _settlement_message(row: dict[str, Any], *, marker: str | None = None) -> str:
    state = str(row.get("state") or "open")
    parts = [
        "**Paper position settlement**",
        f"{row.get('league')} · {row.get('team_a')} vs {row.get('team_b')}",
        f"Position: `{row.get('position_id')}` · selection: `{row.get('selection_id')}`",
        (
            f"Entry: {float(row.get('decimal_odds') or 0):.3f} odds · "
            f"{float(row.get('stake_units') or 0):.2f}u"
        ),
        f"State: `{state}`",
    ]
    if state == "settled":
        parts.append(
            f"Result: `{row.get('result')}` · PnL: "
            f"{float(row.get('pnl_units') or 0):+.2f}u"
        )
    else:
        parts.append("Owner: choose Win, Loss, Push, or Void and confirm the source.")
    return _bounded_message(parts, marker=marker)


def _bounded_message(parts: list[str], *, marker: str | None) -> str:
    body = "\n".join(parts)
    if not marker:
        return body[:DELIVERY_TARGET]
    body_limit = DELIVERY_TARGET - len(marker) - 1
    return f"{body[:body_limit]}\n{marker}"


def _publication_marker(kind: str, identity: str) -> str:
    digest = hashlib.sha256(f"{kind}:{identity}".encode()).hexdigest()[:20]
    return f"[oracle-ref:{kind}:{digest}]"


def _publication_intent(
    store: EvidenceStore,
    row: dict[str, Any],
    *,
    kind: str,
) -> dict[str, str]:
    """Persist a stable marker before a Discord send can occur."""
    if kind not in {"proposal", "position"}:
        raise ValueError("Publication kind must be proposal or position")
    identity_key = f"{kind}_id"
    identity = str(row[identity_key])
    event_type = f"discord_{kind}_publish_intent"
    event_id = (
        f"discord-{kind}-intent-{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
    )
    event = store.get(EvidenceTable.RUN_EVENTS, event_id)
    if event is not None:
        payload = _json_payload(event.get("payload_json"))
        if payload:
            return {key: str(value) for key, value in payload.items()}
    intent_at = datetime.now(UTC).isoformat()
    payload = {
        identity_key: identity,
        "marker": _publication_marker(kind, identity),
        "intent_at": intent_at,
    }
    store.append(
        EvidenceTable.RUN_EVENTS,
        {
            "id": event_id,
            "run_id": row["run_id"],
            "event_at": intent_at,
            "event_type": event_type,
            "status": "pending",
            "idempotency_key": f"discord-{kind}-intent:{identity}",
            "payload_json": payload,
        },
    )
    return payload


async def _recover_message_id(
    channel: Any,
    *,
    marker: str,
    after: str,
    author_id: int,
) -> int | None:
    """Find a previously sent marked message after a crash-before-record window."""
    since = datetime.fromisoformat(after)
    async for message in channel.history(limit=100, after=since, oldest_first=False):
        message_author_id = getattr(getattr(message, "author", None), "id", None)
        if message_author_id == author_id and marker in str(
            getattr(message, "content", "")
        ):
            return int(message.id)
    return None


async def _send_or_recover(
    channel: Any,
    *,
    marker: str,
    after: str,
    author_id: int,
    send: Callable[[], Awaitable[Any]],
) -> tuple[int, bool]:
    """Recover a prior marked send, otherwise send exactly once for this attempt."""
    recovered = await _recover_message_id(
        channel,
        marker=marker,
        after=after,
        author_id=author_id,
    )
    if recovered is not None:
        return recovered, False
    message = await send()
    return int(message.id), True


def _acquire_instance_lock(path: Path) -> TextIO:
    """Hold one non-blocking process lock for the Gateway bot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError("Discord Gateway bot is already running") from error
    return handle


def _settle_owner_position(
    store: EvidenceStore,
    *,
    position_id: str,
    result: SettlementResult,
    source_reference: str,
    actor_id: str,
    note: str | None = None,
) -> str:
    """Owner-only service used by Discord; it has no LLM or market dependency."""
    return settle_paper(
        store,
        position_id=position_id,
        result=result,
        source_reference=source_reference,
        actor_id=actor_id,
        note=note,
    )


def _unpublished_actionable_rows(
    rows: list[dict[str, Any]],
    published: dict[str, int],
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["gate_state"] in {"paper_actionable", "research_only"}
        and row["proposal_id"] not in published
    ]


def _published_proposals(store: EvidenceStore) -> dict[str, int]:
    output: dict[str, int] = {}
    for event in store.list(EvidenceTable.RUN_EVENTS):
        if event["event_type"] != "discord_proposal_published":
            continue
        payload = _json_payload(event.get("payload_json"))
        try:
            output[str(payload["proposal_id"])] = int(payload["message_id"])
        except (KeyError, TypeError, ValueError):
            continue
    return output


def _unpublished_open_position_rows(
    rows: list[dict[str, Any]],
    published: dict[str, int],
) -> list[dict[str, Any]]:
    return [row for row in rows if row["position_id"] not in published]


def _published_positions(store: EvidenceStore) -> dict[str, int]:
    output: dict[str, int] = {}
    for event in store.list(EvidenceTable.RUN_EVENTS):
        if event["event_type"] != "discord_position_published":
            continue
        payload = _json_payload(event.get("payload_json"))
        try:
            output[str(payload["position_id"])] = int(payload["message_id"])
        except (KeyError, TypeError, ValueError):
            continue
    return output


def _record_publication(
    store: EvidenceStore,
    row: dict[str, Any],
    message_id: int,
) -> None:
    proposal_id = str(row["proposal_id"])
    identity = hashlib.sha256(proposal_id.encode()).hexdigest()[:24]
    store.append(
        EvidenceTable.RUN_EVENTS,
        {
            "id": f"discord-published-{identity}",
            "run_id": row["run_id"],
            "event_at": datetime.now(UTC),
            "event_type": "discord_proposal_published",
            "status": "completed",
            "idempotency_key": f"discord-proposal:{proposal_id}",
            "payload_json": {
                "proposal_id": proposal_id,
                "message_id": message_id,
            },
        },
    )


def _record_position_publication(
    store: EvidenceStore,
    row: dict[str, Any],
    message_id: int,
) -> None:
    position_id = str(row["position_id"])
    identity = hashlib.sha256(position_id.encode()).hexdigest()[:24]
    store.append(
        EvidenceTable.RUN_EVENTS,
        {
            "id": f"discord-position-published-{identity}",
            "run_id": row["run_id"],
            "event_at": datetime.now(UTC),
            "event_type": "discord_position_published",
            "status": "completed",
            "idempotency_key": f"discord-position:{position_id}",
            "payload_json": {
                "position_id": position_id,
                "message_id": message_id,
            },
        },
    )


def _report_for_run(run_id: str) -> Path | None:
    import json

    for path in sorted((REPORTS_DIR / "daily").glob("*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("evidence_run_id") == run_id:
            markdown = path.with_suffix(".md")
            return markdown if markdown.is_file() else path
    return None


def _json_payload(value: Any) -> dict[str, Any]:
    import json

    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _is_owner(user_id: int, owner_id: int) -> bool:
    return user_id == owner_id
