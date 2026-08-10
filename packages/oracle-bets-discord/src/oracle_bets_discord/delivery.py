"""Deterministic saved-report publication for Discord webhooks."""

from __future__ import annotations

import json
import os
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from oracle_bets_core.paths import REPORTS_DIR

from oracle_bets_discord.formatting import split_message

DISCORD_API = "https://discord.com/api/v10"

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class DiscordDeliveryMode(StrEnum):
    GATEWAY = "gateway"
    WEBHOOK = "webhook"
    OFF = "off"


def resolve_delivery_mode(
    configured: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> DiscordDeliveryMode:
    """Resolve one transport so webhook and Gateway delivery cannot duplicate."""
    values = environment if environment is not None else os.environ
    raw = str(configured or values.get("DISCORD_DELIVERY_MODE") or "").strip()
    if raw:
        try:
            return DiscordDeliveryMode(raw.casefold())
        except ValueError as error:
            raise ValueError(
                "DISCORD_DELIVERY_MODE must be gateway, webhook, or off"
            ) from error
    gateway_ready = all(
        str(values.get(name) or "").strip()
        for name in ("DISCORD_TOKEN", "DISCORD_CHANNEL_ID", "DISCORD_OWNER_USER_ID")
    )
    if gateway_ready:
        return DiscordDeliveryMode.GATEWAY
    if str(values.get("DISCORD_WEBHOOK_URL") or "").strip():
        return DiscordDeliveryMode.WEBHOOK
    return DiscordDeliveryMode.OFF


def check_gateway_access(
    token: str,
    channel_id: str,
    *,
    session: requests.Session | None = None,
) -> dict[str, str]:
    """Read bot and channel metadata without sending or mutating anything."""
    http = session or requests.Session()
    headers = {"Authorization": f"Bot {token}"}
    user = http.get(f"{DISCORD_API}/users/@me", headers=headers, timeout=15)
    user.raise_for_status()
    channel = http.get(
        f"{DISCORD_API}/channels/{channel_id}", headers=headers, timeout=15
    )
    channel.raise_for_status()
    history = http.get(
        f"{DISCORD_API}/channels/{channel_id}/messages",
        headers=headers,
        params={"limit": 1},
        timeout=15,
    )
    history.raise_for_status()
    return {
        "bot": str(user.json().get("username") or user.json().get("id") or "unknown"),
        "channel": str(
            channel.json().get("name") or channel.json().get("id") or channel_id
        ),
        "history_readable": "yes",
    }


def send_webhook_messages(
    webhook_url: str,
    messages: Sequence[str],
    *,
    session: requests.Session | None = None,
) -> None:
    """Send deterministic bounded messages with mentions disabled."""
    http = session or requests.Session()
    for content in messages:
        for message in split_message(str(content)):
            response = http.post(
                webhook_url,
                json={"content": message, "allowed_mentions": {"parse": []}},
                timeout=15,
            )
            response.raise_for_status()


def publish_saved_report(run_id: str) -> None:
    if resolve_delivery_mode() is not DiscordDeliveryMode.WEBHOOK:
        raise RuntimeError("discord publish requires DISCORD_DELIVERY_MODE=webhook")
    webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        raise RuntimeError("DISCORD_WEBHOOK_URL is required for publish")
    latest = max((REPORTS_DIR / "daily").glob("*.json"), default=None)
    selected = latest if run_id == "latest" and latest else Path(run_id)
    if not selected.is_file():
        raise FileNotFoundError(f"Daily report not found: {run_id}")
    payload = json.loads(selected.read_text(encoding="utf-8"))
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        messages = [json.dumps(payload, ensure_ascii=False)]
    send_webhook_messages(webhook, [str(message) for message in messages])
