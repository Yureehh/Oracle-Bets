"""Exclusive Gateway/off delivery configuration and read-only access checks."""

from __future__ import annotations

import os
from enum import StrEnum
from typing import TYPE_CHECKING

import requests

DISCORD_API = "https://discord.com/api/v10"

if TYPE_CHECKING:
    from collections.abc import Mapping


class DiscordDeliveryMode(StrEnum):
    GATEWAY = "gateway"
    OFF = "off"


def resolve_delivery_mode(
    configured: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> DiscordDeliveryMode:
    """Resolve the single Gateway transport or disable external delivery."""
    values = environment if environment is not None else os.environ
    raw = str(configured or values.get("DISCORD_DELIVERY_MODE") or "").strip()
    if raw:
        try:
            return DiscordDeliveryMode(raw.casefold())
        except ValueError as error:
            raise ValueError("DISCORD_DELIVERY_MODE must be gateway or off") from error
    gateway_ready = all(
        str(values.get(name) or "").strip()
        for name in ("DISCORD_TOKEN", "DISCORD_CHANNEL_ID", "DISCORD_OWNER_USER_ID")
    )
    if gateway_ready:
        return DiscordDeliveryMode.GATEWAY
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
    return {
        "bot": str(user.json().get("username") or user.json().get("id") or "unknown"),
        "channel": str(
            channel.json().get("name") or channel.json().get("id") or channel_id
        ),
        "channel_accessible": "yes",
    }
