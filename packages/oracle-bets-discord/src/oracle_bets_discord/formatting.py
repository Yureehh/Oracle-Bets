"""Shared Discord formatting helpers."""

from __future__ import annotations

MESSAGE_LIMIT: int = 2000
DELIVERY_TARGET: int = 1900


def split_message(content: str, *, limit: int = DELIVERY_TARGET) -> tuple[str, ...]:
    """Split without dropping text, preferring complete section boundaries."""
    if limit <= 0 or limit >= MESSAGE_LIMIT:
        raise ValueError(f"limit must be between 1 and {MESSAGE_LIMIT - 1}")
    messages: list[str] = []
    remaining = content
    while len(remaining) > limit:
        boundary = remaining.rfind("\n\n", 0, limit + 1)
        cut = boundary + 2 if boundary >= 0 else limit
        messages.append(remaining[:cut])
        remaining = remaining[cut:]
    messages.append(remaining)
    return tuple(messages)
