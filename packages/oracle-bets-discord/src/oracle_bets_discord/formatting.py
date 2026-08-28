"""Shared Discord formatting helpers."""

from __future__ import annotations

MESSAGE_LIMIT: int = 2000
DELIVERY_TARGET: int = 1900
_FIRST_PRINTABLE_CODEPOINT = 32


def sanitize_discord_text(value: object) -> str:
    """Disable mentions and strip control characters from external text."""
    text = "".join(
        character
        for character in str(value)
        if character in {"\n", "\t"} or ord(character) >= _FIRST_PRINTABLE_CODEPOINT
    )
    return text.replace("@", "@\u200b")
