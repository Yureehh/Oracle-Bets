"""Shared Discord formatting helpers."""

from __future__ import annotations

from typing import Any

MESSAGE_LIMIT: int = 2000


def handle_command_error(error: Exception, additional_info: str = "") -> str:
    """Return a consistent Discord-safe error message."""
    extra = f"{additional_info.strip()} " if additional_info else ""
    return (
        f"Something went wrong. {extra}If this issue persists, please contact either Yureeh.\n"
        f"Error:\n```{error}```"
    )


def dataframe_to_markdown(df: Any) -> str:
    """Render a DataFrame as a trimmed Discord code block."""
    try:
        md = df.to_markdown(index=False)
        md = "\n".join(line.lstrip() for line in md.split("\n"))
        return f"```{md}```\n\n"[: MESSAGE_LIMIT - 1]
    except Exception as e:
        msg = f"Error converting DataFrame to Markdown: {e}"
        raise ValueError(msg) from e
