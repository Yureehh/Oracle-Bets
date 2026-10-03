"""LoL-specific Discord prediction and profile helpers."""

from __future__ import annotations

import numpy as np
from oracle_bets_core.pd import pd

from oracle_bets_discord.formatting import (
    MESSAGE_LIMIT,
)

# ── config & constants ──────────────────────────────────────────────────── #

_EMPTY_ROSTER: dict[str, str | None] = {
    "top": None,
    "jng": None,
    "mid": None,
    "bot": None,
    "sup": None,
}


# ── small helpers ───────────────────────────────────────────────────────── #


def get_empty_roster() -> dict[str, str | None]:
    """Copy of the empty-roster template."""
    return dict(_EMPTY_ROSTER.items())


# ── schedule / leagues formatting ───────────────────────────────────────── #


def format_schedule_message(schedule_df: pd.DataFrame) -> str:
    messages = format_schedule_messages(schedule_df)
    if not messages:
        return "No upcoming matches found."
    if len(messages) == 1:
        return messages[0]
    return messages[0] + "\nMessage split; run the daily workflow for all chunks."


def format_schedule_messages(
    schedule_df: pd.DataFrame,
    *,
    message_limit: int = MESSAGE_LIMIT,
) -> list[str]:
    if schedule_df.empty or "league" not in schedule_df.columns:
        return ["No upcoming matches found."]

    df = schedule_df.copy()
    df["start_utc"] = pd.to_datetime(df["start_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["start_utc"]).sort_values(
        ["start_utc", "league", "team_a"], kind="mergesort"
    )
    if df.empty:
        return ["No upcoming matches found."]

    chunks: list[str] = []
    current = ""

    for day, day_df in df.groupby(df["start_utc"].dt.date, sort=True):
        day_header = f"**Upcoming LoL Games - {day}**"
        blocks = [day_header]
        blocks.extend(
            format_league(day_df, league, max_matches=None).rstrip()
            for league in np.sort(day_df["league"].unique())
        )
        day_message = "\n\n".join(blocks)
        for block in _split_schedule_block(day_message, message_limit):
            if not current:
                current = block
                continue
            candidate = current + "\n\n" + block
            if len(candidate) >= message_limit:
                chunks.append(current)
                current = block
            else:
                current = candidate

    if current:
        chunks.append(current)
    return chunks or ["No upcoming matches found. Double-check the league names."]


def _split_schedule_block(block: str, message_limit: int) -> list[str]:
    if len(block) < message_limit:
        return [block]
    lines = block.splitlines()
    chunks: list[str] = []
    current = ""
    for raw_line in lines:
        # A single pathological line must never exceed the Discord limit on
        # its own; the API rejects the whole message.
        line = (
            raw_line
            if len(raw_line) < message_limit
            else raw_line[: message_limit - 2] + "…"
        )
        candidate = line if not current else current + "\n" + line
        if len(candidate) >= message_limit:
            if current:
                chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def format_league(
    df: pd.DataFrame,
    league: str,
    *,
    max_matches: int | None = 5,
) -> str:
    league_df = df[df["league"] == league].copy()
    if max_matches is not None:
        league_df = league_df.head(max_matches)
    display_cols = {
        "start_utc": "Start (UTC)",
        "team_a": "Team A",
        "team_b": "Team B",
        "best_of": "Best Of",
        "market_query": "Market Query",
    }
    league_df = league_df[[c for c in display_cols if c in league_df.columns]].rename(
        columns=display_cols
    )
    if "Start (UTC)" in league_df.columns:
        league_df["Start (UTC)"] = pd.to_datetime(
            league_df["Start (UTC)"], errors="coerce", utc=True
        ).dt.strftime("%Y-%m-%d %H:%M")
    md = league_df.to_markdown(index=False)
    md = "\n".join(line.lstrip() for line in md.split("\n"))
    return f"Upcoming {league} Games:\n```{md}```\n\n"
