"""LoL-specific Discord prediction and profile helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from oracle_bets_core.betting import (
    OverUnderSignal,
    decimal_odds_from_probability,
)
from oracle_bets_core.pd import pd

from oracle_bets_discord.formatting import (
    MESSAGE_LIMIT,
)

if TYPE_CHECKING:
    from lol_bets.inference.team import Team

# ── config & constants ──────────────────────────────────────────────────── #

_EMPTY_ROSTER: dict[str, str | None] = {
    "top": None,
    "jng": None,
    "mid": None,
    "bot": None,
    "sup": None,
}
LOW_CONFIDENCE_WARNING_COUNT: int = 2


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


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _odds(value: float) -> str:
    if value <= 0:
        return "∞"
    if value >= 1:
        return "1.00"
    return f"{decimal_odds_from_probability(value):.2f}"


def confidence_label(warnings: list[str]) -> str:
    if len(warnings) >= LOW_CONFIDENCE_WARNING_COUNT:
        return "Low"
    if warnings:
        return "Medium"
    return "High"


def format_warnings(warnings: list[str]) -> str:
    if not warnings:
        return ""
    summary = "; ".join(warning.removesuffix(".") for warning in warnings)
    return f"\n\nWarning: {summary}."


def outcome_probability_source(predictor) -> str:
    calibrator = getattr(predictor, "series_winner_calibrator", None)
    method = getattr(calibrator, "method", None)
    if method and method != "raw":
        return f"calibrated model ({method})"
    if method == "raw":
        return "raw model (selected by calibration)"
    return "raw model"


def context_line(
    team_a: Team,
    team_b: Team,
    account_for_side: bool,
    first_pick_team_name: str | None,
) -> str:
    side = f"{team_a.name}=Blue, {team_b.name}=Red" if account_for_side else "ignored"
    first_pick = first_pick_team_name.strip() if first_pick_team_name else "unknown"
    return f"- Context: side {side} | first pick {first_pick}"


def format_winner_market_output(
    *,
    blue_team_name: str,
    red_team_name: str,
    match_type: str,
    blue_win: float,
    red_win: float,
    probability_source: str,
    warnings: list[str],
    context: str,
    notes: list[str] | None = None,
    blue_range: tuple[float, float] | None = None,
    red_range: tuple[float, float] | None = None,
    uncertainty_confidence: float | None = None,
    drivers: list[str] | None = None,
) -> str:
    market = "Map Winner" if match_type == "bo1" else "Series Winner"
    lines = [
        f"**LoL Markets: {blue_team_name} vs {red_team_name} ({match_type.upper()})**",
        f"Source: {probability_source} | Confidence: {confidence_label(warnings)}",
        *(
            [
                "Evidence: probabilities come directly from the independent "
                "prematch series-winner model."
            ]
            if match_type != "bo1"
            else []
        ),
        "",
        "```text",
        f"{'Market':<18} {'Pick':<16} {'Model':>7} {'Fair':>6}",
        (
            f"{market:<18} {blue_team_name[:16]:<16} "
            f"{_pct(blue_win):>7} {_odds(blue_win):>6}"
        ),
        (
            f"{market:<18} {red_team_name[:16]:<16} "
            f"{_pct(red_win):>7} {_odds(red_win):>6}"
        ),
        "```",
    ]
    if blue_range is not None and red_range is not None:
        uncertainty_label = (
            f"{uncertainty_confidence * 100:.0f}%"
            if uncertainty_confidence is not None
            else "held-out"
        )
        lines.append(
            f"Probability range ({uncertainty_label} calibration uncertainty): "
            f"{blue_team_name} {_pct(blue_range[0])}–{_pct(blue_range[1])}; "
            f"{red_team_name} {_pct(red_range[0])}–{_pct(red_range[1])}."
        )
    if drivers:
        lines.append("Main model drivers (descriptive, not causal):")
        lines.extend(f"- {driver}" for driver in drivers)
    lines.append(context.removeprefix("- "))
    # Informational notes are displayed alongside warnings but do not count
    # against the confidence tier shown above.
    return "\n".join(lines) + format_warnings([*warnings, *(notes or [])])


def format_research_forecasts(
    *,
    team_a: str,
    team_b: str,
    map_prediction: dict | None,
    prop_values: dict[str, float],
) -> str:
    """Render compact non-actionable Map 1 and scalar forecasts."""
    lines = ["**Research-only forecasts**"]
    if map_prediction is not None:
        team_a_map = float(map_prediction["team1_win_probability"])
        team_b_map = float(map_prediction["team2_win_probability"])
        lines.append(
            f"- Map 1: {team_a} {_pct(team_a_map)} (fair {_odds(team_a_map)}) · "
            f"{team_b} {_pct(team_b_map)} (fair {_odds(team_b_map)})"
        )
    labels = {
        "gamelength": ("Length", "m"),
        "total_kills": ("Kills", ""),
        "total_towers": ("Towers", ""),
    }
    values = [
        f"{label} {float(prop_values[key]):.1f}{suffix}"
        for key, (label, suffix) in labels.items()
        if key in prop_values
    ]
    if values:
        lines.append("- Point means: " + " · ".join(values))
        lines.append("- Prop over/under probabilities require an explicit market line.")
    if len(lines) == 1:
        return ""
    lines.append("These forecasts cannot create paper proposals.")
    return "\n".join(lines)


def _format_prop_line(label: str, signal: OverUnderSignal) -> str:
    lines = [
        f"{label} {signal.line:g}",
        f"- Over: {_pct(signal.over_probability)} | fair {signal.over_fair_odds:.2f}",
        f"- Under: {_pct(signal.under_probability)} | fair {signal.under_fair_odds:.2f}",
    ]
    if signal.over_edge is not None or signal.under_edge is not None:
        over_edge = (
            f"{signal.over_edge * 100:+.1f}%" if signal.over_edge is not None else "n/a"
        )
        under_edge = (
            f"{signal.under_edge * 100:+.1f}%"
            if signal.under_edge is not None
            else "n/a"
        )
        lines.append(f"- Edge: Over {over_edge} | Under {under_edge}")
    if (
        signal.over_half_kelly_fraction is not None
        or signal.under_half_kelly_fraction is not None
    ):
        over_stake = (
            f"{signal.over_half_kelly_fraction * 100:.1f}%"
            if signal.over_half_kelly_fraction is not None
            else "n/a"
        )
        under_stake = (
            f"{signal.under_half_kelly_fraction * 100:.1f}%"
            if signal.under_half_kelly_fraction is not None
            else "n/a"
        )
        lines.append(f"- Half-Kelly: Over {over_stake} | Under {under_stake}")
    return "\n".join(lines)


def _format_manual_research_note(
    blue_team_name: str, red_team_name: str, line_signals: dict[str, OverUnderSignal]
) -> str:
    _ = blue_team_name, red_team_name
    if not any(
        (signal.over_edge or 0) > 0 or (signal.under_edge or 0) > 0
        for signal in line_signals.values()
    ):
        return ""
    return (
        "\n\nPaper research only. Record any real owner decision manually in "
        "canonical evidence; Oracle Bets never places a bet."
    )


def format_prop_market_output(
    *,
    blue_team_name: str,
    red_team_name: str,
    gamelength: float,
    total_kills: float,
    total_towers: float,
    line_signals: dict[str, OverUnderSignal],
    calibration_sources: dict[str, str],
    warnings: list[str],
    notes: list[str] | None = None,
) -> str:
    output = (
        f"**LoL Props: {blue_team_name} vs {red_team_name}**\n\n"
        f"- Length: **{gamelength:.1f}m**\n"
        f"- Kills: **{total_kills:.1f}**\n"
        f"- Towers: **{total_towers:.1f}**\n"
    )
    for label, signal in line_signals.items():
        output += "\n" + _format_prop_line(label, signal)
    output += _format_manual_research_note(
        blue_team_name,
        red_team_name,
        line_signals,
    )
    if calibration_sources:
        sources = " | ".join(
            f"{label}: {source}" for label, source in calibration_sources.items()
        )
        output += f"\n\n- Probability source: {sources}"
    output += f"\n\nConfidence: **{confidence_label(warnings)}**"
    output += format_warnings([*warnings, *(notes or [])])
    return output
