"""LoL-specific Discord prediction and profile helpers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from oracle_bets_core.betting import (
    OverUnderSignal,
    decimal_odds_from_probability,
    expected_edge,
    kelly_fraction,
)
from oracle_bets_core.logger import logger
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
# Defaults; override with ORACLE_BETS_BANKROLL / ORACLE_BETS_KELLY_FRACTION /
# ORACLE_BETS_STAKE_CAP so displayed stakes match the reader's actual bankroll.
POLYMARKET_BANKROLL: float = 500.0
POLYMARKET_KELLY_MULTIPLIER: float = 0.25
VALUE_EDGE_THRESHOLD: float = 0.05
WATCH_EDGE_THRESHOLD: float = 0.02
BINARY_SELECTION_COUNT: int = 2


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using default %s.", name, raw, default)
        return default


@dataclass(frozen=True)
class StakingConfig:
    """Bankroll and Kelly sizing used for *display only* (no bets are placed)."""

    bankroll: float
    kelly_fraction: float
    stake_cap: float | None

    @property
    def kelly_label(self) -> str:
        if self.kelly_fraction > 0 and (1.0 / self.kelly_fraction).is_integer():
            return f"1/{int(1.0 / self.kelly_fraction)}"
        return f"{self.kelly_fraction:g}x"

    @property
    def cap_label(self) -> str:
        if self.stake_cap is None:
            return "No stake cap"
        return f"Stake cap: €{self.stake_cap:.0f}"


def get_staking_config() -> StakingConfig:
    """Resolve staking display settings from the environment with safe defaults."""
    bankroll = _env_float("ORACLE_BETS_BANKROLL", POLYMARKET_BANKROLL)
    kelly = _env_float("ORACLE_BETS_KELLY_FRACTION", POLYMARKET_KELLY_MULTIPLIER)
    cap = _env_float("ORACLE_BETS_STAKE_CAP", None)
    if bankroll is None or bankroll <= 0:
        bankroll = POLYMARKET_BANKROLL
    if kelly is None or not (0 < kelly <= 1):
        kelly = POLYMARKET_KELLY_MULTIPLIER
    if cap is not None and cap <= 0:
        cap = None
    return StakingConfig(bankroll=bankroll, kelly_fraction=kelly, stake_cap=cap)


@dataclass(frozen=True)
class WinnerMarketRow:
    market: str
    pick: str
    probability: float
    poly_price: float | None = None


@dataclass(frozen=True)
class PriceQuote:
    market: str
    selections: tuple[tuple[str, float], ...]


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
    calibrator = getattr(predictor, "outcome_calibrator", None)
    method = getattr(calibrator, "method", None)
    if method and method != "raw":
        return f"calibrated model ({method})"
    if method == "raw":
        return "raw model (selected by calibration)"
    return "raw model"


def prop_probability_source(predictor, prop_name: str) -> str:
    calibrator = getattr(predictor, f"{prop_name}_prop_calibrator", None)
    method = getattr(calibrator, "method", None)
    return f"prop calibrator ({method})" if method else "residual sigma fallback"


def context_line(
    team_a: Team,
    team_b: Team,
    account_for_side: bool,
    first_pick_team_name: str | None,
) -> str:
    side = f"{team_a.name}=Blue, {team_b.name}=Red" if account_for_side else "ignored"
    first_pick = first_pick_team_name.strip() if first_pick_team_name else "unknown"
    return f"- Context: side {side} | first pick {first_pick}"


def _selection_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _team_keys(team_name: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9]+", team_name)
    keys = {_selection_key(team_name)}
    if words:
        keys.add(_selection_key("".join(word[0] for word in words)))
        keys.add(_selection_key(words[0]))
    return keys


def _parse_polymarket_price(raw: str) -> float:
    value = raw.strip().casefold().removesuffix("c")
    try:
        price = float(value)
    except ValueError as e:
        msg = f"Invalid Polymarket price: {raw!r}."
        raise ValueError(msg) from e
    if price > 1:
        price = price / 100.0
    if not 0 < price < 1:
        msg = f"Polymarket price must be between 0 and 1, got {raw!r}."
        raise ValueError(msg)
    return price


def parse_polymarket_price_quotes(raw_quotes: list[str] | None) -> list[PriceQuote]:
    quotes: list[PriceQuote] = []
    for raw_quote in raw_quotes or []:
        if ":" not in raw_quote:
            msg = f"Price quote must include a market label, got {raw_quote!r}."
            raise ValueError(msg)
        market, raw_selections = raw_quote.split(":", 1)
        selections: list[tuple[str, float]] = []
        for raw_selection in raw_selections.split(","):
            if "=" not in raw_selection:
                msg = f"Price selection must look like Name=74c, got {raw_selection!r}."
                raise ValueError(msg)
            name, price = raw_selection.split("=", 1)
            selections.append((name.strip(), _parse_polymarket_price(price)))
        if not selections:
            msg = f"Price quote has no selections: {raw_quote!r}."
            raise ValueError(msg)
        quotes.append(PriceQuote(market=market.strip(), selections=tuple(selections)))
    return quotes


def _base_market_rows(
    match_type: str,
    team_a_name: str,
    team_b_name: str,
    series_a: float,
    series_b: float,
) -> list[WinnerMarketRow]:
    market = "Map Winner" if match_type == "bo1" else "Series Winner"
    return [
        WinnerMarketRow(market, team_a_name, series_a),
        WinnerMarketRow(market, team_b_name, series_b),
    ]


def _market_key(market: str) -> str:
    text = market.casefold()
    game_match = re.search(r"(?:game|g)\s*([1-5]).*winner|^g\s*([1-5])$", text)
    if game_match:
        game_number = game_match.group(1) or game_match.group(2)
        return f"game{game_number}winner"
    total_match = re.search(r"total\s*games?.*([234]\.5)|^tg\s*([34]\.5)$", text)
    if total_match:
        line = total_match.group(1) or total_match.group(2)
        return f"totalgames{line}"
    handicap_match = re.search(r"handicap.*([12]\.5)|^h\s*([12]\.5)$", text)
    if handicap_match:
        line = handicap_match.group(1) or handicap_match.group(2)
        return f"handicap{line}"
    if "moneyline" in text or "series" in text or "match winner" in text:
        return "serieswinner"
    if "map winner" in text:
        return "mapwinner"
    return _selection_key(market)


def _row_market_key(row: WinnerMarketRow) -> str:
    total_map_match = re.search(r"([234]\.5)\s*maps", row.market.casefold())
    if total_map_match and row.pick.casefold() in {"over", "under"}:
        return f"{row.pick.casefold()}{total_map_match.group(1)}"
    key = _market_key(row.market)
    if key.startswith("totalgames"):
        line = key.removeprefix("totalgames")
        side = "over" if row.pick.casefold() == "over" else "under"
        return f"{side}{line}"
    return key


def _select_row_for_quote(  # noqa: PLR0911, PLR0912
    rows: list[WinnerMarketRow],
    quote: PriceQuote,
    selection: str,
    selection_index: int,
    team_a_name: str,
    team_b_name: str,
) -> WinnerMarketRow | None:
    quote_key = _market_key(quote.market)
    if quote_key.startswith("totalgames"):
        side_key = _selection_key(selection)
        if side_key == "1":
            side_key = "over"
        elif side_key == "2":
            side_key = "under"
        quote_key = f"{side_key}{quote_key.removeprefix('totalgames')}"
    candidates = [row for row in rows if _row_market_key(row) == quote_key]
    if not candidates:
        return None

    selection_key = _selection_key(selection)
    team_a_keys = _team_keys(team_a_name)
    team_b_keys = _team_keys(team_b_name)
    for row in candidates:
        row_key = _selection_key(row.pick)
        if selection_key == "1" and team_a_name in row.pick:
            return row
        if selection_key == "2" and team_b_name in row.pick:
            return row
        if selection_key == "1" and row_key == "over":
            return row
        if selection_key == "2" and row_key == "under":
            return row
        if selection_key == row_key:
            return row
        if selection_key in {"over", "under"} and selection_key == row_key:
            return row
        if selection_key in team_a_keys and team_a_name in row.pick:
            return row
        if selection_key in team_b_keys and team_b_name in row.pick:
            return row

    if len(candidates) == BINARY_SELECTION_COUNT and selection_index < len(candidates):
        return candidates[selection_index]
    return None


def _price_rows(
    rows: list[WinnerMarketRow],
    quotes: list[PriceQuote],
    team_a_name: str,
    team_b_name: str,
) -> tuple[list[WinnerMarketRow], set[tuple[str, str]], list[str]]:
    priced: list[WinnerMarketRow] = []
    priced_keys: set[tuple[str, str]] = set()
    unsupported: list[str] = []
    for quote in quotes:
        matched = False
        for idx, (selection, price) in enumerate(quote.selections):
            row = _select_row_for_quote(
                rows, quote, selection, idx, team_a_name, team_b_name
            )
            if row is None:
                continue
            matched = True
            priced_row = WinnerMarketRow(
                market=row.market,
                pick=row.pick,
                probability=row.probability,
                poly_price=price,
            )
            priced.append(priced_row)
            priced_keys.add((priced_row.market, priced_row.pick))
        if not matched:
            unsupported.append(quote.market)
    priced.sort(key=lambda row: _edge(row), reverse=True)
    return priced, priced_keys, unsupported


def _edge(row: WinnerMarketRow) -> float:
    if row.poly_price is None:
        return float("-inf")
    return expected_edge(1.0 / row.poly_price, row.probability)


def _action(edge: float | None) -> str:
    if edge is None:
        return "PRICE NEEDED"
    if edge >= VALUE_EDGE_THRESHOLD:
        return "VALUE"
    if edge >= WATCH_EDGE_THRESHOLD:
        return "WATCH"
    return "PASS"


def _format_market_table_row(
    row: WinnerMarketRow, staking: StakingConfig | None = None
) -> str:
    staking = staking or get_staking_config()
    fair = _odds(row.probability)
    if row.poly_price is None:
        return (
            f"{row.market[:18]:<18} {row.pick[:12]:<12} {_pct(row.probability):>7} "
            f"{'--':>5} {fair:>6} {'--':>7} {'--':>9} {'--':>8} PRICE NEEDED"
        )
    odds = 1.0 / row.poly_price
    edge = expected_edge(odds, row.probability)
    kelly = kelly_fraction(odds, row.probability, fraction=staking.kelly_fraction)
    stake = staking.bankroll * kelly
    if staking.stake_cap is not None:
        stake = min(stake, staking.stake_cap)
    if edge < 0:
        stake = 0.0
    return (
        f"{row.market[:18]:<18} {row.pick[:12]:<12} {_pct(row.probability):>7} "
        f"{row.poly_price * 100:>4.0f}c {fair:>6} {edge * 100:>+6.1f}% "
        f"{kelly * 100:>8.1f}% €{stake:>6.2f} {_action(edge)}"
    )


def _default_unpriced_rows(
    rows: list[WinnerMarketRow],
    priced_keys: set[tuple[str, str]],
) -> list[WinnerMarketRow]:
    return [row for row in rows if (row.market, row.pick) not in priced_keys]


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
    price_quotes: list[str] | None = None,
    notes: list[str] | None = None,
    blue_range: tuple[float, float] | None = None,
    red_range: tuple[float, float] | None = None,
    uncertainty_confidence: float | None = None,
    drivers: list[str] | None = None,
) -> str:
    rows = _base_market_rows(
        match_type, blue_team_name, red_team_name, blue_win, red_win
    )
    quotes = parse_polymarket_price_quotes(price_quotes)
    priced_rows, priced_keys, unsupported = _price_rows(
        rows, quotes, blue_team_name, red_team_name
    )
    unpriced_rows = _default_unpriced_rows(rows, priced_keys)
    table_rows = priced_rows + unpriced_rows

    staking = get_staking_config()
    lines = [
        f"**LoL Markets: {blue_team_name} vs {red_team_name} ({match_type.upper()})**",
    ]
    if priced_rows:
        lines.append(
            f"Bankroll: €{staking.bankroll:.0f} | Kelly: {staking.kelly_label} | "
            f"{staking.cap_label}"
        )
    lines.extend(
        [
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
            (
                f"{'Market':<18} {'Pick':<12} {'Model':>7} {'Poly':>5} "
                f"{'Fair':>6} {'Edge':>7} {'Kelly':>9} {'Stake':>8} Action"
            ),
            *[_format_market_table_row(row, staking) for row in table_rows],
            "```",
        ]
    )
    if unsupported:
        lines.append(
            "Unsupported prices ignored: " + ", ".join(sorted(set(unsupported)))
        )
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
