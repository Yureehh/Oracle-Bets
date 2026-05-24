"""LoL-specific Discord prediction and profile helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import lol_bets.inference.match_predictor as match_predictor_module
import numpy as np
from lol_bets.inference.team import Team
from oracle_bets_core.betting import OverUnderSignal, decimal_odds_from_probability
from oracle_bets_core.io_utils import parquet_loader
from oracle_bets_core.paths import FLATTENED_PLAYERS, FLATTENED_TEAMS
from oracle_bets_core.pd import pd

from oracle_bets_discord.formatting import (
    MESSAGE_LIMIT,
    dataframe_to_markdown,
    handle_command_error,
)
from oracle_bets_discord.predictions.best_ofs import BestOfs

# ── config & constants ──────────────────────────────────────────────────── #

_EMPTY_ROSTER: dict[str, str | None] = {
    "top": None,
    "jng": None,
    "mid": None,
    "bot": None,
    "sup": None,
}
VALID_MATCH_TYPES: list[str] = ["bo1", "bo2", "bo3", "bo5"]
POSITIONS: tuple[str, ...] = ("top", "jng", "mid", "bot", "sup")
WEEKS_FOR_DELAY: int = 3
LOW_CONFIDENCE_WARNING_COUNT: int = 2

PLEASE_PROVIDE_TEAMS = "Please provide both a blue and red team name."
TEAMS_MUST_BE_DIFFERENT = "The two teams must be different."
FIRST_PICK_TEAM_MUST_MATCH = "First-pick team must match one of the two teams."

# ── lazy, single-shot predictor ─────────────────────────────────────────── #


@lru_cache(maxsize=1)
def get_match_predictor() -> match_predictor_module.MatchPredictor:
    return match_predictor_module.MatchPredictor()


# ── small helpers ───────────────────────────────────────────────────────── #


def get_empty_roster() -> dict[str, str | None]:
    """Copy of the empty-roster template."""
    return dict(_EMPTY_ROSTER.items())


# ── schedule / leagues formatting ───────────────────────────────────────── #


def format_leagues_message(leagues: list[str]) -> str:
    if not leagues:
        return "No leagues found. Please check the league names."
    formatted = "\n".join(f"- {lg}" for lg in leagues)
    return f"Leagues playing in the next 7 days are:\n{formatted}"


def format_schedule_message(schedule_df: pd.DataFrame) -> str:
    if schedule_df.empty or "league" not in schedule_df.columns:
        return "No upcoming matches found."

    parts: list[str] = []
    too_long_alert = "Message too long. Please specify a narrower filter."

    for league in np.sort(schedule_df["league"].unique()):
        block = format_league(schedule_df, league)
        est_len = len("\n".join(parts)) + len(block)
        if est_len > MESSAGE_LIMIT - len(too_long_alert):
            parts.append(too_long_alert)
            break
        parts.append(block)

    final = "\n".join(parts)
    return final or "No upcoming matches found. Double-check the league names."


def format_league(df: pd.DataFrame, league: str) -> str:
    league_df = df[df["league"] == league].head(5).copy()
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
    return f"Upcoming {league} Games (Next 5 Matches Within 7 Days):\n```{md}```\n\n"


# ── cached parquet reads for profiles ───────────────────────────────────── #


@lru_cache(maxsize=2)
def _parquet_cached(path: str) -> pd.DataFrame:
    return parquet_loader(Path(path))


def get_player_data(entity_name: str, players_path: Path) -> pd.DataFrame | None:
    try:
        df = _parquet_cached(str(players_path))
        if "playername" not in df.columns:
            msg = "Column 'playername' not found in players data."
            raise KeyError(msg)  # noqa: TRY301
        filtered = df[df["playername"].str.casefold() == entity_name.casefold()]
        return None if filtered.empty else filtered
    except Exception as e:
        msg = f"Error processing player data: {e}"
        raise ValueError(msg) from e


def get_team_data(entity_name: str, teams_path: Path) -> pd.DataFrame | None:
    try:
        df = _parquet_cached(str(teams_path))
        if "teamname" not in df.columns:
            msg = "Column 'teamname' not found in teams data."
            raise KeyError(msg)  # noqa: TRY301
        filtered = df[df["teamname"].str.casefold() == entity_name.casefold()]
        return None if filtered.empty else filtered
    except Exception as e:
        msg = f"Error processing team data: {e}"
        raise ValueError(msg) from e


# ── profile formatting ──────────────────────────────────────────────────── #


def format_player_profile(data: pd.DataFrame, truncate: bool = False) -> str:
    row = data.iloc[0]
    stats_names = [
        "Position",
        "Team",
        "Elo",
        "Glicko2 Score",
        "Plackett-Luce Score",
        "TrueSkill Score",
        "Blue Side Win Rate",
        "Red Side Win Rate",
        "K/D/A Ratio",
        "K/D/A at 15",
        "Gold Diff At 15",
        "CS Diff At 15",
        "XP Diff At 15",
        "CSPM",
        "DPM",
        "EGPM",
        "VSPM",
        "Damage Share",
        "XP Efficiency",
    ]
    if truncate:
        stats_names = stats_names[:8]

    def val(k: str, *alts: str) -> float:
        """Prefer EMA versions if present, else raw; returns np.nan if missing."""
        for a in (k, *alts):
            if a in row.index:
                return row[a]
        return np.nan

    kd15 = (
        f"{val('ema_killsat15'):.2f} / {val('ema_deathsat15'):.2f} / {val('ema_assistsat15'):.2f}"
        if all(
            x in row.index
            for x in ("ema_killsat15", "ema_deathsat15", "ema_assistsat15")
        )
        else "N/A"
    )

    stats_values = [
        str(row.get("position", "")).capitalize(),
        str(row.get("teamname", "N/A")),
        f"{val('elo'):.2f}" if "elo" in row.index else "N/A",
        f"{val('glicko2_mu', 'gl2_mu'):.2f}"
        if any(c in row.index for c in ("glicko2_mu", "gl2_mu"))
        else "N/A",
        f"{val('pl_mu'):.2f}" if "pl_mu" in row.index else "N/A",
        f"{val('trueskill_mu'):.2f}" if "trueskill_mu" in row.index else "N/A",
        f"{float(val('ema_blue_side', 'blue_side', 'side_blue') or 0) * 100:.2f}%",
        f"{float(val('ema_red_side', 'red_side', 'side_red') or 0) * 100:.2f}%",
        f"{float(val('ema_kda', 'kda') or 0):.2f}",
        kd15,
        f"{float(val('ema_golddiffat15', 'golddiffat15') or 0):.2f}",
        f"{float(val('ema_csdiffat15', 'csdiffat15') or 0):.2f}",
        f"{float(val('ema_xpdiffat15', 'xpdiffat15') or 0):.2f}",
        f"{float(val('ema_cspm', 'cspm') or 0):.2f}",
        f"{float(val('ema_dpm', 'dpm') or 0):.2f}",
        f"{float(val('ema_egpm', 'egpm') or 0):.2f}",
        f"{float(val('ema_vspm', 'vspm') or 0):.2f}",
        f"{float(val('ema_damageshare', 'damageshare') or 0) * 100:.2f}%",
        f"{float(val('ema_xp_efficiency', 'xp_efficiency') or 0):.2f}",
    ]
    if truncate:
        stats_values = stats_values[:8]

    df_out = pd.DataFrame({"Stat": stats_names, "Value": stats_values})
    return dataframe_to_markdown(df_out)


def format_team_profile(data: pd.DataFrame) -> str:
    row = data.iloc[0]
    stats_names = [
        "Elo",
        "Glicko-2 Score",
        "Plackett-Luce Score",
        "TrueSkill Score",
        "League Elo",
        "Patch Win Rate",
        "Season Win Rate",
        "Blue Side Win Rate",
        "Red Side Win Rate",
        "AVG Gamelength in Minutes",
    ]
    stats_values = [
        f"{row.get('elo', np.nan):.2f}" if "elo" in row.index else "N/A",
        f"{row.get('glicko2_mu', np.nan):.2f}" if "glicko2_mu" in row.index else "N/A",
        f"{row.get('pl_mu', np.nan):.2f}" if "pl_mu" in row.index else "N/A",
        f"{row.get('trueskill_mu', np.nan):.2f}"
        if "trueskill_mu" in row.index
        else "N/A",
        f"{row.get('league_elo', np.nan):.2f}" if "league_elo" in row.index else "N/A",
        f"{float(row.get('ema_patch_win_rate', 0)) * 100:.2f}%",
        f"{float(row.get('ema_season_win_rate', 0)) * 100:.2f}%",
        f"{float(row.get('ema_blue_side', 0)) * 100:.2f}%",
        f"{float(row.get('ema_red_side', 0)) * 100:.2f}%",
        f"{float(row.get('ema_gamelength', row.get('gamelength', 0))):.2f}",
    ]
    df_out = pd.DataFrame({"Stat": stats_names, "Value": stats_values})
    return dataframe_to_markdown(df_out)


# ── async profile getters ───────────────────────────────────────────────── #


async def get_formatted_team_profile(team_name: str) -> tuple[str | None, str | None]:
    try:
        team_profile = get_team_data(team_name, FLATTENED_TEAMS)
        if team_profile is not None and not team_profile.empty:
            return format_team_profile(team_profile), None
        return None, f"Data for team '{team_name}' not found in the database."
    except Exception as e:
        return None, handle_command_error(e, "Team profile retrieval failed.")


async def get_formatted_player_profile(
    player_name: str, truncate: bool = False
) -> tuple[str | None, str | None]:
    try:
        player_profile = get_player_data(player_name, FLATTENED_PLAYERS)
        if player_profile is not None and not player_profile.empty:
            return format_player_profile(player_profile, truncate), None
        return None, f"Data for player '{player_name}' not found in the database."
    except Exception as e:
        return None, handle_command_error(e, "Player profile retrieval failed.")


# ── prediction helpers ──────────────────────────────────────────────────── #


def add_roster_to_output(output: str, team_a: Team, team_b: Team) -> str:
    def fmt(team: Team) -> str:
        parts: list[str] = []
        for role in POSITIONS:
            name = team.roster.get(role)
            parts.append(str(name) if name else "N/A")
        return " \t-  \t".join(parts)

    output += "\n## Found Rosters\n"
    output += f"**{team_a.name}:**\t {fmt(team_a)}\n"
    output += f"**{team_b.name}:**\t {fmt(team_b)}"
    return output


def add_break_flags_to_output(
    output: str,
    break_blue_flag: bool,
    blue_team_name: str,
    break_red_flag: bool,
    red_team_name: str,
) -> str:
    if break_blue_flag:
        output += f"\n\nCAREFUL! {blue_team_name} has not played in the last {WEEKS_FOR_DELAY} weeks."
    if break_red_flag:
        output += f"\n\nCAREFUL! {red_team_name} has not played in the last {WEEKS_FOR_DELAY} weeks."
    return output


def process_roster(
    roster_str: str | None, positions: list[str] | None = None
) -> dict[str, str]:
    if not roster_str:
        msg = "Roster string cannot be empty."
        raise ValueError(msg)
    positions = positions or list(POSITIONS)
    players = [p.strip() for p in roster_str.split(",")]
    if len(players) != len(positions):
        msg = f"Roster does not contain the correct number of players: expected {len(positions)}, got {len(players)}."
        raise ValueError(msg)
    return dict(zip(positions, players, strict=False))


def strip_team_names(team1: str | None, team2: str | None) -> tuple[str, str]:
    return (team1 or "").strip(), (team2 or "").strip()


def resolve_first_pick(
    team_a_name: str,
    team_b_name: str,
    first_pick_team_name: str | None,
) -> tuple[bool | None, bool | None]:
    if not first_pick_team_name:
        return None, None
    first_pick = first_pick_team_name.strip().casefold()
    if first_pick == team_a_name.casefold():
        return True, False
    if first_pick == team_b_name.casefold():
        return False, True
    msg = FIRST_PICK_TEAM_MUST_MATCH
    raise ValueError(msg)


def add_selection_context_to_output(
    output: str,
    team_a: Team,
    team_b: Team,
    account_for_side: bool,
    first_pick_team_name: str | None,
) -> str:
    if not account_for_side and not first_pick_team_name:
        return output
    output += "\n\n## Selection Context"
    if account_for_side:
        output += f"\n- Map side: {team_a.name}=Blue, {team_b.name}=Red"
    if first_pick_team_name:
        output += f"\n- First pick: {first_pick_team_name.strip()}"
    else:
        output += "\n- First pick: unknown/neutral"
    return output


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _odds(value: float) -> str:
    return f"{decimal_odds_from_probability(value):.2f}"


def _roster_count(team: Team) -> int:
    return sum(1 for role in POSITIONS if team.roster.get(role))


def _confidence_label(warnings: list[str]) -> str:
    if len(warnings) >= LOW_CONFIDENCE_WARNING_COUNT:
        return "Low"
    if warnings:
        return "Medium"
    return "High"


def _format_warnings(warnings: list[str]) -> str:
    if not warnings:
        return "\nWarnings\n- None."
    return "\nWarnings\n" + "\n".join(f"- {warning}" for warning in warnings)


def _format_prop_line(label: str, signal: OverUnderSignal) -> str:
    lines = [
        f"\n{label} Line {signal.line:g}",
        f"- Over probability: {_pct(signal.over_probability)}",
        f"- Under probability: {_pct(signal.under_probability)}",
        f"- Fair odds: Over {signal.over_fair_odds:.2f} | Under {signal.under_fair_odds:.2f}",
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
        lines.append(
            f"- Suggested stake: half-Kelly Over {over_stake} | Under {under_stake}"
        )
    return "\n".join(lines)


def _market_odds(probability: float, edge: float | None) -> float | None:
    if edge is None or probability <= 0:
        return None
    return (1.0 + edge) / probability


def _format_record_bet_commands(
    blue_team_name: str, red_team_name: str, line_signals: dict[str, OverUnderSignal]
) -> str:
    commands: list[str] = []
    event = f"{blue_team_name} vs {red_team_name}"
    market_names = {"Length": "length", "Kills": "kills", "Towers": "towers"}
    for label, signal in line_signals.items():
        market = market_names.get(label, label.casefold())
        for selection, probability, edge, odds in (
            (
                "over",
                signal.over_probability,
                signal.over_edge,
                _market_odds(signal.over_probability, signal.over_edge),
            ),
            (
                "under",
                signal.under_probability,
                signal.under_edge,
                _market_odds(signal.under_probability, signal.under_edge),
            ),
        ):
            if edge is None or edge <= 0 or odds is None:
                continue
            commands.append(
                f'!bet record --event "{event}" --market {market} '
                f"--selection {selection} --line {signal.line:g} --odds {odds:.2f} "
                f'--prob {probability:.3f} --edge {edge:.3f} --sport "League of Legends"'
            )
    if not commands:
        return ""
    return (
        "\n\nRecord Positive-Edge Bets Manually\n"
        "Copy one of these after you place the bet yourself:\n"
        + "\n".join(f"- `{command}`" for command in commands[:3])
    )


def format_prop_market_output(
    *,
    blue_team_name: str,
    red_team_name: str,
    gamelength: float,
    total_kills: float,
    total_towers: float,
    line_signals: dict[str, OverUnderSignal],
    warnings: list[str],
) -> str:
    output = (
        f"**Prop Predictions: {blue_team_name} vs {red_team_name}**\n\n"
        "Projected Totals\n"
        f"- Expected game length: **{gamelength:.1f} minutes**\n"
        f"- Expected total kills: **{total_kills:.1f}**\n"
        f"- Expected total towers: **{total_towers:.1f}**\n"
    )
    for label, signal in line_signals.items():
        output += _format_prop_line(label, signal)
    output += _format_record_bet_commands(blue_team_name, red_team_name, line_signals)
    output += (
        "\n\nMeaning\n"
        "- Expected total is the model's central estimate for one map.\n"
        "- Over/Under probability uses historical model error, not just the mean.\n"
        "- Fair odds are no-vig decimal odds implied by the model probability.\n"
        "- Edge compares model probability to market odds when odds are supplied.\n"
        "- Half-Kelly is a bankroll fraction suggestion, never an auto-bet.\n"
        "- Map 3/4/5 props are interpreted only after that map is confirmed."
    )
    output += f"\n\nConfidence: **{_confidence_label(warnings)}**"
    output += _format_warnings(warnings)
    return output


# ── main async prediction entrypoints ───────────────────────────────────── #


async def predict_and_format_result(
    ctx,
    blue_team_name: str,
    red_team_name: str,
    blue_roster_str: str | None,
    red_roster_str: str | None,
    match_type: str,
    account_for_side: bool,
    first_pick_team_name: str | None = None,
) -> None:
    """Create teams, predict outcomes, and format result for bo1/bo3/bo5."""
    if match_type not in VALID_MATCH_TYPES:
        await ctx.send(
            content=f"Invalid match type: {match_type}. Please specify 'bo1', 'bo2', 'bo3', or 'bo5'."
        )
        return

    msg = await ctx.send(content="```Calculating win probabilities...```")
    try:
        blue_roster = (
            process_roster(blue_roster_str) if blue_roster_str else get_empty_roster()
        )
        red_roster = (
            process_roster(red_roster_str) if red_roster_str else get_empty_roster()
        )
        blue_first_pick, red_first_pick = resolve_first_pick(
            blue_team_name, red_team_name, first_pick_team_name
        )
        blue_team = Team(
            name=blue_team_name,
            side="Blue",
            first_pick=blue_first_pick,
            roster=blue_roster,
        )
        red_team = Team(
            name=red_team_name,
            side="Red",
            first_pick=red_first_pick,
            roster=red_roster,
        )
        # inactivity flags (robust to missing dates)
        today = pd.Timestamp.today().normalize()
        days_delay = int(WEEKS_FOR_DELAY) * 7

        def _parse_date(x) -> pd.Timestamp | pd.NaT:  # pyright: ignore[reportInvalidTypeForm]
            d = pd.to_datetime(x, errors="coerce")
            return d.normalize() if pd.notna(d) else pd.NaT

        blue_last = _parse_date(blue_team.team_stats.get("date"))
        red_last = _parse_date(red_team.team_stats.get("date"))
        break_blue_flag = bool(
            pd.notna(blue_last) and (today - blue_last).days >= days_delay
        )
        break_red_flag = bool(
            pd.notna(red_last) and (today - red_last).days >= days_delay
        )
        predictor = get_match_predictor()
        # Predict both orientations and average
        p1 = predictor.predict_match(
            blue_team, red_team, account_for_side=account_for_side
        )
        p2 = predictor.predict_match(
            red_team, blue_team, account_for_side=account_for_side
        )
        # p1: team1=blue; p2: team2=blue (flipped)  # noqa: ERA001
        blue_win = (p1["team1_win_probability"] + p2["team2_win_probability"]) / 2.0
        red_win = 1.0 - blue_win
        warnings: list[str] = []
        if predictor.outcome_calibrator is None:
            warnings.append(
                "Outcome calibration artifact is missing; raw model probability is being used."
            )
        if _roster_count(blue_team) < len(POSITIONS) or _roster_count(red_team) < len(
            POSITIONS
        ):
            warnings.append(
                "One or both rosters are incomplete, so roster features use fallback state."
            )
        if not account_for_side:
            warnings.append("Side selection is ignored for this command.")
        # series breakdown
        if match_type == "bo1":
            output = BestOfs.best_of_one(
                blue_team_name, blue_win, red_team_name, red_win
            )
        elif match_type == "bo2":
            output = BestOfs.best_of_two(
                blue_team_name, blue_win, red_team_name, red_win
            )
        elif match_type == "bo3":
            output = BestOfs.best_of_three(
                blue_team_name, blue_win, red_team_name, red_win
            )
        else:  # "bo5"
            output = BestOfs.best_of_five(
                blue_team_name, blue_win, red_team_name, red_win
            )
        output += (
            "\n\n## Winner Market Read"
            f"\n- {blue_team_name} fair odds: {_odds(blue_win)}"
            f"\n- {red_team_name} fair odds: {_odds(red_win)}"
            f"\n- Probability source: {'calibrated model' if predictor.outcome_calibrator is not None else 'raw model'}"
            f"\n- Confidence: {_confidence_label(warnings)}"
            "\n- Meaning: compare fair odds to the market price; a bet only has edge when the market pays above fair odds."
        )
        output = add_selection_context_to_output(
            output, blue_team, red_team, account_for_side, first_pick_team_name
        )
        output = add_roster_to_output(output, blue_team, red_team)
        output = add_break_flags_to_output(
            output, break_blue_flag, blue_team_name, break_red_flag, red_team_name
        )
        output += _format_warnings(warnings)
        # stay under Discord limit
        await msg.edit(content=output[: MESSAGE_LIMIT - 1])

    except Exception as e:
        await msg.edit(
            content=handle_command_error(e, "Could not complete the prediction.")
        )


async def predict_and_format_props(
    ctx,
    blue_team_name: str,
    red_team_name: str,
    blue_roster_str: str | None,
    red_roster_str: str | None,
    account_for_side: bool,
    first_pick_team_name: str | None = None,
    kills_line: float | None = None,
    kills_over_odds: float | None = None,
    kills_under_odds: float | None = None,
    towers_line: float | None = None,
    towers_over_odds: float | None = None,
    towers_under_odds: float | None = None,
    length_line: float | None = None,
    length_over_odds: float | None = None,
    length_under_odds: float | None = None,
) -> None:
    """Predict game props (gamelength, total kills, total towers) for a single game."""
    msg = await ctx.send(content="```Calculating prop predictions...```")
    try:
        blue_roster = (
            process_roster(blue_roster_str) if blue_roster_str else get_empty_roster()
        )
        red_roster = (
            process_roster(red_roster_str) if red_roster_str else get_empty_roster()
        )
        blue_first_pick, red_first_pick = resolve_first_pick(
            blue_team_name, red_team_name, first_pick_team_name
        )
        blue_team = Team(
            name=blue_team_name,
            side="Blue",
            first_pick=blue_first_pick,
            roster=blue_roster,
        )
        red_team = Team(
            name=red_team_name,
            side="Red",
            first_pick=red_first_pick,
            roster=red_roster,
        )

        predictor = get_match_predictor()
        gamelength = predictor.predict_gamelength(
            blue_team, red_team, account_for_side=account_for_side
        )
        total_kills = predictor.predict_total_kills(
            blue_team, red_team, account_for_side=account_for_side
        )
        total_towers = predictor.predict_total_towers(
            blue_team, red_team, account_for_side=account_for_side
        )

        warnings: list[str] = []
        if _roster_count(blue_team) < len(POSITIONS) or _roster_count(red_team) < len(
            POSITIONS
        ):
            warnings.append(
                "One or both rosters are incomplete, so roster features use fallback state."
            )
        if not account_for_side:
            warnings.append("Side selection is ignored for this command.")

        line_signals: dict[str, OverUnderSignal] = {}
        for label, prop_name, mean, line, over_odds, under_odds in (
            (
                "Length",
                "gamelength",
                gamelength,
                length_line,
                length_over_odds,
                length_under_odds,
            ),
            (
                "Kills",
                "total_kills",
                total_kills,
                kills_line,
                kills_over_odds,
                kills_under_odds,
            ),
            (
                "Towers",
                "total_towers",
                total_towers,
                towers_line,
                towers_over_odds,
                towers_under_odds,
            ),
        ):
            if line is None:
                continue
            try:
                line_signals[label] = predictor.price_prop_line(
                    prop_name=prop_name,
                    mean=mean,
                    line=line,
                    over_odds=over_odds,
                    under_odds=under_odds,
                )
            except RuntimeError as e:
                warnings.append(str(e))

        output = format_prop_market_output(
            blue_team_name=blue_team_name,
            red_team_name=red_team_name,
            gamelength=gamelength,
            total_kills=total_kills,
            total_towers=total_towers,
            line_signals=line_signals,
            warnings=warnings,
        )
        output = add_selection_context_to_output(
            output, blue_team, red_team, account_for_side, first_pick_team_name
        )
        await msg.edit(content=output[: MESSAGE_LIMIT - 1])
    except Exception as e:
        await msg.edit(
            content=handle_command_error(e, "Could not complete prop predictions.")
        )


async def validate_and_predict(
    ctx,
    blue_team_name: str,
    red_team_name: str,
    blue_roster_str: str | None,
    red_roster_str: str | None,
    match_type: str,
    side_consideration: bool,
    first_pick_team_name: str | None = None,
):
    """Validates inputs and triggers prediction."""
    blue_team_name, red_team_name = strip_team_names(blue_team_name, red_team_name)
    if not blue_team_name or not red_team_name:
        await ctx.send(PLEASE_PROVIDE_TEAMS)
        return
    if blue_team_name.casefold() == red_team_name.casefold():
        await ctx.send(TEAMS_MUST_BE_DIFFERENT)
        return
    await predict_and_format_result(
        ctx,
        blue_team_name,
        red_team_name,
        blue_roster_str,
        red_roster_str,
        match_type,
        side_consideration,
        first_pick_team_name,
    )


async def validate_and_predict_props(
    ctx,
    blue_team_name: str,
    red_team_name: str,
    blue_roster_str: str | None,
    red_roster_str: str | None,
    side_consideration: bool,
    first_pick_team_name: str | None = None,
    kills_line: float | None = None,
    kills_over_odds: float | None = None,
    kills_under_odds: float | None = None,
    towers_line: float | None = None,
    towers_over_odds: float | None = None,
    towers_under_odds: float | None = None,
    length_line: float | None = None,
    length_over_odds: float | None = None,
    length_under_odds: float | None = None,
):
    blue_team_name, red_team_name = strip_team_names(blue_team_name, red_team_name)
    if not blue_team_name or not red_team_name:
        await ctx.send(PLEASE_PROVIDE_TEAMS)
        return
    if blue_team_name.casefold() == red_team_name.casefold():
        await ctx.send(TEAMS_MUST_BE_DIFFERENT)
        return
    await predict_and_format_props(
        ctx,
        blue_team_name,
        red_team_name,
        blue_roster_str,
        red_roster_str,
        side_consideration,
        first_pick_team_name,
        kills_line,
        kills_over_odds,
        kills_under_odds,
        towers_line,
        towers_over_odds,
        towers_under_odds,
        length_line,
        length_over_odds,
        length_under_odds,
    )
