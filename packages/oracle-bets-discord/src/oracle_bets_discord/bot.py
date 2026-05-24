"""
Oracle Bets Discord bot.

Commands for registered prediction modules, market search, and betting utilities.
"""

from __future__ import annotations

import os
import shlex
from datetime import UTC, datetime
from typing import Any

import discord
from discord.ext import commands
from dotenv import load_dotenv
from lol_bets.data_generation.ingestion.schedule import (
    fetch_and_store_schedule,
    get_or_update_schedule,
)
from lol_bets.inference.team import Team
from oracle_bets_core.betting import build_edge_signal
from oracle_bets_core.logger import logger
from oracle_bets_core.markets import PolymarketGammaAdapter
from oracle_bets_core.paths import SCHEDULE
from oracle_bets_dashboard.database import (
    delete_bet,
    get_all_bets,
    init_db,
    insert_bankroll_entry,
    insert_bet,
    settle_bet,
)

from oracle_bets_discord.betting import (
    calculate_kelly_criterion,
    calculate_odds,
    calculate_prob,
    convert_odds,
)
from oracle_bets_discord.formatting import (
    MESSAGE_LIMIT,
    dataframe_to_markdown,
    handle_command_error,
)
from oracle_bets_discord.predictions.lol import (
    format_leagues_message,
    format_schedule_message,
    get_formatted_player_profile,
    get_formatted_team_profile,
    validate_and_predict,
    validate_and_predict_props,
)
from oracle_bets_discord.registry import default_registry

# ── env & bot setup ─────────────────────────────────────────────────────── #

load_dotenv()

DISCORD_TOKEN_ENV = "DISCORD_TOKEN"  # noqa: S105
BOT_COMMAND_PREFIX = "!"
BOT_DESCRIPTION = "LoL esports prediction & betting helper."
OPTIONAL_STARTUP_ARTIFACT_PREFIXES = ("gamelength", "total kills", "total towers")

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix=BOT_COMMAND_PREFIX,
    description=BOT_DESCRIPTION,
    intents=intents,
)

# ── helpers ─────────────────────────────────────────────────────────────── #


def _split_two_rosters(rosters: str | None) -> tuple[str | None, str | None]:
    """
    Accepts either:
      - "p1,p2,p3,p4,p5 | q1,q2,q3,q4,q5"  (pipe-separated rosters), or
      - "p1,p2,p3,p4,p5" (blue only; red left empty), or
      - None
    Returns (blue_roster_str, red_roster_str)
    """
    if not rosters:
        return None, None
    if "|" in rosters:
        left, right = rosters.split("|", 1)
        return left.strip(), right.strip()
    return rosters.strip(), None


def _parse_lol_options(options: str | None) -> dict[str, Any]:
    """Parse Discord-friendly GNU-style flags from a trailing option string."""
    parsed: dict[str, Any] = {"rosters": None}
    if not options:
        return parsed
    tokens = shlex.split(options)
    numeric_flags = {
        "--kills-line": "kills_line",
        "--kills-over-odds": "kills_over_odds",
        "--kills-under-odds": "kills_under_odds",
        "--towers-line": "towers_line",
        "--towers-over-odds": "towers_over_odds",
        "--towers-under-odds": "towers_under_odds",
        "--length-line": "length_line",
        "--length-over-odds": "length_over_odds",
        "--length-under-odds": "length_under_odds",
    }
    text_flags = {
        "--first-pick": "first_pick_team_name",
        "--market": "market",
        "--query": "query",
        "--rosters": "rosters",
    }
    bool_flags = {"--bo1": "bo1", "--bo2": "bo2", "--bo3": "bo3", "--bo5": "bo5"}
    positional: list[str] = []
    i = 0
    while i < len(tokens):
        option = tokens[i]
        if option in numeric_flags:
            i += 1
            if i >= len(tokens):
                raise ValueError(f"Missing value for {option}.")
            parsed[numeric_flags[option]] = float(tokens[i])
        elif option == "--side":
            i += 1
            if i >= len(tokens):
                raise ValueError("Missing value for --side.")
            side = tokens[i].strip().casefold()
            if side not in {"blue", "red"}:
                raise ValueError("--side must be Blue or Red.")
            parsed["side"] = side.title()
        elif option in text_flags:
            i += 1
            if i >= len(tokens):
                raise ValueError(f"Missing value for {option}.")
            parsed[text_flags[option]] = tokens[i]
        elif option in bool_flags:
            parsed[bool_flags[option]] = True
        else:
            positional.append(option)
        i += 1
    if positional and parsed["rosters"] is None:
        parsed["rosters"] = " ".join(positional)
    return parsed


def _orient_for_side(
    team_a_name: str | None, team_b_name: str | None, side: str | None
) -> tuple[str | None, str | None, bool]:
    if side is None:
        return team_a_name, team_b_name, False
    if side == "Red":
        return team_b_name, team_a_name, True
    return team_a_name, team_b_name, True


def _match_type_from_options(options: dict[str, Any]) -> str:
    for match_type in ("bo5", "bo3", "bo2", "bo1"):
        if options.get(match_type):
            return match_type
    return "bo1"


def _parse_bet_options(options: str | None) -> dict[str, Any]:
    """Parse Discord ledger flags for `!bet` commands."""
    parsed: dict[str, Any] = {}
    if not options:
        return parsed
    tokens = shlex.split(options)
    text_flags = {
        "--event": "event_name",
        "--event-date": "event_date",
        "--sport": "sport",
        "--league": "league",
        "--market": "market_type",
        "--selection": "selection",
        "--side": "side",
        "--book": "bookmaker",
        "--bookmaker": "bookmaker",
        "--ref": "bet_slip_ref",
        "--source": "source_prediction",
        "--note": "notes",
        "--notes": "notes",
        "--tags": "tags",
        "--currency": "currency",
        "--status": "status",
        "--result": "result",
    }
    numeric_flags = {
        "--odds": "odds_decimal",
        "--stake": "stake",
        "--prob": "model_probability",
        "--edge": "edge",
        "--kelly": "kelly_fraction",
        "--line": "line",
        "--map": "map_number",
        "--bo": "best_of",
        "--balance": "balance",
        "--deposit": "deposit",
        "--withdrawal": "withdrawal",
        "--limit": "limit",
    }
    bool_flags = {"--smoke": "is_smoke", "--live": "is_live"}
    i = 0
    while i < len(tokens):
        option = tokens[i]
        if option in text_flags:
            i += 1
            if i >= len(tokens):
                raise ValueError(f"Missing value for {option}.")
            parsed[text_flags[option]] = tokens[i]
        elif option in numeric_flags:
            i += 1
            if i >= len(tokens):
                raise ValueError(f"Missing value for {option}.")
            value = float(tokens[i])
            parsed[numeric_flags[option]] = (
                int(value)
                if numeric_flags[option] in {"map_number", "best_of", "limit"}
                else value
            )
        elif option in bool_flags:
            parsed[bool_flags[option]] = True
        else:
            raise ValueError(f"Unknown option: {option}.")
        i += 1
    return parsed


def _format_bet_row(bet: dict[str, Any]) -> str:
    status = bet.get("result") or bet.get("status")
    stake = bet.get("stake")
    stake_text = f"{stake:.2f}" if stake is not None else "n/a"
    line = bet.get("line")
    line_text = f" line {line:g}" if line is not None else ""
    return (
        f"#{bet['id']} | {bet['event_name']} | {bet['market_type']}{line_text} | "
        f"{bet['selection']} @ {bet['odds_decimal']:.2f} | stake {stake_text} | {status}"
    )


def _raise_for_blocking_artifact_failures(module) -> None:
    health = module.artifact_health()
    blocking = [
        check
        for check in health.checks
        if not check.ok
        and not check.name.startswith(OPTIONAL_STARTUP_ARTIFACT_PREFIXES)
    ]
    optional = [
        check
        for check in health.checks
        if not check.ok and check.name.startswith(OPTIONAL_STARTUP_ARTIFACT_PREFIXES)
    ]
    if blocking:
        failed = [
            f"{check.name}: {check.path} ({check.reason or 'missing/unreadable'})"
            for check in blocking
        ]
        msg = f"Artifact health check failed for {module.id}: " + "; ".join(failed)
        raise RuntimeError(msg)
    if optional:
        logger.warning(
            "Starting Discord bot with unavailable prop commands for %s: %s",
            module.id,
            "; ".join(f"{check.name} ({check.reason})" for check in optional),
        )


async def _send_market_search(ctx: commands.Context, query: str) -> None:
    msg = await ctx.send(content="```Searching markets...```")
    try:
        quotes = PolymarketGammaAdapter().search(query, limit=10)
        if not quotes:
            await msg.edit(content="No active markets found.")
            return
        lines = ["**Active Market Quotes**"]
        for quote in quotes[:10]:
            price = (
                f"{quote.implied_probability * 100:.1f}%"
                if quote.implied_probability is not None
                else "n/a"
            )
            lines.append(f"- {quote.question} | {quote.outcome}: **{price}**")
        lines.append(
            '\nAfter manually placing a bet, record it with `!bet record --event "Team A vs Team B" --market winner --selection "Team A" --odds 1.85 --stake 25 --book polymarket`.'
        )
        await msg.edit(content="\n".join(lines)[: MESSAGE_LIMIT - 1])
    except Exception as e:
        await msg.edit(content=handle_command_error(e, "Market search failed."))


# ── lifecycle ───────────────────────────────────────────────────────────── #


@bot.event
async def on_ready():
    logger.info("%s has connected to Discord!", bot.user)


# ── utility commands ────────────────────────────────────────────────────── #


@bot.command(name="code", aliases=["github", "repository", "git", "source"])
async def code(ctx: commands.Context):
    """Link to the repository."""
    response = (
        "This model is open source. Ideas & contributions welcome!\n"
        "GitHub: https://github.com/Yureehh/Oracle-Bets"
    )
    await ctx.send(response)


@bot.command(name="leagues", aliases=["league", "show_leagues", "show_league"])
async def leagues(ctx: commands.Context):
    """List supported leagues in the schedule file."""
    try:
        leagues = get_or_update_schedule(save_path=SCHEDULE, force_refresh=True)[
            "league"
        ].unique()
        await ctx.send(format_leagues_message(sorted(leagues)))
    except Exception as e:
        await ctx.send(
            handle_command_error(e, "Could not retrieve league information.")
        )


@bot.command(name="schedule")
async def schedule(ctx: commands.Context, leagues: str | None = None):
    """Show upcoming schedule. Optionally filter by comma-separated leagues."""
    try:
        schedule_df = get_or_update_schedule(
            leagues=leagues, save_path=SCHEDULE, force_refresh=True
        )
        await ctx.send(format_schedule_message(schedule_df))
    except Exception as e:
        await ctx.send(handle_command_error(e, "Could not retrieve schedule."))


@bot.command(
    name="schedule_update",
    aliases=["update_schedule", "refresh_schedule", "schedule_refresh"],
)
async def schedule_update(
    ctx: commands.Context,
    leagues: str | None = None,
    window_days: str | None = None,
):
    """Fetch and store upcoming schedule, then display it."""
    days = 7
    if window_days:
        if not window_days.isdigit() or int(window_days) <= 0:
            await ctx.send("Window days must be a positive integer.")
            return
        days = int(window_days)
    try:
        schedule_df = fetch_and_store_schedule(
            window_days=days, leagues=leagues, save_path=SCHEDULE
        )
        await ctx.send(format_schedule_message(schedule_df))
    except Exception as e:
        await ctx.send(handle_command_error(e, "Could not update schedule."))


# ── roster / profiles ───────────────────────────────────────────────────── #


@bot.command(name="team_roster", aliases=["roster"])
async def roster(ctx: commands.Context, team: str | None = None):
    """Show roster for a team."""
    if not team:
        await ctx.send("Please provide a team name.")
        return
    msg = await ctx.send(content="```Extracting...```")
    try:
        team_df = Team(name=team).get_team_info()
        await msg.edit(content=dataframe_to_markdown(team_df))
    except Exception as e:
        await msg.edit(
            content=handle_command_error(e, "Could not extract roster information.")
        )


@bot.command(name="team_rosters", aliases=["rosters"])
async def rosters(ctx: commands.Context, teams: str | None = None):
    """Show rosters for multiple teams. Usage: !rosters T1, G2, JDG"""
    if not teams:
        await ctx.send(
            "Provide team names separated by commas, e.g., `!rosters T1, G2`."
        )
        return
    msg = await ctx.send(content="```Extracting...```")
    output = ""
    for team_name in (t.strip() for t in teams.split(",")):
        try:
            team_df = Team(name=team_name).get_team_info()
            output += dataframe_to_markdown(team_df)
        except Exception as e:
            output += handle_command_error(
                e, f"Could not extract roster for {team_name}.\n"
            )
    await msg.edit(content=output)


@bot.command(name="team_profile", aliases=["team"])
async def team_profile(ctx: commands.Context, team_name: str | None = None):
    """Show team profile."""
    if not team_name:
        await ctx.send("Please provide a team name.")
        return
    try:
        profile, error = await get_formatted_team_profile(team_name)
        await ctx.send(profile or error)
    except Exception as e:
        await ctx.send(handle_command_error(e, "Could not retrieve team profile."))


@bot.command(name="player_profile", aliases=["player"])
async def player_profile(
    ctx: commands.Context, player_name: str | None = None, verbose: str = "False"
):
    """Show player profile. Add `True` for more stats: `!player Faker True`"""
    if not player_name:
        await ctx.send("Please provide a player name.")
        return
    show_more = verbose.lower() in {"true", "1", "t", "y", "yes"}
    try:
        profile, error = await get_formatted_player_profile(player_name, show_more)
        await ctx.send(profile or error)
    except Exception as e:
        await ctx.send(handle_command_error(e, "Could not retrieve player profile."))


# ── predictions (bo1/bo2/bo3/bo5) ───────────────────────────────────────── #


@bot.command(name="bo1", aliases=["predict", "prediction", "match", "BO1"])
async def bo1(
    ctx: commands.Context,
    team_a_name: str | None = None,
    team_b_name: str | None = None,
    rosters: str | None = None,
):
    """
    Predict a neutral best-of-one. Optional rosters:
    `!bo1 T1 G2 "t1top,t1jng,t1mid,t1adc,t1sup | g2top,g2jng,g2mid,g2adc,g2sup"`
    """
    blue_roster_str, red_roster_str = _split_two_rosters(rosters)
    await validate_and_predict(
        ctx,
        team_a_name,
        team_b_name,
        blue_roster_str,
        red_roster_str,
        "bo1",
        False,
    )


@bot.command(
    name="sided_bo1",
    aliases=["sided_predict", "sided_prediction", "sided_match", "sided_BO1"],
)
async def sided_bo1(
    ctx: commands.Context,
    blue_team_name: str | None = None,
    red_team_name: str | None = None,
    rosters: str | None = None,
):
    """Best-of-one with explicit map side: first team Blue, second team Red."""
    blue_roster_str, red_roster_str = _split_two_rosters(rosters)
    await validate_and_predict(
        ctx, blue_team_name, red_team_name, blue_roster_str, red_roster_str, "bo1", True
    )


@bot.command(
    name="first_selection_bo1",
    aliases=["fs_bo1", "selection_bo1", "first_pick_bo1"],
)
async def first_selection_bo1(
    ctx: commands.Context,
    blue_team_name: str | None = None,
    red_team_name: str | None = None,
    first_pick_team_name: str | None = None,
    rosters: str | None = None,
):
    """Best-of-one with explicit map side and First Selection first-pick team."""
    if not first_pick_team_name:
        await ctx.send("Provide the first-pick team name.")
        return
    blue_roster_str, red_roster_str = _split_two_rosters(rosters)
    await validate_and_predict(
        ctx,
        blue_team_name,
        red_team_name,
        blue_roster_str,
        red_roster_str,
        "bo1",
        True,
        first_pick_team_name,
    )


@bot.group(name="lol", invoke_without_command=True)
async def lol_group(ctx: commands.Context):
    """LoL command group with copy-paste market prediction examples."""
    await ctx.send(
        "**LoL Bot Commands**\n"
        '- `!lol predict "Team WE" "LNG Esports"`\n'
        '- `!lol predict "Team WE" "LNG Esports" --side Blue --first-pick "Team WE"`\n'
        '- `!lol predict "Team WE" "LNG Esports" --bo5`\n'
        '- `!lol props "Team WE" "LNG Esports" --kills-line 26.5 --kills-over-odds 1.85`\n'
        '- `!lol props "Team WE" "LNG Esports" --towers-line 12.5 --length-line 31.5`\n'
        "- Predictions are decision support only: model probability, fair odds, "
        "edge, and half-Kelly are shown when available."
    )


@lol_group.command(name="predict")
async def lol_predict(
    ctx: commands.Context,
    team_a_name: str | None = None,
    team_b_name: str | None = None,
    *,
    options: str = "",
):
    """Winner prediction with optional --side, --first-pick, --bo3, or --bo5."""
    try:
        parsed = _parse_lol_options(options)
        blue_name, red_name, account_for_side = _orient_for_side(
            team_a_name, team_b_name, parsed.get("side")
        )
        blue_roster_str, red_roster_str = _split_two_rosters(parsed.get("rosters"))
        await validate_and_predict(
            ctx,
            blue_name,
            red_name,
            blue_roster_str,
            red_roster_str,
            _match_type_from_options(parsed),
            account_for_side,
            parsed.get("first_pick_team_name"),
        )
    except ValueError as e:
        await ctx.send(str(e))


@lol_group.command(name="props")
async def lol_props(
    ctx: commands.Context,
    team_a_name: str | None = None,
    team_b_name: str | None = None,
    *,
    options: str = "",
):
    """Prop projections and line pricing for kills, towers, and game length."""
    try:
        parsed = _parse_lol_options(options)
        blue_name, red_name, account_for_side = _orient_for_side(
            team_a_name, team_b_name, parsed.get("side")
        )
        blue_roster_str, red_roster_str = _split_two_rosters(parsed.get("rosters"))
        await validate_and_predict_props(
            ctx,
            blue_name,
            red_name,
            blue_roster_str,
            red_roster_str,
            account_for_side,
            parsed.get("first_pick_team_name"),
            parsed.get("kills_line"),
            parsed.get("kills_over_odds"),
            parsed.get("kills_under_odds"),
            parsed.get("towers_line"),
            parsed.get("towers_over_odds"),
            parsed.get("towers_under_odds"),
            parsed.get("length_line"),
            parsed.get("length_over_odds"),
            parsed.get("length_under_odds"),
        )
    except ValueError as e:
        await ctx.send(str(e))


@lol_group.command(name="edge")
async def lol_edge(
    ctx: commands.Context,
    team_a_name: str | None = None,
    team_b_name: str | None = None,
    *,
    options: str = "",
):
    """Read-only market discovery helper for LoL Polymarket searches."""
    try:
        parsed = _parse_lol_options(options)
    except ValueError as e:
        await ctx.send(str(e))
        return
    market = str(parsed.get("market") or "polymarket").casefold()
    if market != "polymarket":
        await ctx.send(
            "Only read-only Polymarket search is supported for LoL edge scans right now."
        )
        return
    query = parsed.get("query") or " ".join(
        part for part in (team_a_name, team_b_name, "LoL") if part
    )
    if not query:
        await ctx.send('Provide teams or a query, e.g. `!lol edge "T1" "G2"`.')
        return
    await _send_market_search(ctx, query)


@bot.command(name="props", aliases=["props_bo1", "bo1_props"])
async def props(
    ctx: commands.Context,
    team_a_name: str | None = None,
    team_b_name: str | None = None,
    *,
    options: str = "",
):
    """Predict neutral single-game props: gamelength, total kills, total towers."""
    try:
        parsed = _parse_lol_options(options)
        blue_roster_str, red_roster_str = _split_two_rosters(parsed.get("rosters"))
        await validate_and_predict_props(
            ctx,
            team_a_name,
            team_b_name,
            blue_roster_str,
            red_roster_str,
            False,
            parsed.get("first_pick_team_name"),
            parsed.get("kills_line"),
            parsed.get("kills_over_odds"),
            parsed.get("kills_under_odds"),
            parsed.get("towers_line"),
            parsed.get("towers_over_odds"),
            parsed.get("towers_under_odds"),
            parsed.get("length_line"),
            parsed.get("length_over_odds"),
            parsed.get("length_under_odds"),
        )
    except ValueError as e:
        await ctx.send(str(e))


@bot.command(name="sided_props", aliases=["props_sided", "sided_props_bo1"])
async def sided_props(
    ctx: commands.Context,
    blue_team_name: str | None = None,
    red_team_name: str | None = None,
    *,
    options: str = "",
):
    """Predict single-game props with explicit map side."""
    try:
        parsed = _parse_lol_options(options)
        blue_roster_str, red_roster_str = _split_two_rosters(parsed.get("rosters"))
        await validate_and_predict_props(
            ctx,
            blue_team_name,
            red_team_name,
            blue_roster_str,
            red_roster_str,
            True,
            parsed.get("first_pick_team_name"),
            parsed.get("kills_line"),
            parsed.get("kills_over_odds"),
            parsed.get("kills_under_odds"),
            parsed.get("towers_line"),
            parsed.get("towers_over_odds"),
            parsed.get("towers_under_odds"),
            parsed.get("length_line"),
            parsed.get("length_over_odds"),
            parsed.get("length_under_odds"),
        )
    except ValueError as e:
        await ctx.send(str(e))


@bot.command(
    name="first_selection_props",
    aliases=["fs_props", "selection_props", "first_pick_props"],
)
async def first_selection_props(
    ctx: commands.Context,
    blue_team_name: str | None = None,
    red_team_name: str | None = None,
    first_pick_team_name: str | None = None,
    rosters: str | None = None,
):
    """Predict props with explicit map side and First Selection first-pick team."""
    if not first_pick_team_name:
        await ctx.send("Provide the first-pick team name.")
        return
    blue_roster_str, red_roster_str = _split_two_rosters(rosters)
    await validate_and_predict_props(
        ctx,
        blue_team_name,
        red_team_name,
        blue_roster_str,
        red_roster_str,
        True,
        first_pick_team_name,
    )


@bot.command(name="bo2", aliases=["BO2"])
async def bo2(
    ctx: commands.Context,
    blue_team_name: str | None = None,
    red_team_name: str | None = None,
    rosters: str | None = None,
):
    """
    Predict a best-of-two (supports 2-0 / 1-1 / 0-2 series formats).
    Optional rosters string: see `!bo1`.
    """
    blue_roster_str, red_roster_str = _split_two_rosters(rosters)
    await validate_and_predict(
        ctx,
        blue_team_name,
        red_team_name,
        blue_roster_str,
        red_roster_str,
        "bo2",
        False,
    )


@bot.command(name="bo3", aliases=["BO3"])
async def bo3(
    ctx: commands.Context,
    blue_team_name: str | None = None,
    red_team_name: str | None = None,
    rosters: str | None = None,
):
    """Predict a best-of-three. Optional rosters string: see `!bo1`."""
    blue_roster_str, red_roster_str = _split_two_rosters(rosters)
    await validate_and_predict(
        ctx,
        blue_team_name,
        red_team_name,
        blue_roster_str,
        red_roster_str,
        "bo3",
        False,
    )


@bot.command(name="bo5", aliases=["BO5"])
async def bo5(
    ctx: commands.Context,
    blue_team_name: str | None = None,
    red_team_name: str | None = None,
    rosters: str | None = None,
):
    """Predict a best-of-five. Optional rosters string: see `!bo1`."""
    blue_roster_str, red_roster_str = _split_two_rosters(rosters)
    await validate_and_predict(
        ctx,
        blue_team_name,
        red_team_name,
        blue_roster_str,
        red_roster_str,
        "bo5",
        False,
    )


# ── betting utils ───────────────────────────────────────────────────────── #


@bot.command(name="odds", aliases=["prob_to_odds", "win_probability_to_odds"])
async def convert_win_probability_to_odds(
    ctx: commands.Context, win_probability: str | None = None, to_decimal: str = "True"
):
    """Convert probability (e.g., `0.64` or `64%`) to odds. `to_decimal=True|False`"""
    if not win_probability:
        await ctx.send("Please provide a win probability.")
        return
    to_dec = to_decimal.lower() in {"true", "1", "t", "y", "yes"}
    try:
        p = convert_odds(win_probability)
        if not (0 <= p <= 1):
            msg = "Win probability must be between 0% and 100%."
            raise ValueError(msg)  # noqa: TRY301
        odds = calculate_odds(p, to_dec)
        kind = "decimal" if to_dec else "fractional"
        await ctx.send(f"{kind.capitalize()} odds for {p * 100:.2f}%: {odds}")
    except ValueError as ve:
        await ctx.send(str(ve))


@bot.command(name="prob", aliases=["odds_to_prob", "odds_to_win_probability"])
async def convert_odds_to_win_probability(
    ctx: commands.Context, odds: str | None = None
):
    """Convert decimal odds to win probability."""
    if not odds:
        await ctx.send("Please provide odds.")
        return
    try:
        o = float(odds)
        if o <= 1:
            msg = "Odds must be greater than 1."
            raise ValueError(msg)  # noqa: TRY301
        p = calculate_prob(o)
        await ctx.send(f"Implied win probability for odds {o}: {p * 100:.2f}%")
    except ValueError as ve:
        await ctx.send(str(ve))


@bot.command(name="kelly", aliases=["kelly_criterion"])
async def kelly_criterion(
    ctx: commands.Context,
    bookmaker_odds: str | None = None,
    win_probability: str | None = None,
):
    """Half-Kelly suggestion for decimal odds and probability (e.g., `!kelly 2.1 55%`)."""
    if not bookmaker_odds or not win_probability:
        await ctx.send(
            "Provide both bookmaker odds and a win probability (e.g., `!kelly 2.1 55%`)."
        )
        return
    try:
        o = float(bookmaker_odds)
        p = convert_odds(win_probability)
        if not (0 <= p <= 1):
            msg = "Win probability must be between 0% and 100%."
            raise ValueError(msg)  # noqa: TRY301
        if o <= 1:
            msg = "Bookmaker odds must be greater than 1."
            raise ValueError(msg)  # noqa: TRY301
        f = calculate_kelly_criterion(o, p)  # half-Kelly
        await ctx.send(
            f"For odds {o} and win prob {p * 100:.2f}%: bet **{f * 100:.2f}%** of bankroll (half-Kelly)."
        )
    except ValueError as ve:
        await ctx.send(str(ve))


@bot.command(name="edge", aliases=["ev", "value"])
async def edge(
    ctx: commands.Context,
    decimal_odds: str | None = None,
    win_probability: str | None = None,
):
    """Show fair odds, edge, and half-Kelly for a model probability."""
    if not decimal_odds or not win_probability:
        await ctx.send(
            "Provide decimal odds and win probability, e.g. `!edge 2.10 55%`."
        )
        return
    try:
        signal = build_edge_signal(
            model_probability=convert_odds(win_probability),
            market_odds=float(decimal_odds),
        )
        await ctx.send(
            "\n".join(
                [
                    "**Market Edge**",
                    f"- Model probability: **{signal.model_probability * 100:.2f}%**",
                    f"- Market implied probability: **{signal.implied_probability * 100:.2f}%**",
                    f"- Fair odds: **{signal.fair_odds:.3f}**",
                    f"- Edge: **{signal.edge * 100:.2f}%**",
                    f"- Half-Kelly stake: **{signal.half_kelly_fraction * 100:.2f}%** of bankroll",
                ]
            )
        )
    except ValueError as ve:
        await ctx.send(str(ve))


# ── ledger commands ────────────────────────────────────────────────────── #


@bot.group(name="bet", invoke_without_command=True)
async def bet_group(ctx: commands.Context):
    """Manual bet ledger commands."""
    await ctx.send(
        "**Bet Ledger Commands**\n"
        '- `!bet record --event "Team WE vs LNG" --market kills --selection over --line 26.5 --odds 1.85 --stake 25 --book polymarket --prob 0.557 --edge 0.031 --map 1 --league LPL`\n'
        "- `!bet list --status open`\n"
        "- `!bet settle 12 --result win`\n"
        "- `!bet delete 12`\n"
        '- `!bet bankroll --balance 1250 --note "after LPL slate"`\n'
        "This records bets you manually placed. It never places orders."
    )


@bet_group.command(name="record")
async def bet_record(ctx: commands.Context, *, options: str = ""):
    """Record a manually placed bet in the local ledger."""
    try:
        parsed = _parse_bet_options(options)
        selection = str(parsed.get("selection") or "").strip()
        over_under = (
            selection.title() if selection.casefold() in {"over", "under"} else None
        )
        data = {
            "event_date": parsed.get("event_date")
            or datetime.now(UTC).date().isoformat(),
            "sport": parsed.get("sport") or "League of Legends",
            "league": parsed.get("league"),
            "event_name": parsed.get("event_name"),
            "market_type": parsed.get("market_type"),
            "selection": selection.title() if over_under else selection,
            "side": parsed.get("side"),
            "odds_decimal": parsed.get("odds_decimal"),
            "stake": parsed.get("stake"),
            "currency": parsed.get("currency") or "EUR",
            "model_probability": parsed.get("model_probability"),
            "edge": parsed.get("edge"),
            "kelly_fraction": parsed.get("kelly_fraction"),
            "is_smoke": bool(parsed.get("is_smoke")),
            "is_live": bool(parsed.get("is_live")),
            "bookmaker": parsed.get("bookmaker"),
            "bet_slip_ref": parsed.get("bet_slip_ref"),
            "source_prediction": parsed.get("source_prediction")
            or "discord !bet record",
            "notes": parsed.get("notes"),
            "tags": parsed.get("tags"),
            "line": parsed.get("line"),
            "over_under": over_under,
            "best_of": parsed.get("best_of"),
            "map_number": parsed.get("map_number"),
        }
        missing = [
            flag
            for flag, key in (
                ("--event", "event_name"),
                ("--market", "market_type"),
                ("--selection", "selection"),
                ("--odds", "odds_decimal"),
            )
            if data.get(key) in {None, ""}
        ]
        if missing:
            await ctx.send(f"Missing required option(s): {', '.join(missing)}.")
            return
        init_db()
        bet_id = insert_bet(data)
        await ctx.send(
            "\n".join(
                [
                    f"Recorded bet #{bet_id}.",
                    _format_bet_row({**data, "id": bet_id, "status": "pending"}),
                    "Ledger only: no order was placed by the bot.",
                ]
            )
        )
    except ValueError as e:
        await ctx.send(str(e))


@bet_group.command(name="list")
async def bet_list(ctx: commands.Context, *, options: str = ""):
    """List recent ledger bets."""
    try:
        parsed = _parse_bet_options(options)
        status = str(parsed.get("status") or "").casefold()
        limit = int(parsed.get("limit") or 10)
        if status == "open":
            status = "pending"
        init_db()
        bets = get_all_bets()
        if status:
            bets = [
                bet
                for bet in bets
                if str(bet.get("status") or "").casefold() == status
                or str(bet.get("result") or "").casefold() == status
            ]
        if not bets:
            await ctx.send("No matching bets found.")
            return
        lines = ["**Recent Bets**", *[_format_bet_row(bet) for bet in bets[:limit]]]
        await ctx.send("\n".join(lines)[: MESSAGE_LIMIT - 1])
    except ValueError as e:
        await ctx.send(str(e))


@bet_group.command(name="settle")
async def bet_settle(
    ctx: commands.Context, bet_id: int | None = None, *, options: str = ""
):
    """Settle a bet as win/loss/push/void."""
    if bet_id is None:
        await ctx.send("Provide a bet id, e.g. `!bet settle 12 --result win`.")
        return
    try:
        parsed = _parse_bet_options(options)
        result = parsed.get("result")
        if not result:
            await ctx.send("Provide `--result win|loss|push|void`.")
            return
        init_db()
        settled = settle_bet(bet_id, str(result))
        await ctx.send(
            f"Settled bet #{bet_id}: {settled['result']} | P&L {settled['pnl']:+.2f}."
        )
    except ValueError as e:
        await ctx.send(str(e))


@bet_group.command(name="delete")
async def bet_delete(ctx: commands.Context, bet_id: int | None = None):
    """Delete a ledger bet by id."""
    if bet_id is None:
        await ctx.send("Provide a bet id, e.g. `!bet delete 12`.")
        return
    init_db()
    delete_bet(bet_id)
    await ctx.send(f"Deleted bet #{bet_id}.")


@bet_group.command(name="bankroll")
async def bet_bankroll(ctx: commands.Context, *, options: str = ""):
    """Log a bankroll balance entry."""
    try:
        parsed = _parse_bet_options(options)
        if "balance" not in parsed:
            await ctx.send("Provide `--balance`, e.g. `!bet bankroll --balance 1250`.")
            return
        init_db()
        entry_id = insert_bankroll_entry(
            balance=float(parsed["balance"]),
            deposit=float(parsed.get("deposit") or 0),
            withdrawal=float(parsed.get("withdrawal") or 0),
            notes=str(parsed.get("notes") or ""),
        )
        await ctx.send(f"Logged bankroll entry #{entry_id}.")
    except ValueError as e:
        await ctx.send(str(e))


@bot.command(name="markets", aliases=["market_search", "polymarket"])
async def markets(ctx: commands.Context, *, query: str | None = None):
    """Search read-only Polymarket markets."""
    if not query:
        await ctx.send("Provide a search query, e.g. `!markets LoL T1`.")
        return
    await _send_market_search(ctx, query)


# ── admin ───────────────────────────────────────────────────────────────── #


@bot.command(name="kill", aliases=["stop"])
@commands.is_owner()
async def kill(ctx: commands.Context):
    """Shutdown (owner only)."""
    try:
        logger.info("Shutting down the bot...")
        await bot.close()
        logger.info("Bot shut down.")
    except Exception as e:
        logger.error("Failed to shut down bot properly: %s", e)
        await ctx.send("Failed to shut down the bot properly.")


# ── runner ──────────────────────────────────────────────────────────────── #


def run_bot() -> None:
    """Start the Discord bot."""
    try:
        for module in default_registry().all():
            _raise_for_blocking_artifact_failures(module)
        token = os.getenv(DISCORD_TOKEN_ENV)
        if not token:
            logger.error("Environment variable %s is missing.", DISCORD_TOKEN_ENV)
            return
        bot.run(token)
    except Exception as e:
        logger.error("Failed to start bot: %s", e)


if __name__ == "__main__":
    run_bot()
