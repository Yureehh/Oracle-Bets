# run_bot.py
"""
League of Legends Esports Prediction Bot.

Commands for predictions, team/player info, schedules, and betting utilities.
"""

from __future__ import annotations

import os

import discord
from discord.ext import commands
from dotenv import load_dotenv
from ingestion.data_generation.schedule import (
    fetch_and_store_schedule,
    get_or_update_schedule,
)

from discord_predictions.discord_utils import (
    calculate_kelly_criterion,
    calculate_odds,
    calculate_prob,
    convert_odds,
    convert_to_discord_markdown,
    format_leagues_message,
    format_schedule_message,
    get_formatted_player_profile,
    get_formatted_team_profile,
    handle_command_error,
    validate_and_predict,
    validate_and_predict_props,
)
from discord_predictions.team import Team
from utils.logger import logger
from utils.paths import SCHEDULE

# ── env & bot setup ─────────────────────────────────────────────────────── #

load_dotenv()

DISCORD_TOKEN_ENV = "DISCORD_TOKEN"  # noqa: S105
BOT_COMMAND_PREFIX = "!"
BOT_DESCRIPTION = "LoL esports prediction & betting helper."

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
        await msg.edit(content=convert_to_discord_markdown(team_df))
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
            output += convert_to_discord_markdown(team_df)
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
    blue_team_name: str | None = None,
    red_team_name: str | None = None,
    rosters: str | None = None,
):
    """
    Predict a best-of-one. Optional rosters:
    `!bo1 T1 G2 "t1top,t1jng,t1mid,t1adc,t1sup | g2top,g2jng,g2mid,g2adc,g2sup"`
    """
    blue_roster_str, red_roster_str = _split_two_rosters(rosters)
    await validate_and_predict(
        ctx,
        blue_team_name,
        red_team_name,
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
    """Best-of-one with side consideration (Blue/Red advantages)."""
    blue_roster_str, red_roster_str = _split_two_rosters(rosters)
    await validate_and_predict(
        ctx, blue_team_name, red_team_name, blue_roster_str, red_roster_str, "bo1", True
    )


@bot.command(name="props", aliases=["props_bo1", "bo1_props"])
async def props(
    ctx: commands.Context,
    blue_team_name: str | None = None,
    red_team_name: str | None = None,
    rosters: str | None = None,
):
    """Predict single-game props: gamelength, total kills, total towers."""
    blue_roster_str, red_roster_str = _split_two_rosters(rosters)
    await validate_and_predict_props(
        ctx, blue_team_name, red_team_name, blue_roster_str, red_roster_str, False
    )


@bot.command(name="sided_props", aliases=["props_sided", "sided_props_bo1"])
async def sided_props(
    ctx: commands.Context,
    blue_team_name: str | None = None,
    red_team_name: str | None = None,
    rosters: str | None = None,
):
    """Predict single-game props with side consideration."""
    blue_roster_str, red_roster_str = _split_two_rosters(rosters)
    await validate_and_predict_props(
        ctx, blue_team_name, red_team_name, blue_roster_str, red_roster_str, True
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
        token = os.getenv(DISCORD_TOKEN_ENV)
        if not token:
            logger.error("Environment variable %s is missing.", DISCORD_TOKEN_ENV)
            return
        bot.run(token)
    except Exception as e:
        logger.error("Failed to start bot: %s", e)


if __name__ == "__main__":
    run_bot()
