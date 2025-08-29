"""
League of Legends Esports Prediction Bot.

This bot provides commands for a League of Legends esports prediction model.
Users can request predictions and view information such as team rosters,
player profiles, match schedules, and betting odds.
"""

import os

import discord
from discord.ext import commands
from dotenv import load_dotenv

from discord_predictions.discord import (
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
)
from ingestion.schedule import PandaScoreSchedule
from utils.entities.team import Team
from utils.logger import logger
from utils.paths import SCHEDULE

# Load environment variables from .env file
load_dotenv()

# Constants
DISCORD_TOKEN_ENV = "DISCORD_TOKEN"
PANDASCORE_API_KEY_ENV = "PANDASCORE_API_KEY"  # pragma: allowlist secret
BOT_COMMAND_PREFIX = "!"
BOT_DESCRIPTION = "A comprehensive League of Legends esports prediction bot."

# Initialize bot with command prefix and description
intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent
bot = commands.Bot(
    command_prefix=BOT_COMMAND_PREFIX,
    description=BOT_DESCRIPTION,
    intents=intents,
)


@bot.event
async def on_ready():
    """Event handler for when the bot is ready."""
    logger.info(f"{bot.user} has connected to Discord!")


# Utility Commands


@bot.command(name="code", aliases=["github", "repository", "git", "source"])
async def code(ctx):
    """Sends a message with the GitHub repository link."""
    response = (
        "This model is entirely open-source!\nWe'd love to discuss ideas or contributions!\n"
        "Check the link at: https://github.com/MRittinghouse/esports-analytics"
    )
    await ctx.send(response)


@bot.command(name="leagues", aliases=["league", "show_leagues", "show_league"])
async def leagues(ctx):
    """Displays the list of supported leagues."""
    try:
        leagues = PandaScoreSchedule.load_schedule(SCHEDULE)["league"].unique()
        response = format_leagues_message(sorted(leagues))
    except Exception as e:
        response = handle_command_error(
            e, additional_info="Could not retrieve league information."
        )
    await ctx.send(response)


@bot.command(name="schedule")
async def schedule(ctx, leagues: str | None = None):
    """Displays the upcoming schedule for the specified leagues."""
    try:
        schedule_df = PandaScoreSchedule.load_schedule(SCHEDULE, leagues)
        response = format_schedule_message(schedule_df)
    except Exception as e:
        response = handle_command_error(
            e, additional_info="Could not retrieve schedule."
        )
    await ctx.send(response)


# Profile and Roster Commands


@bot.command(name="team_roster", aliases=["roster"])
async def roster(ctx, team: str | None = None):
    """Displays the roster for the specified team."""
    if not team:
        await ctx.send("Please provide a team name.")
        return
    message = await ctx.send(content="```Extracting...```")
    try:
        team_data = Team(name=team).get_team_info()
        output = convert_to_discord_markdown(team_data)
    except Exception as e:
        output = handle_command_error(
            e, additional_info="Could not extract roster information."
        )
    await message.edit(content=output)


@bot.command(name="team_rosters", aliases=["rosters"])
async def rosters(ctx, teams: str | None = None):
    """Displays the roster for the specified teams."""
    if not teams:
        await ctx.send("Please provide a list of team names separated by commas.")
        return
    message = await ctx.send(content="```Extracting...```")
    output = ""
    team_names = [team.strip() for team in teams.split(",")]
    for team_name in team_names:
        try:
            team_data = Team(name=team_name).get_team_info()
            output += convert_to_discord_markdown(team_data)
        except Exception as e:
            output += handle_command_error(
                e,
                additional_info=f"Could not extract roster information for {team_name}.\n",
            )
    await message.edit(content=output)


@bot.command(name="team_profile", aliases=["team"])
async def team_profile(ctx, team_name: str | None = None):
    """Displays the profile for the specified team."""
    if not team_name:
        await ctx.send("Please provide a team name.")
        return
    try:
        profile, error = await get_formatted_team_profile(team_name)
        response = profile if profile else error
    except Exception as e:
        response = handle_command_error(
            e, additional_info="Could not retrieve team profile."
        )
    await ctx.send(response)


@bot.command(name="player_profile", aliases=["player"])
async def player_profile(ctx, player_name: str | None = None, verbose: str = "False"):
    """Displays the profile for the specified player."""
    if not player_name:
        await ctx.send("Please provide a player name.")
        return
    verbose = verbose.lower() in ["true", "1", "t", "y", "yes"]
    try:
        profile, error = await get_formatted_player_profile(player_name, verbose)
        response = profile if profile else error
    except Exception as e:
        response = handle_command_error(
            e, additional_info="Could not retrieve player profile."
        )
    await ctx.send(response)


# Prediction Commands


@bot.command(name="bo1", aliases=["predict", "prediction", "match", "BO1"])
async def bo1(
    ctx,
    blue_team_name: str | None = None,
    red_team_name: str | None = None,
    rosters: str | None = None,
):
    """Predicts the outcome of a best-of-one match between two teams."""
    blue_roster_str, red_roster_str = (
        ([*rosters.split(","), None, None])[:2] if rosters else (None, None)
    )
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
    ctx,
    blue_team_name: str | None = None,
    red_team_name: str | None = None,
    rosters: str | None = None,
):
    """Predicts the outcome of a best-of-one match between two teams with side considerations."""
    blue_roster_str, red_roster_str = (
        ([*rosters.split(","), None, None])[:2] if rosters else (None, None)
    )
    await validate_and_predict(
        ctx, blue_team_name, red_team_name, blue_roster_str, red_roster_str, "bo1", True
    )


@bot.command(name="bo2", aliases=["BO2"])
async def bo2(
    ctx,
    blue_team_name: str | None = None,
    red_team_name: str | None = None,
    rosters: str | None = None,
):
    """Predicts the outcome of a best-of-two match between two teams."""
    blue_roster_str, red_roster_str = (
        ([*rosters.split(","), None, None])[:2] if rosters else (None, None)
    )
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
    ctx,
    blue_team_name: str | None = None,
    red_team_name: str | None = None,
    rosters: str | None = None,
):
    """Predicts the outcome of a best-of-three match between two teams."""
    blue_roster_str, red_roster_str = (
        ([*rosters.split(","), None, None])[:2] if rosters else (None, None)
    )
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
    ctx,
    blue_team_name: str | None = None,
    red_team_name: str | None = None,
    rosters: str | None = None,
):
    """Predicts the outcome of a best-of-five match between two teams."""
    blue_roster_str, red_roster_str = (
        ([*rosters.split(","), None, None])[:2] if rosters else (None, None)
    )
    await validate_and_predict(
        ctx,
        blue_team_name,
        red_team_name,
        blue_roster_str,
        red_roster_str,
        "bo5",
        False,
    )


# Odds and Betting Commands


@bot.command(name="odds", aliases=["prob_to_odds", "win_probability_to_odds"])
async def convert_win_probability_to_odds(
    ctx, win_probability: str | None = None, to_decimal: str = "True"
):
    """Converts a win probability to odds."""
    if not win_probability:
        await ctx.send("Please provide a win probability.")
        return
    to_decimal = to_decimal.lower() in ["true", "1", "t", "y", "yes"]
    try:
        win_probability = convert_odds(win_probability)
        if not (0 <= win_probability <= 1):
            msg = "Win probability must be between 0% and 100%."
            raise ValueError(msg)
        odds = calculate_odds(win_probability, to_decimal)
        odds_type = "decimal" if to_decimal else "fractional"
        await ctx.send(
            f"The {odds_type} odds for a win probability of {win_probability * 100:.2f}% are {odds}."
        )
    except ValueError as ve:
        await ctx.send(str(ve))


@bot.command(name="prob", aliases=["odds_to_prob", "odds_to_win_probability"])
async def convert_odds_to_win_probability(ctx, odds: str | None = None):
    """Converts decimal odds to a win probability."""
    if not odds:
        await ctx.send("Please provide odds.")
        return
    try:
        odds = float(odds)
        if odds <= 1:
            msg = "Odds must be greater than 1."
            raise ValueError(msg)
        win_probability = calculate_prob(odds)
        await ctx.send(
            f"The win probability for odds of {odds} is {win_probability * 100:.2f}%."
        )
    except ValueError as ve:
        await ctx.send(str(ve))


@bot.command(name="kelly", aliases=["kelly_criterion"])
async def kelly_criterion(
    ctx, bookmaker_odds: str | None = None, win_probability: str | None = None
):
    """Calculates the Kelly Criterion based on the given win probability and bookmaker odds."""
    if not bookmaker_odds or not win_probability:
        await ctx.send("Please provide both bookmaker odds and a win probability.")
        return
    try:
        bookmaker_odds = float(bookmaker_odds)
        win_probability = convert_odds(win_probability)
        if not (0 <= win_probability <= 1):
            msg = "Win probability must be between 0% and 100%."
            raise ValueError(msg)
        if bookmaker_odds <= 1:
            msg = "Bookmaker odds must be greater than 1."
            raise ValueError(msg)
        kelly_fraction = calculate_kelly_criterion(bookmaker_odds, win_probability)
        await ctx.send(
            f"Given bookmaker odds of {bookmaker_odds} and a win probability of {win_probability * 100:.2f}%:\n"
            f"The Kelly Criterion suggests betting {kelly_fraction * 100:.2f}% of your bankroll."
        )
    except ValueError as ve:
        await ctx.send(str(ve))


# Administrative Commands


@bot.command(name="kill", aliases=["stop"])
@commands.is_owner()
async def kill(ctx):
    """Shuts down the bot. Only the bot owner can use this command."""
    try:
        logger.info("Shutting down the bot...")
        await bot.close()
        logger.info("Bot has been successfully shut down.")
    except Exception as e:
        logger.error(f"Failed to shut down bot properly: {e}")
        await ctx.send("Failed to shut down the bot properly.")


# Bot Runner


def run_bot():
    """Runs the Discord bot."""
    try:
        discord_token = os.getenv(DISCORD_TOKEN_ENV)
        if not discord_token:
            logger.error("Discord token is not set in environment variables.")
            return
        bot.run(discord_token)
    except Exception as e:
        logger.error(f"Failed to start the bot: {e}")


if __name__ == "__main__":
    run_bot()
