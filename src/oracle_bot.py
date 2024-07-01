"""
This bot shows commands for a League of Legends esports prediction model.
It allows users to call down predictions and view various information.
"""

import discord
from discord.ext import commands

from src.discord_predictions.discord import (
    calculate_kelly_criterion,
    calculate_odds,
    calculate_prob,
    convert_odds,
    convert_to_discord_markdown,
    format_leagues_message,
    format_schedule_message,
    get_allowed_models,
    get_formatted_player_profile,
    get_formatted_team_profile,
    get_validation_metrics,
    handle_command_error,
    predict_and_format_result,
    send_validation_result,
)
from src.ingestion.schedule import PandaScoreSchedule
from utils.logger import logger
from utils.paths import SCHEDULE
from utils.secrets import get_secret_value
from utils.team import Team

# Constants
DISCORD_TOKEN_ENV = "DISCORD_TOKEN"
BOT_COMMAND_PREFIX = "!"
BOT_DESCRIPTION = "A comprehensive League of Legends esports prediction bot."
PLEASE_PROVIDE = "Please provide both a blue and red team name."
DIFFERENT_TEAMS = "The two teams must be different."

# Initialize bot with command prefix and description
bot = commands.Bot(
    command_prefix=BOT_COMMAND_PREFIX,
    description=BOT_DESCRIPTION,
    intents=discord.Intents.all(),
)


@bot.event
async def on_ready():
    """Event handler for when the bot is ready."""
    logger.info(f"{bot.user} has connected to Discord!")


@bot.command(name="code", aliases=["github", "repository", "git", "source"])
async def code(ctx):
    """Sends a message with the GitHub repository link."""
    try:
        response = (
            "This model is entirely open-source!\nWe'd love to talk about ideas or contributions!\n"
            "Check the link at: https://github.com/MRittinghouse/esports-analytics"
        )
    except Exception as e:
        response = handle_command_error(e, additional_info="Could not retrieve code information.")
    await ctx.send(response)


@bot.command(name="leagues", aliases=["league", "show_leagues", "show_league"])
async def leagues(ctx):
    """Displays the list of supported leagues."""
    try:
        leagues = PandaScoreSchedule.load_schedule(SCHEDULE)["league"].unique()
        response = format_leagues_message(sorted(leagues))
    except Exception as e:
        response = handle_command_error(e, additional_info="Could not retrieve league information.")
    await ctx.send(response)


@bot.command(name="schedule")
async def schedule(ctx, leagues: str = None):
    """Displays the upcoming schedule for the specified leagues."""
    try:
        schedule_df = PandaScoreSchedule.load_schedule(SCHEDULE, leagues)
        response = format_schedule_message(schedule_df)
    except Exception as e:
        response = handle_command_error(e, additional_info="Could not retrieve schedule.")
    await ctx.send(response)


@bot.command(name="team_roster", aliases=["roster"])
async def roster(ctx, team=None):
    """Displays the roster for the specified team."""
    if not team:
        await ctx.send("Please provide a team name.")
        return
    message = await ctx.send(content="```Extracting...```")
    try:
        team_data = Team(name=team).get_team_info()
        output = convert_to_discord_markdown(team_data)
    except Exception as e:
        output = handle_command_error(e, additional_info="Could not extract roster information.")
    await message.edit(content=output)


@bot.command(name="team_profile", aliases=["team"])
async def team_profile(ctx, team_name: str = None, verbose: bool = False):
    """Displays the profile for the specified team."""
    if team_name is None:
        await ctx.send("Please provide a team name.")
        return
    if not isinstance(verbose, bool):
        verbose = verbose.lower() in ["true", "1", "t", "y", "yes"]
    try:
        profile, error = await get_formatted_team_profile(team_name, verbose)
        response = profile if profile else error
    except Exception as e:
        response = handle_command_error(e, additional_info="Could not retrieve team profile.")
    await ctx.send(response)


@bot.command(name="player_profile", aliases=["player"])
async def player_profile(ctx, player_name: str = None, verbose: bool = False):
    """Displays the profile for the specified player."""
    if player_name is None:
        await ctx.send("Please provide a player name.")
        return
    if not isinstance(verbose, bool):
        verbose = verbose.lower() in ["true", "1", "t", "y", "yes"]
    try:
        profile, error = await get_formatted_player_profile(player_name, verbose)
        response = profile if profile else error
    except Exception as e:
        response = handle_command_error(e, additional_info="Could not retrieve player profile.")
    await ctx.send(response)


@bot.command(name="validate", aliases=["validation"])
async def validate(ctx, model: str = None, graph: bool = False):
    """Validates the specified model and displays the metrics."""
    if model is None:
        await ctx.send("Please provide a model to validate.")
        return
    try:
        all_models = get_allowed_models()
        if model not in all_models:
            await ctx.send(content=f"Model must be one of: {' | '.join(all_models)}")
            return
        metrics, images = get_validation_metrics(model, graph)
        await send_validation_result(ctx, metrics, images)
    except Exception as e:
        await ctx.send(handle_command_error(e, "Validation failed."))


@bot.command(name="bo1", aliases=["predict", "prediction", "match"])
async def bo1(
    ctx, blue_team_name: str = None, red_team_name: str = None, blue_roster_str: str = None, red_roster_str: str = None
):
    """Predicts the outcome of a best-of-one match between two teams."""
    if not blue_team_name or not red_team_name:
        await ctx.send(PLEASE_PROVIDE)
        return
    if blue_team_name == red_team_name:
        await ctx.send(DIFFERENT_TEAMS)
        return
    await predict_and_format_result(ctx, blue_team_name, red_team_name, blue_roster_str, red_roster_str, "bo1", False)


@bot.command(name="sided_bo1", aliases=["sided_predict", "sided_prediction", "sided_match"])
async def sided_bo1(
    ctx, blue_team_name: str = None, red_team_name: str = None, blue_roster_str: str = None, red_roster_str: str = None
):
    """Predicts the outcome of a best-of-one match between two teams with side considerations."""
    if not blue_team_name or not red_team_name:
        await ctx.send(PLEASE_PROVIDE)
        return
    if blue_team_name == red_team_name:
        await ctx.send(DIFFERENT_TEAMS)
        return
    await predict_and_format_result(ctx, blue_team_name, red_team_name, blue_roster_str, red_roster_str, "bo1", True)


@bot.command(name="bo3")
async def bo3(
    ctx, blue_team_name: str = None, red_team_name: str = None, blue_roster_str: str = None, red_roster_str: str = None
):
    """Predicts the outcome of a best-of-three match between two teams."""
    if not blue_team_name or not red_team_name:
        await ctx.send(PLEASE_PROVIDE)
        return
    if blue_team_name == red_team_name:
        await ctx.send(DIFFERENT_TEAMS)
        return
    await predict_and_format_result(ctx, blue_team_name, red_team_name, blue_roster_str, red_roster_str, "bo3", False)


@bot.command(name="bo5")
async def bo5(
    ctx, blue_team_name: str = None, red_team_name: str = None, blue_roster_str: str = None, red_roster_str: str = None
):
    """Predicts the outcome of a best-of-five match between two teams."""
    if not (blue_team_name or red_team_name):
        await ctx.send(PLEASE_PROVIDE)
        return
    if blue_team_name == red_team_name:
        await ctx.send(DIFFERENT_TEAMS)
        return
    await predict_and_format_result(ctx, blue_team_name, red_team_name, blue_roster_str, red_roster_str, "bo5", False)


@bot.command(name="odds", aliases=["prob_to_odds", "win_probability_to_odds"])
async def convert_win_probability_to_odds(ctx, win_probability: str = None, to_decimal: bool = True):
    """Converts a win probability to odds."""
    if not win_probability:
        await ctx.send("Please provide a win probability.")
        return

    try:
        win_probability = convert_odds(win_probability)
        if not (0 <= win_probability <= 1):
            raise ValueError("Win probability must be between 0% and 100%.")
    except ValueError as ve:
        await ctx.send(str(ve))
        return

    odds = calculate_odds(win_probability, to_decimal)
    odds_type = "decimal" if to_decimal else "fractional"
    await ctx.send(f"The {odds_type} odds for a win probability of {win_probability * 100:.2f}% are {odds}.")


@bot.command(name="to_prob", aliases=["odds_to_prob", "odds_to_win_probability"])
async def convert_odds_to_win_probability(ctx, odds: str = None):
    """Converts decimal odds to a win probability."""
    if not odds:
        await ctx.send("Please provide odds.")
        return

    try:
        odds = float(odds)
        if odds <= 1:
            raise ValueError("Odds must be greater than 1.")
    except ValueError as ve:
        await ctx.send(str(ve))
        return

    win_probability = calculate_prob(odds)
    await ctx.send(f"The win probability for odds of {odds} is {win_probability * 100:.2f}%.")


@bot.command(name="kelly", aliases=["kelly_criterion"])
async def kelly_criterion(ctx, bookmaker_odds: str = None, win_probability: str = None):
    """Calculates the Kelly Criterion based on the given win probability and bookmaker odds."""
    if not win_probability or not bookmaker_odds:
        await ctx.send("Please provide both a win probability and bookmaker odds.")
        return

    try:
        bookmaker_odds = float(bookmaker_odds)
        win_probability = convert_odds(win_probability)

        if not (0 <= win_probability <= 1):
            raise ValueError("Win probability must be between 0% and 100%.")
        if bookmaker_odds <= 1:
            raise ValueError("Bookmaker odds must be greater than 1.")
    except ValueError as ve:
        await ctx.send(str(ve))
        return

    kelly_fraction = calculate_kelly_criterion(bookmaker_odds, win_probability)
    await ctx.send(
        f"Given the bookmaker odds of {bookmaker_odds}, and a win probability of {win_probability * 100:.2f}%\n"
        f"the Kelly Criterion suggests betting {kelly_fraction * 100:.2f}% of your bankroll."
    )


@bot.command(name="kill", aliases=["stop"])
@commands.is_owner()
async def kill(ctx):
    """Kills the bot. Only the bot owner can use this command."""
    try:
        logger.info("Shutting down the bot...")
        await bot.close()
        logger.info("Bot has been successfully shut down.")
    except Exception as e:
        logger.error(f"Failed to shut down bot properly: {e}")
        await ctx.send("Failed to shut down the bot properly.")


def run_bot():
    """Runs the Discord bot."""
    try:
        bot.run(get_secret_value("lol_oracle", "DISCORD_TOKEN"))
    except Exception as e:
        logger.error(f"Failed to start the bot: {e}")


if __name__ == "__main__":
    run_bot()
