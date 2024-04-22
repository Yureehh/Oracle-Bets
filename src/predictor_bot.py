"""
This bot is a wrapper for a League of Legends esports prediction model.
It allows users to call down predictions and view various information.
"""

import os

import pandas as pd
from dotenv import load_dotenv

import discord
from discord.ext import commands
from src.data_ingest.schedule import PandaScoreSchedule
from src.discord.discord import (
    convert_to_discord_markdown,
    format_prediction_message,
    format_schedule_message,
    get_allowed_models,
    get_empty_roster,
    get_formatted_player_profile,
    get_formatted_team_profile,
    get_match_prediction,
    get_validation_metrics,
    handle_command_error,
    predict_and_format_result,
    process_roster,
    send_validation_result,
)
from utils.logger import logger
from utils.paths import PROCESSED_DIR
from utils.team import Team
from utils.utils import setup_pandas

load_dotenv()

bot = commands.Bot(
    command_prefix="!",
    description="A League of Legends esports prediction bot.",
    intents=discord.Intents.all(),
)


@bot.event
async def on_ready():
    """Event handler for when the bot is ready."""
    logger.info(f"{bot.user} has connected to Discord!")


@bot.command(name="code", aliases=["github"])
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


@bot.command(name="schedule")
async def schedule(ctx, leagues: str = None):
    """Displays the upcoming schedule for the specified leagues."""
    try:
        schedule_df = PandaScoreSchedule.load_schedule(PROCESSED_DIR / "schedule.csv", leagues)
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
        team_data = Team(name=team)
        output = convert_to_discord_markdown(team_data.get_team_info())
    except Exception as e:
        output = handle_command_error(e, additional_info="Could not extract roster information.")
    await message.edit(content=output)


@bot.command(name="team_profile", aliases=["team"])
async def team_profile(ctx, entity: str = None, verbose: bool = False):
    """Displays the profile for the specified team."""
    if not entity:
        await ctx.send("Please provide a team name.")
        return
    if not isinstance(verbose, bool):
        verbose = verbose.lower() in ["true", "1", "t", "y", "yes"]
    profile, error = await get_formatted_team_profile(entity, verbose)
    if profile:
        await ctx.send(profile)
    else:
        await ctx.send(error)


@bot.command(name="player_profile", aliases=["player"])
async def player_profile(ctx, entity: str = None, verbose: bool = False):
    """Displays the profile for the specified player."""
    if not entity:
        await ctx.send("Please provide a player name.")
        return
    if not isinstance(verbose, bool):
        verbose = verbose.lower() in ["true", "1", "t", "y", "yes"]
    profile, error = await get_formatted_player_profile(entity, verbose)
    if profile:
        await ctx.send(profile)
    else:
        await ctx.send(error)


@bot.command(name="validate", aliases=["validation"])
async def validate(ctx, method=None, graph=False):
    """Validates the specified model and displays the metrics."""
    if not method:
        output = "Please provide a method to validate."
        await ctx.send(content=output)
        return
    allowed_models = get_allowed_models()
    if method not in allowed_models:
        output = "Method must be one of:\n" + ", ".join(allowed_models) + "."
        await ctx.send(content=output)
        return
    try:
        metrics, images = get_validation_metrics(method, graph)
        await send_validation_result(ctx, metrics, images if images else None)
    except ValueError as ve:
        output = handle_command_error(ve, additional_info="Could not validate the model.")
        await ctx.send(content=output)
    except Exception as e:
        output = handle_command_error(e, additional_info="Could not validate the model.")
        await ctx.send(content=output)


@bot.command(name="predict", aliases=["prediction"])
async def predict(
    ctx,
    blue_team_name: str = None,
    red_team_name: str = None,
    verbose: bool = False,
    blue_roster_str: str = None,
    red_roster_str: str = None,
):
    """Predicts the outcome of a match between two teams."""
    if not blue_team_name or not red_team_name:
        await ctx.send("Please provide both a blue and red team name.")
        return
    message = await ctx.send("```Calculating prediction...```")
    try:
        blue_roster = (
            process_roster(blue_roster_str)
            if blue_roster_str and blue_roster_str != "{}" and blue_roster_str != get_empty_roster()
            else get_empty_roster()
        )
        red_roster = (
            process_roster(red_roster_str)
            if red_roster_str and red_roster_str != "{}" and red_roster_str != get_empty_roster()
            else get_empty_roster()
        )

        blue_team = Team(name=blue_team_name, side="Blue", roster=blue_roster)
        red_team = Team(name=red_team_name, side="Red", roster=red_roster)

        prediction = get_match_prediction(blue_team, red_team)
        blue_profile, _ = await get_formatted_team_profile(blue_team_name, True) if verbose else None, None
        red_profile, _ = await get_formatted_team_profile(red_team_name, True) if verbose else None, None
        output = format_prediction_message(prediction, blue_profile, red_profile)
    except Exception as e:
        output = handle_command_error(e, additional_info="Could not complete the prediction.")

    await message.edit(content=output)


@bot.command(name="odds", aliases=["prob_to_odds", "win_probability_to_odds"])
async def convert_win_probability_to_odds(ctx, win_probability: str = None, to_decimal: bool = True):
    """Converts a win probability to odds."""
    if not win_probability:
        await ctx.send("Please provide a win probability.")
        return

    win_probability = str(win_probability)

    try:
        if win_probability.endswith("%"):
            win_probability = float(win_probability.strip("%")) / 100
        else:
            win_probability = float(win_probability)
    except ValueError:
        await ctx.send("Invalid win probability format. Please provide a numeric value or a percentage.")
        return

    if win_probability < 0 or win_probability > 1:
        await ctx.send("Win probability must be between 0 and 1 or between 0% and 100%.")
        return

    if to_decimal:
        odds = round((1 / win_probability), 2)
        await ctx.send(f"The decimal odds for a win probability of {win_probability * 100}% are {odds}.")
    else:
        odds = round((win_probability / (1 - win_probability)), 2)
        await ctx.send(f"The fractional odds for a win probability of {win_probability * 100}% are {odds}.")


@bot.command(name="bo3", aliases=["best_of_3"])
async def best_of_3(
    ctx,
    blue_team_name: str = None,
    red_team_name: str = None,
    blue_roster_str: str = None,
    red_roster_str: str = None,
):
    """Predicts the outcome of a best-of-three match between two teams."""
    if not blue_team_name or not red_team_name:
        await ctx.send("Please provide both a blue and red team name.")
        return
    if blue_team_name == red_team_name:
        await ctx.send("Please provide two different team names.")
        return
    await predict_and_format_result(ctx, blue_team_name, red_team_name, blue_roster_str, red_roster_str, "bo3")


@bot.command(name="bo5", aliases=["best_of_5"])
async def best_of_5(
    ctx,
    blue_team_name: str = None,
    red_team_name: str = None,
    blue_roster_str: str = None,
    red_roster_str: str = None,
):
    """Predicts the outcome of a best-of-five match between two teams."""
    if not blue_team_name or not red_team_name:
        await ctx.send("Please provide both a blue and red team name.")
        return
    if blue_team_name == red_team_name:
        await ctx.send("Please provide two different team names.")
        return
    await predict_and_format_result(ctx, blue_team_name, red_team_name, blue_roster_str, red_roster_str, "bo5")


@bot.command(name="kill", aliases=["stop"])
@commands.is_owner()
async def kill(ctx):
    """Kills the bot. Only the bot owner can use this command."""
    logger.info("Killing bot...")
    if not await bot.is_owner(ctx.author):
        await ctx.send("You are not authorized to kill the bot.")
        return

    # TODO: fix the unclosed connection error
    await bot.close()


def run_bot():
    """Runs the Discord bot."""
    bot.run(os.getenv("DISCORD_TOKEN"))


if __name__ == "__main__":
    setup_pandas(pd)
    run_bot()
