"""
Discord

This module contains utility functions for interacting with the discord bot.
It includes functions for handling commands, formatting messages, and sending predictions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from discord import File

import discord_predictions.match_predictor as mp
from discord_predictions.best_ofs import BestOfs
from utils.paths import DISCORD_CONFIG, FIGURES_DIR, FLATTENED_PLAYERS, FLATTENED_TEAMS, INSIGHTS_DIR
from utils.team import Team
from utils.utils import json_loader, parquet_loader

config = json_loader(DISCORD_CONFIG)
match_predictor = mp.MatchPredictor()

# Constants
MESSAGE_LIMIT = config.get("MESSAGE_LIMIT", 2000)
EMPTY_ROSTER = config["EMPTY_ROSTER"].copy()
VALID_MATCH_TYPES = ["bo1", "bo3", "bo5"]
POSITIONS = ["top", "jng", "mid", "bot", "sup"]
MODEL_FILES = config["MODEL_FILES"]


def get_empty_roster() -> Dict[str, str]:
    """Returns a copy of the empty roster configuration."""
    return EMPTY_ROSTER.copy()


def read_discord_image(image_path: Path, filename: str = "image.png") -> File:
    """Reads an image from the given path and returns a discord File object."""
    with image_path.open("rb") as file:
        return File(file, filename=filename)


def handle_command_error(error: Exception, additional_info: str = "") -> str:
    """Template for formatting and handling command errors."""
    return (
        f"Something went wrong. {additional_info} If this issue persists, please contact either Yureeh or ProjektZero. Error: \n"
        f"```{error}```"
    )


def format_leagues_message(leagues: List[str]) -> str:
    """Formats a list of leagues into a Discord message."""
    if not leagues:
        return "No leagues found. Please check the league names."
    return "Leagues playing in the next 7 days are:\n" + "\n".join(f"- {league}" for league in leagues)


def format_schedule_message(schedule_df: pd.DataFrame) -> str:
    """Formats the message displaying scheduled games from a DataFrame."""
    if schedule_df.empty or "league" not in schedule_df.columns:
        return "No upcoming matches found."

    message = ""
    too_long_alert = "Message too long. Please specify a narrower filter."

    for league in np.sort(schedule_df["league"].unique()):
        formatted_league = format_league(schedule_df, league)
        if len(message) + len(formatted_league) > MESSAGE_LIMIT - len(too_long_alert):
            return message + too_long_alert
        message += formatted_league

    return message or "No upcoming matches found. Double-check the league names."


def format_league(df: pd.DataFrame, league: str) -> str:
    """Formats a league section for a schedule message."""
    league_df = df[df["league"] == league].head(5)
    league_df = league_df.assign(league=league_df["league"].str.split().str[:3].str.join(" "))
    markdown = league_df.to_markdown(index=False)
    clean_markdown = "\n".join(line.lstrip() for line in markdown.split("\n"))
    return f"Upcoming {league} Games (Next 5 Matches Within 7 Days):\n```{clean_markdown}```\n\n"


def get_validation_metrics(model: str, get_graph: bool = False) -> Tuple[str, List[File]]:
    """Retrieves validation metrics and optional graphs for the specified model."""
    if model not in MODEL_FILES:
        raise ValueError(f"Model '{model}' is not supported.")

    metrics_filename, cf_graph_filename, ha_graph_filename, aot_graph_filename = MODEL_FILES[model]
    metrics_file_path = INSIGHTS_DIR / metrics_filename
    graph_file_path, ha_graph_file_path, aot_graph_file_path = (
        FIGURES_DIR / cf_graph_filename,
        FIGURES_DIR / ha_graph_filename,
        FIGURES_DIR / aot_graph_filename,
    )

    metrics = json_loader(metrics_file_path)
    metrics_df = pd.DataFrame(metrics, index=[0]).round(2)
    metrics_md = convert_to_discord_markdown(metrics_df)

    if get_graph:
        validation_graph = read_discord_image(graph_file_path, cf_graph_filename)
        historical_accuracy_graph = read_discord_image(ha_graph_file_path, ha_graph_filename)
        accuracy_over_time_graph = read_discord_image(aot_graph_file_path, aot_graph_filename)
        return metrics_md, [validation_graph, historical_accuracy_graph, accuracy_over_time_graph]

    return metrics_md, None


def get_player_data(entity_name: str, players_path: Path) -> Optional[pd.DataFrame]:
    """Retrieves player data from the specified file path."""
    try:
        players_df = parquet_loader(players_path)
        filtered_players = players_df[players_df["playername"].str.lower() == entity_name.lower()]
        return filtered_players if not filtered_players.empty else None
    except Exception as e:
        raise ValueError(f"Error processing player data: {e}")


def get_team_data(entity_name: str, teams_path: Path) -> Optional[pd.DataFrame]:
    """Retrieves team data from the specified file path."""
    try:
        teams_df = parquet_loader(teams_path)
        filtered_teams = teams_df[teams_df["teamname"].str.lower() == entity_name.lower()]
        return filtered_teams if not filtered_teams.empty else None
    except Exception as e:
        raise ValueError(f"Error processing team data: {e}")


def format_player_profile(data: pd.DataFrame, truncate: bool = False) -> str:
    """Formats the player profile data into a Discord-friendly Markdown format."""
    stats_names = [
        "Position",
        "Team",
        "League",
        "Elo",
        "Plackett-Luce Score",
        "TrueSkill Score",
        "EGPM Dominance",
        "Blue Side Win Rate",
        "Red Side Win Rate",
        "K/D/A Ratio",
        "K/D/A at 15",
        "Gold Diff At 15",
        "CS Diff At 15",
        "XP Diff At 15",
        "Avg. Game Time",
        "CSPM",
        "DPM",
        "EGPM",
        "VSPM",
        "Gold Share",
        "Damage Share",
        "Gold Efficiency",
        "XP Efficiency",
    ]
    stats_values = [
        data["position"].iloc[0].capitalize(),
        data["teamname"].iloc[0],
        data["league"].iloc[0],
        f"{data['elo'].iloc[0]:.2f}",
        f"{data['gl2_mu'].iloc[0]:.2f}",
        f"{data['pl_mu'].iloc[0]:.2f}",
        f"{data['trueskill_mu'].iloc[0]:.2f}",
        f"{data['ema_blue_side'].iloc[0] * 100:.2f}%",
        f"{data['ema_red_side'].iloc[0] * 100:.2f}%",
        f"{data['ema_kda'].iloc[0]:.2f}",
        f"{data['ema_killsat15'].iloc[0]:.2f} / {data['ema_deathsat15'].iloc[0]:.2f} / {data['ema_assistsat15'].iloc[0]:.2f}",
        f"{data['ema_golddiffat15'].iloc[0]:.2f}",
        f"{data['ema_csdiffat15'].iloc[0]:.2f}",
        f"{data['ema_xpdiffat15'].iloc[0]:.2f}",
        f"{data['ema_gamelength'].iloc[0]:.2f} mins",
        f"{data['ema_cspm'].iloc[0]:.2f}",
        f"{data['ema_dpm'].iloc[0]:.2f}",
        f"{data['ema_egpm'].iloc[0]:.2f}",
        f"{data['ema_vspm'].iloc[0]:.2f}",
        f"{data['ema_earnedgoldshare'].iloc[0] * 100:.2f}%",
        f"{data['ema_damageshare'].iloc[0] * 100:.2f}%",
        f"{data['ema_gold_efficiency'].iloc[0]:.2f}",
        f"{data['ema_xp_efficiency'].iloc[0]:.2f}",
    ]

    player_profile_df = pd.DataFrame(
        {
            "Stat": stats_names if not truncate else stats_names[:9],
            "Value": stats_values if not truncate else stats_values[:9],
        }
    )
    return convert_to_discord_markdown(player_profile_df)


def format_team_profile(data: pd.DataFrame, truncate: bool = False) -> str:
    """Formats the team profile data into a Discord-friendly Markdown format."""
    stats_names = [
        "Elo",
        "Glicko-2 Score",
        "Plackett-Luce Score",
        "TrueSkill Score",
        "League Elo",
        "Patch Win Rate",
        "Blue Side Win Rate",
        "Red Side Win Rate",
    ]
    stats_values = [
        f"{data['elo'].iloc[0]:.2f}",
        f"{data['gl2_mu'].iloc[0]:.2f}",
        f"{data['pl_mu'].iloc[0]:.2f}",
        f"{data['trueskill_mu'].iloc[0]:.2f}",
        f"{data['league_elo'].iloc[0]:.2f}",
        f"{data['ema_patch_win_rate'].iloc[0] * 100:.2f}%",
        f"{data['ema_blue_side'].iloc[0] * 100:.2f}%",
        f"{data['ema_red_side'].iloc[0] * 100:.2f}%",
    ]

    team_profile_df = pd.DataFrame(
        {
            "Stat": stats_names if not truncate else stats_names[:9],
            "Value": stats_values if not truncate else stats_values[:9],
        }
    )
    return convert_to_discord_markdown(team_profile_df)


def convert_to_discord_markdown(df: pd.DataFrame) -> str:
    """Converts a DataFrame to a Discord-friendly Markdown format."""
    try:
        markdown_text = df.to_markdown(index=False)
        discord_friendly_md = "\n".join(line.lstrip() for line in markdown_text.split("\n"))
        return f"```{discord_friendly_md}``` \n\n"
    except Exception as e:
        raise ValueError(f"Error converting DataFrame to Markdown: {e}")


async def get_formatted_team_profile(team_name: str, truncate: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """Retrieves and formats the team profile."""
    try:
        team_profile = get_team_data(team_name, FLATTENED_TEAMS)
        if team_profile is not None and not team_profile.empty:
            profile_md = format_team_profile(team_profile, truncate)
            return profile_md, None
        return None, f"Data for team {team_name} not found in database."
    except Exception as e:
        return None, handle_command_error(e, additional_info="Team profile retrieval failed.")


async def get_formatted_player_profile(player_name: str, truncate: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """Retrieves and formats the player profile."""
    try:
        player_profile = get_player_data(player_name, FLATTENED_PLAYERS)
        if player_profile is not None and not player_profile.empty:
            profile_md = format_player_profile(player_profile, truncate)
            return profile_md, None
        return None, f"Data for player {player_name} not found in database."
    except Exception as e:
        return None, handle_command_error(e, additional_info="Player profile retrieval failed.")


async def send_validation_result(ctx, metrics: str, images: Optional[List[File]] = None):
    """Sends the validation result to the context. Optionally sends images if provided."""
    try:
        if images:
            await ctx.send(content=metrics, files=images)
        else:
            await ctx.send(content=metrics)
    except Exception as e:
        await ctx.send(content=f"Failed to send validation result: {e}")


async def predict_and_format_result(
    ctx, blue_team_name, red_team_name, blue_roster_str, red_roster_str, match_type, account_for_side
):
    """Creates teams, predicts match outcomes, and formats the result based on the match type."""
    if match_type not in VALID_MATCH_TYPES:
        await ctx.send(content=f"Invalid match type: {match_type}. Please specify either 'bo1', 'bo3' or 'bo5'.")
        return

    message = await ctx.send(content="```Calculating win probabilities...```")

    try:
        blue_roster = process_roster(blue_roster_str) if blue_roster_str else get_empty_roster()
        red_roster = process_roster(red_roster_str) if red_roster_str else get_empty_roster()

        blue_team = Team(name=blue_team_name, side="Blue", roster=blue_roster)
        red_team = Team(name=red_team_name, side="Red", roster=red_roster)

        first_prediction = match_predictor.predict_match(blue_team, red_team, account_for_side=account_for_side)
        second_prediction = match_predictor.predict_match(red_team, blue_team, account_for_side=account_for_side)

        final_team1_win = (first_prediction[0][1] + second_prediction[0][0]) / 2
        final_team2_win = (first_prediction[0][0] + second_prediction[0][1]) / 2

        if match_type == "bo1":
            output = BestOfs.best_of_one(blue_team_name, final_team1_win, red_team_name, final_team2_win)
        elif match_type == "bo3":
            output = BestOfs.best_of_three(blue_team_name, final_team1_win, red_team_name, final_team2_win)
        elif match_type == "bo5":
            output = BestOfs.best_of_five(blue_team_name, final_team1_win, red_team_name, final_team2_win)

        await message.edit(content=output)

    except Exception as e:
        await ctx.send(content=handle_command_error(e, additional_info="Could not complete the prediction."))


def process_roster(roster_str: str, positions: Optional[List[str]] = None) -> Dict[str, str]:
    """Convert a comma-separated string of player names into a dictionary mapping each position to a player's name."""
    if positions is None:
        positions = POSITIONS
    players = [player.strip() for player in roster_str.split(",")]

    if len(players) != len(positions):
        raise ValueError(
            f"Roster does not contain the correct number of players: expected {len(positions)}, got {len(players)}."
        )

    return dict(zip(positions, players))


def get_allowed_models() -> List[str]:
    """Retrieves a list of allowed models from the configuration."""
    try:
        return list(MODEL_FILES.keys())
    except KeyError:
        raise KeyError("MODEL_FILES configuration is missing or corrupt.")


def calculate_odds(win_probability: float, to_decimal: bool) -> float:
    """Helper function to calculate odds based on win probability."""
    if to_decimal:
        return round((1 / win_probability), 2)
    return round((win_probability / (1 - win_probability)), 2)


def calculate_prob(win_probability: float) -> float:
    """Converts a decimal odds to probability format."""
    return round((1 / win_probability), 4)


def convert_odds(odds: float) -> float:
    """Converts odds to probability format for Kelly Criterion calculation."""
    if odds.endswith("%"):
        odds = float(odds.strip("%")) / 100
    else:
        odds = float(odds)
    return odds


def calculate_kelly_criterion(bookmaker_odds: float, win_probability: float) -> float:
    """Helper function to calculate the half Kelly Criterion based on win probability and bookmaker odds."""
    loss_probability = 1 - win_probability
    net_odds = bookmaker_odds - 1
    return (net_odds * win_probability - loss_probability) / net_odds / 4
