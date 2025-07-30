"""
Discord Utilities Module

This module contains utility functions for interacting with the Discord bot.
It includes functions for handling commands, formatting messages, and sending predictions.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import discord_predictions.match_predictor as match_predictor_module
from discord_predictions.best_ofs import BestOfs
from utils.paths import DISCORD_CONFIG, FLATTENED_PLAYERS, FLATTENED_TEAMS
from utils.team import Team
from utils.utils import json_loader, parquet_loader

# Load configuration and initialize match predictor
CONFIG = json_loader(DISCORD_CONFIG)
MATCH_PREDICTOR = match_predictor_module.MatchPredictor()

# Constants
MESSAGE_LIMIT: int = CONFIG.get("MESSAGE_LIMIT", 2000)
EMPTY_ROSTER: dict[str, str] = CONFIG["EMPTY_ROSTER"].copy()
VALID_MATCH_TYPES: list[str] = ["bo1", "bo3", "bo5"]
POSITIONS: list[str] = ["top", "jng", "mid", "bot", "sup"]
WEEKS_FOR_DELAY: int = CONFIG.get("WEEKS_FOR_DELAY", 3)
PLEASE_PROVIDE_TEAMS = "Please provide both a blue and red team name."
TEAMS_MUST_BE_DIFFERENT = "The two teams must be different."


def get_empty_roster() -> dict[str, str]:
    """
    Returns a copy of the empty roster configuration.

    Returns:
        Dict[str, str]: Empty roster dictionary.

    """
    return EMPTY_ROSTER.copy()


def handle_command_error(error: Exception, additional_info: str = "") -> str:
    """
    Formats and handles command errors.

    Args:
        error (Exception): The exception that was raised.
        additional_info (str, optional): Additional information about the error. Defaults to "".

    Returns:
        str: Formatted error message.

    """
    return (
        f"Something went wrong. {additional_info} If this issue persists, please contact either Yureeh or ProjektZero. "
        f"Error: \n```{error}```"
    )


def format_leagues_message(leagues: list[str]) -> str:
    """
    Formats a list of leagues into a Discord message.

    Args:
        leagues (List[str]): List of league names.

    Returns:
        str: Formatted message string.

    """
    if not leagues:
        return "No leagues found. Please check the league names."
    formatted_leagues = "\n".join(f"- {league}" for league in leagues)
    return f"Leagues playing in the next 7 days are:\n{formatted_leagues}"


def format_schedule_message(schedule_df: pd.DataFrame) -> str:
    """
    Formats the message displaying scheduled games from a DataFrame.

    Args:
        schedule_df (pd.DataFrame): DataFrame containing schedule information.

    Returns:
        str: Formatted schedule message.

    """
    if schedule_df.empty or "league" not in schedule_df.columns:
        return "No upcoming matches found."

    message_parts = []
    too_long_alert = "Message too long. Please specify a narrower filter."

    for league in np.sort(schedule_df["league"].unique()):
        formatted_league = format_league(schedule_df, league)
        estimated_length = len("\n".join(message_parts)) + len(formatted_league)
        if estimated_length > MESSAGE_LIMIT - len(too_long_alert):
            message_parts.append(too_long_alert)
            break
        message_parts.append(formatted_league)

    final_message = "\n".join(message_parts)
    return final_message or "No upcoming matches found. Double-check the league names."


def format_league(df: pd.DataFrame, league: str) -> str:
    """
    Formats a league section for a schedule message.

    Args:
        df (pd.DataFrame): DataFrame containing schedule information.
        league (str): The league name to format.

    Returns:
        str: Formatted league schedule string.

    """
    league_df = df[df["league"] == league].head(5).copy()
    # Modify 'league' column if needed; ensure this doesn't affect other parts
    league_df["league"] = league_df["league"].str.split().str[:3].str.join(" ")
    markdown = league_df.to_markdown(index=False)
    clean_markdown = "\n".join(line.lstrip() for line in markdown.split("\n"))
    return f"Upcoming {league} Games (Next 5 Matches Within 7 Days):\n```{clean_markdown}```\n\n"


def get_player_data(entity_name: str, players_path: Path) -> pd.DataFrame | None:
    """
    Retrieves player data from the specified file path.

    Args:
        entity_name (str): Name of the player.
        players_path (Path): Path to the players data file.

    Returns:
        Optional[pd.DataFrame]: DataFrame with player data or None if not found.

    """
    try:
        players_df = parquet_loader(players_path)
        if "playername" not in players_df.columns:
            msg = "Column 'playername' not found in players data."
            raise KeyError(msg)
        filtered_players = players_df[
            players_df["playername"].str.lower() == entity_name.lower()
        ]
        return filtered_players if not filtered_players.empty else None
    except Exception as e:
        msg = f"Error processing player data: {e}"
        raise ValueError(msg) from e


def get_team_data(entity_name: str, teams_path: Path) -> pd.DataFrame | None:
    """
    Retrieves team data from the specified file path.

    Args:
        entity_name (str): Name of the team.
        teams_path (Path): Path to the teams data file.

    Returns:
        Optional[pd.DataFrame]: DataFrame with team data or None if not found.

    """
    try:
        teams_df = parquet_loader(teams_path)
        if "teamname" not in teams_df.columns:
            msg = "Column 'teamname' not found in teams data."
            raise KeyError(msg)
        filtered_teams = teams_df[
            teams_df["teamname"].str.lower() == entity_name.lower()
        ]
        return filtered_teams if not filtered_teams.empty else None
    except Exception as e:
        msg = f"Error processing team data: {e}"
        raise ValueError(msg) from e


def format_profile(
    data: pd.DataFrame, stats_names: list[str], truncate: bool = False
) -> str:
    """
    Formats the profile data into a Discord-friendly Markdown format.

    Args:
        data (pd.DataFrame): DataFrame containing profile data.
        stats_names (List[str]): List of statistics names to include.
        truncate (bool, optional): Whether to truncate the stats. Defaults to False.

    Returns:
        str: Formatted profile string.

    """
    if truncate:
        stats_names = stats_names[:9]

    stats_values = []
    for stat in stats_names:
        value = data[stat.lower()].iloc[0] if stat.lower() in data.columns else "N/A"
        stats_values.append(f"{value}")

    profile_df = pd.DataFrame(
        {
            "Stat": stats_names,
            "Value": stats_values,
        }
    )
    return convert_to_discord_markdown(profile_df)


def format_player_profile(data: pd.DataFrame, truncate: bool = False) -> str:
    """
    Formats the player profile data into a Discord-friendly Markdown format.

    Args:
        data (pd.DataFrame): DataFrame containing player profile data.
        truncate (bool, optional): Whether to truncate the stats. Defaults to False.

    Returns:
        str: Formatted player profile string.

    """
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
        "Gold Share",
        "Damage Share",
        "Gold Efficiency",
        "XP Efficiency",
    ]
    if truncate:
        stats_names = stats_names[:8]

    stats_values = [
        data["position"].iloc[0].capitalize(),
        data["teamname"].iloc[0],
        f"{data['elo'].iloc[0]:.2f}",
        f"{data['gl2_mu'].iloc[0]:.2f}",
        f"{data['pl_mu'].iloc[0]:.2f}",
        f"{data['trueskill_mu'].iloc[0]:.2f}",
        f"{data['ema_blue_side'].iloc[0] * 100:.2f}%",
        f"{data['ema_red_side'].iloc[0] * 100:.2f}%",
        f"{data['ema_kda'].iloc[0]:.2f}",
        f"{data['ema_killsat15'].iloc[0]:.2f} / "
        f"{data['ema_deathsat15'].iloc[0]:.2f} / "
        f"{data['ema_assistsat15'].iloc[0]:.2f}",
        f"{data['ema_golddiffat15'].iloc[0]:.2f}",
        f"{data['ema_csdiffat15'].iloc[0]:.2f}",
        f"{data['ema_xpdiffat15'].iloc[0]:.2f}",
        f"{data['ema_cspm'].iloc[0]:.2f}",
        f"{data['ema_dpm'].iloc[0]:.2f}",
        f"{data['ema_egpm'].iloc[0]:.2f}",
        f"{data['ema_vspm'].iloc[0]:.2f}",
        f"{data['ema_earnedgoldshare'].iloc[0] * 100:.2f}%",
        f"{data['ema_damageshare'].iloc[0] * 100:.2f}%",
        f"{data['ema_gold_efficiency'].iloc[0]:.2f}",
        f"{data['ema_xp_efficiency'].iloc[0]:.2f}",
    ]

    if truncate:
        stats_values = stats_values[:8]

    player_profile_df = pd.DataFrame(
        {
            "Stat": stats_names,
            "Value": stats_values,
        }
    )
    return convert_to_discord_markdown(player_profile_df)


def format_team_profile(data: pd.DataFrame) -> str:
    """
    Formats the team profile data into a Discord-friendly Markdown format.

    Args:
        data (pd.DataFrame): DataFrame containing team profile data.

    Returns:
        str: Formatted team profile string.

    """
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
        f"{data['elo'].iloc[0]:.2f}",
        f"{data['gl2_mu'].iloc[0]:.2f}",
        f"{data['pl_mu'].iloc[0]:.2f}",
        f"{data['trueskill_mu'].iloc[0]:.2f}",
        f"{data['league_elo'].iloc[0]:.2f}",
        f"{data['ema_patch_win_rate'].iloc[0] * 100:.2f}%",
        f"{data['ema_season_win_rate'].iloc[0] * 100:.2f}%",
        f"{data['ema_blue_side'].iloc[0] * 100:.2f}%",
        f"{data['ema_red_side'].iloc[0] * 100:.2f}%",
        f"{data['ema_gamelength'].iloc[0]:.2f}",
    ]

    team_profile_df = pd.DataFrame(
        {
            "Stat": stats_names,
            "Value": stats_values,
        }
    )
    return convert_to_discord_markdown(team_profile_df)


def convert_to_discord_markdown(df: pd.DataFrame) -> str:
    """
    Converts a DataFrame to a Discord-friendly Markdown format.

    Args:
        df (pd.DataFrame): DataFrame to convert.

    Returns:
        str: Formatted Markdown string.

    """
    try:
        markdown_text = df.to_markdown(index=False)
        discord_friendly_md = "\n".join(
            line.lstrip() for line in markdown_text.split("\n")
        )
        return f"```{discord_friendly_md}```\n\n"
    except Exception as e:
        msg = f"Error converting DataFrame to Markdown: {e}"
        raise ValueError(msg) from e


async def get_formatted_team_profile(team_name: str) -> tuple[str | None, str | None]:
    """
    Retrieves and formats the team profile.

    Args:
        team_name (str): Name of the team.

    Returns:
        Tuple[Optional[str], Optional[str]]: Formatted team profile or error message.

    """
    try:
        team_profile = get_team_data(team_name, FLATTENED_TEAMS)
        if team_profile is not None and not team_profile.empty:
            profile_md = format_team_profile(team_profile)
            return profile_md, None
        return None, f"Data for team '{team_name}' not found in the database."
    except Exception as e:
        return None, handle_command_error(
            e, additional_info="Team profile retrieval failed."
        )


async def get_formatted_player_profile(
    player_name: str, truncate: bool = False
) -> tuple[str | None, str | None]:
    """
    Retrieves and formats the player profile.

    Args:
        player_name (str): Name of the player.
        truncate (bool, optional): Whether to truncate the stats. Defaults to False.

    Returns:
        Tuple[Optional[str], Optional[str]]: Formatted player profile or error message.

    """
    try:
        player_profile = get_player_data(player_name, FLATTENED_PLAYERS)
        if player_profile is not None and not player_profile.empty:
            profile_md = format_player_profile(player_profile, truncate)
            return profile_md, None
        return None, f"Data for player '{player_name}' not found in the database."
    except Exception as e:
        return None, handle_command_error(
            e, additional_info="Player profile retrieval failed."
        )


async def predict_and_format_result(
    ctx,
    blue_team_name: str,
    red_team_name: str,
    blue_roster_str: str | None,
    red_roster_str: str | None,
    match_type: str,
    account_for_side: bool,
) -> None:
    """
    Creates teams, predicts match outcomes, and formats the result based on the match type.

    Args:
        ctx: Discord context.
        blue_team_name (str): Name of the blue team.
        red_team_name (str): Name of the red team.
        blue_roster_str (Optional[str]): Comma-separated string of blue team players.
        red_roster_str (Optional[str]): Comma-separated string of red team players.
        match_type (str): Type of match ('bo1', 'bo3', 'bo5').
        account_for_side (bool): Whether to account for side in predictions.

    """
    if match_type not in VALID_MATCH_TYPES:
        await ctx.send(
            content=f"Invalid match type: {match_type}. Please specify either 'bo1', 'bo3', or 'bo5'."
        )
        return

    message = await ctx.send(content="```Calculating win probabilities...```")

    try:
        blue_roster = (
            process_roster(blue_roster_str) if blue_roster_str else get_empty_roster()
        )
        red_roster = (
            process_roster(red_roster_str) if red_roster_str else get_empty_roster()
        )

        blue_team = Team(name=blue_team_name, side="Blue", roster=blue_roster)
        red_team = Team(name=red_team_name, side="Red", roster=red_roster)

        today_date = pd.Timestamp.today().normalize()
        blue_last_played = pd.Timestamp(blue_team.team_stats["date"]).normalize()
        red_last_played = pd.Timestamp(red_team.team_stats["date"]).normalize()

        days_delay = WEEKS_FOR_DELAY * 7
        break_blue_flag = (today_date - blue_last_played).days >= days_delay
        break_red_flag = (today_date - red_last_played).days >= days_delay

        first_prediction = MATCH_PREDICTOR.predict_match(
            blue_team, red_team, account_for_side=account_for_side
        )
        second_prediction = MATCH_PREDICTOR.predict_match(
            red_team, blue_team, account_for_side=account_for_side
        )

        final_team1_win = (first_prediction[0][1] + second_prediction[0][0]) / 2
        final_team2_win = (first_prediction[0][0] + second_prediction[0][1]) / 2

        if match_type == "bo1":
            output = BestOfs.best_of_one(
                blue_team_name, final_team1_win, red_team_name, final_team2_win
            )
        elif match_type == "bo3":
            output = BestOfs.best_of_three(
                blue_team_name, final_team1_win, red_team_name, final_team2_win
            )
        elif match_type == "bo5":
            output = BestOfs.best_of_five(
                blue_team_name, final_team1_win, red_team_name, final_team2_win
            )

        output = add_roster_to_output(output, blue_team, red_team)
        output = add_break_flags_to_output(
            output, break_blue_flag, blue_team_name, break_red_flag, red_team_name
        )

        await message.edit(content=output)

    except Exception as e:
        await message.edit(
            content=handle_command_error(
                e, additional_info="Could not complete the prediction."
            )
        )


def add_roster_to_output(output: str, blue_team: Team, red_team: Team) -> str:
    """
    Adds roster information to the output message.

    Args:
        output (str): Current output message.
        blue_team (Team): Blue team object.
        red_team (Team): Red team object.

    Returns:
        str: Updated output message with roster information.

    """
    blue_players = "\t-\t".join(blue_team.roster.values())
    red_players = "\t-\t".join(red_team.roster.values())
    output += "\n## Found Rosters\n"
    output += f"**Blue Team:**\t {blue_players}\n"
    output += f"**Red Team:**\t {red_players}"
    return output


def add_break_flags_to_output(
    output: str,
    break_blue_flag: bool,
    blue_team_name: str,
    break_red_flag: bool,
    red_team_name: str,
) -> str:
    """
    Adds break flags to the output message if teams have not played recently.

    Args:
        output (str): Current output message.
        break_blue_flag (bool): Flag indicating if blue team has a break.
        blue_team_name (str): Name of the blue team.
        break_red_flag (bool): Flag indicating if red team has a break.
        red_team_name (str): Name of the red team.

    Returns:
        str: Updated output message with break flags.

    """
    if break_blue_flag:
        output += f"\n\nCAREFUL! {blue_team_name.capitalize()} has not played in the last {WEEKS_FOR_DELAY} weeks."
    if break_red_flag:
        output += f"\n\nCAREFUL! {red_team_name.capitalize()} has not played in the last {WEEKS_FOR_DELAY} weeks."
    return output


def process_roster(
    roster_str: str | None, positions: list[str] | None = None
) -> dict[str, str]:
    """
    Converts a comma-separated string of player names into a dictionary mapping each position to a player's name.

    Args:
        roster_str (Optional[str]): Comma-separated string of player names.
        positions (Optional[List[str]], optional): List of positions. Defaults to None.

    Raises:
        ValueError: If the number of players does not match the number of positions.

    Returns:
        Dict[str, str]: Dictionary mapping positions to player names.

    """
    if not roster_str:
        msg = "Roster string cannot be empty."
        raise ValueError(msg)
    if positions is None:
        positions = POSITIONS
    players = [player.strip() for player in roster_str.split(",")]

    if len(players) != len(positions):
        msg = f"Roster does not contain the correct number of players: expected {len(positions)}, got {len(players)}."
        raise ValueError(msg)

    return dict(zip(positions, players, strict=False))


def calculate_odds(win_probability: float, to_decimal: bool) -> float | str:
    """
    Calculates odds based on win probability.

    Args:
        win_probability (float): Probability of winning (between 0 and 1).
        to_decimal (bool): If True, converts to decimal odds; else, to fractional odds.

    Returns:
        Union[float, str]: Calculated odds or a message if odds are undefined.

    """
    if win_probability <= 0 or win_probability >= 1:
        return "Odds are undefined for win probabilities of 0% or 100%."

    if to_decimal:
        return round(1 / win_probability, 2)
    return round(win_probability / (1 - win_probability), 2)


def calculate_prob(odds: float) -> float:
    """
    Converts decimal odds to win probability.

    Args:
        odds (float): Decimal odds.

    Returns:
        float: Probability value between 0 and 1.

    """
    if odds <= 0:
        msg = "Odds must be greater than 0."
        raise ValueError(msg)
    return round(1 / odds, 4)


def convert_odds(odds: float | str) -> float:
    """
    Converts odds to a probability value.

    Args:
        odds (Union[float, str]): Odds value, either as a decimal or percentage string.

    Returns:
        float: Probability value between 0 and 1.

    """
    if isinstance(odds, str) and odds.endswith("%"):
        odds_value = float(odds.strip("%")) / 100
    else:
        odds_value = float(odds)
    return odds_value


def calculate_kelly_criterion(bookmaker_odds: float, win_probability: float) -> float:
    """
    Calculates the half Kelly Criterion based on win probability and bookmaker odds.

    Args:
        bookmaker_odds (float): Bookmaker's odds.
        win_probability (float): Probability of winning.

    Returns:
        float: Kelly Criterion value.

    """
    loss_probability = 1 - win_probability
    net_odds = bookmaker_odds - 1
    if net_odds == 0:
        return 0.0  # Avoid division by zero
    kelly = (
        (net_odds * win_probability - loss_probability) / net_odds / 2
    )  # Dividing by 2 for half Kelly
    return max(kelly, 0.0)  # Ensure non-negative


def strip_team_names(team1: str, team2: str) -> tuple[str, str]:
    """
    Strips team names of any leading or trailing whitespace.

    Args:
        team1 (str): Name of the first team.
        team2 (str): Name of the second team.

    Returns:
        Tuple[str, str]: Stripped team names.

    """
    return team1.strip(), team2.strip()


async def validate_and_predict(
    ctx,
    blue_team_name: str,
    red_team_name: str,
    blue_roster_str: str | None,
    red_roster_str: str | None,
    match_type: str,
    side_consideration: bool,
):
    """Validates team names and invokes prediction."""
    blue_team_name, red_team_name = strip_team_names(blue_team_name, red_team_name)
    if not blue_team_name or not red_team_name:
        await ctx.send(PLEASE_PROVIDE_TEAMS)
        return
    if blue_team_name == red_team_name:
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
    )
