import numpy as np
import pandas as pd

import discord
import src.discord.match_predictor as mp
from utils.paths import DISCORD_CONFIG, FIGURES_DIR, METRICS_DIR, PROCESSED_DIR
from utils.team import Team
from utils.utils import json_loader

config = json_loader(DISCORD_CONFIG)
match_predictor = mp.MatchPredictor()


def get_empty_roster():
    return config["EMPTY_ROSTER"].copy()


def read_discord_image(image_path, filename="image.png"):
    with open(image_path, "rb") as file:
        return discord.File(file, filename=filename)


def handle_command_error(error, additional_info=""):
    """
    Template for handling command errors.
    """
    return (
        f"Something went wrong. {additional_info} If this issue persists, please contact either Yureeh or ProjektZero. Error: \n"
        f"```{error}```"
    )


def format_schedule_message(schedule_df):
    if schedule_df.empty or "league" not in schedule_df.columns:
        return "No upcoming matches found."

    message = ""
    too_long_alert = "Message too long. Please specify a narrower filter."

    for league in np.sort(schedule_df["league"].unique()):
        formatted_league = format_league(schedule_df, league)
        if len(message) + len(formatted_league) > config["MESSAGE_LIMIT"] - len(too_long_alert):
            return message + too_long_alert
        message += formatted_league

    return message or "No upcoming matches found. Double-check the league names."


def format_league(df, league):
    league_df = df[df["league"] == league].head(5)
    markdown = league_df.to_markdown(index=False)
    clean_markdown = "\n".join(line.lstrip() for line in markdown.split("\n"))
    return f"Upcoming {league} Games (Next 5 Matches Within 7 Days):\n```{clean_markdown}```\n\n"


def get_validation_metrics(model, get_graph=False):
    if model not in config["MODEL_FILES"]:
        raise ValueError(f"Model '{model}' is not supported.")

    metrics_filename, graph_filename, ha_graph_filename = config["MODEL_FILES"][model]
    metrics_file_path = METRICS_DIR / metrics_filename
    graph_file_path, ha_graph_file_path = FIGURES_DIR / graph_filename, FIGURES_DIR / ha_graph_filename

    metrics = json_loader(metrics_file_path)
    metrics = pd.DataFrame(metrics, index=[0]).round(2)
    metrics_md = convert_to_discord_markdown(metrics)

    if get_graph:
        validation_graph = read_discord_image(graph_file_path, "validation_graph.png")
        historical_accuracy_graph = read_discord_image(ha_graph_file_path, "historical_accuracy_graph.png")
        return metrics_md, [validation_graph, historical_accuracy_graph]

    return metrics_md, []


def format_prediction_message(prediction, blue_team_profile_md=None, red_team_profile_md=None):
    if prediction.empty:
        return "No prediction data available."

    # Creating a simplified DataFrame for the prediction message
    prediction_summary = pd.DataFrame(
        {
            "Team": [prediction.loc[0, "team1"], prediction.loc[0, "team2"]],
            "Win Likelihood": [
                f"{prediction.loc[0, 'team1_win_chance'] * 100:.2f}%",
                f"{prediction.loc[0, 'team2_win_chance'] * 100:.2f}%",
            ],
            "Roster": [
                list(prediction.loc[0, "team1_roster"].values()),
                list(prediction.loc[0, "team2_roster"].values()),
            ],
        }
    )

    prediction_md = f"\n**Prediction:**\n{convert_to_discord_markdown(prediction_summary)}"

    # Initialize the full message with the prediction markdown
    full_message = prediction_md

    # Append the blue team profile markdown if provided
    if blue_team_profile_md:
        full_message += f"**Blue Team Profile:**\n{blue_team_profile_md[0]}"
    # Append the red team profile markdown if provided
    if red_team_profile_md:
        full_message += f"**Red Team Profile:**\n{red_team_profile_md[0]}"

    return full_message


def load_csv_data(file_path):
    return pd.read_csv(file_path)


def get_player_data(entity_name, players_path):
    players_df = load_csv_data(players_path)
    entity_name_lower = entity_name.lower()
    filtered_players = players_df[players_df.playername.str.lower() == entity_name_lower]
    return filtered_players if not filtered_players.empty else None


def get_team_data(entity_name, teams_path):
    teams_df = load_csv_data(teams_path)
    entity_name_lower = entity_name.lower()
    filtered_teams = teams_df[teams_df.teamname.str.lower() == entity_name_lower]
    return filtered_teams if not filtered_teams.empty else None


def format_player_profile(data, truncate=False):
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
        f"{data['player_elo'].iloc[0]:.2f}",
        f"{data['player_pl_mu'].iloc[0]:.2f}",
        f"{data['player_trueskill_mu'].iloc[0]:.2f}",
        f"{data['egpm_dominance_ratio'].iloc[0]:.2f}",
        f"{data['blue_side_wr'].iloc[0] * 100:.2f}%",
        f"{data['red_side_wr'].iloc[0] * 100:.2f}%",
        f"{data['kda'].iloc[0]:.2f}",
        f"{data['killsat15'].iloc[0]:.2f} / {data['deathsat15'].iloc[0]:.2f} / {data['assistsat15'].iloc[0]:.2f}",
        f"{data['golddiffat15'].iloc[0]:.2f}",
        f"{data['csdiffat15'].iloc[0]:.2f}",
        f"{data['xpdiffat15'].iloc[0]:.2f}",
        f"{data['gamelength'].iloc[0]:.2f} mins",
        f"{data['cspm'].iloc[0]:.2f}",
        f"{data['dpm'].iloc[0]:.2f}",
        f"{data['egpm'].iloc[0]:.2f}",
        f"{data['vspm'].iloc[0]:.2f}",
        f"{data['earnedgoldshare'].iloc[0] * 100:.2f}%",
        f"{data['damageshare'].iloc[0] * 100:.2f}%",
        f"{data['gold_efficiency'].iloc[0]:.2f}",
        f"{data['xp_efficiency'].iloc[0]:.2f}",
    ]

    if truncate:
        stats_names = stats_names[:9]
        stats_values = stats_values[:9]

    player_profile_df = pd.DataFrame({"Stat": stats_names, "Value": stats_values})
    return convert_to_discord_markdown(player_profile_df)


def format_team_profile(data, truncate=False):
    stats_names = [
        "League",
        "Elo",
        "Plackett-Luce Score",
        "TrueSkill Score",
        "EGPM Dominance",
        "Blue Side Win Rate",
        "Red Side Win Rate",
        "K/D/A Ratio",
        "Team Kills",
        "Team Deaths",
        "Gold Diff At 15",
        "CS Diff At 15",
        "XP Diff At 15",
        "EGPM",
        "Avg. Game Time",
        "VoidGrubs",
        "Heralds",
        "Drakes",
        "Barons",
        "Towers",
        "TowerPlates",
    ]
    stats_values = [
        data["league"].iloc[0],
        f"{data['team_elo'].iloc[0]:.2f}",
        f"{data['team_pl_mu'].iloc[0]:.2f}",
        f"{data['team_trueskill_sum_mu'].iloc[0]:.2f}",
        f"{data['egpm_dominance_ratio'].iloc[0]:.2f}",
        f"{data['blue_side_wr'].iloc[0] * 100:.2f}%",
        f"{data['red_side_wr'].iloc[0] * 100:.2f}%",
        f"{data['kda'].iloc[0]:.2f}",
        f"{data['teamkills'].iloc[0]:.2f}",
        f"{data['teamdeaths'].iloc[0]:.2f}",
        f"{data['golddiffat15'].iloc[0]:.2f}",
        f"{data['csdiffat15'].iloc[0]:.2f}",
        f"{data['xpdiffat15'].iloc[0]:.2f}",
        f"{data['egpm'].iloc[0]:.2f}",
        f"{data['gamelength'].iloc[0]:.2f} mins",
        f"{data['void_grubs'].iloc[0]:.2f}",
        f"{data['heralds'].iloc[0]:.2f}",
        f"{data['dragons'].iloc[0]:.2f}",
        f"{data['barons'].iloc[0]:.2f}",
        f"{data['towers'].iloc[0]:.2f}",
        f"{data['turretplates'].iloc[0]:.2f}",
    ]

    if truncate:
        stats_names = stats_names[:7]
        stats_values = stats_values[:7]

    team_profile_df = pd.DataFrame({"Stat": stats_names, "Value": stats_values})
    return convert_to_discord_markdown(team_profile_df)


def format_entity_profile(data, is_player, truncate=False):
    return format_player_profile(data, truncate) if is_player else format_team_profile(data, truncate)


def convert_to_discord_markdown(df: pd.DataFrame) -> str:
    """
    Converts a DataFrame to a Discord-friendly Markdown format.

    Parameters:
        df (pd.DataFrame): The DataFrame to be converted.

    Returns:
        str: The Discord-friendly Markdown string.
    """
    # Convert the DataFrame to Markdown format
    df_md = df.to_markdown(index=False)
    df_md = "\n".join(line.lstrip() for line in df_md.split("\n"))
    return f"```{df_md}``` \n\n"


async def get_formatted_team_profile(team_name, truncate=False):
    try:
        team_profile = get_team_data(team_name, PROCESSED_DIR / "flattened_teams.csv")
        if team_profile is not None and not team_profile.empty:
            profile_md = format_team_profile(team_profile, truncate)
            return profile_md, None
        else:
            return None, f"Data for team {team_name} not found in database."
    except Exception as e:
        return None, handle_command_error(e, additional_info="Could not retrieve team information.")


async def get_formatted_player_profile(player_name, truncate=False):
    try:
        player_profile = get_player_data(player_name, PROCESSED_DIR / "flattened_players.csv")
        if player_profile is not None and not player_profile.empty:
            profile_md = format_player_profile(player_profile, truncate)
            return profile_md, None
        else:
            return None, f"Data for player {player_name} not found in database."
    except Exception as e:
        return None, handle_command_error(e, additional_info="Could not retrieve player information.")


async def send_validation_result(ctx, metrics, images=None):
    """
    Sends the validation result to the context. Optionally sends images if provided.
    """
    if images:
        await ctx.send(content=metrics, files=images)
    else:
        await ctx.send(content=metrics)


async def predict_and_format_result(ctx, blue_team_name, red_team_name, blue_roster_str, red_roster_str, match_type):
    """
    A helper function to create teams, predict match outcomes, and format the result based on the match type.
    """
    if match_type not in ["bo3", "bo5"]:
        await ctx.send(content=f"Invalid match type: {match_type}. Please specify either 'bo3' or 'bo5'.")
        return

    message = await ctx.send(content="```Calculating win probabilities...```")

    try:
        # Rosters processing
        blue_roster = process_roster(blue_roster_str) if blue_roster_str else get_empty_roster()
        red_roster = process_roster(red_roster_str) if red_roster_str else get_empty_roster()

        # Team creation
        blue_team = Team(name=blue_team_name, side="Blue", roster=blue_roster)
        red_team = Team(name=red_team_name, side="Red", roster=red_roster)

        prediction = match_predictor.predict_match(blue_team, red_team, account_for_side=False)

        if match_type == "bo3":
            output = match_predictor.best_of_three(
                blue_team_name,
                prediction.iloc[0]["team1_win_chance"],
                red_team_name,
                prediction.iloc[0]["team2_win_chance"],
            )
        elif match_type == "bo5":
            output = match_predictor.best_of_five(
                blue_team_name,
                prediction.iloc[0]["team1_win_chance"],
                red_team_name,
                prediction.iloc[0]["team2_win_chance"],
            )
        else:
            raise ValueError("Invalid match type. Please specify either 'bo3' or 'bo5'.")

        await message.edit(content=output)

    except Exception as e:
        await ctx.send(content=handle_command_error(e, additional_info="Could not complete the prediction."))


def process_roster(roster_str, positions=["top", "jng", "mid", "bot", "sup"]):
    """
    Convert a comma-separated string of player names into a dictionary with positions.
    """
    players = [player.strip() for player in roster_str.split(",")]

    if len(players) != len(positions):
        raise ValueError("Roster does not contain the correct number of players.")

    return dict(zip(positions, players))


def get_match_prediction(blue_team, red_team):
    """
    Uses the match predictor to forecast outcomes between two teams.
    """
    return match_predictor.predict_match(blue_team, red_team)


def get_allowed_models():
    return list(config["MODEL_FILES"].keys())
