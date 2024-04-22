"""
TrueSkill rating model

This module provides functionality to rate players using the TrueSkill model.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import trueskill

from utils.paths import DEFAULT_PARAMETERS
from utils.utils import get_sorting_keys, json_loader

config = json_loader(DEFAULT_PARAMETERS)

# Default parameters for Elo rating system
DEFAULT_MU = config["trueskill"]["mu"]
DEFAULT_SIGMA = config["trueskill"]["sigma"]


def initialize_player_ratings(
    player_data: pd.DataFrame,
    initial_mu: float = DEFAULT_MU,
    initial_sigma: float = DEFAULT_SIGMA,
    ts_env: trueskill.TrueSkill = trueskill.TrueSkill(),
) -> Dict:
    """
    Initialize player ratings using the TrueSkill model.
    """
    return {
        player_id: ts_env.create_rating(mu=initial_mu, sigma=initial_sigma)
        for player_id in player_data["playerid"].unique()
    }


def preprocess_data(player_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess player data by computing 'team_egpm' and sorting the DataFrame.
    """
    required_columns = [
        "gameid",
        "date",
        "league",
        "playerid",
        "side",
        "teamname",
        "teamid",
        "position",
        "result",
        "egpm",
        "team_kpm",
        "ckpm",
    ]

    # Check for the existence of required columns to avoid runtime errors
    missing_columns = [col for col in required_columns if col not in player_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Compute 'team_egpm' without the unnecessary copy operation
    team_egpm = (
        player_data.groupby(["gameid", "teamid"], as_index=False)["egpm"].sum().rename(columns={"egpm": "team_egpm"})
    )

    # Merge the calculated 'team_egpm' back into the original DataFrame
    merged_data = player_data.merge(team_egpm, on=["gameid", "teamid"], how="left")

    # Sort the DataFrame; consider inplace sorting if the original order is not needed later
    merged_data.sort_values(by=get_sorting_keys("team"), inplace=True)

    # Select and return only the required columns, ensuring the DataFrame is tidy
    required_columns.append("team_egpm")
    return merged_data[required_columns].reset_index(drop=True)


def generate_match_array(df: pd.DataFrame) -> List:
    """
    Generate a list of match details for each game in the DataFrame.
    """
    match_arrays = []
    for _, group in df.groupby("gameid"):
        match_details = []
        # Basic match info
        match_info = group.iloc[0][["gameid", "date", "league", "ckpm"]].to_list()
        match_details.extend(match_info)
        # Team-specific details
        for side in ["Blue", "Red"]:
            team_data = group[group["side"] == side]
            team_performance = team_data.iloc[0][["team_egpm", "team_kpm"]].tolist()
            team_ids = team_data.iloc[0][["teamname", "teamid"]].tolist()
            player_ids = team_data["playerid"].tolist()
            result = team_data["result"].iloc[0]

            team_details = team_ids + team_performance + player_ids + [result]
            match_details.extend(team_details)

        match_arrays.append(match_details)

    return match_arrays


def compute_win_probability(
    team1: Tuple[trueskill.Rating],
    team2: Tuple[trueskill.Rating],
    ts_env: trueskill.TrueSkill,
) -> float:
    """
    Compute the TrueSkill probability of a team winning based on the mu and sigma values of its players.
    """
    delta_mu = sum(player.mu for player in team1) - sum(player.mu for player in team2)
    sum_sigma = sum(player.sigma**2 for player in team1 + team2)
    size = len(team1) + len(team2)
    denominator = math.sqrt(size * (ts_env.beta**2) + sum_sigma)
    return ts_env.cdf(delta_mu / denominator)


def update_player_ratings(
    match_details: pd.Series,
    rating_dict: Dict[str, trueskill.Rating],
    gameid_dict: Dict[str, pd.Series],
    ts_env: trueskill.TrueSkill,
) -> pd.Series:
    """
    Update player ratings based on match outcomes and compute additional metrics for analysis,
    returning old mu and sigma values before the update, while still updating the player ratings.
    """
    gameid = match_details["gameid"]

    # Check if game has already been processed to avoid duplicative updates
    if gameid in gameid_dict:
        return gameid_dict[gameid]

    blue_team_result = match_details["blue_result"]
    blue_player_ids = [
        match_details["blue_player1"],
        match_details["blue_player2"],
        match_details["blue_player3"],
        match_details["blue_player4"],
        match_details["blue_player5"],
    ]
    red_player_ids = [
        match_details["red_player1"],
        match_details["red_player2"],
        match_details["red_player3"],
        match_details["red_player4"],
        match_details["red_player5"],
    ]

    # Collect initial (old) mu and sigma values before the update
    old_mu_sigma_values = [rating_dict[player_id].mu for player_id in blue_player_ids + red_player_ids] + [
        rating_dict[player_id].sigma for player_id in blue_player_ids + red_player_ids
    ]

    # Create rating groups for the TrueSkill update
    blue_players = [rating_dict[id] for id in blue_player_ids]
    red_players = [rating_dict[id] for id in red_player_ids]
    rating_groups = (blue_players, red_players)

    # Determine ranks based on match result
    ranks = [0, 1] if blue_team_result == 1 else [1, 0]

    # Update ratings using TrueSkill's rate function
    rated_rating_groups = ts_env.rate(rating_groups, ranks=ranks)

    # Compute win probability for the blue team based on old ratings
    blue_team_win_prob = compute_win_probability(blue_players, red_players, ts_env)

    # Update the rating dictionary with new ratings after capturing old values
    for i, player_ids in enumerate([blue_player_ids, red_player_ids]):
        for j, player_id in enumerate(player_ids):
            rating_dict[player_id] = rated_rating_groups[i][j]

    # Prepare return values using old mu and sigma values, along with the win probability
    ts_preview = pd.Series([blue_team_win_prob] + old_mu_sigma_values)
    ts_preview = ts_preview.round(3)

    # Record the results in gameid_dict to prevent reprocessing
    gameid_dict[gameid] = ts_preview
    return ts_preview


def merge_player_stats(
    player_data: pd.DataFrame,
    match_df: pd.DataFrame,
    positions: List[str] = ["top", "jng", "mid", "bot", "sup"],
    team_colors: List[str] = ["blue", "red"],
) -> pd.DataFrame:
    """
    Merge player statistics back into the player data DataFrame.
    """
    # Initialize an empty DataFrame to store aggregated player statistics
    aggregated_player_stats = pd.DataFrame()

    for index, position in enumerate(positions, start=1):  # start=1 to match player1, player2, etc.
        for team_color in team_colors:
            opponent_color = "red" if team_color == "blue" else "blue"
            player_key = f"{team_color}_player{index}"
            mu_key = f"{team_color}_player{index}_mu"
            sigma_key = f"{team_color}_player{index}_sigma"
            opponent_mu_key = f"{opponent_color}_player{index}_mu"
            opponent_sigma_key = f"{opponent_color}_player{index}_sigma"

            # Adjust selection of relevant columns for merging
            relevant_columns = [
                "gameid",
                "date",
                f"{team_color}_teamid",
                player_key,
                mu_key,
                sigma_key,
                opponent_mu_key,
                opponent_sigma_key,
            ]

            # Adjust renaming for consistency with player_data
            player_stats = match_df[relevant_columns].rename(
                columns={
                    f"{team_color}_teamid": "teamid",
                    player_key: "playerid",
                    mu_key: "trueskill_mu",
                    sigma_key: "trueskill_sigma",
                    opponent_mu_key: "trueskill_opponent_mu",
                    opponent_sigma_key: "trueskill_opponent_sigma",
                }
            )

            # Append the player statistics to the aggregated DataFrame
            aggregated_player_stats = pd.concat([aggregated_player_stats, player_stats], ignore_index=True)

    # Merge player stats back into the player_data DataFrame
    player_data = pd.merge(
        player_data,
        aggregated_player_stats[
            [
                "gameid",
                "date",
                "teamid",
                "playerid",
                "trueskill_mu",
                "trueskill_sigma",
                "trueskill_opponent_mu",
                "trueskill_opponent_sigma",
            ]
        ],
        on=["gameid", "date", "teamid", "playerid"],
        how="left",
    )

    return player_data


def calculate_and_merge_team_statistics(
    match_df: pd.DataFrame,
    team_data: pd.DataFrame,
    team_colors: List[str] = ["blue", "red"],
) -> pd.DataFrame:
    """
    Calculate team statistics based on TrueSkill ratings and merge them back into team data.

    Parameters:
    - match_df: DataFrame containing match and player statistics.
    - team_data: DataFrame to merge the calculated team statistics into.
    - team_colors: List of team colors to differentiate teams.
    - positions: List of player positions in a team.

    Returns:
    - Updated team_data DataFrame with merged team statistics.
    """
    # Initialize an empty DataFrame to store aggregated team statistics
    aggregated_team_stats = pd.DataFrame()

    for team_color in team_colors:
        opponent_color = "red" if team_color == "blue" else "blue"

        player_mu_columns = [f"{team_color}_player{i}_mu" for i in range(1, 6)]
        player_sigma_columns = [f"{team_color}_player{i}_sigma" for i in range(1, 6)]

        opponent_mu_columns = [f"{opponent_color}_player{i}_mu" for i in range(1, 6)]
        opponent_sigma_columns = [f"{opponent_color}_player{i}_sigma" for i in range(1, 6)]

        # Calculate sum of mu, sigma squared for the team and opponent
        match_df[f"{team_color}_sum_mu"] = match_df[player_mu_columns].sum(axis=1).round(3)
        match_df[f"{team_color}_sigma_squared"] = match_df[player_sigma_columns].pow(2).sum(axis=1).round(3)
        match_df[f"{team_color}_opponent_sum_mu"] = match_df[opponent_mu_columns].sum(axis=1).round(3)
        match_df[f"{team_color}_opponent_sigma_squared"] = match_df[opponent_sigma_columns].pow(2).sum(axis=1).round(3)
        match_df[f"{team_color}_trueskill_diff"] = match_df[f"{team_color}_win_probability"] - 0.5

        # Prepare team and opponent statistics for merging
        team_statistics_cols = [
            "gameid",
            "date",
            f"{team_color}_teamname",
            f"{team_color}_teamid",
            f"{team_color}_sum_mu",
            f"{team_color}_sigma_squared",
            f"{team_color}_opponent_sum_mu",
            f"{team_color}_opponent_sigma_squared",
            f"{team_color}_trueskill_diff",
        ]

        team_statistics = match_df[team_statistics_cols].rename(
            columns={
                f"{team_color}_teamname": "teamname",
                f"{team_color}_teamid": "teamid",
                f"{team_color}_sum_mu": "trueskill_sum_mu",
                f"{team_color}_sigma_squared": "trueskill_sigma_squared",
                f"{team_color}_opponent_sum_mu": "trueskill_opponent_sum_mu",
                f"{team_color}_opponent_sigma_squared": "trueskill_opponent_sigma_squared",
                f"{team_color}_trueskill_diff": "trueskill_diff",
            }
        )

        # Append the team statistics to the aggregated DataFrame
        aggregated_team_stats = pd.concat([aggregated_team_stats, team_statistics], ignore_index=True)

    team_data = pd.merge(
        team_data,
        aggregated_team_stats,
        on=["gameid", "date", "teamname", "teamid"],
        how="left",
    ).reset_index(drop=True)

    return team_data


def trueskill_model(
    player_data: pd.DataFrame,
    team_data: pd.DataFrame,
    initial_mu: float = DEFAULT_MU,
    initial_sigma: float = DEFAULT_SIGMA,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict]:
    """
    Calculate TrueSkill ratings for players and teams based on match results.
    """
    gameid_dict = {}
    ts_env = trueskill.TrueSkill(draw_probability=0.0)

    # Initialize a default ratings dict for every player
    player_ratings_dict = initialize_player_ratings(player_data, initial_mu, initial_sigma, ts_env)

    # Preprocess the player data adding team_egpm and filtering columns
    processed_player_data = preprocess_data(player_data)

    # Generate a list with match details for each game, then convert to a DataFrame
    match_arrays = generate_match_array(processed_player_data)
    match_col_names = [
        "gameid",
        "date",
        "league",
        "ckpm",
        "blue_teamname",
        "blue_teamid",
        "team_egpm",
        "team_kpm",
        "blue_player1",
        "blue_player2",
        "blue_player3",
        "blue_player4",
        "blue_player5",
        "blue_result",
        "red_teamname",
        "red_teamid",
        "red_team_egpm",
        "red_team_kpm",
        "red_player1",
        "red_player2",
        "red_player3",
        "red_player4",
        "red_player5",
        "red_result",
    ]
    match_df = pd.DataFrame(match_arrays, columns=match_col_names)

    # Update player ratings and compute additional metrics
    true_skill_col_names = [
        "blue_win_probability",
        "blue_player1_mu",
        "blue_player2_mu",
        "blue_player3_mu",
        "blue_player4_mu",
        "blue_player5_mu",
        "red_player1_mu",
        "red_player2_mu",
        "red_player3_mu",
        "red_player4_mu",
        "red_player5_mu",
        "blue_player1_sigma",
        "blue_player2_sigma",
        "blue_player3_sigma",
        "blue_player4_sigma",
        "blue_player5_sigma",
        "red_player1_sigma",
        "red_player2_sigma",
        "red_player3_sigma",
        "red_player4_sigma",
        "red_player5_sigma",
    ]
    updates = match_df.apply(
        lambda x: update_player_ratings(x, player_ratings_dict, gameid_dict, ts_env),
        axis=1,
        result_type="expand",
    )
    updates.columns = true_skill_col_names
    match_df[true_skill_col_names] = updates

    match_df["red_win_probability"] = 1 - match_df["blue_win_probability"]
    # Add the expected result column based on the win probability
    match_df["blue_expected_result"] = np.where(match_df["blue_win_probability"] > 0.5, 1, 0)

    team_data = calculate_and_merge_team_statistics(match_df, team_data)

    # Sort the team_data DataFrame by date and gameid
    team_data.sort_values(by=get_sorting_keys("team"), ascending=True, inplace=True)

    # Apply the function to merge player statistics
    player_data = merge_player_stats(player_data, match_df)

    # Reset index and sort player_data for consistency
    player_data.reset_index(drop=True, inplace=True)

    # Sort player_data by date, league, gameid, teamid, side, and position
    player_data.sort_values(by=get_sorting_keys("player"), inplace=True)

    player_data.reset_index(drop=True, inplace=True)
    return player_data, team_data, player_ratings_dict
