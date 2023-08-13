from typing import Dict, Tuple

import pandas as pd
from openskill.models import PlackettLuce


def calculate_plackett_luce(
    df: pd.DataFrame, initial_mu: float = 25.0, initial_sigma: float = (25.0 / 3.0)
) -> pd.DataFrame:
    """
    Compute Plackett-Luce ratings for each player based on the provided DataFrame.
    Reading:
    https://janzert.com/halite/rating-report/
    https://openskill.me/en/stable/manual.html

    Args:
        df (pd.DataFrame): DataFrame containing game data.
        initial_mu (float): Initial mu value to use for calculating ratings.
        initial_sigma (float): Initial sigma value for ratings (expected: 1/3 mu).

    Returns:
        Dict[str, Tuple[float, float]]: A dictionary with player IDs as keys and their
                                        (mu, sigma) ratings as values.
    """
    # Sort values for data integrity & consistency
    sort_keys = ["date", "league", "gameid", "teamid", "position", "result"]
    df = df.sort_values(sort_keys).reset_index(drop=True)

    # Create the Plackett-Luce model
    model = PlackettLuce(mu=initial_mu, sigma=initial_sigma)

    # Dictionary to store player ratings
    player_ratings = {player: model.rating() for player in df["playerid"].unique()}

    # Arrays for new columns
    mus, sigmas, pre_mus, pre_sigmas, win_likelihoods = [], [], [], [], []

    unique_game_ids = df["gameid"].unique()

    for game_id in unique_game_ids:
        game_data = df[df["gameid"] == game_id]
        blue_players = game_data[game_data["side"] == "Blue"]["playerid"].tolist()
        red_players = game_data[game_data["side"] == "Red"]["playerid"].tolist()

        # Store pre-match ratings
        for player in game_data["playerid"]:
            pre_mus.append(player_ratings[player].mu)
            pre_sigmas.append(player_ratings[player].sigma)

        team1 = [player_ratings[player] for player in blue_players]
        team2 = [player_ratings[player] for player in red_players]

        # Calculate win likelihoods for the teams based on their pre-match ratings.
        win_probs = model.predict_win([team1, team2])

        # Assign the win likelihoods to players
        for player in blue_players:
            win_likelihoods.append(win_probs[0])
        for player in red_players:
            win_likelihoods.append(win_probs[1])

        # Rate and update ratings
        updated_teams = model.rate([team1, team2], ranks=[1, 0])

        for idx, player in enumerate(blue_players):
            player_ratings[player] = updated_teams[0][idx]
            mus.append(updated_teams[0][idx].mu)
            sigmas.append(updated_teams[0][idx].sigma)
        for idx, player in enumerate(red_players):
            player_ratings[player] = updated_teams[1][idx]
            mus.append(updated_teams[1][idx].mu)
            sigmas.append(updated_teams[1][idx].sigma)

    # Add the collected values to the dataframe
    df["mu"] = mus
    df["sigma"] = sigmas
    df["pre_match_mu"] = pre_mus
    df["pre_match_sigma"] = pre_sigmas
    df["pl_win_likelihood"] = win_likelihoods

    return df
