import itertools
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import trueskill


def trueskill_model(
    player_data: pd.DataFrame, team_data: pd.DataFrame, initial_sigma: float = 8.33
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[Any, Any]]:
    r"""
    Calculate team ranking using Microsoft's TrueSkill 1 algorithm.
    Reference: https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/

    Parameters
    ----------
    player_data : DataFrame
        Pandas DataFrame containing Oracle's Elixir player data.
    team_data : DataFrame
        Pandas DataFrame containing Oracle's Elixir team data.

    Returns
    -------
    A Pandas dataframe containing the latest TrueSkill scores for the TEAMS
        in the league specified within the leagues parameter.
    This will be an expanded version of the team_data input.
    """
    # Initialize TrueSkill Player Ratings Dictionary
    ts = trueskill.TrueSkill(draw_probability=0.0)
    player_ratings_dict = dict()
    for i in player_data["playerid"].unique():
        player_ratings_dict[i] = ts.create_rating(sigma=initial_sigma)

    def setup_match(df: pd.DataFrame) -> np.array:
        # Prepare DataFrame
        df = df.sort_values(["date", "gameid", "side", "position"]).reset_index()
        df = (
            df[
                [
                    "playerid",
                    "playername",
                    "date",
                    "result",
                    "teamname",
                    "league",
                    "ckpm",
                    "gameid",
                    "team_egpm",
                    "team_kpm",
                    "teamid",
                ]
            ]
            .copy()
            .values
        )

        # Define Initial Variables
        matches_count = int(len(df) / 10)
        pointer = 0
        output_array = []

        for m in range(matches_count):
            # Define Index Position
            if pointer == 0:
                ind = m
            else:
                ind = m * 10

            # If you ever have to modify these, the index values correspond to the "df" from line 53.
            match_array = [
                df[ind, 7],  # gameid
                df[ind, 2],  # date
                df[ind, 5],  # league
                df[ind, 6],  # ckpm
                df[ind, 8],  # blue earned gpm
                df[ind, 9],  # blue kpm
                df[ind, 4],  # blue team name
                df[ind, 10],  # blue team id
                df[ind + 4, 0],  # blue top
                df[ind + 1, 0],  # blue jng
                df[ind + 2, 0],  # blue mid
                df[ind, 0],  # blue bot
                df[ind + 3, 0],  # blue sup
                df[ind, 3],  # blue result
                df[ind + 5, 4],  # red team name
                df[ind + 5, 10],  # red team id
                df[ind + 5, 8],  # red earned gpm
                df[ind + 5, 9],  # red kpm
                df[ind + 9, 0],  # red top
                df[ind + 6, 0],  # red jng
                df[ind + 7, 0],  # red mid
                df[ind + 5, 0],  # red bot
                df[ind + 8, 0],
            ]  # red sup
            output_array.append(match_array)
            pointer += 1

        return output_array

    col_names = [
        "gameid",
        "date",
        "league",
        "ckpm",
        "blue_earned_gpm",
        "blue_kpm",
        "blue_team",
        "blue_team_id",
        "blue_top_name",
        "blue_jng_name",
        "blue_mid_name",
        "blue_bot_name",
        "blue_sup_name",
        "blue_team_result",
        "red_team",
        "red_team_id",
        "red_earned_gpm",
        "red_kpm",
        "red_top_name",
        "red_jng_name",
        "red_mid_name",
        "red_bot_name",
        "red_sup_name",
    ]

    lcs_rating = pd.DataFrame(setup_match(input_data), columns=col_names)

    analyzed_gameids = {}

    def win_probability(team1: dict, team2: dict, trueskill_global_env) -> float:
        """
        Compute the TrueSkill probability of a team to win based on mu and sigma values.
        """
        delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
        sum_sigma = sum(r.sigma**2 for r in itertools.chain(team1, team2))
        size = len(team1) + len(team2)
        denominator = math.sqrt(size * (trueskill_global_env.beta**2) + sum_sigma)

        return trueskill_global_env.cdf(delta_mu / denominator)

    def update_trueskill(
        rating_dict,
        gameid_dict,
        gameid,
        blue_team_result,
        blue_top_name,
        blue_jng_name,
        blue_mid_name,
        blue_bot_name,
        blue_sup_name,
        red_top_name,
        red_jng_name,
        red_mid_name,
        red_bot_name,
        red_sup_name,
    ):
        """
        Compute individual changes to a player's mu and sigma values as a result of a given match.
        """
        rating_groups = [
            (
                rating_dict[blue_top_name],
                rating_dict[blue_jng_name],
                rating_dict[blue_mid_name],
                rating_dict[blue_bot_name],
                rating_dict[blue_sup_name],
            ),
            (
                rating_dict[red_top_name],
                rating_dict[red_jng_name],
                rating_dict[red_mid_name],
                rating_dict[red_bot_name],
                rating_dict[red_sup_name],
            ),
        ]
        blue_mu = rating_groups[0][0].mu
        blue_sigma = rating_groups[0][0].sigma
        red_mu = rating_groups[1][0].mu
        red_sigma = rating_groups[1][0].sigma
        blue_team_win_prob = win_probability(rating_groups[0], rating_groups[1], ts)

        # Get Mu by position
        blue_top_mu = rating_dict[blue_top_name].mu
        blue_jng_mu = rating_dict[blue_jng_name].mu
        blue_mid_mu = rating_dict[blue_mid_name].mu
        blue_bot_mu = rating_dict[blue_bot_name].mu
        blue_sup_mu = rating_dict[blue_sup_name].mu
        red_top_mu = rating_dict[red_top_name].mu
        red_jng_mu = rating_dict[red_jng_name].mu
        red_mid_mu = rating_dict[red_mid_name].mu
        red_bot_mu = rating_dict[red_bot_name].mu
        red_sup_mu = rating_dict[red_sup_name].mu

        # Get Sigma by position
        blue_top_sigma = rating_dict[blue_top_name].sigma
        blue_jng_sigma = rating_dict[blue_jng_name].sigma
        blue_mid_sigma = rating_dict[blue_mid_name].sigma
        blue_bot_sigma = rating_dict[blue_bot_name].sigma
        blue_sup_sigma = rating_dict[blue_sup_name].sigma
        red_top_sigma = rating_dict[red_top_name].sigma
        red_jng_sigma = rating_dict[red_jng_name].sigma
        red_mid_sigma = rating_dict[red_mid_name].sigma
        red_bot_sigma = rating_dict[red_bot_name].sigma
        red_sup_sigma = rating_dict[red_sup_name].sigma

        # Update ratings_features
        if blue_team_result == 1:
            # For ranks, 0 represents the winner
            rated_rating_groups = ts.rate(rating_groups, ranks=[0, 1])
        else:
            rated_rating_groups = ts.rate(rating_groups, ranks=[1, 0])

        # Return values for new columns
        ts_update = pd.Series(
            [
                blue_team_win_prob,
                blue_mu,
                blue_sigma,
                red_mu,
                red_sigma,
                blue_top_mu,
                blue_top_sigma,
                blue_jng_mu,
                blue_jng_sigma,
                blue_mid_mu,
                blue_mid_sigma,
                blue_bot_mu,
                blue_bot_sigma,
                blue_sup_mu,
                blue_sup_sigma,
                red_top_mu,
                red_top_sigma,
                red_jng_mu,
                red_jng_sigma,
                red_mid_mu,
                red_mid_sigma,
                red_bot_mu,
                red_bot_sigma,
                red_sup_mu,
                red_sup_sigma,
            ]
        )

        # Update the rating dictionary
        rating_dict[blue_top_name] = rated_rating_groups[0][0]
        rating_dict[blue_jng_name] = rated_rating_groups[0][1]
        rating_dict[blue_mid_name] = rated_rating_groups[0][2]
        rating_dict[blue_bot_name] = rated_rating_groups[0][3]
        rating_dict[blue_sup_name] = rated_rating_groups[0][4]
        rating_dict[red_top_name] = rated_rating_groups[1][0]
        rating_dict[red_jng_name] = rated_rating_groups[1][1]
        rating_dict[red_mid_name] = rated_rating_groups[1][2]
        rating_dict[red_bot_name] = rated_rating_groups[1][3]
        rating_dict[red_sup_name] = rated_rating_groups[1][4]

        # Conditional handling to prevent gameIDs from duplicative updating TS ratings_features
        if gameid in gameid_dict:
            return gameid_dict[gameid]
        else:
            gameid_dict[gameid] = ts_update
            return ts_update

    lcs_rating[
        [
            "blue_team_win_prob",
            "blue_mu",
            "blue_sigma",
            "red_mu",
            "red_sigma",
            "blue_top_mu",
            "blue_top_sigma",
            "blue_jng_mu",
            "blue_jng_sigma",
            "blue_mid_mu",
            "blue_mid_sigma",
            "blue_bot_mu",
            "blue_bot_sigma",
            "blue_sup_mu",
            "blue_sup_sigma",
            "red_top_mu",
            "red_top_sigma",
            "red_jng_mu",
            "red_jng_sigma",
            "red_mid_mu",
            "red_mid_sigma",
            "red_bot_mu",
            "red_bot_sigma",
            "red_sup_mu",
            "red_sup_sigma",
        ]
    ] = lcs_rating.apply(
        lambda row: update_trueskill(
            player_ratings_dict,
            analyzed_gameids,
            row["gameid"],
            row["blue_team_result"],
            row["blue_top_name"],
            row["blue_jng_name"],
            row["blue_mid_name"],
            row["blue_bot_name"],
            row["blue_sup_name"],
            row["red_top_name"],
            row["red_jng_name"],
            row["red_mid_name"],
            row["red_bot_name"],
            row["red_sup_name"],
        ),
        axis=1,
    )
    lcs_rating["blue_expected_result"] = np.where(
        lcs_rating["blue_team_win_prob"] >= 0.5, 1, 0
    )

    # Merge New Information Into Team Data
    blue_mu = [
        "blue_top_mu",
        "blue_jng_mu",
        "blue_mid_mu",
        "blue_bot_mu",
        "blue_sup_mu",
    ]
    blue_sigma = [
        "blue_top_sigma",
        "blue_jng_sigma",
        "blue_mid_sigma",
        "blue_bot_sigma",
        "blue_sup_sigma",
    ]
    red_mu = ["red_top_mu", "red_jng_mu", "red_mid_mu", "red_bot_mu", "red_sup_mu"]
    red_sigma = [
        "red_top_sigma",
        "red_jng_sigma",
        "red_mid_sigma",
        "red_bot_sigma",
        "red_sup_sigma",
    ]

    blue = lcs_rating[
        ["gameid", "date", "blue_team", "blue_team_id", "blue_team_win_prob"]
    ].copy()
    blue["blue_sum_mu"] = lcs_rating[blue_mu].sum(axis=1)
    blue["blue_sigma_squared"] = lcs_rating.apply(
        lambda row: sum([row[x] ** 2 for x in blue_sigma]), axis=1
    )
    blue["opponent_sum_mu"] = lcs_rating[red_mu].sum(axis=1)
    blue["opponent_sigma_squared"] = lcs_rating.apply(
        lambda row: sum([row[x] ** 2 for x in red_sigma]), axis=1
    )
    blue["trueskill_diff"] = blue["blue_team_win_prob"] - 0.50
    blue = blue.rename(
        columns={
            "blue_team": "teamname",
            "blue_team_id": "teamid",
            "blue_sum_mu": "trueskill_sum_mu",
            "blue_sigma_squared": "trueskill_sigma_squared",
            "blue_team_win_prob": "trueskill_win_perc",
        }
    )

    red = lcs_rating[["gameid", "date", "red_team", "red_team_id"]].copy()
    red["red_team_win_prob"] = 1 - lcs_rating["blue_team_win_prob"]
    red["red_sum_mu"] = lcs_rating[red_mu].sum(axis=1)
    red["red_sigma_squared"] = lcs_rating.apply(
        lambda row: sum([row[x] ** 2 for x in red_sigma]), axis=1
    )
    red["opponent_sum_mu"] = lcs_rating[blue_mu].sum(axis=1)
    red["opponent_sigma_squared"] = lcs_rating.apply(
        lambda row: sum([row[x] ** 2 for x in blue_sigma]), axis=1
    )
    red["trueskill_diff"] = red["red_team_win_prob"] - 0.50
    red = red.rename(
        columns={
            "red_team": "teamname",
            "red_team_id": "teamid",
            "red_sum_mu": "trueskill_sum_mu",
            "red_sigma_squared": "trueskill_sigma_squared",
            "red_team_win_prob": "trueskill_win_perc",
        }
    )

    # Merge Things Back Together
    team_trueskill = pd.concat([blue, red], ignore_index=True)
    team_trueskill = team_trueskill.astype({"gameid": "str"})

    team_data = team_data.astype({"gameid": "str"})
    team_data = pd.merge(
        left=team_data,
        right=team_trueskill,
        how="left",
        left_on=["gameid", "date", "teamname", "teamid"],
        right_on=["gameid", "date", "teamname", "teamid"],
    ).reset_index(drop=True)
    team_data.sort_values(by=["date", "gameid", "side"], ascending=True, inplace=True)

    # Merge New Information To Player Data
    # Blue
    blue_top = (
        lcs_rating[
            [
                "gameid",
                "date",
                "blue_team_id",
                "blue_top_name",
                "blue_top_mu",
                "blue_top_sigma",
            ]
        ]
        .copy()
        .rename(
            columns={
                "blue_team_id": "teamid",
                "blue_top_name": "playerid",
                "blue_top_mu": "trueskill_mu",
                "blue_top_sigma": "trueskill_sigma",
            }
        )
    )
    blue_jng = (
        lcs_rating[
            [
                "gameid",
                "date",
                "blue_team_id",
                "blue_jng_name",
                "blue_jng_mu",
                "blue_jng_sigma",
            ]
        ]
        .copy()
        .rename(
            columns={
                "blue_team_id": "teamid",
                "blue_jng_name": "playerid",
                "blue_jng_mu": "trueskill_mu",
                "blue_jng_sigma": "trueskill_sigma",
            }
        )
    )
    blue_mid = (
        lcs_rating[
            [
                "gameid",
                "date",
                "blue_team_id",
                "blue_mid_name",
                "blue_mid_mu",
                "blue_mid_sigma",
            ]
        ]
        .copy()
        .rename(
            columns={
                "blue_team_id": "teamid",
                "blue_mid_name": "playerid",
                "blue_mid_mu": "trueskill_mu",
                "blue_mid_sigma": "trueskill_sigma",
            }
        )
    )
    blue_bot = (
        lcs_rating[
            [
                "gameid",
                "date",
                "blue_team_id",
                "blue_bot_name",
                "blue_bot_mu",
                "blue_bot_sigma",
            ]
        ]
        .copy()
        .rename(
            columns={
                "blue_team_id": "teamid",
                "blue_bot_name": "playerid",
                "blue_bot_mu": "trueskill_mu",
                "blue_bot_sigma": "trueskill_sigma",
            }
        )
    )
    blue_sup = (
        lcs_rating[
            [
                "gameid",
                "date",
                "blue_team_id",
                "blue_sup_name",
                "blue_sup_mu",
                "blue_sup_sigma",
            ]
        ]
        .copy()
        .rename(
            columns={
                "blue_team_id": "teamid",
                "blue_sup_name": "playerid",
                "blue_sup_mu": "trueskill_mu",
                "blue_sup_sigma": "trueskill_sigma",
            }
        )
    )

    # Red
    red_top = (
        lcs_rating[
            [
                "gameid",
                "date",
                "red_team_id",
                "red_top_name",
                "red_top_mu",
                "red_top_sigma",
            ]
        ]
        .copy()
        .rename(
            columns={
                "red_team_id": "teamid",
                "red_top_name": "playerid",
                "red_top_mu": "trueskill_mu",
                "red_top_sigma": "trueskill_sigma",
            }
        )
    )
    red_jng = (
        lcs_rating[
            [
                "gameid",
                "date",
                "red_team_id",
                "red_jng_name",
                "red_jng_mu",
                "red_jng_sigma",
            ]
        ]
        .copy()
        .rename(
            columns={
                "red_team_id": "teamid",
                "red_jng_name": "playerid",
                "red_jng_mu": "trueskill_mu",
                "red_jng_sigma": "trueskill_sigma",
            }
        )
    )
    red_mid = (
        lcs_rating[
            [
                "gameid",
                "date",
                "red_team_id",
                "red_mid_name",
                "red_mid_mu",
                "red_mid_sigma",
            ]
        ]
        .copy()
        .rename(
            columns={
                "red_team_id": "teamid",
                "red_mid_name": "playerid",
                "red_mid_mu": "trueskill_mu",
                "red_mid_sigma": "trueskill_sigma",
            }
        )
    )
    red_bot = (
        lcs_rating[
            [
                "gameid",
                "date",
                "red_team_id",
                "red_bot_name",
                "red_bot_mu",
                "red_bot_sigma",
            ]
        ]
        .copy()
        .rename(
            columns={
                "red_team_id": "teamid",
                "red_bot_name": "playerid",
                "red_bot_mu": "trueskill_mu",
                "red_bot_sigma": "trueskill_sigma",
            }
        )
    )
    red_sup = (
        lcs_rating[
            [
                "gameid",
                "date",
                "red_team_id",
                "red_sup_name",
                "red_sup_mu",
                "red_sup_sigma",
            ]
        ]
        .copy()
        .rename(
            columns={
                "red_team_id": "teamid",
                "red_sup_name": "playerid",
                "red_sup_mu": "trueskill_mu",
                "red_sup_sigma": "trueskill_sigma",
            }
        )
    )

    # Concat
    player_trueskill = pd.concat(
        [
            blue_top,
            blue_jng,
            blue_mid,
            blue_bot,
            blue_sup,
            red_top,
            red_jng,
            red_mid,
            red_bot,
            red_sup,
        ],
        axis=0,
    )
    player_trueskill.sort_values(
        by=["gameid", "date", "teamid"], ascending=True, inplace=True
    )
    player_data = pd.merge(
        left=player_data,
        right=player_trueskill,
        how="left",
        left_on=["gameid", "date", "teamid", "playerid"],
        right_on=["gameid", "date", "teamid", "playerid"],
    ).reset_index(drop=True)

    player_data["opponent_mu"] = oe.get_opponent(
        player_data["trueskill_mu"].to_list(), "player"
    )
    player_data["opponent_sigma"] = oe.get_opponent(
        player_data["trueskill_sigma"].to_list(), "player"
    )
    player_data.sort_values(
        by=["date", "league", "gameid", "teamname", "position"],
        ascending=True,
        inplace=True,
    )
    player_data.reset_index(drop=True)

    return player_data, team_data, player_ratings_dict
