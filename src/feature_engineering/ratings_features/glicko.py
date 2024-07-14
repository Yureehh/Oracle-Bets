"""
Glicko-2 rating model

This module provides functionality to rate teams or players using the Glicko-2 model.
"""

import math
from typing import Dict, List, Tuple, Union

import pandas as pd
from glicko2 import Glicko2, Rating
from tqdm import tqdm

from src.utils.paths import CONSIDERED_LEAGUES, DEFAULT_MODELS_PARAMETERS, LEAGUE_ELO
from src.utils.utils import get_sorting_keys, json_loader

# Load configuration
config = json_loader(DEFAULT_MODELS_PARAMETERS)
DEFAULT_MU = config["glicko2"]["mu"]
DEFAULT_PHI = config["glicko2"]["phi"]
DEFAULT_SIGMA = config["glicko2"]["sigma"]

# Constants
WIN = 1


def initialize_ratings(
    df: pd.DataFrame, entity_key: str, model: Glicko2
) -> Dict[Union[int, str], Dict[str, Union[Rating, int, None]]]:
    """
    Initialize ratings for all entities identified by unique IDs in the DataFrame using provided Glicko-2 parameters.
    """
    unique_entities = df[entity_key].unique()
    return {
        entity: {
            "rating": model.create_rating(model.mu, model.phi, model.sigma),
            "season": df["season"].min(),
            "league": None,
        }
        for entity in unique_entities
    }


def calculate_mean_rating(ratings: List[Rating]) -> Rating:
    """
    Calculate the mean rating for a team.
    """
    mean_values = {
        "mu": sum(r.mu for r in ratings) / len(ratings),
        "phi": sum(r.phi for r in ratings) / len(ratings),
        "sigma": sum(r.sigma for r in ratings) / len(ratings),
    }
    return Rating(mean_values["mu"], mean_values["phi"], mean_values["sigma"])


def rate_match_using_mean(
    model: Glicko2, team_ratings: List[Rating], opp_mean_rating: Rating, result: int
) -> List[Rating]:
    """
    Rate each player in a team using the team's mean rating against the opposing team's mean rating.
    """
    if result == WIN:  # Team wins
        return [model.rate_1vs1(rating, opp_mean_rating)[0] for rating in team_ratings]
    else:  # Team loses
        return [model.rate_1vs1(opp_mean_rating, rating)[1] for rating in team_ratings]


def dynamic_percentage_reset_glicko2(
    ratings: Dict[str, Dict[str, Rating]], baseline_mu: float, baseline_phi: float, current_season: int
) -> None:
    """
    Apply dynamic percentage reset to Glicko-2 ratings at the beginning of a new season.
    """
    for entity, data in ratings.items():
        if data["season"] < current_season:
            rating = data["rating"]
            delta_mu = abs(rating.mu - baseline_mu)
            delta_phi = abs(rating.phi - baseline_phi)
            reset_factor_mu = 1 / (math.log2(delta_mu + 1) + 1)
            reset_factor_phi = 1 / (math.log2(delta_phi + 1) + 1)
            new_mu = baseline_mu + (rating.mu - baseline_mu) * reset_factor_mu
            new_phi = baseline_phi + (rating.phi - baseline_phi) * reset_factor_phi
            ratings[entity]["rating"] = Rating(new_mu, new_phi, rating.sigma)
            ratings[entity]["season"] = current_season


def handle_player_swap(
    player_id: Union[int, str],
    new_league: str,
    ratings: Dict[Union[int, str], Dict[str, Union[Rating, int, None]]],
    baseline_mu: float,
    baseline_phi: float,
    model: Glicko2,
) -> None:
    """Handle player swap between leagues and reset Glicko-2 rating."""
    league_elo_dict = {}
    major_leagues = json_loader(CONSIDERED_LEAGUES)["major_leagues"]
    current_league = ratings[player_id]["league"]

    # Check if LEAGUE_ELO parquet file exists
    if LEAGUE_ELO.exists():
        league_elo_df = pd.read_parquet(LEAGUE_ELO)
        league_elo_dict = league_elo_df.set_index("league")["elo"].to_dict()

    if league_elo_dict and new_league in major_leagues and current_league in major_leagues:
        current_league_elo = league_elo_dict.get(current_league, baseline_mu)
        new_league_elo = league_elo_dict.get(new_league, baseline_mu)
        elo_increment = (
            max(0, current_league_elo - new_league_elo) / 2
        )  # Ensure increment is non-negative and divide by 2
        new_mu = baseline_mu + elo_increment
    else:
        new_mu = baseline_mu

    ratings[player_id]["rating"] = model.create_rating(
        mu=new_mu, phi=baseline_phi, sigma=ratings[player_id]["rating"].sigma
    )
    ratings[player_id]["league"] = new_league


def process_game(
    df_sorted: pd.DataFrame,
    game_group: pd.DataFrame,
    ratings: Dict[Union[int, str], Dict[str, Union[Rating, int, None]]],
    model: Glicko2,
    entity: str,
    entity_key: str,
    baseline_mu: float,
    baseline_phi: float,
) -> None:
    """
    Process each game and update Glicko-2 ratings for both sides.
    """
    current_season = game_group.iloc[0]["season"]
    cross_competition_leagues = json_loader(CONSIDERED_LEAGUES)["cross_league_competitions"]
    dynamic_percentage_reset_glicko2(ratings, baseline_mu, baseline_phi, current_season)

    # Get the player IDs involved in the current game group
    player_ids = game_group[entity_key].unique()

    # Handle player swaps before processing game ratings
    for player_id in player_ids:
        player_data = ratings[player_id]
        new_league = game_group[game_group[entity_key] == player_id]["league"].iloc[0]
        if player_data["league"] != new_league and new_league not in cross_competition_leagues:
            handle_player_swap(player_id, new_league, ratings, baseline_mu, baseline_phi, model)

    blue_rows, red_rows = split_teams_by_side(game_group, entity)
    blue_ratings = [ratings[player]["rating"] for player in blue_rows[entity_key]]
    red_ratings = [ratings[player]["rating"] for player in red_rows[entity_key]]

    mean_blue_rating = calculate_mean_rating(blue_ratings)
    mean_red_rating = calculate_mean_rating(red_ratings)

    mean_blue_impact = sum(model.reduce_impact(rating) for rating in blue_ratings) / len(blue_ratings)
    mean_red_impact = sum(model.reduce_impact(rating) for rating in red_ratings) / len(red_ratings)

    result = blue_rows.iloc[0]["result"]
    win_probs = [
        model.expect_score(mean_blue_rating, mean_red_rating, mean_blue_impact),
        model.expect_score(mean_red_rating, mean_blue_rating, mean_red_impact),
    ]

    updated_blue_ratings = rate_match_using_mean(model, blue_ratings, mean_red_rating, result)
    updated_red_ratings = rate_match_using_mean(model, red_ratings, mean_blue_rating, 1 - result)
    updated_ratings = [updated_blue_ratings, updated_red_ratings]

    # Update ratings for each player
    for i, player in enumerate(blue_rows.itertuples()):
        ratings[getattr(player, entity_key)]["rating"] = updated_ratings[0][i]
        df_sorted.loc[
            player.Index,
            [
                "gl2_mu_before",
                "gl2_phi_before",
                "gl2_win_likelihood",
                "gl2_mu_after",
                "gl2_phi_after",
                "opp_gl2_mu_before",
                "opp_gl2_phi_before",
            ],
        ] = [
            blue_ratings[i].mu,
            blue_ratings[i].phi,
            win_probs[0],
            updated_ratings[0][i].mu,
            updated_ratings[0][i].phi,
            red_ratings[i].mu,
            red_ratings[i].phi,
        ]

    for i, player in enumerate(red_rows.itertuples()):
        ratings[getattr(player, entity_key)]["rating"] = updated_ratings[1][i]
        df_sorted.loc[
            player.Index,
            [
                "gl2_mu_before",
                "gl2_phi_before",
                "gl2_win_likelihood",
                "gl2_mu_after",
                "gl2_phi_after",
                "opp_gl2_mu_before",
                "opp_gl2_phi_before",
            ],
        ] = [
            red_ratings[i].mu,
            red_ratings[i].phi,
            win_probs[1],
            updated_ratings[1][i].mu,
            updated_ratings[1][i].phi,
            blue_ratings[i].mu,
            blue_ratings[i].phi,
        ]


def split_teams_by_side(game_group: pd.DataFrame, entity: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sort game data into Blue and Red teams, ordering by 'position' if the entity type is 'player'.
    """
    blue_team = (
        game_group[game_group["side"] == "Blue"].sort_values(by="position")
        if entity == "player"
        else game_group[game_group["side"] == "Blue"]
    )
    red_team = (
        game_group[game_group["side"] == "Red"].sort_values(by="position")
        if entity == "player"
        else game_group[game_group["side"] == "Red"]
    )
    return blue_team, red_team


def calculate_glicko2(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Calculate and update Glicko-2 ratings for entities within a DataFrame.
    """
    entity_key = "teamid" if entity.lower() == "team" else "playerid"
    sorted_matches_df = df.sort_values(get_sorting_keys(entity)).reset_index(drop=True)
    model = Glicko2(mu=DEFAULT_MU, phi=DEFAULT_PHI, sigma=DEFAULT_SIGMA)
    ratings = initialize_ratings(sorted_matches_df, entity_key, model)

    # Process each game once, handling both sides simultaneously
    for _, game_group in tqdm(sorted_matches_df.groupby(["date", "gameid"])):
        process_game(sorted_matches_df, game_group, ratings, model, entity, entity_key, DEFAULT_MU, DEFAULT_PHI)

    return sorted_matches_df
