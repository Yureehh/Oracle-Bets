"""
Plackett-Luce Rating Model

This module provides functionality to rate teams or players using the Plackett-Luce model.
"""

import math
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import pandas as pd
from openskill.models import PlackettLuce
from tqdm import tqdm

from src.utils.paths import CONSIDERED_LEAGUES, DEFAULT_MODELS_PARAMETERS, LEAGUE_ELO
from src.utils.utils import get_sorting_keys, json_loader

# Load configuration
config = json_loader(DEFAULT_MODELS_PARAMETERS)
DEFAULT_MU = config["plackett_luce"]["mu"]
DEFAULT_SIGMA = config["plackett_luce"]["sigma"]


def initialize_ratings(
    df: pd.DataFrame, entity_key: str, model: PlackettLuce
) -> Dict[Union[int, str], Dict[str, Union[PlackettLuce.rating, int, str]]]:
    """
    Initialize ratings for all entities identified by unique IDs in the DataFrame using the Plackett-Luce model.

    Args:
        df (pd.DataFrame): DataFrame containing match data.
        entity_key (str): Column name representing the entity ID.
        model (PlackettLuce): Instance of the PlackettLuce model.

    Returns:
        Dict[Union[int, str], Dict[str, Union[PlackettLuce.rating, int, str]]]: Initialized ratings for all entities.
    """
    unique_entities = df[entity_key].unique()
    return {
        entity: {
            "rating": model.rating(mu=DEFAULT_MU, sigma=DEFAULT_SIGMA),
            "season": df["season"].min(),
            "league": None,
        }
        for entity in unique_entities
    }


def update_ratings(
    model: PlackettLuce,
    entities_ratings: Tuple[List[PlackettLuce.rating], List[PlackettLuce.rating]],
    ranks: List[int],
) -> List[List[PlackettLuce.rating]]:
    """
    Update ratings based on the outcomes of matches using the Plackett-Luce model.

    Args:
        model (PlackettLuce): Instance of the PlackettLuce model.
        entities_ratings (Tuple[List[PlackettLuce.rating], List[PlackettLuce.rating]]): Ratings of entities.
        ranks (List[int]): Ranks of entities.

    Returns:
        List[List[PlackettLuce.rating]]: Updated ratings for all entities.
    """
    updated_entities = model.rate([deepcopy(ratings) for ratings in entities_ratings], ranks=ranks)
    return updated_entities


def predict_win_probability(
    model: PlackettLuce, team1: List[PlackettLuce.rating], team2: List[PlackettLuce.rating]
) -> List[float]:
    """
    Predict the win probability between two teams or players based on their current ratings.

    Args:
        model (PlackettLuce): Instance of the PlackettLuce model.
        team1 (List[PlackettLuce.rating]): Ratings of team 1.
        team2 (List[PlackettLuce.rating]): Ratings of team 2.

    Returns:
        List[float]: Win probabilities for each team.
    """
    return model.predict_win([team1, team2])


def sort_teams_by_side(game_group: pd.DataFrame, entity: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sort game data into Blue and Red teams, ordering by 'position' if the entity type is 'player'.

    Args:
        game_group (pd.DataFrame): DataFrame containing match data for a single game.
        entity (str): The type of entity, e.g., 'player' or 'team'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames for Blue and Red teams.
    """
    if entity == "player":
        blue_team = game_group[game_group["side"] == "Blue"].sort_values(by="position")
        red_team = game_group[game_group["side"] == "Red"].sort_values(by="position")
    else:
        blue_team = game_group[game_group["side"] == "Blue"]
        red_team = game_group[game_group["side"] == "Red"]
    return blue_team, red_team


def determine_match_result(team_data: pd.DataFrame) -> List[int]:
    """
    Determine the result of a match based on the 'result' column in the team data.

    Args:
        team_data (pd.DataFrame): DataFrame containing match data for a single team.

    Returns:
        List[int]: Match result as ranks.
    """
    return [0, 1] if team_data.iloc[0]["result"] == 1 else [1, 0]


def dynamic_percentage_reset_plackett_luce(
    ratings: Dict[Union[int, str], Dict[str, Union[PlackettLuce.rating, int, str]]],
    baseline_mu: float,
    baseline_sigma: float,
    current_season: int,
    model: PlackettLuce,
) -> None:
    """
    Apply dynamic percentage reset to Plackett-Luce ratings at the beginning of a new season.

    Args:
        ratings (Dict[Union[int, str], Dict[str, Union[PlackettLuce.rating, int, str]]]): Current ratings of entities.
        baseline_mu (float): Baseline MU value for ratings.
        baseline_sigma (float): Baseline SIGMA value for ratings.
        current_season (int): The current season.
        model (PlackettLuce): Instance of the PlackettLuce model.
    """
    for data in ratings.values():
        if data["season"] < current_season:
            rating = data["rating"]
            delta_mu = abs(rating.mu - baseline_mu)
            delta_sigma = abs(rating.sigma - baseline_sigma)
            reset_factor_mu = 1 / (math.log2(delta_mu + 1) + 1)
            reset_factor_sigma = 1 / (math.log2(delta_sigma + 1) + 1)
            new_mu = baseline_mu + (rating.mu - baseline_mu) * reset_factor_mu
            new_sigma = baseline_sigma + (rating.sigma - baseline_sigma) * reset_factor_sigma
            data["rating"] = model.rating(mu=new_mu, sigma=new_sigma)
            data["season"] = current_season


def handle_entity_swap(
    entity_id: Union[int, str],
    new_league: str,
    ratings: Dict[Union[int, str], Dict[str, Union[PlackettLuce.rating, int, str]]],
    baseline_mu: float,
    baseline_sigma: float,
    model: PlackettLuce,
) -> None:
    """
    Handle entity swaps between leagues.

    Args:
        entity_id (Union[int, str]): ID of the entity.
        new_league (str): New league of the entity.
        ratings (Dict[Union[int, str], Dict[str, Union[PlackettLuce.rating, int, str]]]): Current ratings of entities.
        baseline_mu (float): Baseline MU value for ratings.
        baseline_sigma (float): Baseline SIGMA value for ratings.
        model (PlackettLuce): Instance of the PlackettLuce model.
    """
    major_leagues = json_loader(CONSIDERED_LEAGUES)["major_leagues"]
    cross_competition_leagues = json_loader(CONSIDERED_LEAGUES)["cross_league_competitions"]
    current_league = ratings[entity_id].get("league")

    if new_league == current_league or new_league in cross_competition_leagues:
        return  # No action needed

    league_elo_dict = {}
    if LEAGUE_ELO.exists():
        league_elo_df = pd.read_parquet(LEAGUE_ELO)
        league_elo_dict = league_elo_df.set_index("league")["elo"].to_dict()

    if league_elo_dict and new_league in major_leagues and current_league in major_leagues:
        current_league_elo = league_elo_dict.get(current_league, baseline_mu)
        new_league_elo = league_elo_dict.get(new_league, baseline_mu)
        elo_increment = max(0, current_league_elo - new_league_elo) / 2
        new_mu = baseline_mu + elo_increment
    else:
        new_mu = baseline_mu

    ratings[entity_id]["rating"] = model.rating(mu=new_mu, sigma=baseline_sigma)
    ratings[entity_id]["league"] = new_league


def process_game(
    df_sorted: pd.DataFrame,
    game_group: pd.DataFrame,
    ratings: Dict[Union[int, str], Dict[str, Union[PlackettLuce.rating, int, str]]],
    model: PlackettLuce,
    entity: str,
    entity_key: str,
    baseline_mu: float,
    baseline_sigma: float,
) -> None:
    """
    Process each game and update Plackett-Luce ratings for both sides.

    Args:
        df_sorted (pd.DataFrame): The sorted DataFrame containing match data.
        game_group (pd.DataFrame): The game group DataFrame.
        ratings (Dict[Union[int, str], Dict[str, Union[PlackettLuce.rating, int, str]]]): Dictionary of current ratings.
        model (PlackettLuce): Instance of the PlackettLuce model.
        entity (str): The type of entity, e.g., 'player' or 'team'.
        entity_key (str): The column name for the entity identifier.
        baseline_mu (float): Baseline MU value for ratings.
        baseline_sigma (float): Baseline SIGMA value for ratings.
    """
    current_season = game_group.iloc[0]["season"]
    dynamic_percentage_reset_plackett_luce(ratings, baseline_mu, baseline_sigma, current_season, model)

    # Get the entity IDs involved in the current game group
    entity_ids = game_group[entity_key].unique()

    # Handle entity swaps before processing game ratings
    for entity_id in entity_ids:
        if entity_id not in ratings:
            ratings[entity_id] = {
                "rating": model.rating(mu=baseline_mu, sigma=baseline_sigma),
                "season": current_season,
                "league": None,
            }
        new_league = game_group[game_group[entity_key] == entity_id]["league"].iloc[0]
        handle_entity_swap(entity_id, new_league, ratings, baseline_mu, baseline_sigma, model)

    blue_rows, red_rows = sort_teams_by_side(game_group, entity)
    blue_ratings = [ratings[entity]["rating"] for entity in blue_rows[entity_key]]
    red_ratings = [ratings[entity]["rating"] for entity in red_rows[entity_key]]

    ranks = determine_match_result(blue_rows)
    win_probs = predict_win_probability(model, blue_ratings, red_ratings)
    updated_ratings = update_ratings(model, (blue_ratings, red_ratings), ranks)

    for i, player in enumerate(blue_rows.itertuples()):
        entity_id = getattr(player, entity_key)
        ratings[entity_id]["rating"] = updated_ratings[0][i]
        df_sorted.loc[
            player.Index,
            [
                "pl_mu_before",
                "pl_sigma_before",
                "pl_win_likelihood",
                "pl_mu_after",
                "pl_sigma_after",
                "opp_pl_mu_before",
                "opp_pl_sigma_before",
            ],
        ] = (
            blue_ratings[i].mu,
            blue_ratings[i].sigma,
            win_probs[0],
            updated_ratings[0][i].mu,
            updated_ratings[0][i].sigma,
            red_ratings[i].mu,
            red_ratings[i].sigma,
        )

    for i, player in enumerate(red_rows.itertuples()):
        entity_id = getattr(player, entity_key)
        ratings[entity_id]["rating"] = updated_ratings[1][i]
        df_sorted.loc[
            player.Index,
            [
                "pl_mu_before",
                "pl_sigma_before",
                "pl_win_likelihood",
                "pl_mu_after",
                "pl_sigma_after",
                "opp_pl_mu_before",
                "opp_pl_sigma_before",
            ],
        ] = (
            red_ratings[i].mu,
            red_ratings[i].sigma,
            1 - win_probs[0],
            updated_ratings[1][i].mu,
            updated_ratings[1][i].sigma,
            blue_ratings[i].mu,
            blue_ratings[i].sigma,
        )


def calculate_plackett_luce(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Calculate and update Plackett-Luce ratings for entities within a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing match data.
        entity (str): The type of entity, e.g., 'team' or 'player'.

    Returns:
        pd.DataFrame: Updated DataFrame with Plackett-Luce ratings.
    """
    if entity.lower() not in ["team", "player"]:
        raise ValueError("Entity must be 'team' or 'player'")

    entity_key = "teamid" if entity.lower() == "team" else "playerid"

    required_columns = ["season", "date", "gameid", entity_key, "league", "side", "result"]
    if entity == "player":
        required_columns.append("position")

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_columns}")

    df_sorted = df.sort_values(get_sorting_keys(entity)).reset_index(drop=True)
    model = PlackettLuce(mu=DEFAULT_MU, sigma=DEFAULT_SIGMA)
    ratings = initialize_ratings(df_sorted, entity_key, model)

    # Initialize columns for Plackett-Luce ratings
    for col in ["pl_mu_before", "pl_sigma_before", "pl_win_likelihood", "pl_mu_after", "pl_sigma_after"]:
        df_sorted[col] = None

    for _, game_group in tqdm(df_sorted.groupby(["date", "gameid"]), desc="Processing games"):
        process_game(df_sorted, game_group, ratings, model, entity, entity_key, DEFAULT_MU, DEFAULT_SIGMA)

    return df_sorted
