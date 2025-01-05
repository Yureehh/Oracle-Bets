"""
Trueskill Rating Model

This module provides functionality to rate teams or players using the TrueSkill model.
"""

import itertools
import math
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import pandas as pd
import trueskill
from tqdm import tqdm
from trueskill import Rating, TrueSkill

from src.utils.paths import CONSIDERED_LEAGUES, DEFAULT_MODELS_PARAMETERS, LEAGUE_ELO
from src.utils.utils import get_sorting_keys, json_loader

# Load configuration
config = json_loader(DEFAULT_MODELS_PARAMETERS)
DEFAULT_MU = config["trueskill"]["mu"]
DEFAULT_SIGMA = config["trueskill"]["sigma"]
DEFAULT_BETA = DEFAULT_SIGMA / 2  # Define a constant for beta


def initialize_ratings(
    df: pd.DataFrame, entity_key: str, model: TrueSkill
) -> Dict[Union[int, str], Dict[str, Union[Rating, int, str]]]:
    """
    Initialize ratings for all entities identified by unique IDs in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the match data.
        entity_key (str): Key to identify entities (e.g., 'teamid' or 'playerid').
        model (TrueSkill): TrueSkill model instance.

    Returns:
        Dict[Union[int, str], Dict[str, Union[Rating, int, str]]]: Initialized ratings for each entity.
    """
    unique_entities = df[entity_key].unique()
    initial_mu = model.mu
    initial_sigma = model.sigma
    return {
        entity: {
            "rating": model.create_rating(mu=initial_mu, sigma=initial_sigma),
            "season": df["season"].min(),
            "league": None,
        }
        for entity in unique_entities
    }


def update_ratings(
    model: TrueSkill, entities_ratings: Tuple[List[Rating], List[Rating]], ranks: List[int]
) -> List[List[Rating]]:
    """
    Rate entities based on match outcomes and update their ratings.

    Args:
        model (TrueSkill): TrueSkill model instance.
        entities_ratings (Tuple[List[Rating], List[Rating]]): Current ratings of the entities.
        ranks (List[int]): Ranks based on match results.

    Returns:
        List[List[Rating]]: Updated ratings for the entities.
    """
    return model.rate(deepcopy(entities_ratings), ranks=ranks)


def win_probability(team1: List[Rating], team2: List[Rating], beta: float) -> float:
    """
    Calculate the win probability of team1 against team2 based on their ratings.

    Args:
        team1 (List[Rating]): Ratings of the first team.
        team2 (List[Rating]): Ratings of the second team.
        beta (float): Skill variance parameter.

    Returns:
        float: Win probability of team1 against team2.
    """
    delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
    sum_sigma = sum(r.sigma**2 for r in itertools.chain(team1, team2))
    size = len(team1) + len(team2)
    denom = math.sqrt(size * (beta**2) + sum_sigma)
    return trueskill.global_env().cdf(delta_mu / denom)


def split_teams_by_side(game_group: pd.DataFrame, entity_key: str, entity: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split game data into separate teams based on the side ('Blue' or 'Red')
    and sort by position if entity type is 'player'.

    Args:
        game_group (pd.DataFrame): DataFrame containing the match data for a game.
        entity_key (str): Key to identify entities (e.g., 'teamid' or 'playerid').
        entity (str): Type of entity, 'team' or 'player'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames for the Blue and Red teams.
    """
    if entity == "player":
        blue_team = game_group[game_group["side"] == "Blue"].sort_values(by="position")
        red_team = game_group[game_group["side"] == "Red"].sort_values(by="position")
    else:
        blue_team = game_group[game_group["side"] == "Blue"]
        red_team = game_group[game_group["side"] == "Red"]
    return blue_team, red_team


def extract_ratings(
    blue_team: pd.DataFrame,
    red_team: pd.DataFrame,
    ratings: Dict[Union[int, str], Dict[str, Union[Rating, int, str]]],
    entity_key: str,
) -> Tuple[List[Rating], List[Rating]]:
    """
    Extract ratings for players or teams from the ratings dictionary.

    Args:
        blue_team (pd.DataFrame): DataFrame for the Blue team.
        red_team (pd.DataFrame): DataFrame for the Red team.
        ratings (Dict[Union[int, str], Dict[str, Union[Rating, int, str]]]): Current ratings for all entities.
        entity_key (str): Key to identify entities (e.g., 'teamid' or 'playerid').

    Returns:
        Tuple[List[Rating], List[Rating]]: Ratings for the Blue and Red teams.
    """
    blue_ratings = [ratings[entity]["rating"] for entity in blue_team[entity_key]]
    red_ratings = [ratings[entity]["rating"] for entity in red_team[entity_key]]
    return blue_ratings, red_ratings


def determine_game_result(team_data: pd.DataFrame) -> List[int]:
    """
    Determine game results based on the 'result' column where a value of 1 indicates a win for the team.

    Args:
        team_data (pd.DataFrame): DataFrame containing the match data for a team.

    Returns:
        List[int]: List indicating the rank positions based on the match result.
    """
    return [0, 1] if team_data.iloc[0]["result"] == 1 else [1, 0]


def dynamic_percentage_reset_trueskill(
    ratings: Dict[Union[int, str], Dict[str, Union[Rating, int, str]]],
    baseline_mu: float,
    baseline_sigma: float,
    current_season: int,
) -> None:
    """
    Apply dynamic percentage reset to TrueSkill ratings at the beginning of a new season.

    Args:
        ratings (Dict[Union[int, str], Dict[str, Union[Rating, int, str]]]): Current ratings for all entities.
        baseline_mu (float): Baseline value for the rating mean.
        baseline_sigma (float): Baseline value for the rating deviation.
        current_season (int): Current season number.
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
            data["rating"] = trueskill.Rating(mu=new_mu, sigma=new_sigma)
            data["season"] = current_season


def handle_entity_swap(
    entity_id: Union[int, str],
    new_league: str,
    ratings: Dict[Union[int, str], Dict[str, Union[Rating, int, str]]],
    baseline_mu: float,
    baseline_sigma: float,
) -> None:
    """
    Handle the rating reset for entities who swap leagues.

    Args:
        entity_id (Union[int, str]): The ID of the entity.
        new_league (str): The new league of the entity.
        ratings (Dict[Union[int, str], Dict[str, Union[Rating, int, str]]]): Current ratings for all entities.
        baseline_mu (float): Baseline value for the rating mean.
        baseline_sigma (float): Baseline value for the rating deviation.
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

    ratings[entity_id]["rating"] = trueskill.Rating(mu=new_mu, sigma=baseline_sigma)
    ratings[entity_id]["league"] = new_league


def process_game(
    df_sorted: pd.DataFrame,
    game_group: pd.DataFrame,
    ratings: Dict[Union[int, str], Dict[str, Union[Rating, int, str]]],
    model: TrueSkill,
    entity: str,
    entity_key: str,
    baseline_mu: float,
    baseline_sigma: float,
) -> None:
    """
    Process each game and update TrueSkill ratings for both sides.

    Args:
        df_sorted (pd.DataFrame): Sorted DataFrame containing the match data.
        game_group (pd.DataFrame): DataFrame for a specific game group.
        ratings (Dict[Union[int, str], Dict[str, Union[Rating, int, str]]]): Current ratings for all entities.
        model (TrueSkill): TrueSkill model instance.
        entity (str): Type of entity, 'team' or 'player'.
        entity_key (str): Key to identify entities (e.g., 'teamid' or 'playerid').
        baseline_mu (float): Baseline value for the rating mean.
        baseline_sigma (float): Baseline value for the rating deviation.
    """
    current_season = game_group.iloc[0]["season"]
    dynamic_percentage_reset_trueskill(ratings, baseline_mu, baseline_sigma, current_season)

    # Get the entity IDs involved in the current game group
    entity_ids = game_group[entity_key].unique()

    # Handle entity swaps before processing game ratings
    for entity_id in entity_ids:
        if entity_id not in ratings:
            ratings[entity_id] = {
                "rating": trueskill.Rating(mu=baseline_mu, sigma=baseline_sigma),
                "season": current_season,
                "league": None,
            }
        new_league = game_group[game_group[entity_key] == entity_id]["league"].iloc[0]
        handle_entity_swap(entity_id, new_league, ratings, baseline_mu, baseline_sigma)

    blue_team, red_team = split_teams_by_side(game_group, entity_key, entity)
    blue_ratings, red_ratings = extract_ratings(blue_team, red_team, ratings, entity_key)
    ranks = determine_game_result(blue_team)
    win_prob = win_probability(blue_ratings, red_ratings, beta=model.beta)
    updated_ratings = update_ratings(model, (blue_ratings, red_ratings), ranks)

    for i, player in enumerate(blue_team.itertuples()):
        entity_id = getattr(player, entity_key)
        ratings[entity_id]["rating"] = updated_ratings[0][i]
        df_sorted.loc[
            player.Index,
            [
                "trueskill_mu_before",
                "trueskill_sigma_before",
                "trueskill_win_likelihood",
                "trueskill_mu_after",
                "trueskill_sigma_after",
                "opp_trueskill_mu_before",
                "opp_trueskill_sigma_before",
            ],
        ] = [
            blue_ratings[i].mu,
            blue_ratings[i].sigma,
            win_prob,
            updated_ratings[0][i].mu,
            updated_ratings[0][i].sigma,
            red_ratings[i].mu,
            red_ratings[i].sigma,
        ]

    for i, player in enumerate(red_team.itertuples()):
        entity_id = getattr(player, entity_key)
        ratings[entity_id]["rating"] = updated_ratings[1][i]
        df_sorted.loc[
            player.Index,
            [
                "trueskill_mu_before",
                "trueskill_sigma_before",
                "trueskill_win_likelihood",
                "trueskill_mu_after",
                "trueskill_sigma_after",
                "opp_trueskill_mu_before",
                "opp_trueskill_sigma_before",
            ],
        ] = [
            red_ratings[i].mu,
            red_ratings[i].sigma,
            1 - win_prob,
            updated_ratings[1][i].mu,
            updated_ratings[1][i].sigma,
            blue_ratings[i].mu,
            blue_ratings[i].sigma,
        ]


def calculate_trueskill(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Calculate and update TrueSkill ratings for entities within a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the match data.
        entity (str): Type of entity, 'team' or 'player'.

    Returns:
        pd.DataFrame: DataFrame with updated TrueSkill ratings.
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
    model = TrueSkill(mu=DEFAULT_MU, sigma=DEFAULT_SIGMA, draw_probability=0.0, beta=DEFAULT_BETA)
    ratings = initialize_ratings(df_sorted, entity_key, model)

    # Initialize columns for TrueSkill ratings
    for col in [
        "trueskill_mu_before",
        "trueskill_sigma_before",
        "trueskill_win_likelihood",
        "trueskill_mu_after",
        "trueskill_sigma_after",
    ]:
        df_sorted[col] = None

    for _, game_group in tqdm(df_sorted.groupby(["date", "gameid"]), desc="Processing games"):
        process_game(
            df_sorted,
            game_group,
            ratings,
            model,
            entity,
            entity_key,
            DEFAULT_MU,
            DEFAULT_SIGMA,
        )

    return df_sorted
