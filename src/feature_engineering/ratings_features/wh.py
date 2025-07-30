"""
Whole History Rating System

This module calculates Whole History Ratings (WHR) for teams or players based on match results.
"""

import sys

import pandas as pd
from tqdm import tqdm
from whr import whole_history_rating

from utils.io_utils import get_sorting_keys, json_loader
from utils.paths import DEFAULT_MODELS_PARAMETERS, WHOLE_HISTORY_RATING_PATH

# Load configuration parameters
config = json_loader(DEFAULT_MODELS_PARAMETERS)
DEFAULT_W2 = config["whr"]["w2"]
DEFAULT_UNCASED = config["whr"]["uncased"]

DEFAULT_RATING = [0, 0, 1]  # Default rating if not available


def format_game_data(
    blue_id: int | str,
    red_id: int | str,
    result: int,
    game_date: str,
) -> str:
    """
    Format data for a single game into the required string format for WHR.

    Args:
        blue_id (Union[int, str]): Identifier for the blue team/player.
        red_id (Union[int, str]): Identifier for the red team/player.
        result (int): Result of the game (1 if blue team/player won, 0 otherwise).
        game_date (str): Date of the game in 'YYYYMMDDHHMMSS' format.

    Returns:
        str: Formatted game data string for WHR.

    """
    winner = "B" if result == 1 else "W"
    return f"{blue_id};{red_id};{winner};{game_date};0"


def get_player_rating(
    whr: whole_history_rating.Base, player_id: int | str
) -> list[float]:
    """
    Retrieve the latest player rating or default if not available.

    Args:
        whr (whole_history_rating.Base): WHR model instance.
        player_id (Union[int, str]): Identifier for the player.

    Returns:
        List[float]: Latest rating [time, mu, sigma] or default rating.

    """
    ratings = whr.ratings_for_player(player_id)
    return ratings[-1] if ratings else DEFAULT_RATING


def process_game_group(
    group: pd.DataFrame,
    whr: whole_history_rating.Base,
    df: pd.DataFrame,
    entity_column: str,
) -> None:
    """
    Process a group of games and update ratings.

    Args:
        group (pd.DataFrame): Group of games to process.
        whr (whole_history_rating.Base): WHR model instance.
        df (pd.DataFrame): DataFrame to update with ratings.
        entity_column (str): Column name for the entity identifier ('teamid' or 'playerid').

    """
    games_batch = []
    blue_rows = group[group["side"] == "Blue"]
    red_rows = group[group["side"] == "Red"]

    for blue_row, red_row in zip(
        blue_rows.itertuples(), red_rows.itertuples(), strict=False
    ):
        result = blue_row.result
        blue_id = blue_row.entity_column
        red_id = red_row.entity_column
        game_date = blue_row.date.strftime("%Y%m%d%H%M%S")

        game_data = format_game_data(blue_id, red_id, result, game_date)
        games_batch.append(game_data)

        blue_rating = get_player_rating(whr, blue_id)
        red_rating = get_player_rating(whr, red_id)

        win_likelihood = whr.probability_future_match(blue_id, red_id)

        # Store ratings and win likelihoods in DataFrame
        index_mapping = [
            "whr_mu_before",
            "whr_sigma_before",
            "whr_win_likelihood",
            "opp_whr_mu_before",
            "opp_whr_sigma_before",
        ]
        df.loc[blue_row.Index, index_mapping] = [
            blue_rating[1],
            blue_rating[2],
            win_likelihood[0],
            red_rating[1],
            red_rating[2],
        ]
        df.loc[red_row.Index, index_mapping] = [
            red_rating[1],
            red_rating[2],
            win_likelihood[1],
            blue_rating[1],
            blue_rating[2],
        ]

    # Process games and update ratings
    whr.load_games(games_batch, separator=";")
    whr.iterate(len(games_batch) * 2)

    rating_attributes = ["whr_mu_after", "whr_sigma_after"]
    for blue_row, red_row in zip(
        blue_rows.itertuples(), red_rows.itertuples(), strict=False
    ):
        blue_id = getattr(blue_row, entity_column)
        red_id = getattr(red_row, entity_column)

        blue_rating = get_player_rating(whr, blue_id)
        red_rating = get_player_rating(whr, red_id)

        df.loc[blue_row.Index, rating_attributes] = [
            blue_rating[1],
            blue_rating[2],
        ]
        df.loc[red_row.Index, rating_attributes] = [
            red_rating[1],
            red_rating[2],
        ]


def calculate_whr(
    df: pd.DataFrame,
    entity: str,
    w2: float = DEFAULT_W2,
    uncased: bool = DEFAULT_UNCASED,
    store_model: bool = True,
) -> pd.DataFrame:
    """
    Main function to calculate WHR for a dataset.

    Args:
        df (pd.DataFrame): DataFrame containing match data.
        entity (str): Entity type ('team' or 'player').
        w2 (float): WHR parameter controlling the weight of the prior.
        uncased (bool): Whether to treat player IDs as case-insensitive.
        store_model (bool): Whether to save the WHR model after calculation.

    Returns:
        pd.DataFrame: Updated DataFrame with WHR ratings.

    """
    if entity.lower() not in ["team", "player"]:
        msg = "Entity must be 'team' or 'player'"
        raise ValueError(msg)

    entity_column = "teamid" if entity.lower() == "team" else "playerid"
    required_columns = ["date", "gameid", "side", "result", entity_column]

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        msg = f"Input DataFrame is missing required columns: {missing_columns}"
        raise ValueError(msg)

    sort_keys = get_sorting_keys(entity)
    df_sorted = df.sort_values(by=sort_keys).reset_index(drop=True)

    # Initialize columns for WHR ratings
    rating_columns = [
        "whr_mu_before",
        "whr_sigma_before",
        "whr_win_likelihood",
        "opp_whr_mu_before",
        "opp_whr_sigma_before",
        "whr_mu_after",
        "whr_sigma_after",
    ]
    for col in rating_columns:
        df_sorted[col] = None

    whr = whole_history_rating.Base({"w2": w2, "uncased": uncased, "debug": False})

    for _, game_group in tqdm(
        df_sorted.groupby(["date", "gameid"]), desc="Processing games"
    ):
        process_game_group(game_group, whr, df_sorted, entity_column)

    whr.auto_iterate()

    if store_model:
        sys.setrecursionlimit(10000)
        whr.save_base(WHOLE_HISTORY_RATING_PATH)

    return df_sorted
