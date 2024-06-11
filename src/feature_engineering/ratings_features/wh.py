"""
Whole History Rating System

This script calculates Whole History Ratings (WHR) for teams or players based on match results.
"""

import sys

from tqdm import tqdm
from whr import whole_history_rating

from utils.paths import DEFAULT_MODELS_PARAMETERS, WHOLE_HISTORY_RATING_PATH
from utils.utils import get_sorting_keys, json_loader

# Load configuration parameters
config = json_loader(DEFAULT_MODELS_PARAMETERS)
DEFAULT_W2 = config["whr"]["w2"]
DEFAULT_UNCASED = config["whr"]["uncased"]

DEFAULT_RATING = [0, 0, 1]  # Default rating if not available


def format_game_data(blue_row, red_row, entity_column, result):
    """
    Format data for a single game into the required string format for WHR.
    """
    blue_id = getattr(blue_row, entity_column)
    red_id = getattr(red_row, entity_column)
    game_date = blue_row.date.replace("-", "").replace(" ", "").replace(":", "")
    winner = "B" if result == 1 else "W"
    return f"{blue_id};{red_id};{winner};{game_date};0"


def get_player_rating(whr, player_id):
    """
    Retrieve the latest player rating or default if not available.
    """
    ratings = whr.ratings_for_player(player_id)
    return ratings[-1] if ratings else DEFAULT_RATING


def process_game_group(group, whr, df):
    """
    Process a group of games and update ratings.
    """
    games_batch = []
    entity_column = "teamname"
    blue_rows = group[group["side"] == "Blue"]
    red_rows = group[group["side"] == "Red"]

    for blue_row, red_row in zip(blue_rows.itertuples(), red_rows.itertuples()):
        result = blue_row.result
        blue_id = getattr(blue_row, entity_column)
        red_id = getattr(red_row, entity_column)

        games_batch.append(format_game_data(blue_row, red_row, entity_column, result))

        blue_rating = get_player_rating(whr, blue_id)
        red_rating = get_player_rating(whr, red_id)

        win_likelihood = whr.probability_future_match(blue_id, red_id)

        # Store ratings and win likelihoods in dataframe
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
    for blue_row, red_row in zip(blue_rows.itertuples(), red_rows.itertuples()):
        blue_id = getattr(blue_row, entity_column)
        red_id = getattr(red_row, entity_column)

        df.loc[blue_row.Index, rating_attributes] = [
            whr.ratings_for_player(blue_id)[-1][1],
            whr.ratings_for_player(blue_id)[-1][2],
        ]
        df.loc[red_row.Index, rating_attributes] = [
            whr.ratings_for_player(red_id)[-1][1],
            whr.ratings_for_player(red_id)[-1][2],
        ]


def calculate_whr(df, entity, w2=DEFAULT_W2, uncased=DEFAULT_UNCASED, store_model=True):
    """
    Main function to calculate WHR for a dataset.
    """
    sort_keys = get_sorting_keys(entity)
    df_sorted = df.sort_values(by=sort_keys).reset_index(drop=True)
    whr = whole_history_rating.Base({"w2": w2, "uncased": uncased, "debug": False})

    for _, game_group in tqdm(df_sorted.groupby(["date", "gameid"])):
        process_game_group(game_group, whr, df_sorted)

    whr.auto_iterate()

    if store_model:
        sys.setrecursionlimit(10000)
        whr.save_base(WHOLE_HISTORY_RATING_PATH)

    return df_sorted
