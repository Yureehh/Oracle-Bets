"""
Entity statistics

This script contains functions to calculate entity-specific statistics, such as KDA, kill participation, and more.
It uses EMA (Exponential Moving Average) to calculate the statistics for 'before' and 'after' periods.
"""

from typing import List

import pandas as pd
from tqdm import tqdm

from src.ingestion.oracles_elixir import get_opponent
from utils.paths import DEFAULT_MODELS_PARAMETERS, FLATTENED_PLAYER_CONFIG, FLATTENED_TEAM_CONFIG
from utils.utils import get_identity, get_sorting_keys, json_loader

# Constants
config_params = json_loader(DEFAULT_MODELS_PARAMETERS)
HALF_LIFE = config_params["half_life"]
EPSILON = 1e-8  # Small constant to prevent division by zero


# Helper Functions
def select_columns_for_entity(entity: str) -> List[str]:
    """
    Select relevant columns for EMA statistics based on the entity type.

    Parameters:
        entity (str): The type of entity ('team' or 'player').

    Returns:
        List[str]: List of selected columns for EMA calculations.

    Raises:
        ValueError: If the entity type is neither 'team' nor 'player'.
    """
    if entity not in ["team", "player"]:
        raise ValueError("Entity must be either 'team' or 'player'.")

    config = FLATTENED_TEAM_CONFIG if entity == "team" else FLATTENED_PLAYER_CONFIG
    entity_cols = json_loader(config)["flattened_cols"]
    avoid_cols = [
        "ema_red_side_after",
        "ema_blue_side_after",
        "ema_patch_win_rate_after",
        "ema_season_win_rate_after",
        "patch_avg_gamelength",
    ]

    return [
        col.replace("ema_", "").replace("_after", "")
        for col in entity_cols
        if "ema" in col and "after" in col and "_std" not in col and col not in avoid_cols
    ]


def apply_ema_and_std(df: pd.DataFrame, identity: str, columns: List[str], half_life: float) -> pd.DataFrame:
    """
    Apply EMA calculations and standard deviation to selected columns for entities,
    maintaining distinctions between 'before' and 'after' periods.
    """
    # Containers for new columns
    new_cols_before, new_cols_after = {}, {}

    for col in tqdm(columns):
        group = df.groupby(identity)[col]

        # Calculate EMA and Standard Deviation for 'before'
        ema_before = group.transform(lambda x: x.ewm(halflife=half_life, ignore_na=True).mean().shift().bfill())
        std_before = group.transform(lambda x: x.ewm(halflife=half_life, ignore_na=True).std().shift().bfill())
        new_cols_before[f"ema_{col}_before"] = ema_before
        new_cols_before[f"ema_{col}_std_before"] = std_before

        # Calculate EMA and Standard Deviation for 'after'
        ema_after = group.transform(lambda x: x.ewm(halflife=half_life, ignore_na=True).mean())
        std_after = group.transform(lambda x: x.ewm(halflife=half_life, ignore_na=True).std())
        new_cols_after[f"ema_{col}_after"] = ema_after
        new_cols_after[f"ema_{col}_std_after"] = std_after

    # Concatenate the new columns with the original DataFrame
    df = pd.concat([df, pd.DataFrame(new_cols_before), pd.DataFrame(new_cols_after)], axis=1)
    return df


def apply_opponent_stats(df: pd.DataFrame, entity: str, columns: List[str]) -> pd.DataFrame:
    """
    Get the opponent entity's values for the EMA statistics calculated previously.

    Parameters:
        df (pd.DataFrame): DataFrame containing the entity data.
        entity (str): The type of entity ('team' or 'player').
        columns (List[str]): List of columns to apply opponent stats.

    Returns:
        pd.DataFrame: DataFrame with new opponent statistics columns added.
    """
    ema_cols = [item for col in columns for item in (f"ema_{col}_before", f"ema_{col}_std_before")]
    new_cols = {f"opp_{col}": get_opponent(df[col], entity=entity) for col in tqdm(ema_cols)}
    return pd.concat([df, pd.DataFrame(new_cols)], axis=1)


def enrich_entity_ema_statistics(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Enrich the DataFrame with entity-specific EMA statistics and their standard deviations.

    Parameters:
        df (pd.DataFrame): DataFrame containing the entity data.
        entity (str): The type of entity ('team' or 'player').

    Returns:
        pd.DataFrame: DataFrame with enriched EMA statistics.

    Raises:
        ValueError: If the entity type is neither 'team' nor 'player'.
    """
    if entity not in ["team", "player"]:
        raise ValueError("Entity must be either 'team' or 'player'.")

    df.sort_values(get_sorting_keys(entity), inplace=True)
    identity = get_identity(entity)
    columns = select_columns_for_entity(entity)

    df = apply_ema_and_std(df, identity, columns, HALF_LIFE)
    df = apply_opponent_stats(df, entity, columns)

    return df
