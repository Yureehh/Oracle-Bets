"""
Entity Statistics Module

This module contains functions to calculate entity-specific statistics, such as KDA, kill participation, and more.
It uses Exponential Moving Average (EMA) to calculate statistics for 'before' and 'after' periods.
"""

from typing import List

import fireducks.pandas as pd
from tqdm import tqdm

from ingestion.oracles_elixir import get_opponent
from src.utils.paths import DEFAULT_MODELS_PARAMETERS, FLATTENED_PLAYER_CONFIG, FLATTENED_TEAM_CONFIG
from src.utils.utils import get_identity, get_sorting_keys, json_loader

# Constants
config_params = json_loader(DEFAULT_MODELS_PARAMETERS)
HALF_LIFE = config_params["half_life"]
EPSILON = 1e-8  # Small constant to prevent division by zero


def select_columns_for_entity(entity: str) -> List[str]:
    """
    Select relevant columns for EMA statistics based on the entity type.

    Args:
        entity (str): The type of entity ('team' or 'player').

    Returns:
        List[str]: List of selected columns for EMA calculations.

    Raises:
        ValueError: If the entity type is neither 'team' nor 'player'.
    """
    if entity not in {"team", "player"}:
        raise ValueError("Entity must be either 'team' or 'player'.")

    config_path = FLATTENED_TEAM_CONFIG if entity == "team" else FLATTENED_PLAYER_CONFIG
    entity_cols = json_loader(config_path)["flattened_cols"]

    avoid_cols = {
        "ema_red_side_after",
        "ema_blue_side_after",
        "ema_patch_win_rate_after",
        "ema_season_win_rate_after",
        "patch_avg_gamelength",
        "total_kills",
        "total_towers",
    }

    selected_cols = [
        col.replace("ema_", "").replace("_after", "")
        for col in entity_cols
        if ("ema" in col and "after" in col and "_std" not in col and col not in avoid_cols)
    ]

    return selected_cols


def apply_ema_and_std(df: pd.DataFrame, identity: str, columns: List[str], half_life: float) -> pd.DataFrame:
    """
    Apply EMA calculations and standard deviation to selected columns for entities,
    maintaining distinctions between 'before' and 'after' periods.

    Args:
        df (pd.DataFrame): The input DataFrame.
        identity (str): The identity column to group by.
        columns (List[str]): List of columns to calculate EMA and STD for.
        half_life (float): The half-life parameter for EMA calculations.

    Returns:
        pd.DataFrame: DataFrame with new EMA and STD columns added, ensuring no duplicates.
    """
    # Initialize dictionaries to collect new columns
    ema_before_cols = {}
    std_before_cols = {}
    ema_after_cols = {}
    std_after_cols = {}

    for col in tqdm(columns, desc="Calculating EMA and STD"):
        group = df.groupby(identity)[col]

        # Calculate EMA and Standard Deviation for 'before'
        ema_before = group.transform(lambda x: x.ewm(halflife=half_life, ignore_na=True).mean()).shift().bfill()
        std_before = group.transform(lambda x: x.ewm(halflife=half_life, ignore_na=True).std()).shift().bfill()
        ema_before_cols[f"ema_{col}_before"] = ema_before
        std_before_cols[f"ema_{col}_std_before"] = std_before

        # Calculate EMA and Standard Deviation for 'after'
        ema_after = group.transform(lambda x: x.ewm(halflife=half_life, ignore_na=True).mean())
        std_after = group.transform(lambda x: x.ewm(halflife=half_life, ignore_na=True).std())
        ema_after_cols[f"ema_{col}_after"] = ema_after
        std_after_cols[f"ema_{col}_std_after"] = std_after

    # Combine all new columns into a DataFrame
    new_columns = pd.DataFrame({**ema_before_cols, **std_before_cols, **ema_after_cols, **std_after_cols})

    # Concatenate new columns to the original DataFrame
    df = pd.concat([df, new_columns], axis=1)

    return df


def apply_opponent_stats(df: pd.DataFrame, entity: str, columns: List[str]) -> pd.DataFrame:
    """
    Get the opponent entity's values for the EMA statistics calculated previously.

    Args:
        df (pd.DataFrame): DataFrame containing the entity data.
        entity (str): The type of entity ('team' or 'player').
        columns (List[str]): List of columns to apply opponent stats.

    Returns:
        pd.DataFrame: DataFrame with new opponent statistics columns added.
    """
    ema_cols = [f"ema_{col}_before" for col in columns] + [f"ema_{col}_std_before" for col in columns]

    # Create a dictionary to store opponent stats
    new_cols = {}
    for col in tqdm(ema_cols, desc="Calculating opponent stats"):
        new_cols[f"opp_{col}"] = get_opponent(df[col], entity=entity)

    # Concatenate all new columns at once to avoid fragmentation
    new_cols_df = pd.DataFrame(new_cols, index=df.index)

    # Use pd.concat to combine the original DataFrame with the new columns
    df = pd.concat([df, new_cols_df], axis=1)

    return df


def enrich_entity_ema_statistics(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Enrich the DataFrame with entity-specific EMA statistics and their standard deviations.

    Args:
        df (pd.DataFrame): DataFrame containing the entity data.
        entity (str): The type of entity ('team' or 'player').

    Returns:
        pd.DataFrame: DataFrame with enriched EMA statistics.

    Raises:
        ValueError: If the entity type is neither 'team' nor 'player'.
    """
    if entity not in {"team", "player"}:
        raise ValueError("Entity must be either 'team' or 'player'.")

    df = df.sort_values(get_sorting_keys(entity)).reset_index(drop=True)
    identity = get_identity(entity)
    columns = select_columns_for_entity(entity)

    df = apply_ema_and_std(df, identity, columns, HALF_LIFE)
    df = apply_opponent_stats(df, entity, columns)

    return df
