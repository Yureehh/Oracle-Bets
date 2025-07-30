"""
Entity Statistics Module

This module contains functions to calculate entity-specific statistics, such as KDA, kill participation, and more.
It uses Exponential Moving Average (EMA) to calculate statistics for 'before' and 'after' periods.
"""

import pandas as pd
from tqdm import tqdm

from ingestion.oracles_elixir import get_opponent
from utils.io_utils import get_identity, get_sorting_keys, json_loader
from utils.paths import (
    DEFAULT_MODELS_PARAMETERS,
    FLATTENED_PLAYER_CONFIG,
    FLATTENED_TEAM_CONFIG,
)

# Constants
config_params = json_loader(DEFAULT_MODELS_PARAMETERS)
HALF_LIFE = config_params["half_life"]


def select_columns_for_entity(entity: str) -> list[str]:
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
        msg = "Entity must be either 'team' or 'player'."
        raise ValueError(msg)

    config_path = FLATTENED_TEAM_CONFIG if entity == "team" else FLATTENED_PLAYER_CONFIG
    entity_cols = json_loader(config_path)["flattened_cols"]

    # TODO: Here i have to escluse all the columns that are target features
    # or that i compute with other performance features scripts.
    # Once i finish revamping the code i should double check  the columns to esclude
    avoid_cols = {
        # Side wr cols
        "ema_red_side_after",
        "ema_blue_side_after",
        # Patch and season wr cols
        "ema_patch_win_rate_after",
        "ema_season_win_rate_after",
        # Target feature cols
        "season_avg_gamelength",
        "patch_avg_gamelength",
        "team_season_avg_gamelength",
        "team_patch_avg_gamelength",
        "team_win_avg_season_gamelength",
        "team_lose_avg_season_gamelength",
        "team_win_avg_patch_gamelength",
        "team_lose_avg_patch_gamelength",
        "total_kills",
        "total_towers",
    }

    # Only select columns with "ema" and "after", but skip any in avoid_cols
    return [
        col.replace("ema_", "").replace("_after", "")
        for col in entity_cols
        if ("ema" in col and "after" in col and col not in avoid_cols)
    ]


def apply_ema(
    df: pd.DataFrame, identity: str, columns: list[str], half_life: float
) -> pd.DataFrame:
    """
    Apply EMA calculations to selected columns for entities,
    maintaining distinctions between 'before' and 'after' periods.

    Args:
        df (pd.DataFrame): The input DataFrame.
        identity (str): The identity column to group by.
        columns (List[str]): List of columns to calculate EMA for.
        half_life (float): The half-life parameter for EMA calculations.

    Returns:
        pd.DataFrame: DataFrame with new EMA columns added, ensuring no duplicates.

    """
    # Initialize dictionaries to collect new columns
    ema_before_cols = {}
    ema_after_cols = {}

    for col in tqdm(columns, desc="Calculating EMA"):
        group = df.groupby(identity)[col]

        # Calculate 'before' EMA: shift to avoid data leakage into current row
        ema_before = (
            group.transform(lambda x: x.ewm(halflife=half_life, ignore_na=True).mean())
            .shift()
            .bfill()
        )
        ema_before_cols[f"ema_{col}_before"] = ema_before

        # Calculate 'after' EMA: no shift, as it represents the updated stats
        ema_after = group.transform(
            lambda x: x.ewm(halflife=half_life, ignore_na=True).mean()
        )
        ema_after_cols[f"ema_{col}_after"] = ema_after

    # Combine all new columns into a DataFrame
    new_columns = pd.DataFrame({**ema_before_cols, **ema_after_cols})

    # Concatenate new columns to the original DataFrame
    return pd.concat([df, new_columns], axis=1)


def apply_opponent_stats(
    df: pd.DataFrame, entity: str, columns: list[str]
) -> pd.DataFrame:
    """
    Get the opponent entity's values for the EMA statistics calculated previously.
    Currently, only 'before' columns are used to fetch opponent stats.

    Args:
        df (pd.DataFrame): DataFrame containing the entity data.
        entity (str): The type of entity ('team' or 'player').
        columns (List[str]): List of columns to apply opponent stats.

    Returns:
        pd.DataFrame: DataFrame with new opponent statistics columns added.

    """
    # Adjust if you also need "after" columns for opponent stats
    ema_cols = [f"ema_{col}_before" for col in columns]

    new_cols = {}
    for col in tqdm(ema_cols, desc="Calculating opponent stats"):
        new_cols[f"opp_{col}"] = get_opponent(df[col], entity=entity)

    # Concatenate all new columns at once to avoid fragmentation
    new_cols_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_cols_df], axis=1)


def enrich_entity_ema_statistics(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Enrich the DataFrame with entity-specific EMA statistics.

    Args:
        df (pd.DataFrame): DataFrame containing the entity data.
        entity (str): The type of entity ('team' or 'player').

    Returns:
        pd.DataFrame: DataFrame with enriched EMA statistics.

    Raises:
        ValueError: If the entity type is neither 'team' nor 'player'.

    """
    if entity not in {"team", "player"}:
        msg = "Entity must be either 'team' or 'player'."
        raise ValueError(msg)

    # Sort and reset index to ensure consistent row ordering
    df = df.sort_values(get_sorting_keys(entity)).reset_index(drop=True)
    identity = get_identity(entity)

    # Select columns to apply EMA
    columns = select_columns_for_entity(entity)

    # Apply EMA calculations
    df = apply_ema(df, identity, columns, HALF_LIFE)

    # Apply opponent stats to the 'before' columns
    return apply_opponent_stats(df, entity, columns)
