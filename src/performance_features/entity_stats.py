# -*- coding: utf-8 -*-
"""
Entity statistics

This script contains functions to calculate entity-specific statistics, such as KDA, kill participation and so on.
It uses EMA (Exponential Moving Average) to calculate the statistics for 'before' and 'after' periods.
"""
import pandas as pd

from utils.paths import DEFAULT_PARAMETERS
from utils.utils import get_identity, get_sorting_keys, json_loader

HALF_LIFE = json_loader(DEFAULT_PARAMETERS)["half_life"]


def calculate_entity_kda(df):
    """Calculate the Kill-Death-Assist ratio for entities."""
    df["kda"] = (df["kills"] + df["assists"]) / df["deaths"].replace(0, 1)
    return df


def calculate_kill_participation(df):
    """Calculate the kill participation for entities."""
    team_kills = df.groupby(["gameid", "teamid"])["kills"].transform("sum")
    df["kill_participation"] = (df["kills"] + df["assists"]) / team_kills
    return df


def elaborate_stats(df, entity):
    """Elaborate the DataFrame with entity-specific statistics."""
    df = calculate_entity_kda(df)
    if entity == "player":
        df = calculate_kill_participation(df)
    return df


def select_columns_for_entity(entity):
    """Select relevant columns for EMA statistics based on the entity type."""
    base_columns = [
        "gamelength",
        "kills",
        "deaths",
        "assists",
        "kda",
        "goldat10",
        "xpat10",
        "csat10",
        "golddiffat10",
        "xpdiffat10",
        "csdiffat10",
        "goldat15",
        "xpat15",
        "csat15",
        "golddiffat15",
        "xpdiffat15",
        "csdiffat15",
        "egpm",
        "ckpm",
    ]

    team_extra_columns = [
        "firstblood",
        "dragons",
        "void_grubs",
        "heralds",
        "barons",
        "elders",
        "towers",
        "turretplates",
        "teamkills",
        "teamdeaths",
        "gspd",
        "team_kpm",
    ]
    player_extra_columns = [
        "damageshare",
        "kill_participation",
        "total_cs",
        "earnedgoldshare",
        "damagetochampions",
        "damagetakenperminute",
        "damagemitigatedperminute",
        "controlwardsbought",
        "visionscore",
        "totalgold",
        "gpr",
        "killsat15",
        "assistsat15",
        "deathsat15",
        "dpm",
        "wpm",
        "wcpm",
        "vspm",
        "cspm",
        "gold_efficiency",
        "xp_efficiency",
    ]

    if entity == "team":
        return base_columns + list(set(team_extra_columns) - set(base_columns))
    elif entity == "player":
        return base_columns + list(set(player_extra_columns) - set(base_columns))
    else:
        raise ValueError("Entity must be either team or player.")


def apply_entity_ema_std(df, identity, columns, half_life):
    """Apply EMA calculations, standard deviation to selected columns for entities,
    maintaining distinctions between 'before' and 'after' periods in an optimized manner to avoid DF fragmentation.
    """

    # Containers for new columns
    new_cols_before, new_cols_after = {}, {}

    for col in columns:
        # Grouping by identity for each column
        group = df.groupby(identity)[col]

        # Calculate EMA, Standard Deviation and Growth for 'before'
        ema_before = group.transform(lambda x: x.ewm(halflife=half_life, ignore_na=True).mean().shift().bfill())
        std_before = group.transform(lambda x: x.ewm(halflife=half_life, ignore_na=True).std().shift().bfill())

        # Store calculations in containers
        new_cols_before[f"ema_{col}_before"] = ema_before
        new_cols_before[f"ema_{col}_std_before"] = std_before

        # Calculate EMA, Standard Deviation and Growth for 'after'
        ema_after = group.transform(lambda x: x.ewm(halflife=half_life, ignore_na=True).mean())
        std_after = group.transform(lambda x: x.ewm(halflife=half_life, ignore_na=True).std())

        # Store calculations in containers
        new_cols_after[f"ema_{col}_after"] = ema_after
        new_cols_after[f"ema_{col}_std_after"] = std_after

    # Convert dictionaries to DataFrames
    new_cols_before_df = pd.DataFrame(new_cols_before)
    new_cols_after_df = pd.DataFrame(new_cols_after)

    # Concatenate the new columns with the original DataFrame to avoid fragmentation
    df = pd.concat([df, new_cols_before_df, new_cols_after_df], axis=1)

    return df


def enrich_entity_ema_statistics(df, entity):
    """Enrich the DataFrame with entity-specific EMA statistics and their standard deviations."""
    df.sort_values(get_sorting_keys(entity), inplace=True)
    df = elaborate_stats(df, entity)

    identity = get_identity(entity)
    columns = select_columns_for_entity(entity)

    df = apply_entity_ema_std(df, identity, columns, HALF_LIFE)
    return df
