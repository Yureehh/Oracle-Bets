# Assuming the necessary imports are done from your utility modules
import warnings

from utils.paths import DEFAULT_PARAMETERS
from utils.utils import get_identity, get_sorting_keys, json_loader

HALF_LIFE = json_loader(DEFAULT_PARAMETERS)["half_life"]

# TODO: improve and remove the warnings
warnings.filterwarnings("ignore")


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


#! TODO: double check for new potential columns, also try to include as many columns as possible in both ones
def select_columns_for_entity(entity):
    """Select relevant columns for EMA statistics based on the entity type."""
    base_columns = [
        "gamelength",
        "kills",
        "deaths",
        "assists",
        "kda",
        "goldat15",
        "xpat15",
        "csat15",
        "golddiffat15",
        "xpdiffat15",
        "csdiffat15",
        "egpm",
        "ckpm",
    ]

    team_extra_columns = ["firstblood", "dragons", "barons", "towers"]
    player_extra_columns = [
        "damageshare",
        "kill_participation",
        "total_cs",
        "earnedgoldshare",
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


def apply_entity_ema_std_and_growth(df, identity, columns, half_life):
    """Apply EMA calculations, standard deviation, and growth to selected columns for entities,
    maintaining distinctions between 'before' and 'after' periods."""

    for col in columns:
        group = df.groupby(identity)[col]

        # EMA calculations for 'before' and 'after' using respective halflives
        df[f"ema_{col}_before"] = group.transform(
            lambda x: x.ewm(halflife=half_life, ignore_na=True).mean().shift().bfill()
        )
        df[f"ema_{col}_after"] = group.transform(
            lambda x: x.ewm(halflife=half_life, ignore_na=True).mean()
        )

        # Standard Deviation calculations for 'before' and 'after'
        df[f"ema_{col}_std_before"] = group.transform(
            lambda x: x.ewm(halflife=half_life, ignore_na=True).std().shift().bfill()
        )
        df[f"ema_{col}_std_after"] = group.transform(
            lambda x: x.ewm(halflife=half_life, ignore_na=True).std()
        )

        # Growth calculations as the difference between the actual metric and its EMA for 'before' and 'after'
        df[f"ema_{col}_growth_before"] = df[col] - df[f"ema_{col}_before"]
        df[f"ema_{col}_growth_after"] = df[col] - df[f"ema_{col}_after"]

    return df


def enrich_entity_ema_statistics(df, entity):
    """Enrich the DataFrame with entity-specific EMA statistics and their standard deviations."""
    df.sort_values(get_sorting_keys(entity), inplace=True)
    df = elaborate_stats(df, entity)

    identity = get_identity(entity)
    columns = select_columns_for_entity(entity)

    df = apply_entity_ema_std_and_growth(df, identity, columns, HALF_LIFE)
    return df
