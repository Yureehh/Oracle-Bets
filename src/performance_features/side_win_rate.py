import numpy as np
import pandas as pd

import src.data_ingest.oracles_elixir as oe
from utils.paths import DEFAULT_PARAMETERS
from utils.utils import get_identity, get_sorting_keys, json_loader

HALF_LIFE = json_loader(DEFAULT_PARAMETERS)["half_life"]


def compute_ema_side(df: pd.DataFrame, side: str, identity: str) -> pd.DataFrame:
    """
    Compute Exponentially Weighted Mean (EWM) for a specific side (Red or Blue).

    Parameters
    ----------
    df : DataFrame
        DataFrame filtered by the specific side.
    side : str
        Side for which EWM is to be computed ('Red' or 'Blue').
    identity : str
        Column name to group by ('playerid' or 'teamid').

    Returns
    -------
    DataFrame
        DataFrame with EWM computed for the specified side.
    """
    side_df = df[df["side"] == side].copy()
    side_df[f"ema_{side.lower()}_side_before"] = side_df.groupby(identity)[
        "result"
    ].transform(
        lambda x: x.ewm(halflife=HALF_LIFE, ignore_na=True).mean().shift().bfill()
    )
    side_df[f"ema_{side.lower()}_side_after"] = side_df.groupby(identity)[
        "result"
    ].transform(lambda x: x.ewm(halflife=HALF_LIFE, ignore_na=True).mean())

    # Reapply a bfill to the first to replace NaNs with the first non-NaN value.
    side_df[f"ema_{side.lower()}_side_before"] = side_df.groupby(identity)[
        f"ema_{side.lower()}_side_before"
    ].transform(lambda x: x.bfill().ffill())

    return side_df


def calculate_ema_side_win_perc(row):
    side = row["side"].lower()  # Ensure the side value is lowercase for consistency.
    opposite_side = "red" if side == "blue" else "blue"  # Determine the opposite side.

    ema_side_col = f"ema_{side}_side_before"
    ema_opp_side_col = f"ema_opp_{opposite_side}_side_before"

    # Check if the required columns have non-null values.
    if pd.notnull(row[ema_side_col]) and pd.notnull(row[ema_opp_side_col]):
        ema_side_value = row[ema_side_col]
        ema_opp_side_value = row[ema_opp_side_col]
        # Calculate the EMA side win percentage with epsilon to avoid division by zero.
        ema_side_win_perc = round(
            ema_side_value / (ema_side_value + ema_opp_side_value + 1e-8), 3
        )
        return ema_side_win_perc
    else:
        return np.nan


def side_win_rate_ewm_performance(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Generate an Exponentially Weighted Mean (EWM) model for side win rates.
    """
    if entity.lower() not in ["player", "team"]:
        raise ValueError("Entity must be either 'player' or 'team'.")

    # Setup parameters
    identity = get_identity(entity)

    df.sort_values(get_sorting_keys(entity), inplace=True)

    # Compute EMA for both Red and Blue sides
    red_side = compute_ema_side(df, "Red", identity)
    blue_side = compute_ema_side(df, "Blue", identity)

    # Merge Red and Blue side DataFrames
    merged = (
        pd.concat([red_side, blue_side], ignore_index=True)
        .sort_values(get_sorting_keys(entity))
        .reset_index(drop=True)
    )

    columns = [f"ema_{color}_side_before" for color in ["blue", "red"]] + [
        f"ema_{color}_side_after" for color in ["blue", "red"]
    ]

    # Apply forward and back fill within each group for the specified EWM columns
    for column in columns:
        # This row is needed to "carry forward" the metrics of opponent side when dealing with the "other" one.
        # So basically this carries forward red side metrics when dealing with blue side and vice versa.
        merged[column] = merged.groupby([identity])[column].transform(
            lambda x: x.ffill().bfill()
        )

    merged.sort_values(get_sorting_keys(entity), inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # Compute Opponent Columns
    for color in ["red", "blue"]:
        # Here I get the last value of the opponent side EMA before the processed row.
        # It will basically use last value of the opponent EMA {opposite_color}_side_ema_before before the current row.
        merged[f"ema_opp_{color}_side_before"] = oe.get_opponent(
            merged[f"ema_{color}_side_before"].to_list(), entity
        )

    # Predict Win Probability By Side Win Rate averaging playuer side win rate and opponent side win rate
    merged["ema_side_win_perc"] = merged.apply(calculate_ema_side_win_perc, axis=1)

    return merged
