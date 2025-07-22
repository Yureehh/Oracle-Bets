"""
Side Win Rate Module

This module provides functionality to compute the side win rate for a given entity,
using an Exponentially Weighted Mean (EWM) model.
"""

import pandas as pd

from ingestion.oracles_elixir import get_opponent
from src.utils.paths import DEFAULT_MODELS_PARAMETERS
from src.utils.utils import get_identity, get_sorting_keys, json_loader

# Constants
config_params = json_loader(DEFAULT_MODELS_PARAMETERS)
HALF_LIFE = config_params["half_life"]
EPSILON = 1e-8  # Small constant to prevent division by zero


def compute_ema_side(
    df: pd.DataFrame, side: str, identity: str, half_life: float
) -> pd.DataFrame:
    """
    Compute Exponentially Weighted Mean (EWM) for a specific side ('Red' or 'Blue').

    Args:
        df (pd.DataFrame): The input DataFrame containing match data.
        side (str): The side to compute EWM for ('Red' or 'Blue').
        identity (str): The identity column to group by (e.g., 'playerid' or 'teamid').
        half_life (float): The half-life for the EWM calculation.

    Returns:
        pd.DataFrame: DataFrame with computed EWM for the specified side,
                      containing columns for "before" and "after" EWM.

    """
    # Filter rows for the given side and copy to avoid modifying original df
    side_df = df[df["side"] == side].copy()
    side_lower = side.lower()
    ema_col = f"ema_{side_lower}_side"

    # Group by identity to compute "before" and "after" EWM
    group = side_df.groupby(identity)["result"]

    # "before" uses .shift() to ensure current row's data isn't included in its own historical average
    side_df[f"{ema_col}_before"] = (
        group.transform(
            lambda x: x.ewm(halflife=half_life, adjust=False, ignore_na=True).mean()
        )
        .shift()
        .bfill()  # bfill ensures no NaN at the first record. You may consider leaving it NaN.
    )

    # "after" includes the current row (no shift)
    side_df[f"{ema_col}_after"] = group.transform(
        lambda x: x.ewm(halflife=half_life, adjust=False, ignore_na=True).mean()
    )

    return side_df


def calculate_side_win_likelihood(row: pd.Series) -> float | None:
    """
    Calculate the EMA side win likelihood for a given row.

    Args:
        row (pd.Series): A row from the DataFrame containing match data.

    Returns:
        Optional[float]: Calculated EMA side win likelihood, or None if not computable.

    """
    side = row["side"].lower()  # "red" or "blue"
    opposite_side = "red" if side == "blue" else "blue"

    # Column for the entity's side
    ema_side_col = f"ema_{side}_side_before"
    # Opponent column: "opp_ema_red_side_before" or "opp_ema_blue_side_before"
    opp_ema_side_col = f"opp_{ema_side_col.replace(side, opposite_side)}"

    ema_side = row.get(ema_side_col)
    opp_ema_side = row.get(opp_ema_side_col)

    if pd.notnull(ema_side) and pd.notnull(opp_ema_side):
        total = ema_side + opp_ema_side + EPSILON
        return round(ema_side / total, 3)
    return None


def side_win_rate_ewm_performance(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Generate an Exponentially Weighted Mean (EWM) model for side win rates.

    Args:
        df (pd.DataFrame): The input DataFrame containing match data.
        entity (str): The entity type ('player' or 'team').

    Returns:
        pd.DataFrame: A DataFrame with columns for EWM side win rates before/after,
                      opponent side EWM, and a side win likelihood score.

    """
    if entity.lower() not in {"player", "team"}:
        msg = "Entity must be either 'player' or 'team'."
        raise ValueError(msg)

    # Sort the DataFrame to ensure rows are in correct order for EWM
    df = df.sort_values(get_sorting_keys(entity)).reset_index(drop=True)
    identity = get_identity(entity)

    # --- 1) Compute EWM for both Red and Blue sides ---
    red_side_df = compute_ema_side(df, "Red", identity, HALF_LIFE)
    blue_side_df = compute_ema_side(df, "Blue", identity, HALF_LIFE)

    # Combine both side DataFrames
    combined_df = pd.concat([red_side_df, blue_side_df], ignore_index=True)
    combined_df = combined_df.sort_values(get_sorting_keys(entity)).reset_index(
        drop=True
    )

    # The columns to fill forward/backward
    ema_columns = [
        "ema_blue_side_before",
        "ema_red_side_before",
        "ema_blue_side_after",
        "ema_red_side_after",
    ]

    # Fill missing values within each group (ffill then bfill)
    combined_df[ema_columns] = (
        combined_df.groupby(identity)[ema_columns]
        .apply(lambda grp: grp.ffill().bfill())
        .reset_index(drop=True)
    )

    # --- 2) Compute opponent EMA side values (using "before" columns) ---
    for clr in ["red", "blue"]:
        ema_col = f"ema_{clr}_side_before"
        opp_ema_col = f"opp_{ema_col}"
        combined_df[opp_ema_col] = get_opponent(
            combined_df[ema_col].tolist(), entity=entity
        )

    # --- 3) Calculate side win likelihood ---
    combined_df["side_win_likelihood"] = combined_df.apply(
        calculate_side_win_likelihood, axis=1
    )

    # Final sort to maintain consistent ordering
    return combined_df.sort_values(get_sorting_keys(entity)).reset_index(drop=True)
