"""Shared game-level feature construction for LoL prop models."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from oracle_bets_core.pd import pd
from pandas.api.types import is_numeric_dtype

PROP_TARGETS: tuple[str, ...] = ("gamelength", "total_kills", "total_towers")
SIDE_ORDER: tuple[str, str] = ("Blue", "Red")
SIDE_PREFIX: dict[str, str] = {"Blue": "blue", "Red": "red"}
TARGET_TOLERANCE = 1e-9


def is_prop_target(target_col: str) -> bool:
    return target_col in PROP_TARGETS


def build_game_level_prop_features(
    X_side: pd.DataFrame,
    meta_side: pd.DataFrame,
    y_side: pd.Series | None = None,
    *,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame]:
    """
    Collapse two side-POV rows into one game-level prop row.

    The output intentionally contains both side-specific state and symmetric
    aggregates. This matches map-level prop markets where the target is one
    total for the full game, not one label per team.
    """
    _require_meta(meta_side)
    frame = pd.concat(
        [
            meta_side[
                [
                    "gameid",
                    "side",
                    *[
                        c
                        for c in ("date", "league", "season", "patch")
                        if c in meta_side.columns
                    ],
                ]
            ],
            X_side,
        ],
        axis=1,
    ).copy()
    frame["side"] = frame["side"].astype(str).str.strip().str.title()
    if y_side is not None:
        frame["__target"] = pd.to_numeric(y_side, errors="coerce")

    feature_rows: list[dict[str, Any]] = []
    target_rows: list[float] = []
    meta_rows: list[dict[str, Any]] = []
    feature_cols = list(X_side.columns)
    numeric_cols = [
        c for c in feature_cols if c in X_side and is_numeric_dtype(X_side[c])
    ]

    for gameid, game in frame.groupby("gameid", sort=False):
        side_rows = _validated_side_rows(game, gameid)
        meta_rows.append(_meta_payload(side_rows, gameid))

        out: dict[str, Any] = {}
        for side in SIDE_ORDER:
            prefix = SIDE_PREFIX[side]
            row = side_rows.loc[side]
            for col in feature_cols:
                out[f"{prefix}_{col}"] = row.get(col, np.nan)

        blue = side_rows.loc["Blue"]
        red = side_rows.loc["Red"]
        for col in numeric_cols:
            blue_val = _finite_or_nan(blue.get(col))
            red_val = _finite_or_nan(red.get(col))
            out[f"mean_{col}"] = np.nanmean([blue_val, red_val])
            out[f"absdiff_{col}"] = (
                abs(blue_val - red_val) if _both_finite(blue_val, red_val) else np.nan
            )
            out[f"sum_{col}"] = (
                blue_val + red_val if _both_finite(blue_val, red_val) else np.nan
            )

        feature_rows.append(out)
        if y_side is not None:
            target_rows.append(_validated_game_target(side_rows, gameid, target_col))

    X_game = pd.DataFrame(feature_rows)
    meta_game = pd.DataFrame(meta_rows)
    y_game = pd.Series(target_rows, name=target_col) if y_side is not None else None
    return X_game, y_game, meta_game


def _require_meta(meta_side: pd.DataFrame) -> None:
    missing = {"gameid", "side"} - set(meta_side.columns)
    if missing:
        msg = f"Prop feature builder missing metadata columns: {sorted(missing)}"
        raise ValueError(msg)


def _validated_side_rows(game: pd.DataFrame, gameid: Any) -> pd.DataFrame:
    side_rows = game.drop_duplicates("side", keep="first").set_index("side")
    missing = set(SIDE_ORDER) - set(side_rows.index)
    if missing:
        msg = (
            f"Game '{gameid}' is missing side rows for prop training: {sorted(missing)}"
        )
        raise ValueError(msg)
    return side_rows.loc[list(SIDE_ORDER)]


def _validated_game_target(
    side_rows: pd.DataFrame, gameid: Any, target_col: str | None
) -> float:
    values = (
        pd.to_numeric(side_rows["__target"], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )
    if len(values) != len(SIDE_ORDER):
        msg = f"Game '{gameid}' has missing prop target '{target_col}'."
        raise ValueError(msg)
    if not math.isclose(
        float(values[0]), float(values[1]), rel_tol=0.0, abs_tol=TARGET_TOLERANCE
    ):
        msg = f"Game '{gameid}' has side-level disagreement for prop target '{target_col}'."
        raise ValueError(msg)
    return float(values[0])


def _meta_payload(side_rows: pd.DataFrame, gameid: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"gameid": gameid}
    for col in ("date", "league", "season", "patch"):
        if col in side_rows.columns:
            payload[col] = (
                side_rows[col].dropna().iloc[0]
                if side_rows[col].notna().any()
                else np.nan
            )
    return payload


def _finite_or_nan(value: Any) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else np.nan


def _both_finite(left: float, right: float) -> bool:
    return bool(np.isfinite(left) and np.isfinite(right))
