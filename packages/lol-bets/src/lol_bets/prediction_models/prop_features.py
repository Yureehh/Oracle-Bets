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
TEAMS_PER_GAME = 2
MATCHUP_EXCLUDED_FEATURES = frozenset({"first_pick", "side_win_likelihood"})
MATCHUP_INVARIANT_NUMERIC = frozenset({"best_of", "maps_completed", "next_map_number"})
MATCHUP_IDENTITY_METADATA = (
    "series_id",
    "source_gameid",
    "target_gameid",
    "next_map_number",
)


def is_prop_target(target_col: str) -> bool:
    return target_col in PROP_TARGETS


def build_game_level_outcome_features(
    X_team: pd.DataFrame,
    meta_team: pd.DataFrame,
    y_team: pd.Series | None = None,
    *,
    target_col: str = "result",
) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame]:
    """
    Build one canonical, side-free winner row per game.

    Team IDs determine the canonical order. Numeric features become signed
    Team-A minus Team-B deltas, while categorical features become sorted pairs.
    Reversing a caller therefore builds the same feature row.
    """
    required = {"gameid", "teamid", "teamname"}
    missing = required - set(meta_team.columns)
    if missing:
        msg = f"Outcome matchup builder missing metadata columns: {sorted(missing)}"
        raise ValueError(msg)

    frame = pd.concat(
        [meta_team.reset_index(drop=True), X_team.reset_index(drop=True)], axis=1
    )
    if y_team is not None:
        frame["__target"] = pd.to_numeric(y_team, errors="coerce").to_numpy()
    feature_cols = [
        column for column in X_team.columns if column not in MATCHUP_EXCLUDED_FEATURES
    ]
    numeric_cols = [
        column for column in feature_cols if is_numeric_dtype(X_team[column])
    ]

    feature_rows: list[dict[str, Any]] = []
    target_rows: list[float] = []
    meta_rows: list[dict[str, Any]] = []
    for gameid, game in frame.groupby("gameid", sort=False):
        if len(game) != TEAMS_PER_GAME:
            msg = f"Game '{gameid}' must have exactly two teams for outcome training."
            raise ValueError(msg)
        ordered = game.sort_values(["teamid", "teamname"], kind="mergesort")
        first, second = ordered.iloc[0], ordered.iloc[1]
        out: dict[str, Any] = {}
        for column in feature_cols:
            left, right = first.get(column), second.get(column)
            if column in numeric_cols:
                if column in MATCHUP_INVARIANT_NUMERIC:
                    if _finite_or_nan(left) != _finite_or_nan(right):
                        raise ValueError(
                            f"Game '{gameid}' has conflicting context '{column}'."
                        )
                    out[f"context_{column}"] = _finite_or_nan(left)
                else:
                    out[f"delta_{column}"] = _finite_or_nan(left) - _finite_or_nan(
                        right
                    )
            else:
                values = sorted(
                    str(value) for value in (left, right) if pd.notna(value)
                )
                out[f"pair_{column}"] = "|".join(values) if values else "Unknown"
        feature_rows.append(out)
        meta_payload = {
            "gameid": gameid,
            "date": first.get("date"),
            "league": first.get("league"),
            "season": first.get("season"),
            "patch": first.get("patch"),
            "split": first.get("split"),
            "playoffs": first.get("playoffs"),
            "league_region": first.get("league_region"),
            "league_tier": first.get("league_tier"),
            "strength_pool": first.get("strength_pool"),
            "best_of": first.get("best_of"),
            "is_bo1": first.get("is_bo1"),
            "is_bo3": first.get("is_bo3"),
            "is_bo5": first.get("is_bo5"),
            "canonical_teamid": first["teamid"],
            "canonical_teamname": first["teamname"],
        }
        meta_payload.update(_matchup_identity_metadata(game, gameid))
        meta_rows.append(meta_payload)
        if y_team is not None:
            target = _finite_or_nan(first.get("__target"))
            if not np.isfinite(target) or target not in {0.0, 1.0}:
                msg = f"Game '{gameid}' has invalid canonical winner target."
                raise ValueError(msg)
            target_rows.append(target)

    X_game = pd.DataFrame(feature_rows)
    y_game = pd.Series(target_rows, name=target_col) if y_team is not None else None
    return X_game, y_game, pd.DataFrame(meta_rows)


def _matchup_identity_metadata(game: pd.DataFrame, gameid: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for column in MATCHUP_IDENTITY_METADATA:
        if column not in game.columns:
            continue
        values = game[column].dropna().unique().tolist()
        if len(values) != 1:
            msg = f"Game '{gameid}' has conflicting identity metadata '{column}'."
            raise ValueError(msg)
        payload[column] = values[0]
    return payload


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
    meta_cols = [
        "gameid",
        "side",
        *[c for c in ("date", "league", "season", "patch") if c in meta_side.columns],
    ]
    feature_side = X_side.drop(columns=meta_cols, errors="ignore")
    frame = pd.concat(
        [meta_side[meta_cols], feature_side],
        axis=1,
    ).copy()
    frame["side"] = frame["side"].astype(str).str.strip().str.title()
    if y_side is not None:
        frame["__target"] = pd.to_numeric(y_side, errors="coerce")

    feature_rows: list[dict[str, Any]] = []
    target_rows: list[float] = []
    meta_rows: list[dict[str, Any]] = []
    feature_cols = list(feature_side.columns)
    numeric_cols = [
        c
        for c in feature_cols
        if c in feature_side and is_numeric_dtype(feature_side[c])
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
            out[f"mean_{col}"] = _mean_or_nan(blue_val, red_val)
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
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return np.nan
    return parsed if np.isfinite(parsed) else np.nan


def _mean_or_nan(left: float, right: float) -> float:
    values = [value for value in (left, right) if np.isfinite(value)]
    return float(np.mean(values)) if values else np.nan


def _both_finite(left: float, right: float) -> bool:
    return bool(np.isfinite(left) and np.isfinite(right))
