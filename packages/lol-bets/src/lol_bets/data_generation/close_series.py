"""Rolling out-of-fold close-series membership for next-map research."""

from __future__ import annotations

from typing import Any

import numpy as np
from oracle_bets_core.paths import (
    CLOSE_SERIES_OOF,
    NEXT_MAP_PLAYER_DATA,
    NEXT_MAP_TEAM_DATA,
    SERIES_MANIFEST,
    SERIES_WINNER_TEAM_DATA,
    TRAINING_PLAYER_DATA,
    TRAINING_TEAM_DATA,
)
from oracle_bets_core.pd import pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lol_bets.data_generation.series import _next_map_training_tables

RATING_COLUMNS = (
    "elo",
    "glicko2_mu",
    "glicko2_phi",
    "pl_mu",
    "pl_sigma",
    "trueskill_mu",
    "trueskill_sigma",
    "elo_win_likelihood",
    "glicko2_win_likelihood",
    "pl_win_likelihood",
    "trueskill_win_likelihood",
)
MIN_OOF_TRAINING_SERIES = 100
OOF_BLOCKS = 6
RANDOM_STATE = 42
SOURCE_MODEL = "direct_series_rating_logistic_oof_v1"
BINARY_CLASSES = 2
TEAM_ROWS_PER_SERIES = 2


def rolling_close_series_oof(
    series_team_rows: pd.DataFrame,
    *,
    minimum_training_series: int = MIN_OOF_TRAINING_SERIES,
    blocks: int = OOF_BLOCKS,
) -> pd.DataFrame:
    """Predict each retained fold using only strictly earlier series."""
    required = {"gameid", "date", "teamid", "result", *RATING_COLUMNS}
    missing = required - set(series_team_rows.columns)
    if missing:
        raise ValueError(f"close-series OOF source is missing: {sorted(missing)}")
    if minimum_training_series < 2 or blocks < 3:  # noqa: PLR2004
        raise ValueError("close-series OOF configuration is too small")
    rows = _canonical_rating_rows(series_team_rows)
    timestamps = pd.Index(rows["date"].drop_duplicates().sort_values())
    if len(timestamps) < blocks:
        raise ValueError("close-series OOF requires enough unique timestamps")
    predictions: list[dict[str, Any]] = []
    for fold, validation_dates in enumerate(np.array_split(timestamps, blocks)[1:], 1):
        validation_start = validation_dates.min()
        train = rows.loc[rows["date"].lt(validation_start)]
        validation = rows.loc[rows["date"].isin(set(validation_dates))]
        if (
            len(train) < minimum_training_series
            or train["actual"].nunique() != BINARY_CLASSES
        ):
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=2000, random_state=RANDOM_STATE),
        )
        model.fit(train.loc[:, RATING_COLUMNS], train["actual"])
        probability = model.predict_proba(validation.loc[:, RATING_COLUMNS])[:, 1]
        cutoff = train["date"].max()
        for series_id, fixture_at, value in zip(
            validation["series_id"],
            validation["date"],
            probability,
            strict=True,
        ):
            predictions.append(
                {
                    "series_id": series_id,
                    "close_series_probability": float(value),
                    "close_series_source_model": SOURCE_MODEL,
                    "close_series_fold": f"fold-{fold}",
                    "close_series_training_cutoff": cutoff,
                    "close_series_forecast_at": validation_start
                    - pd.Timedelta(microseconds=1),
                    "close_series_oof": True,
                    "fixture_at": fixture_at,
                }
            )
    return pd.DataFrame(predictions)


def build_close_series_oof_artifacts() -> dict[str, Any]:
    """Publish OOF membership and rebuild next-map tables from close series only."""
    manifest = pd.read_parquet(SERIES_MANIFEST)
    teams = pd.read_parquet(SERIES_WINNER_TEAM_DATA)
    oof = rolling_close_series_oof(teams)
    if oof.empty:
        raise RuntimeError(
            "No valid rolling close-series OOF predictions were produced"
        )
    augmented = manifest.merge(
        oof.drop(columns=["fixture_at"]),
        on="series_id",
        how="inner",
        validate="1:1",
    )
    next_teams, next_players = _next_map_training_tables(
        augmented,
        pd.read_parquet(TRAINING_TEAM_DATA),
        pd.read_parquet(TRAINING_PLAYER_DATA),
    )
    for path, frame in (
        (CLOSE_SERIES_OOF, oof),
        (NEXT_MAP_TEAM_DATA, next_teams),
        (NEXT_MAP_PLAYER_DATA, next_players),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    return {
        "oof_series": int(len(oof)),
        "close_series": int(augmented["series_id"].nunique()),
        "next_map_rows": int(len(next_teams)),
        "source_model": SOURCE_MODEL,
    }


def _canonical_rating_rows(team_rows: pd.DataFrame) -> pd.DataFrame:
    source = team_rows.copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce", utc=True)
    records: list[dict[str, Any]] = []
    for series_id, group in source.groupby("gameid", sort=False):
        ordered = group.sort_values("teamid", kind="mergesort")
        if (
            len(ordered) != TEAM_ROWS_PER_SERIES
            or ordered["teamid"].nunique() != TEAM_ROWS_PER_SERIES
            or ordered["date"].isna().any()
            or ordered["result"].sum() != 1
        ):
            continue
        first, second = ordered.iloc[0], ordered.iloc[1]
        record = {
            "series_id": str(series_id),
            "date": ordered["date"].min(),
            "actual": int(first["result"]),
        }
        record.update(
            {
                column: float(first[column]) - float(second[column])
                for column in RATING_COLUMNS
            }
        )
        records.append(record)
    rows = pd.DataFrame(records).sort_values("date", kind="mergesort")
    if rows.empty or rows.loc[:, RATING_COLUMNS].isna().any().any():
        raise ValueError("close-series OOF rating rows are empty or incomplete")
    return rows.reset_index(drop=True)
