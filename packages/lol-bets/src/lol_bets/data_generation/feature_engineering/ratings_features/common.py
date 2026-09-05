"""Shared, behavior-neutral plumbing for entity rating systems."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from oracle_bets_core.io_utils import get_sorting_keys
from oracle_bets_core.league_taxonomy import get_league_taxonomy
from oracle_bets_core.logger import logger
from oracle_bets_core.pd import pd

if TYPE_CHECKING:
    import logging
    from pathlib import Path


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a numeric value to an inclusive range."""
    return max(min_value, min(value, max_value))


def is_major_league(league: str) -> bool:
    return get_league_taxonomy(league)["tier"] == "major"


def is_cross_league_competition(league: str) -> bool:
    return get_league_taxonomy(league)["tier"] == "cross"


def preprocess_rating_dataframe(
    df: pd.DataFrame, entity: str, pipeline_logger: logging.Logger
) -> pd.DataFrame:
    """Validate and normalize the common team/player rating input contract."""
    entity = entity.lower()
    if entity not in {"team", "player"}:
        raise ValueError("Entity must be 'team' or 'player'")

    identity = "teamid" if entity == "team" else "playerid"
    required = {"season", "date", "gameid", identity, "league", "side", "result"}
    if entity == "player":
        required.add("position")
    if missing := required - set(df.columns):
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(out["date"]):
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        if null_count := int(out["date"].isna().sum()):
            _warn(
                f"{null_count} 'date' entries could not be converted; dropping them.",
                pipeline_logger,
            )
            out = out.dropna(subset=["date"]).copy()

    missing_leagues = int(out["league"].isna().sum())
    missing_results = int(out["result"].isna().sum())
    if missing_leagues or missing_results:
        _warn(
            f"{missing_leagues} 'league' and {missing_results} 'result' missing; "
            "dropping them.",
            pipeline_logger,
        )
        out = out.dropna(subset=["league", "result"]).reset_index(drop=True)

    if not pd.api.types.is_numeric_dtype(out["result"]):
        out["result"] = (
            out["result"]
            .map(
                {
                    "W": 1,
                    "Win": 1,
                    "win": 1,
                    True: 1,
                    "L": 0,
                    "Loss": 0,
                    "loss": 0,
                    False: 0,
                }
            )
            .astype("float64")
        )
    return out.sort_values(by=get_sorting_keys(entity), kind="mergesort").reset_index(
        drop=True
    )


def load_hyperparameters(
    path: Path, pipeline_logger: logging.Logger
) -> dict[str, float]:
    """Load a rating parameter JSON file, falling back to an empty mapping."""
    try:
        with path.open() as stream:
            logger.info(f"Loading hyperparameters from {path}")
            pipeline_logger.info(f"Loading hyperparameters from {path}")
            return json.load(stream)
    except FileNotFoundError:
        return {}
    except Exception as error:
        logger.error(f"Failed to load hyperparameters from {path}: {error}")
        pipeline_logger.exception(f"Failed to load hyperparameters from {path}")
        return {}


def save_hyperparameters(
    params: dict[str, float], path: Path, pipeline_logger: logging.Logger
) -> None:
    """Write a rating parameter JSON file."""
    logger.info(f"Storing hyperparameters to {path}")
    pipeline_logger.info(f"Storing hyperparameters to {path}")
    try:
        with path.open("w") as stream:
            json.dump(params, stream)
    except Exception as error:
        logger.error(f"Failed to save hyperparameters to {path}: {error}")
        pipeline_logger.exception(f"Failed to save hyperparameters to {path}")


def split_and_validate_data(
    df: pd.DataFrame, entity: str, pipeline_logger: logging.Logger
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the common chronological tuning split and validate game groups."""
    ordered = df.sort_values(by=["date", "gameid", "side"]).reset_index(drop=True)
    if ordered.empty:
        _warn("DataFrame is empty after sorting.", pipeline_logger)
        return pd.DataFrame(), pd.DataFrame()

    try:
        split_date = pd.to_datetime(f"{ordered['date'].dt.year.max()}-01-01")
    except AttributeError as error:
        logger.error(f"Error accessing 'date' column with .dt accessor: {error}")
        pipeline_logger.exception(
            f"Error accessing 'date' column with .dt accessor: {error}"
        )
        return pd.DataFrame(), pd.DataFrame()

    train = ordered[ordered["date"] < split_date].reset_index(drop=True)
    valid = ordered[ordered["date"] >= split_date].reset_index(drop=True)
    if train.empty or valid.empty:
        boundary = ordered["date"].quantile(0.8)
        train = ordered[ordered["date"] < boundary].reset_index(drop=True)
        valid = ordered[ordered["date"] >= boundary].reset_index(drop=True)

    expected_count = 10 if entity.lower() == "player" else 2
    for subset, name in ((train, "Training"), (valid, "Validation")):
        if (
            not subset.empty
            and not (subset.groupby("gameid").size() == expected_count).all()
        ):
            _warn(
                f"{name} data has gameids with incorrect number of entities.",
                pipeline_logger,
            )
            return pd.DataFrame(), pd.DataFrame()
    return train, valid


def _warn(message: str, pipeline_logger: logging.Logger) -> None:
    logger.warning(message)
    pipeline_logger.warning(message)
