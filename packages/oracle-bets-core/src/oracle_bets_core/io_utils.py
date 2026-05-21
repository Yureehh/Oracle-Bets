"""
I/O utility helpers.

Typed, test-friendly wrappers around common file-system operations plus model
(pickle) persistence.  Public API matches the original version so downstream
imports remain unchanged.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Final

from oracle_bets_core.logger import LOG_TOPIC, instantiate_logger
from oracle_bets_core.pd import pd

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
_log = instantiate_logger(LOG_TOPIC.DATA_PIPELINE)


# --------------------------------------------------------------------------- #
# Explicit error models
# --------------------------------------------------------------------------- #
class FileLoadError(RuntimeError):
    """Raised when a file cannot be read or parsed."""


class ModelStoreError(RuntimeError):
    """Raised when pickling / unpickling a model fails."""


class DataFrameStoreError(RuntimeError):
    """Raised when saving a DataFrame to disk fails."""


# --------------------------------------------------------------------------- #
# Sorting helpers
# --------------------------------------------------------------------------- #
_SORT_KEYS: Final = {
    "team": ["date", "league", "gameid", "side"],
    "player": ["date", "league", "gameid", "side", "position"],
}


def get_sorting_keys(entity: str) -> list[str]:
    """Return canonical sort keys for *entity* (“team” or “player”)."""
    try:
        return _SORT_KEYS[entity.lower()]
    except KeyError:
        msg = f"Entity must be 'player' or 'team', not '{entity}'."
        raise ValueError(msg) from None


def get_identity(entity: str) -> str:
    """Return identity column name (“playerid” / “teamid”)."""
    ent = entity.lower()
    if ent in {"player", "team"}:
        return f"{ent}id"
    msg = "Entity must be 'player' or 'team'."
    raise ValueError(msg)


# --------------------------------------------------------------------------- #
# Generic file loaders
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_file(file_path: str | Path, *, file_type: str = "json") -> Any | pd.DataFrame:
    """
    Load *file_path* according to *file_type* (“json”, “csv”, or “parquet”).
    """
    path = Path(file_path)
    if not path.exists():
        msg = f"No such file: '{path}'"
        raise FileLoadError(msg)

    try:
        if file_type == "json":
            return _load_json(path)
        if file_type == "csv":
            return pd.read_csv(path)
        if file_type == "parquet":
            return pd.read_parquet(path)
    except Exception as exc:
        msg = f"Failed parsing '{path}': {exc}"
        raise FileLoadError(msg) from exc

    msg = (
        f"Unsupported file_type '{file_type}'. "
        "Supported types are 'json', 'csv', 'parquet'."
    )
    raise ValueError(msg)


json_loader = lambda p: load_file(p, file_type="json")  # noqa: E731
csv_loader = lambda p: load_file(p, file_type="csv")  # noqa: E731
parquet_loader = lambda p: load_file(p, file_type="parquet")  # noqa: E731


# --------------------------------------------------------------------------- #
# Model persistence
# --------------------------------------------------------------------------- #
def load_model(filepath: str | Path) -> Any:
    """
    Unpickle a model from *filepath*.

    WARNING: Only load models from trusted sources. Unpickling untrusted data is a security risk.
    """
    path = Path(filepath)
    try:
        with path.open("rb") as f:
            # Only load trusted pickle files. Never unpickle untrusted data!
            return pickle.load(f)  # noqa: S301
    except FileNotFoundError as exc:
        msg = f"Model file not found: '{path}'"
        raise ModelStoreError(msg) from exc
    except Exception as exc:
        msg = f"Unpickling failed for '{path}': {exc}"
        raise ModelStoreError(msg) from exc


def store_model(
    path: str | Path,
    model: Any,
    model_name: str,
    logger=_log,
) -> None:
    """Pickle *model* to *path* and log the outcome."""
    dest = Path(path)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            pickle.dump(model, f)
        logger.info("Stored model '%s' → %s", model_name, dest)
    except (OSError, pickle.PicklingError) as exc:
        logger.exception(
            "Could not store model '%s' at '%s': %s",
            model_name,
            dest,
            exc,
        )
        msg = f"Pickling failed for '{dest}': {exc}"
        raise ModelStoreError(msg) from exc


# --------------------------------------------------------------------------- #
# Training data helpers
# --------------------------------------------------------------------------- #
def load_training_data(
    training_team_data: str | Path,
    training_player_data: str | Path,
    logger=_log,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load team/player training parquet files, returning a tuple (team_df, player_df).
    """
    try:
        team_df = parquet_loader(training_team_data)
        player_df = parquet_loader(training_player_data)
        logger.info(
            "Training data loaded (team=%s rows, player=%s rows).",
            len(team_df),
            len(player_df),
        )
        return team_df, player_df
    except Exception as exc:
        logger.exception(f"Training data load failed: {exc}")
        msg = "Unable to load training data."
        raise FileLoadError(msg) from exc


# --------------------------------------------------------------------------- #
# Data-frame persistence
# --------------------------------------------------------------------------- #
def safe_store_df_as_parquet(
    df: pd.DataFrame,
    output_path: str | Path,
    loggers: list[Any] | None = None,
) -> None:
    """
    Save *df* to *output_path* (gzip parquet).  Falls back to polars on pandas
    failure, propagating an exception only if both writers fail.
    """
    loggers = loggers or [_log]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_to_write = pd.DataFrame(df)  # guarantee pandas-compatible

    def _log_to_all(level: str, msg: str, *args: Any) -> None:
        for lg in loggers:
            getattr(lg, level)(msg, *args)

    try:
        df_to_write.to_parquet(out, compression="gzip")
        _log_to_all("info", "Saved DataFrame → %s (pandas)", out)
    except (ImportError, OSError, ValueError) as exc1:
        _log_to_all(
            "warning", "pandas.to_parquet failed (%s), falling back to polars", exc1
        )
        try:
            import polars as pl

            pl.DataFrame(df_to_write).write_parquet(out, compression="gzip")
            _log_to_all("info", "Saved DataFrame → %s (polars)", out)
        except Exception as exc2:
            _log_to_all(
                "exception",
                "Failed to save DataFrame with both pandas and polars: %s",
                exc2,
            )
            msg = f"Could not write parquet at '{out}': {exc2}"
            raise DataFrameStoreError(msg) from exc2
