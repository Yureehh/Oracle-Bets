"""
Data Generator (typed / lightly-refactored).

Daily pipeline:

1. Ingest last *N* seasons of Oracle-Elixir data from S3.
2. Clean & split into team / player sets.
3. Generate engineered features.
4. Add ratings (ELO, Glicko2, PL, TrueSkill) + performance metrics.
5. Persist raw / interim / processed Parquet artefacts.
6. Produce “training_*” and “flattened_*” tables used by models.

The public surface (constructor + `.run()`) and output file names are **unchanged**.
Only internals were tidied (static type-hints, `@dataclass(slots=True)`, early
env-var validation, fewer duplicate log lines).

Requires the same env-vars, configs and utilities as before.
"""

from __future__ import annotations

import datetime as dt
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

import boto3
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

from feature_engineering.features_generator import FeatureGenerator
from feature_engineering.performance_features.performance_metrics import (
    PerformanceMetrics,
)
from feature_engineering.ratings_features.rating_models import Ratings
from ingestion.oracles_elixir import OraclesElixir
from utils.io_utils import get_sorting_keys, json_loader, safe_store_df_as_parquet
from utils.logger import LOG_TOPIC, instantiate_logger, logger
from utils.paths import (
    FLATTENED_PLAYER_CONFIG,
    FLATTENED_TEAM_CONFIG,
    INTERIM_PLAYER_DATA,
    INTERIM_TEAM_DATA,
    PROCESSED_DIR,
    PROCESSED_PLAYERS,
    PROCESSED_TEAMS,
    RAW_DATA,
    TRAINING_PLAYER_CONFIG,
    TRAINING_TEAM_CONFIG,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# ───────────────────────────────  env / logging  ──────────────────────────────
load_dotenv()
data_pipeline_logger = instantiate_logger(LOG_TOPIC.DATA_PIPELINE)


# helper to log to both streams (keeps existing call-sites unchanged)
def _dbl(msg: str) -> None:
    logger.info(msg)
    data_pipeline_logger.info(msg)


# ───────────────────────────────  constants  ──────────────────────────────────
BUCKET_ENV: Final[str] = "BUCKET_NAME"
AWS_ID_ENV: Final[str] = "ACCESS_ID"
AWS_SECRET_ENV: Final[str] = "SECRET_ID"  # noqa: S105

YEARS_BACK: Final[int] = 3
Entity = Literal["team", "player"]


# ───────────────────────────────  helpers  ────────────────────────────────────
def _years_to_process() -> list[str]:
    """Return the most recent `YEARS_BACK` seasons (descending) as strings."""
    now = dt.date.today().year
    return [str(y) for y in range(now, now - YEARS_BACK, -1)]


def _parallelise(
    fn: Callable[..., pd.DataFrame],
    team_df: pd.DataFrame,
    player_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a function on team & player frames concurrently.
    Exceptions in either branch will propagate on .result().
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_team = pool.submit(fn, team_df, entity="team")
        f_player = pool.submit(fn, player_df, entity="player")
        return f_team.result(), f_player.result()


def _require_env(key: str) -> str:
    """Fail-fast env lookup that raises a pipeline-specific error."""
    val = os.getenv(key)
    if not val:
        msg = f"Environment variable '{key}' not set"
        raise DataGeneratorError(msg)
    return val


class DataGeneratorError(RuntimeError):
    """Custom error for DataGenerator exceptions."""


# ───────────────────────────────  pipeline  ──────────────────────────────────
@dataclass(slots=True)
class DataGenerator:
    """End-to-end data update pipeline."""

    # populated in __post_init__
    bucket_name: str = field(init=False)
    s3_session: boto3.Session = field(init=False)
    oracle: OraclesElixir = field(init=False)

    feature_generator: FeatureGenerator = field(default_factory=FeatureGenerator)
    rating_models: Ratings = field(default_factory=Ratings)

    team_data: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)
    player_data: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)

    # ───────────────────────  initialisation  ────────────────────────────
    def __post_init__(self) -> None:
        self.bucket_name = _require_env(BUCKET_ENV)
        self.s3_session = boto3.Session(
            aws_access_key_id=_require_env(AWS_ID_ENV),
            aws_secret_access_key=_require_env(AWS_SECRET_ENV),
        )
        self.oracle = OraclesElixir(session=self.s3_session, bucket=self.bucket_name)
        _dbl("DataGenerator initialised.")

    # ───────────────────────  ingest  ────────────────────────────────────
    def ingest_data_from_s3(self) -> pd.DataFrame:
        years = _years_to_process()
        _dbl(f"Ingesting seasons: {years}")
        try:
            raw = self.oracle.ingest_data(years=years)
            safe_store_df_as_parquet(raw, RAW_DATA, [logger, data_pipeline_logger])
            return raw
        except (BotoCoreError, ClientError) as exc:  # AWS-side errors
            msg = f"S3 ingest failed: {exc}"
            raise DataGeneratorError(msg) from exc

    # ───────────────────────  split & persist  ───────────────────────────
    def clean_and_store_data(self, raw: pd.DataFrame) -> None:
        """Split raw into team/player, sort deterministically, persist interim."""
        self.team_data = (
            self.oracle.clean_data(raw, "team")
            .sort_values(get_sorting_keys("team"))
            .reset_index(drop=True)
        )
        self.player_data = (
            self.oracle.clean_data(raw, "player")
            .sort_values(get_sorting_keys("player"))
            .reset_index(drop=True)
        )

        self._store_team_and_player(
            INTERIM_TEAM_DATA,
            INTERIM_PLAYER_DATA,
            "Interim Parquet files written.",
        )

    # ───────────────────────  enrichment  ────────────────────────────────
    def _enrich_ratings(self) -> None:
        _dbl("Adding ratings …")
        # League ELO for teams first (writes league artefacts used by other models)
        self.team_data = self.rating_models.compute_leagues_elo(self.team_data)
        # Player + Team ratings, in parallel per model
        for fn in (
            self.rating_models.compute_elo,
            self.rating_models.compute_glicko2,
            self.rating_models.compute_plackett_luce,
            self.rating_models.compute_trueskill,
        ):
            self.team_data, self.player_data = _parallelise(
                fn, self.team_data, self.player_data
            )

    def _enrich_performance(self) -> None:
        _dbl("Adding performance metrics …")
        # Entity EMA stats (team+player) in parallel
        self.team_data, self.player_data = _parallelise(
            PerformanceMetrics.add_entity_ema_statistics,
            self.team_data,
            self.player_data,
        )
        # Team-level side/patch/season EMAs (intentionally teams only)
        for fn in (
            PerformanceMetrics.add_side_win_rate_ewm,
            PerformanceMetrics.add_patch_win_rate_ewm,
            PerformanceMetrics.add_season_win_rate_ewm,
        ):
            self.team_data = fn(self.team_data, entity="team")

    def generate_features(self) -> None:
        """Feature generator: team then player."""
        self.team_data = self.feature_generator.generate_new_team_features(
            self.team_data
        )
        self.player_data = self.feature_generator.generate_new_player_features(
            self.player_data
        )

    def enrich_datasets(self) -> None:
        """Run feature gen, ratings, performance; persist processed artefacts."""
        self.generate_features()
        self._enrich_ratings()
        self._enrich_performance()
        self._store_enriched()

    # ───────────────────────  persist processed  ────────────────────────
    def _store_enriched(self) -> None:
        self._store_team_and_player(
            PROCESSED_TEAMS, PROCESSED_PLAYERS, "Processed Parquet files written."
        )

    def _store_team_and_player(
        self, team_path: Path | str, player_path: Path | str, log_msg: str
    ) -> None:
        safe_store_df_as_parquet(
            self.team_data, team_path, [logger, data_pipeline_logger]
        )
        safe_store_df_as_parquet(
            self.player_data, player_path, [logger, data_pipeline_logger]
        )
        _dbl(log_msg)

    # ───────────────────────  training / flattened  ──────────────────────
    def _extract(
        self,
        df: pd.DataFrame,
        conf: Path,
        entity: Entity,
        kind: Literal["training", "flattened"],
    ) -> None:
        """
        Materialise either training (uses *_before columns) or flattened
        inference tables (uses last row per entity, *_after columns).
        """
        cfg = json_loader(conf)
        key = "flattened_cols" if kind == "flattened" else f"{entity}_features"
        cols: list[str] = cfg[key]

        missing = set(cols) - set(df.columns)
        if missing:
            msg = f"{entity} missing cols for {kind}: {missing}"
            raise DataGeneratorError(msg)

        if kind == "flattened":
            # last row per entity, drop _after suffix for cleaner inference columns
            after = {c: c.replace("_after", "") for c in cols if "_after" in c}
            out = (
                df.sort_values([f"{entity}id", "date"])
                .groupby(f"{entity}id", sort=False)
                .tail(1)[cols]
                .rename(columns=after)
            )
            dest = PROCESSED_DIR / f"flattened_{entity}s.parquet"
        else:
            # training uses *_before; auto-append opponent EMA columns
            before_map = {
                c: c.replace("_before", "") for c in cols if c.endswith("_before")
            }

            # Candidate opponent columns only for EMA-before features you listed
            ema_before_cols = [
                c for c in cols if c.startswith("ema_") and c.endswith("_before")
            ]
            opp_candidates = [f"opp_{c}" for c in ema_before_cols]
            opp_existing = [c for c in opp_candidates if c in df.columns]

            # Final column set (dedup while preserving order)
            cols_final = list(dict.fromkeys([*cols, *opp_existing]))

            # Rename only your own *_before columns; opponent cols stay as-is
            out = df.loc[:, cols_final].rename(columns=before_map)
            dest = PROCESSED_DIR / f"training_{entity}_data.parquet"

        safe_store_df_as_parquet(out, dest, [logger, data_pipeline_logger])
        _dbl(f"{kind.title()} {entity} saved: {len(out)} rows.")

    def extract_both_training_data(self) -> None:
        self._extract(self.team_data, TRAINING_TEAM_CONFIG, "team", "training")
        self._extract(self.player_data, TRAINING_PLAYER_CONFIG, "player", "training")

    def flatten_both_inference_data(self) -> None:
        self._extract(self.team_data, FLATTENED_TEAM_CONFIG, "team", "flattened")
        self._extract(self.player_data, FLATTENED_PLAYER_CONFIG, "player", "flattened")

    # ───────────────────────  driver  ────────────────────────────────────
    def run(self) -> None:
        start = dt.datetime.now()
        _dbl("=== Data generation started ===")

        raw = self.ingest_data_from_s3()
        self.clean_and_store_data(raw)
        self.enrich_datasets()
        self.extract_both_training_data()
        self.flatten_both_inference_data()

        elapsed = (dt.datetime.now() - start).total_seconds()
        _dbl(f"=== Data generation finished in {elapsed:.1f}s ===")


# ───────────────────────────────  manual exec  ───────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    try:
        DataGenerator().run()
    except DataGeneratorError:
        data_pipeline_logger.exception("Data generation failed.")
        raise
