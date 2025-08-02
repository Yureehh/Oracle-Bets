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
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

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
    TEAM_REPLACEMENTS_AND_INVALID_GAMES,
    TRAINING_PLAYER_CONFIG,
    TRAINING_TEAM_CONFIG,
)

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
AWS_SECRET_ENV: Final[str] = "SECRET_ID"

YEARS_BACK: Final[int] = 3
MAX_ROWS, MAX_PLAYERS, MAX_TEAMS = 12, 10, 2  # sanity-checks


# ───────────────────────────────  small helpers  ─────────────────────────────
def _years_to_process() -> list[str]:
    now = dt.date.today().year
    return [str(y) for y in range(now, now - YEARS_BACK, -1)]


def _parallelise(
    fn, team_df: pd.DataFrame, player_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run rating/metric fn on team & player frames concurrently."""
    with ThreadPoolExecutor() as pool:
        f_team = pool.submit(fn, team_df, entity="team")
        f_player = pool.submit(fn, player_df, entity="player")
        return f_team.result(), f_player.result()


def _require_env(key: str) -> str:  # fail-fast helper
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Environment variable '{key}' not set")
    return val


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
            raise RuntimeError(f"S3 ingest failed: {exc}") from exc

    # ───────────────────────  cleaning helpers  ──────────────────────────
    @staticmethod
    def _detect_buggy_games(df: pd.DataFrame) -> set[str]:
        grp = df.groupby("gameid")
        bad = grp.filter(
            lambda g: (
                len(g) != MAX_ROWS
                or g["teamid"].nunique() != MAX_TEAMS
                or g["playerid"].nunique() != MAX_PLAYERS
                or g["teamname"].str.contains("unknown", case=False).any()
                or g["playername"].str.contains("unknown", case=False).any()
            )
        )
        return set(bad["gameid"].unique())

    def _remove_buggy_games(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = json_loader(TEAM_REPLACEMENTS_AND_INVALID_GAMES)
        manual = set(cfg["invalid_games"])
        auto = self._detect_buggy_games(df)
        cleaned = df[~df["gameid"].isin(manual | auto)].reset_index(drop=True)
        _dbl(f"Removed {len(manual | auto)} invalid games.")
        return cleaned

    # ───────────────────────  split & persist  ───────────────────────────
    def clean_and_store_data(self, raw: pd.DataFrame) -> None:
        cleaned = self._remove_buggy_games(raw)

        self.team_data = (
            self.oracle.clean_data(cleaned, "team")
            .sort_values(get_sorting_keys("team"))
            .reset_index(drop=True)
        )
        self.player_data = (
            self.oracle.clean_data(cleaned, "player")
            .sort_values(get_sorting_keys("player"))
            .reset_index(drop=True)
        )

        safe_store_df_as_parquet(
            self.team_data, INTERIM_TEAM_DATA, [logger, data_pipeline_logger]
        )
        safe_store_df_as_parquet(
            self.player_data, INTERIM_PLAYER_DATA, [logger, data_pipeline_logger]
        )
        _dbl("Interim Parquet files written.")

    # ───────────────────────  enrichment  ────────────────────────────────
    def _enrich_ratings(self) -> None:
        _dbl("Adding ratings …")
        self.team_data = self.rating_models.compute_leagues_elo(self.team_data)
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
        self.team_data, self.player_data = _parallelise(
            PerformanceMetrics.add_entity_ema_statistics,
            self.team_data,
            self.player_data,
        )
        for fn in (
            PerformanceMetrics.add_side_win_rate_ewm,
            PerformanceMetrics.add_patch_win_rate_ewm,
            PerformanceMetrics.add_season_win_rate_ewm,
        ):
            self.team_data = fn(self.team_data, entity="team")

    def generate_features(self) -> None:
        self.team_data = self.feature_generator.generate_new_team_features(
            self.team_data
        )
        self.player_data = self.feature_generator.generate_new_player_features(
            self.player_data
        )

    def enrich_datasets(self) -> None:
        self.generate_features()
        self._enrich_ratings()
        self._enrich_performance()
        self._store_enriched()

    # ───────────────────────  persist processed  ────────────────────────
    def _store_enriched(self) -> None:
        safe_store_df_as_parquet(
            self.team_data, PROCESSED_TEAMS, [logger, data_pipeline_logger]
        )
        safe_store_df_as_parquet(
            self.player_data, PROCESSED_PLAYERS, [logger, data_pipeline_logger]
        )
        _dbl("Processed Parquet files written.")

    # ───────────────────────  training / flattened  ──────────────────────
    def _extract(
        self,
        df: pd.DataFrame,
        conf: Path,
        entity: str,
        kind: str,  # "training" | "flattened"
    ) -> None:
        cfg = json_loader(conf)
        key = "flattened_cols" if kind == "flattened" else f"{entity}_features"
        cols: list[str] = cfg[key]

        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{entity} missing cols: {missing}")

        if kind == "flattened":
            after = {c: c.replace("_after", "") for c in cols if "_after" in c}
            out = (
                df.sort_values([f"{entity}id", "date"])
                .groupby(f"{entity}id")
                .tail(1)[cols]
                .rename(columns=after)
            )
            dest = PROCESSED_DIR / f"flattened_{entity}s.parquet"
        else:
            before = {c: c.replace("_before", "") for c in cols if "_before" in c}
            out = df[cols].rename(columns=before)
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
    except Exception:  # noqa: BLE001
        data_pipeline_logger.exception("Data generation failed.")
        raise
