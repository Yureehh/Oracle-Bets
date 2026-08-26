"""
Data Generator (typed / lightly-refactored).

Daily pipeline:

1. Ingest last *N* seasons of Oracle-Elixir data from public Google Drive files.
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

from dotenv import load_dotenv
from oracle_bets_core.io_utils import json_loader, safe_store_df_as_parquet
from oracle_bets_core.logger import LOG_TOPIC, instantiate_logger, logger
from oracle_bets_core.paths import (
    DATA_QUALITY_REPORT,
    FLATTENED_PLAYER_CONFIG,
    FLATTENED_TEAM_CONFIG,
    HISTORY_REFRESH_MANIFEST,
    INTERIM_PLAYER_DATA,
    INTERIM_TEAM_DATA,
    PROCESSED_DIR,
    PROCESSED_PLAYERS,
    PROCESSED_TEAMS,
    QUARANTINED_RAW_DATA,
    RAW_CURRENT_POINTER,
    RAW_DATA,
    RAW_GENERATIONS_DIR,
    TRAINING_COMPACT_PLAYER_CONFIG,
    TRAINING_COMPACT_TEAM_CONFIG,
    TRAINING_PLAYER_CONFIG,
    TRAINING_TEAM_CONFIG,
)
from oracle_bets_core.pd import pd

from lol_bets.data_generation.feature_engineering.features_generator import (
    FeatureGenerator,
)
from lol_bets.data_generation.feature_engineering.performance_features.performance_metrics import (
    PerformanceMetrics,
)
from lol_bets.data_generation.feature_engineering.ratings_features.elo import (
    calculate_elo,
)
from lol_bets.data_generation.feature_engineering.ratings_features.glicko import (
    calculate_glicko2,
)
from lol_bets.data_generation.feature_engineering.ratings_features.leagues_elo import (
    calculate_leagues_elo,
)
from lol_bets.data_generation.feature_engineering.ratings_features.plackett_luce import (
    calculate_plackett_luce,
)
from lol_bets.data_generation.feature_engineering.ratings_features.trueskill import (
    calculate_trueskill,
)
from lol_bets.data_generation.ingestion.history import (
    HistoryRefreshMode,
    SourceHistoryError,
    current_history_data_path,
    merge_history,
    publish_history_snapshot,
    refresh_years,
)
from lol_bets.data_generation.ingestion.oracles_elixir import (
    OraclesElixir,
    OraclesElixirError,
)
from lol_bets.data_generation.ingestion.quality import (
    quarantine_oracles_elixir_data,
    write_quality_report,
)
from lol_bets.data_generation.ingestion.source import source_snapshot_id
from lol_bets.prediction_models.feature_contract import default_feature_registry

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
YEARS_BACK: Final[int] = 3
Entity = Literal["team", "player"]


# ───────────────────────────────  helpers  ────────────────────────────────────
def _years_to_process(
    mode: HistoryRefreshMode = HistoryRefreshMode.INCREMENTAL,
) -> list[str]:
    """Return the most recent `YEARS_BACK` seasons (descending) as strings."""
    now = dt.date.today().year
    return [str(year) for year in refresh_years(now, mode, years_back=YEARS_BACK)]


def _parallelise(
    fn: Callable[..., pd.DataFrame],
    team_df: pd.DataFrame,
    player_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a function on team & player frames concurrently with proper error handling.
    Exceptions in either branch propagate with entity context.
    """
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="entity_") as pool:
        futures = {
            pool.submit(fn, team_df, entity="team"): "team",
            pool.submit(fn, player_df, entity="player"): "player",
        }
        results: dict[str, pd.DataFrame] = {}
        for fut, entity in futures.items():
            try:
                results[entity] = fut.result(timeout=3600)  # 1hr timeout
            except Exception as exc:
                msg = f"Failed to process {entity} data: {exc}"
                logger.exception(msg)
                raise DataGeneratorError(msg) from exc
        return results["team"], results["player"]


class DataGeneratorError(RuntimeError):
    """Custom error for DataGenerator exceptions."""


# ───────────────────────────────  pipeline  ──────────────────────────────────
@dataclass(slots=True)
class DataGenerator:
    """End-to-end data update pipeline."""

    history_mode: HistoryRefreshMode = HistoryRefreshMode.INCREMENTAL

    # populated in __post_init__
    oracle: OraclesElixir = field(init=False)

    feature_generator: FeatureGenerator = field(default_factory=FeatureGenerator)

    team_data: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)
    player_data: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)

    # ───────────────────────  initialisation  ────────────────────────────
    def __post_init__(self) -> None:
        if not isinstance(self.history_mode, HistoryRefreshMode):
            self.history_mode = HistoryRefreshMode(self.history_mode)
        self.oracle = OraclesElixir()
        _dbl("DataGenerator initialised.")

    # ───────────────────────  ingest  ────────────────────────────────────
    def ingest_data(self) -> pd.DataFrame:
        years = _years_to_process(self.history_mode)
        _dbl(f"Ingesting seasons: {years}")
        try:
            incoming = self.oracle.ingest_data(years=years)
            if incoming.empty:
                msg = f"No data ingested for years: {years}"
                raise DataGeneratorError(msg)
            incoming = incoming.drop_duplicates().reset_index(drop=True)
            existing = (
                pd.read_parquet(
                    current_history_data_path(
                        RAW_DATA,
                        pointer_path=RAW_CURRENT_POINTER,
                    )
                )
                if RAW_DATA.exists() or RAW_CURRENT_POINTER.exists()
                else pd.DataFrame(columns=incoming.columns)
            )
            raw, manifest = merge_history(
                existing,
                incoming,
                mode=self.history_mode,
                refreshed_at=dt.datetime.now(dt.UTC),
                source_snapshot_id=source_snapshot_id(self.oracle.local_data_dir),
            )
            _dbl(
                f"History refresh ({self.history_mode.value}) produced "
                f"{len(raw):,} rows: +{manifest.added_rows:,}, "
                f"updated {manifest.updated_rows:,}."
            )
            publish_history_snapshot(
                raw,
                manifest,
                raw_path=RAW_DATA,
                manifest_path=HISTORY_REFRESH_MANIFEST,
                generations_dir=RAW_GENERATIONS_DIR,
                pointer_path=RAW_CURRENT_POINTER,
            )
            return raw
        except (OraclesElixirError, SourceHistoryError) as exc:
            msg = f"Oracle Elixir ingest failed: {exc}"
            raise DataGeneratorError(msg) from exc

    # ───────────────────────  split & persist  ───────────────────────────
    def _quality_gate_raw(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Merge exact duplicates and persist quarantined rows plus a manifest."""
        accepted, quarantined, report = quarantine_oracles_elixir_data(
            raw,
            manual_invalid_games=(),
        )
        write_quality_report(report, DATA_QUALITY_REPORT)
        safe_store_df_as_parquet(
            quarantined,
            QUARANTINED_RAW_DATA,
            [logger, data_pipeline_logger],
        )
        _dbl(
            "Raw quality gate accepted "
            f"{report.accepted_games:,} games and quarantined "
            f"{report.quarantined_games:,}."
        )
        return accepted

    def clean_and_store_data(self, raw: pd.DataFrame) -> None:
        """Split raw into team/player, persist interim. Sorting done in clean_data()."""
        accepted = self._quality_gate_raw(raw)
        self.team_data = self.oracle.clean_data(accepted, "team")
        self.player_data = self.oracle.clean_data(accepted, "player")

        self._store_team_and_player(
            INTERIM_TEAM_DATA,
            INTERIM_PLAYER_DATA,
            "Interim Parquet files written.",
        )

    # ───────────────────────  enrichment  ────────────────────────────────
    def _enrich_ratings(self) -> None:
        _dbl("Adding ratings …")
        # League ELO for teams first (writes league artefacts used by other models)
        self.team_data = calculate_leagues_elo(self.team_data, entity="team")
        # Player + Team ratings, in parallel per model
        for fn in (
            calculate_elo,
            calculate_glicko2,
            calculate_plackett_luce,
            calculate_trueskill,
        ):
            self.team_data, self.player_data = _parallelise(
                fn, self.team_data, self.player_data
            )
        self.team_data = self.feature_generator.add_rating_uncertainty(self.team_data)
        self.player_data = self.feature_generator.add_rating_uncertainty(
            self.player_data
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
            self.team_data, player_data=self.player_data
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
    ) -> None:  # sourcery skip: use-named-expression
        """
        Materialise either training (uses *_before columns) or flattened
        inference tables (uses last row per entity, *_after columns).
        """
        config_path = conf
        if kind == "training":
            variant = os.getenv("TRAINING_CONFIG_VARIANT", "").casefold()
            if variant == "compact":
                config_path = (
                    TRAINING_COMPACT_TEAM_CONFIG
                    if entity == "team"
                    else TRAINING_COMPACT_PLAYER_CONFIG
                )

        cfg = json_loader(config_path)
        key = "flattened_cols" if kind == "flattened" else f"{entity}_features"
        cols: list[str] = cfg[key]
        default_feature_registry().validate_materialization(cols)

        missing = set(cols) - set(df.columns)
        if missing:
            msg = f"{entity} missing cols for {kind}: {missing}"
            raise DataGeneratorError(msg)

        if kind == "flattened":
            # last row per entity, drop _after suffix for cleaner inference columns
            after = {c: c.replace("_after", "") for c in cols if "_after" in c}
            identity = f"{entity}name" if entity == "team" else f"{entity}id"
            out = (
                df.sort_values([identity, "date"])
                .groupby(identity, sort=False)
                .tail(1)[cols]
                .rename(columns=after)
            )
            dest = PROCESSED_DIR / f"{entity}s" / f"flattened_{entity}s.parquet"
        else:
            # training uses *_before and explicit diff_ema_* comparison columns
            before_map = {
                c: c.replace("_before", "") for c in cols if c.endswith("_before")
            }
            out = df.loc[:, cols].rename(columns=before_map)
            dest = PROCESSED_DIR / f"{entity}s" / f"training_{entity}s.parquet"

        safe_store_df_as_parquet(out, dest, [logger, data_pipeline_logger])
        _dbl(f"{kind.title()} {entity} saved: {len(out)} rows.")

    def extract_training_data(self) -> None:
        self._extract(self.team_data, TRAINING_TEAM_CONFIG, "team", "training")
        self._extract(self.player_data, TRAINING_PLAYER_CONFIG, "player", "training")

    def flatten_both_inference_data(self) -> None:
        self._extract(self.team_data, FLATTENED_TEAM_CONFIG, "team", "flattened")
        self._extract(self.player_data, FLATTENED_PLAYER_CONFIG, "player", "flattened")

    # ───────────────────────  driver  ────────────────────────────────────
    def run(self) -> None:
        start = dt.datetime.now()
        _dbl("=== Data generation started ===")

        def _timed(_: str) -> float:
            return (dt.datetime.now() - start).total_seconds()

        raw = self.ingest_data()
        _dbl(f"  [1/5] Ingestion: {_timed('ingest'):.1f}s")

        self.clean_and_store_data(raw)
        _dbl(f"  [2/5] Cleaning: {_timed('clean'):.1f}s")
        del raw  # Free memory early

        self.enrich_datasets()
        _dbl(f"  [3/5] Enrichment: {_timed('enrich'):.1f}s")

        self.extract_training_data()
        _dbl(f"  [4/5] Training extraction: {_timed('train'):.1f}s")

        self.flatten_both_inference_data()
        _dbl(f"  [5/5] Flattening: {_timed('flatten'):.1f}s")

        elapsed = (dt.datetime.now() - start).total_seconds()
        _dbl(f"=== Data generation finished in {elapsed:.1f}s ===")


# ───────────────────────────────  manual exec  ───────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    try:
        DataGenerator().run()
    except DataGeneratorError:
        data_pipeline_logger.exception("Data generation failed.")
        raise
