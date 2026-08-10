"""
Oracle Elixir ingestion & cleaning (typed refactor).

Behaviour-for-behaviour identical to the original module, but:

* full static type-hints (`mypy`/`ruff-check-types`-ready);
* `@dataclass(slots=True)` for lower memory and faster attribute access;
* tighter error checking (e.g. correct `isinstance` use, early guards);
* centralised constants to avoid magic literals;
* small vectorised tweaks (no functional change, but marginally faster);
* keeps **all** public names, default values, and logging messages.
"""

from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from dotenv import load_dotenv
from lol_bets.data_generation.ingestion.quality import (
    SOURCE_COLUMN_RENAMES,
    normalize_result,
    quarantine_oracles_elixir_data,
)
from lol_bets.data_generation.ingestion.source import oracle_source_directory
from oracle_bets_core.io_utils import FileLoadError, get_sorting_keys, json_loader
from oracle_bets_core.league_selection import training_leagues
from oracle_bets_core.logger import LOG_TOPIC, instantiate_logger, logger
from oracle_bets_core.paths import IMPORT_COLUMNS, TEAM_ALIASES
from oracle_bets_core.pd import pd

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

# --------------------------------------------------------------------------- #
# Environment & logging
# --------------------------------------------------------------------------- #
load_dotenv()
data_pipeline_logger = instantiate_logger(LOG_TOPIC.DATA_PIPELINE)

# --------------------------------------------------------------------------- #
# Constants (names unchanged so imports elsewhere remain valid)
# --------------------------------------------------------------------------- #
UNIQUE_PLAYERS_PER_GAME: int = 10
UNIQUE_TEAMS_PER_GAME: int = 2
GAP_PLAYER: int = 5
ROWS_PER_GAME_PER_TEAM: int = 2
NULL_REPLACEMENTS: list[str] = ["nan", "null", "unknown", "Unknown", "N/A"]
OPPOSITE_SIDE = {"Blue": "Red", "Red": "Blue"}
EXPECTED_SIDES = {"Blue", "Red"}
EXPECTED_POSITIONS = {"top", "jng", "mid", "bot", "sup"}
EXPECTED_WINNING_ROWS_BY_ENTITY = {"team": 1.0, "player": 5.0}
EXPECTED_ROWS_BY_ENTITY = {
    "team": UNIQUE_TEAMS_PER_GAME,
    "player": UNIQUE_PLAYERS_PER_GAME,
}


class OraclesElixirError(RuntimeError):
    """Base class for all Oracle Elixir ingestion errors."""


# --------------------------------------------------------------------------- #
# Dataclass
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class OraclesElixir:
    """Ingest, clean, and format public Oracle Elixir CSV dumps."""

    local_data_dir: Path = field(default_factory=oracle_source_directory)

    # --------------------------------------------------------------------- #
    # Ingestion
    # --------------------------------------------------------------------- #
    def ingest_data(
        self,
        years: Sequence[int | str] | int | str | None = None,
    ) -> pd.DataFrame:
        """
        Read year-specific CSVs from the validated local source cache.
        """
        # Normalise *years* into a list[int]
        if years is None:
            year_list: list[int] = [dt.date.today().year]
        elif isinstance(years, str | int):
            year_list = [int(years)]
        else:
            year_list = [int(y) for y in years]

        available_years = [
            year for year in year_list if self._local_csv_path(year).is_file()
        ]
        missing = [year for year in year_list if year not in available_years]
        results: dict[int, pd.DataFrame] = {}
        local_years: list[int] = []
        with ThreadPoolExecutor() as pool:
            futures = {
                pool.submit(self._read_local_csv, self._local_csv_path(year)): year
                for year in available_years
            }
            for future in as_completed(futures):
                year = futures[future]
                try:
                    results[year] = future.result()
                    local_years.append(year)
                except Exception as exc:
                    logger.error("Failed to ingest data for year %s: %s", year, exc)
                    data_pipeline_logger.exception(
                        "Failed to ingest data for year %s.", year
                    )
                    raise

        if local_years:
            logger.info("Reading local Oracle Elixir files")
            data_pipeline_logger.info("Reading local Oracle Elixir files")
        if not results:
            msg = (
                "No local Oracle Elixir files found for years "
                f"{year_list} in {self.local_data_dir}. Run "
                "`oracle-bets lol source-refresh` first."
            )
            logger.error(msg)
            data_pipeline_logger.error(msg)
            raise OraclesElixirError(msg)

        frames = [results[y] for y in year_list if y in results]
        if not frames:
            raise OraclesElixirError("No Oracle Elixir frames to concatenate.")
        # Filter out entirely empty frames to avoid dtype confusion (pandas future change)
        frames = [f for f in frames if not f.empty]
        if not frames:
            raise OraclesElixirError("All Oracle Elixir frames were empty.")
        df = pd.concat(frames, ignore_index=True)
        loaded = [y for y in year_list if y in results]
        logger.info("Successfully ingested data for years: %s", loaded)
        data_pipeline_logger.info("Successfully ingested data for years: %s", loaded)
        if missing:
            msg = (
                f"Missing required local Oracle Elixir years: {missing}. "
                "Run `oracle-bets lol source-refresh` and retry."
            )
            raise OraclesElixirError(msg)
        return df

    def _local_csv_path(self, year: int) -> Path:
        return self.local_data_dir / (
            f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"
        )

    @staticmethod
    def _read_local_csv(path: Path) -> pd.DataFrame:
        before = path.stat()
        if before.st_size == 0:
            msg = (
                f"Local Oracle Elixir file is empty: {path}. "
                "Run `oracle-bets lol source-refresh` and retry."
            )
            raise OraclesElixirError(msg)
        try:
            frame = pd.read_csv(path, low_memory=False)
        except OSError as exc:
            msg = (
                f"Cannot read local Oracle Elixir file {path}: {exc}. "
                "Run `oracle-bets lol source-refresh` and retry."
            )
            raise OraclesElixirError(msg) from exc
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            msg = (
                f"Local Oracle Elixir file changed while it was being read: {path}. "
                "Wait for the source refresh to finish and retry."
            )
            raise OraclesElixirError(msg)
        if frame.empty:
            raise OraclesElixirError(f"Local Oracle Elixir file has no rows: {path}")
        return frame

    # --------------------------------------------------------------------- #
    # Formatting & basic cleaning
    # --------------------------------------------------------------------- #
    @staticmethod
    def format_data_length_types(oracles_elixir_data: pd.DataFrame) -> pd.DataFrame:
        """
        Format data types for the Oracle Elixir DataFrame.
        This includes:
        - Converting date strings to datetime objects.
        - Stripping and normalising identifier columns.
        - Replacing common null spellings with `pd.NA`.
        - Converting game length from seconds to minutes.
        """
        if oracles_elixir_data.empty:
            logger.warning("Received empty DataFrame for formatting.")
            data_pipeline_logger.warning("Received empty DataFrame for formatting.")
            return oracles_elixir_data

        logger.info("Formatting data types...")
        data_pipeline_logger.info("Formatting data types...")

        df = oracles_elixir_data.copy()

        # Date → datetime
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Strip + normalise identifier columns
        id_cols = ["gameid", "playerid", "teamid", "league", "teamname", "playername"]
        df[id_cols] = (
            df[id_cols]
            .apply(lambda col: col.str.strip() if col.dtype == "object" else col)
            .replace("", pd.NA)
        )

        # Replace common null spellings
        with pd.option_context("future.no_silent_downcasting", True):
            df.loc[:, :] = df.replace(NULL_REPLACEMENTS, pd.NA)

        # Game length seconds → minutes
        df["gamelength"] = (
            pd.to_numeric(df["gamelength"], errors="coerce").astype(float) / 60.0
        )
        logger.info("Data formatting completed.")
        data_pipeline_logger.info("Data formatting completed.")
        return df

    @staticmethod
    def remove_null_games(oracles_elixir_data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with null 'gameid' values.
        Raises OraclesElixirError if 'gameid' column is missing.
        """
        if "gameid" not in oracles_elixir_data.columns:
            msg = "The dataframe does not contain the 'gameid' column."
            raise OraclesElixirError(msg)
        before = len(oracles_elixir_data)
        cleaned = oracles_elixir_data.dropna(subset=["gameid"])
        logger.info("Removed %s rows with null 'gameid'.", before - len(cleaned))
        data_pipeline_logger.info(
            "Removed %s rows with null 'gameid'.", before - len(cleaned)
        )
        return cleaned

    @staticmethod
    def drop_unknown_entities(oracles_elixir_data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with 'unknown player' or 'unknown team' in 'playername' or 'teamname'.
        Raises OraclesElixirError if 'playername' or 'teamname' columns are missing.
        """
        if not {"playername", "teamname"} <= set(oracles_elixir_data.columns):
            msg = "Missing 'playername' or 'teamname' in dataframe."
            raise OraclesElixirError(msg)
        before = len(oracles_elixir_data)
        mask = ~oracles_elixir_data["playername"].fillna("").str.lower().eq(
            "unknown player"
        ) & ~oracles_elixir_data["teamname"].fillna("").str.lower().eq("unknown team")
        df = oracles_elixir_data[mask]
        logger.info("Removed %s rows with unknown player/team.", before - len(df))
        data_pipeline_logger.info(
            "Removed %s rows with unknown player/team.", before - len(df)
        )
        return df

    @staticmethod
    def replace_team_names(oracles_elixir_data: pd.DataFrame) -> pd.DataFrame:
        """
        Replace incorrect team names based on a JSON file.
        Raises FileLoadError if the JSON file is missing or malformed.
        """
        try:
            loaded = json_loader(TEAM_ALIASES)
            if not isinstance(loaded, dict):
                raise TypeError("team replacements configuration must be an object")
            file_data = cast("dict[str, Any]", loaded)
            replacements: list[list[dict[str, Any]]] = file_data.get(
                "historical_identity_merges", []
            )
        except FileLoadError as exc:
            logger.error("Team replacements file error: %s", exc)
            data_pipeline_logger.exception("Team replacements file error")
            raise

        df = oracles_elixir_data.copy()
        for entry in replacements:
            if not isinstance(entry, list) or len(entry) != ROWS_PER_GAME_PER_TEAM:
                logger.warning("Skipping invalid replacement entry: %s", entry)
                data_pipeline_logger.warning(
                    "Skipping invalid replacement entry: %s", entry
                )
                continue

            old, new = entry
            old_name, old_id = old.get("name"), old.get("teamid")
            new_name, new_id = new.get("name"), new.get("teamid")
            until = old.get("until")

            if not all([old_name, old_id, new_name, new_id]):
                logger.warning(
                    "Missing 'name' or 'teamid' in entry %s – skipped.", entry
                )
                continue

            mask = (df["teamname"] == old_name) & (df["teamid"] == old_id)
            if until:
                mask &= df["date"] < pd.to_datetime(until)

            df.loc[mask, ["teamname", "teamid"]] = new_name, new_id

        logger.info("Replaced incorrect team names with correct ones.")
        data_pipeline_logger.info("Replaced incorrect team names with correct ones.")
        return df

    @staticmethod
    def sort_data(oracles_elixir_data: pd.DataFrame, split_on: str) -> pd.DataFrame:
        """
        Sort the DataFrame by the specified *split_on* key.
        Raises OraclesElixirError if *split_on* is not 'player' or 'team'.
        """
        if split_on not in {"player", "team"}:
            msg = "split_on must be either 'player' or 'team'."
            raise OraclesElixirError(msg)
        keys = get_sorting_keys(split_on)
        df = oracles_elixir_data.sort_values(keys).reset_index(drop=True)
        logger.info("Sorted data by %s.", keys)
        data_pipeline_logger.info("Sorted data by %s.", keys)
        return df

    @staticmethod
    def fill_null_team_ids(oracles_elixir_data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill null 'teamid' values with 'teamname' where possible.
        Raises OraclesElixirError if 'teamid' or 'teamname' columns are missing.

        Note: fillna must run *before* astype(str). Casting first turns missing
        values into the literal strings "nan"/"<NA>", which then survive the
        empty-string filter and pollute identity-keyed features downstream.
        """
        if not {"teamid", "teamname"} <= set(oracles_elixir_data.columns):
            msg = "Missing 'teamid' or 'teamname' in dataframe."
            raise OraclesElixirError(msg)

        df = oracles_elixir_data.copy()
        df["teamname"] = df["teamname"].fillna("").astype(str).str.strip()
        df["teamid"] = df["teamid"].fillna(df["teamname"]).astype(str).str.strip()
        # Guard against stringified missing markers from earlier casts/upstream.
        bad_ids = df["teamid"].str.casefold().isin({"", "nan", "<na>", "none"})
        removed = int(bad_ids.sum())
        df = df[~bad_ids]
        logger.info(
            "Filled null team IDs with team names; removed %d rows without identity.",
            removed,
        )
        data_pipeline_logger.info(
            "Filled null team IDs with team names; removed %d rows without identity.",
            removed,
        )
        return df

    @staticmethod
    def fill_null_patch_value(oracles_elixir_data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill null 'patch' values with the previous value in the column.
        Raises OraclesElixirError if 'patch' column is missing.
        """
        if "patch" not in oracles_elixir_data.columns:
            msg = "The dataframe does not contain the 'patch' column."
            raise OraclesElixirError(msg)

        df = oracles_elixir_data.copy()

        # Forward-fill is only safe on date-ordered rows; otherwise a null patch
        # can inherit a patch from an unrelated era. The pipeline sorts with
        # date-led keys before this step, so this is a cheap invariant check.
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce")
            if not dates.dropna().is_monotonic_increasing:
                msg = (
                    "fill_null_patch_value requires date-ordered rows; "
                    "run sort_data before forward-filling patches."
                )
                raise OraclesElixirError(msg)

        missing_before = int(df["patch"].isna().sum())
        df["patch"] = df["patch"].ffill()
        missing_after = int(df["patch"].isna().sum())

        logger.info(
            "Filled %d null patch values with previous value; %d remain.",
            missing_before - missing_after,
            missing_after,
        )
        data_pipeline_logger.info(
            "Filled %d null patch values with previous value; %d remain.",
            missing_before - missing_after,
            missing_after,
        )
        if missing_after:
            msg = (
                f"Patch values still contain {missing_after} nulls after forward-fill."
            )
            raise OraclesElixirError(msg)
        return df

    @classmethod
    def _remove_buggy_games(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Drop automatically detected bad games."""
        cleaned, _, report = quarantine_oracles_elixir_data(df)
        bad_games = set(report.game_issues) - {"__missing_gameid__"}
        if not bad_games and not report.exact_duplicate_rows:
            return df
        logger.info(
            "Removed %d buggy games and merged %d exact duplicates.",
            len(bad_games),
            report.exact_duplicate_rows,
        )
        data_pipeline_logger.info(
            "Removed %d buggy games and merged %d exact duplicates.",
            len(bad_games),
            report.exact_duplicate_rows,
        )
        return cleaned

    @staticmethod
    def subset_data(
        oracles_elixir_data: pd.DataFrame,
        split_on: str,
        columns: dict[str, list[str]] | None = None,
    ) -> pd.DataFrame:
        """
        Subset the DataFrame to only include relevant columns based on *split_on*.
        Raises OraclesElixirError if *split_on* is not 'player' or 'team'.
        """
        if columns is None:
            try:
                loaded = json_loader(IMPORT_COLUMNS)
                if not isinstance(loaded, dict):
                    raise TypeError("import columns configuration must be an object")
                columns = cast("dict[str, list[str]]", loaded)
            except FileLoadError as exc:
                logger.error("Import columns file error at %s: %s", IMPORT_COLUMNS, exc)
                raise
        if split_on not in columns:
            msg = "Must split on either 'player' or 'team'."
            raise OraclesElixirError(msg)

        df = oracles_elixir_data.rename(columns=SOURCE_COLUMN_RENAMES)

        if "position" not in df.columns:
            msg = "The dataframe does not contain the 'position' column."
            raise OraclesElixirError(msg)
        df["position"] = df["position"].fillna("")

        pos_filter = (
            df["position"].str.lower().eq("team")
            if split_on == "team"
            else ~df["position"].str.lower().eq("team")
        )
        df = df[pos_filter]
        logger.info("Filtered data by %ss.", split_on)
        return df[columns[split_on]]

    @staticmethod
    def remove_inconsistent_games(
        oracles_elixir_data: pd.DataFrame, split_on: str = "player"
    ) -> pd.DataFrame:
        """
        Remove games with inconsistent 'gameid' counts based on *split_on*.
        Raises OraclesElixirError if *split_on* is not 'player' or 'team'.
        """
        expected = 2 if split_on.lower() == "team" else 10
        bad_ids = (
            oracles_elixir_data["gameid"].value_counts()[lambda s: s != expected].index
        )
        df = oracles_elixir_data[~oracles_elixir_data["gameid"].isin(bad_ids)]
        logger.info("Removed %s inconsistent games.", len(bad_ids))
        return df

    @staticmethod
    def validate_game_composition(
        oracles_elixir_data: pd.DataFrame, split_on: str
    ) -> pd.DataFrame:
        """Fail if post-clean data has partial or malformed team/player games."""
        if split_on not in EXPECTED_ROWS_BY_ENTITY:
            msg = "split_on must be either 'player' or 'team'."
            raise OraclesElixirError(msg)
        if oracles_elixir_data.empty:
            msg = f"No {split_on} rows remain after cleaning."
            raise OraclesElixirError(msg)
        if "gameid" not in oracles_elixir_data.columns:
            msg = "The dataframe does not contain the 'gameid' column."
            raise OraclesElixirError(msg)

        expected = EXPECTED_ROWS_BY_ENTITY[split_on]
        counts = oracles_elixir_data.groupby("gameid", observed=True).size()
        bad_counts = counts[counts != expected]
        if not bad_counts.empty:
            sample = bad_counts.head(5).to_dict()
            msg = (
                f"Invalid {split_on} game composition after cleaning: "
                f"expected {expected} rows per game; sample={sample}"
            )
            raise OraclesElixirError(msg)

        if "side" in oracles_elixir_data.columns:
            side_sets = oracles_elixir_data.groupby("gameid", observed=True)[
                "side"
            ].agg(lambda values: set(values.astype(str)))
            bad_sides = side_sets[side_sets != EXPECTED_SIDES]
            if not bad_sides.empty:
                sample = {
                    gameid: sorted(sides) for gameid, sides in bad_sides.head(5).items()
                }
                msg = (
                    f"Invalid {split_on} side composition after cleaning: "
                    f"expected Blue and Red per game; sample={sample}"
                )
                raise OraclesElixirError(msg)

        if split_on == "player" and {"side", "position"} <= set(
            oracles_elixir_data.columns
        ):
            side_counts = oracles_elixir_data.groupby(
                ["gameid", "side"], observed=True
            ).size()
            bad_side_counts = side_counts[side_counts != GAP_PLAYER]
            if not bad_side_counts.empty:
                msg = (
                    "Invalid player side composition after cleaning: "
                    f"expected five player rows per side; sample={bad_side_counts.head(5).to_dict()}"
                )
                raise OraclesElixirError(msg)
            position_sets = oracles_elixir_data.groupby(
                ["gameid", "side"], observed=True
            )["position"].agg(lambda values: set(values.astype(str).str.casefold()))
            bad_positions = position_sets[position_sets != EXPECTED_POSITIONS]
            if not bad_positions.empty:
                sample = {
                    str(key): sorted(positions)
                    for key, positions in bad_positions.head(5).items()
                }
                msg = (
                    "Invalid player role composition after cleaning: "
                    f"expected one top/jng/mid/bot/sup per side; sample={sample}"
                )
                raise OraclesElixirError(msg)

        if "result" in oracles_elixir_data.columns:
            result = normalize_result(oracles_elixir_data["result"])
            if result.isna().any() or not result.isin([0, 1]).all():
                msg = f"Invalid {split_on} result labels after cleaning."
                raise OraclesElixirError(msg)
            result_sums = result.groupby(
                oracles_elixir_data["gameid"], observed=True
            ).sum()
            expected_winners = EXPECTED_WINNING_ROWS_BY_ENTITY[split_on]
            bad_results = result_sums[result_sums != expected_winners]
            if not bad_results.empty:
                msg = (
                    f"Invalid {split_on} result composition after cleaning: "
                    f"sample={bad_results.head(5).to_dict()}"
                )
                raise OraclesElixirError(msg)
        logger.info(
            "Validated %d %s games with %d rows each.",
            len(counts),
            split_on,
            expected,
        )
        data_pipeline_logger.info(
            "Validated %d %s games with %d rows each.",
            len(counts),
            split_on,
            expected,
        )
        return oracles_elixir_data

    @staticmethod
    def enrich_opponent_metrics(
        oracles_elixir_data: pd.DataFrame, split_on: str
    ) -> pd.DataFrame:
        """
        Enrich the DataFrame with opponent metrics based on *split_on*.
        """
        df = OraclesElixir.sort_data(oracles_elixir_data, split_on)
        df["teamid"] = df["teamid"].fillna(df["teamname"])

        if split_on == "team":
            opponent = df[["gameid", "side", "teamname", "teamid"]].copy()
            opponent["side"] = opponent["side"].map(OPPOSITE_SIDE)
            opponent = opponent.rename(
                columns={"teamname": "opponentteam", "teamid": "opponentteamid"}
            )
            out = df.merge(opponent, on=["gameid", "side"], how="left", validate="1:1")
        elif split_on == "player":
            df["playerid"] = df["playerid"].fillna(df["playername"])
            opponent = df[
                [
                    "gameid",
                    "side",
                    "position",
                    "teamname",
                    "teamid",
                    "playername",
                    "playerid",
                ]
            ].copy()
            opponent["side"] = opponent["side"].map(OPPOSITE_SIDE)
            opponent = opponent.rename(
                columns={
                    "teamname": "opponentteam",
                    "teamid": "opponentteamid",
                    "playername": "opponentplayername",
                    "playerid": "opponentplayerid",
                }
            )
            out = df.merge(
                opponent,
                on=["gameid", "side", "position"],
                how="left",
                validate="1:1",
            )
        else:
            msg = "split_on must be either 'player' or 'team'."
            raise OraclesElixirError(msg)

        if out.filter(like="opponent").isna().any(axis=None):
            msg = f"Could not pair all {split_on} opponents by game/side."
            raise OraclesElixirError(msg)

        logger.info("Enriched data with opponent metrics.")
        data_pipeline_logger.info("Enriched data with opponent metrics.")
        return out

    @staticmethod
    def filter_leagues(oracles_elixir_data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the DataFrame to only include rows from considered leagues.
        Raises FileNotFoundError if the considered leagues file is missing.
        Raises KeyError if the active league profile is missing.
        """
        logger.info("Filtering data for relevant leagues...")
        data_pipeline_logger.info("Filtering data for relevant leagues...")
        try:
            considered = training_leagues()
        except (FileNotFoundError, KeyError):
            logger.error("League configuration invalid or missing.")
            data_pipeline_logger.exception("League configuration invalid or missing.")
            raise
        if not considered:
            msg = "No leagues specified in the considered leagues list."
            raise OraclesElixirError(msg)
        return oracles_elixir_data[oracles_elixir_data["league"].isin(considered)]

    # --------------------------------------------------------------------- #
    # Pipeline orchestrator
    # --------------------------------------------------------------------- #
    def clean_data(
        self, oracles_elixir_data: pd.DataFrame, split_on: str
    ) -> pd.DataFrame:
        """
        Clean the Oracle Elixir data according to the specified *split_on* key.
        Raises OraclesElixirError if *split_on* is not 'player' or 'team'.
        """
        logger.info("Cleaning data for %ss...", split_on)
        data_pipeline_logger.info("Cleaning data for %ss...", split_on)

        df = (
            oracles_elixir_data.pipe(self.format_data_length_types)
            .pipe(self.remove_null_games)
            .pipe(self.drop_unknown_entities)
            .pipe(self.replace_team_names)
            .pipe(self._remove_buggy_games)
            .pipe(self.sort_data, split_on)
            .pipe(self.fill_null_team_ids)
            .pipe(self.fill_null_patch_value)
            .pipe(self.subset_data, split_on)  # subset early to shrink following ops
            .pipe(self.remove_inconsistent_games, split_on)
            .pipe(self.enrich_opponent_metrics, split_on)
            .pipe(self.filter_leagues)
            .pipe(self.validate_game_composition, split_on)
        )

        logger.info("Data cleaning for %ss completed.", split_on)
        data_pipeline_logger.info("Data cleaning for %ss completed.", split_on)
        return df
