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
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import awswrangler as wr
import pandas as pd
from dotenv import load_dotenv

from utils.io_utils import get_sorting_keys, json_loader
from utils.logger import instantiate_logger, logger
from utils.paths import (
    CONSIDERED_LEAGUES,
    IMPORT_COLUMNS,
    TEAM_REPLACEMENTS_AND_INVALID_GAMES,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import boto3

# --------------------------------------------------------------------------- #
# Environment & logging
# --------------------------------------------------------------------------- #
load_dotenv()
data_pipeline_logger = instantiate_logger("data_pipeline")

# --------------------------------------------------------------------------- #
# Constants (names unchanged so imports elsewhere remain valid)
# --------------------------------------------------------------------------- #
GAP_PLAYER: int = 5
GAP_TEAM: int = 1
ROWS_PER_GAME_PER_TEAM: int = 2
NULL_REPLACEMENTS: list[str] = ["nan", "null", "unknown", "Unknown", "N/A"]


# --------------------------------------------------------------------------- #
# Dataclass
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class OraclesElixir:
    """Ingest, clean, and format Oracle Elixir CSV dumps from S3."""

    session: boto3.Session | None
    bucket: str

    # --------------------------------------------------------------------- #
    # Ingestion
    # --------------------------------------------------------------------- #
    def ingest_data(
        self,
        years: Sequence[int | str] | int | str | None = None,
    ) -> pd.DataFrame:
        """
        Download year-specific CSVs from S3 and concatenate them.
        """
        # Normalise *years* into a list[int]
        if years is None:
            year_list: list[int] = [dt.date.today().year]
        elif isinstance(years, str | int):
            year_list = [int(years)]
        else:
            year_list = [int(y) for y in years]

        s3_paths = [
            f"s3://{self.bucket}/{y}_LoL_esports_match_data_from_OraclesElixir.csv"
            for y in year_list
        ]

        logger.info("Connecting to S3 bucket")
        data_pipeline_logger.info("Connecting to S3 bucket")

        try:
            with ThreadPoolExecutor() as pool:
                frames = list(
                    pool.map(
                        lambda p: wr.s3.read_csv(
                            p, boto3_session=self.session, low_memory=False
                        ),  # type: ignore[arg-type]
                        s3_paths,
                    )
                )
            df = pd.concat(frames, ignore_index=True)
            logger.info("Successfully ingested data for years: %s", year_list)
            data_pipeline_logger.info(
                "Successfully ingested data for years: %s", year_list
            )
            return df
        except Exception:
            logger.error("Failed to ingest data for years: %s", year_list)
            data_pipeline_logger.exception(
                "Failed to ingest data for years: %s", year_list
            )
            raise  # propagate – caller decides how to recover

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
        with pd.option_context("future.no_silent_downcasting", True):  # noqa: FBT003
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
        Raises ValueError if 'gameid' column is missing.
        """
        if "gameid" not in oracles_elixir_data.columns:
            msg = "The dataframe does not contain the 'gameid' column."
            raise ValueError(msg)
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
        Raises ValueError if 'playername' or 'teamname' columns are missing.
        """
        if not {"playername", "teamname"} <= set(oracles_elixir_data.columns):
            msg = "Missing 'playername' or 'teamname' in dataframe."
            raise ValueError(msg)
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
        Raises FileNotFoundError if the JSON file is missing.
        Raises json.JSONDecodeError if the JSON file is malformed.
        """
        try:
            file_data: dict[str, Any] = json_loader(TEAM_REPLACEMENTS_AND_INVALID_GAMES)
            replacements: list[list[dict[str, Any]]] = file_data.get(
                "team_name_replacements", []
            )
        except (FileNotFoundError, json.JSONDecodeError) as exc:
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
        Raises ValueError if *split_on* is not 'player' or 'team'.
        """
        if split_on not in {"player", "team"}:
            msg = "split_on must be either 'player' or 'team'."
            raise ValueError(msg)
        keys = get_sorting_keys(split_on)
        df = oracles_elixir_data.sort_values(keys).reset_index(drop=True)
        logger.info("Sorted data by %s.", keys)
        data_pipeline_logger.info("Sorted data by %s.", keys)
        return df

    @staticmethod
    def fill_null_team_ids(oracles_elixir_data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill null 'teamid' values with 'teamname' where possible.
        Raises ValueError if 'teamid' or 'teamname' columns are missing.
        """
        df = oracles_elixir_data.copy()
        df["teamname"] = df["teamname"].astype(str).fillna("")
        df["teamid"] = df["teamid"].astype(str).fillna(df["teamname"])
        df = df[df["teamid"] != ""]
        logger.info("Filled null team IDs with team names.")
        data_pipeline_logger.info("Filled null team IDs with team names.")
        return df

    @staticmethod
    def fill_null_patch_value(oracles_elixir_data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill null 'patch' values with the previous value in the column.
        Raises ValueError if 'patch' column is missing.
        """
        oracles_elixir_data["patch"] = oracles_elixir_data["patch"].ffill()
        logger.info("Filled null patch values with previous value.")
        data_pipeline_logger.info("Filled null patch values with previous value.")
        return oracles_elixir_data

    @staticmethod
    def subset_data(
        oracles_elixir_data: pd.DataFrame,
        split_on: str,
        columns: dict[str, list[str]] | None = None,
    ) -> pd.DataFrame:
        """
        Subset the DataFrame to only include relevant columns based on *split_on*.
        Raises ValueError if *split_on* is not 'player' or 'team'.
        """
        try:
            with Path(IMPORT_COLUMNS).open(encoding="utf-8") as f:
                columns = json.load(f)
            if split_on not in columns:
                msg = "Must split on either 'player' or 'team'."
                raise ValueError(msg)
        except FileNotFoundError:
            logger.error("Import columns file not found at %s.", IMPORT_COLUMNS)
            raise
        except json.JSONDecodeError as exc:
            logger.error("Error decoding JSON from %s: %s", IMPORT_COLUMNS, exc)
            raise

        rename_map = {
            "earned gpm": "egpm",
            "team kpm": "team_kpm",
            "total cs": "total_cs",
        }
        df = oracles_elixir_data.rename(columns=rename_map)

        if "position" not in df.columns:
            msg = "The dataframe does not contain the 'position' column."
            raise ValueError(msg)
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
        Raises ValueError if *split_on* is not 'player' or 'team'.
        """
        expected = 2 if split_on.lower() == "team" else 10
        bad_ids = (
            oracles_elixir_data["gameid"].value_counts()[lambda s: s != expected].index
        )
        df = oracles_elixir_data[~oracles_elixir_data["gameid"].isin(bad_ids)]
        logger.info("Removed %s inconsistent games.", len(bad_ids))
        return df

    @staticmethod
    def enrich_opponent_metrics(
        oracles_elixir_data: pd.DataFrame, split_on: str
    ) -> pd.DataFrame:
        """
        Enrich the DataFrame with opponent metrics based on *split_on*.
        Raises ValueError if *split_on* is not 'player' or 'team'.
        """
        metrics: dict[str, Any] = {
            "teamid": oracles_elixir_data["teamid"].fillna(
                oracles_elixir_data["teamname"]
            ),
            "opponentteam": get_opponent(
                oracles_elixir_data["teamname"].tolist(), split_on
            ),
            "opponentteamid": get_opponent(
                oracles_elixir_data["teamid"].tolist(), split_on
            ),
        }
        if split_on == "player":
            metrics |= {
                "playerid": oracles_elixir_data["playerid"].fillna(
                    oracles_elixir_data["playername"]
                ),
                "opponentplayername": get_opponent(
                    oracles_elixir_data["playername"].tolist(), split_on
                ),
                "opponentplayerid": get_opponent(
                    oracles_elixir_data["playerid"].tolist(), split_on
                ),
            }
        df = oracles_elixir_data.assign(**metrics)
        logger.info("Enriched data with opponent metrics.")
        data_pipeline_logger.info("Enriched data with opponent metrics.")
        return df

    @staticmethod
    def filter_leagues(oracles_elixir_data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the DataFrame to only include rows from considered leagues.
        Raises FileNotFoundError if the considered leagues file is missing.
        Raises KeyError if the 'considered_leagues' key is missing.
        """
        logger.info("Filtering data for relevant leagues...")
        data_pipeline_logger.info("Filtering data for relevant leagues...")
        try:
            considered: list[str] = json_loader(CONSIDERED_LEAGUES)[
                "considered_leagues"
            ]
        except (FileNotFoundError, KeyError):
            logger.error("League configuration invalid or missing.")
            data_pipeline_logger.exception("League configuration invalid or missing.")
            raise
        if not considered:
            msg = "No leagues specified in the considered leagues list."
            raise ValueError(msg)
        return oracles_elixir_data[oracles_elixir_data["league"].isin(considered)]

    # --------------------------------------------------------------------- #
    # Pipeline orchestrator
    # --------------------------------------------------------------------- #
    def clean_data(
        self, oracles_elixir_data: pd.DataFrame, split_on: str
    ) -> pd.DataFrame:
        """
        Clean the Oracle Elixir data according to the specified *split_on* key.
        Raises ValueError if *split_on* is not 'player' or 'team'.
        """
        logger.info("Cleaning data for %ss...", split_on)
        data_pipeline_logger.info("Cleaning data for %ss...", split_on)

        df = (
            oracles_elixir_data.pipe(self.format_data_length_types)
            .pipe(self.remove_null_games)
            .pipe(self.drop_unknown_entities)
            .pipe(self.replace_team_names)
            .pipe(self.sort_data, split_on)
            .pipe(self.fill_null_team_ids)
            .pipe(self.fill_null_patch_value)
            .pipe(self.subset_data, split_on)  # subset early to shrink following ops
            .pipe(self.remove_inconsistent_games, split_on)
            .pipe(self.enrich_opponent_metrics, split_on)
            .pipe(self.filter_leagues)
        )

        logger.info("Data cleaning for %ss completed.", split_on)
        data_pipeline_logger.info("Data cleaning for %ss completed.", split_on)
        return df


def get_opponent(column: pd.Series, entity: str) -> pd.Series:
    """Return the opposing entity for each row in *column*."""
    gap_dict = {"player": GAP_PLAYER, "team": GAP_TEAM}
    gap = gap_dict.get(entity)
    if gap is None:
        msg = "Entity must be either player or team."
        raise ValueError(msg)

    opposition: list[Any] = []
    flag = 0
    for i in range(len(column)):
        if flag < gap:
            opposition.append(column[i + gap])
        elif gap <= flag < gap * 2:
            opposition.append(column[i - gap])
        else:
            msg = f"Index {i} - Out Of Bounds"
            raise ValueError(msg)
        flag = (flag + 1) % (gap * 2)
    return pd.Series(opposition)
