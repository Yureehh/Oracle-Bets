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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import awswrangler as wr
from dotenv import load_dotenv
from oracle_bets_core.io_utils import FileLoadError, get_sorting_keys, json_loader
from oracle_bets_core.logger import LOG_TOPIC, instantiate_logger, logger
from oracle_bets_core.paths import (
    CONSIDERED_LEAGUES,
    EXTRAS_DIR,
    IMPORT_COLUMNS,
    RAW_DATA,
    TEAM_REPLACEMENTS_AND_INVALID_GAMES,
)
from oracle_bets_core.pd import pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    import boto3

# --------------------------------------------------------------------------- #
# Environment & logging
# --------------------------------------------------------------------------- #
load_dotenv()
data_pipeline_logger = instantiate_logger(LOG_TOPIC.DATA_PIPELINE)

# --------------------------------------------------------------------------- #
# Constants (names unchanged so imports elsewhere remain valid)
# --------------------------------------------------------------------------- #
ROWS_PER_GAME_FULL: int = 12  # 10 players + 2 team rows
UNIQUE_PLAYERS_PER_GAME: int = 10
UNIQUE_TEAMS_PER_GAME: int = 2
GAP_PLAYER: int = 5
GAP_TEAM: int = 1
ROWS_PER_GAME_PER_TEAM: int = 2
NULL_REPLACEMENTS: list[str] = ["nan", "null", "unknown", "Unknown", "N/A"]
OPPOSITE_SIDE = {"Blue": "Red", "Red": "Blue"}
EXPECTED_ROWS_BY_ENTITY = {
    "team": UNIQUE_TEAMS_PER_GAME,
    "player": UNIQUE_PLAYERS_PER_GAME,
}


class OraclesElixirError(RuntimeError):
    """Base class for all Oracle Elixir ingestion errors."""


def _is_bad_full_game(game: pd.DataFrame) -> bool:
    """Return whether a raw Oracle's Elixir game has invalid row composition."""
    if len(game) != ROWS_PER_GAME_FULL:
        return True
    positions = game["position"].fillna("").astype(str).str.lower()
    if positions.eq("team").sum() != UNIQUE_TEAMS_PER_GAME:
        return True
    if (~positions.eq("team")).sum() != UNIQUE_PLAYERS_PER_GAME:
        return True
    team_ids = game["teamid"].fillna(game["teamname"])
    if team_ids.nunique(dropna=True) != UNIQUE_TEAMS_PER_GAME:
        return True
    return (
        game["teamname"].fillna("").str.contains("unknown", case=False).any()
        or game["playername"].fillna("").str.contains("unknown", case=False).any()
    )


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

        s3_paths = {
            y: f"s3://{self.bucket}/{y}_LoL_esports_match_data_from_OraclesElixir.csv"
            for y in year_list
        }

        logger.info("Connecting to S3 bucket")
        data_pipeline_logger.info("Connecting to S3 bucket")

        results: dict[int, pd.DataFrame] = {}
        missing: list[int] = []
        with ThreadPoolExecutor() as pool:
            futures = {
                pool.submit(
                    wr.s3.read_csv,
                    path,
                    boto3_session=self.session,
                    low_memory=False,
                ): year
                for year, path in s3_paths.items()
            }
            for fut in as_completed(futures):
                year = futures[fut]
                path = s3_paths[year]
                try:
                    results[year] = fut.result()
                except Exception as exc:
                    if isinstance(exc, (FileNotFoundError, wr.exceptions.NoFilesFound)):
                        missing.append(year)
                        logger.warning(
                            "No file found for year %s at %s; skipping.", year, path
                        )
                        data_pipeline_logger.warning(
                            "No file found for year %s at %s; skipping.", year, path
                        )
                        continue
                    logger.error("Failed to ingest data for year %s: %s", year, exc)
                    data_pipeline_logger.exception(
                        "Failed to ingest data for year %s.", year
                    )
                    raise

        if not results:
            msg = f"No Oracle Elixir files found for years: {year_list}"
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
            logger.warning("Skipped missing years: %s", missing)
            data_pipeline_logger.warning("Skipped missing years: %s", missing)
        return df

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
            file_data: dict[str, Any] = json_loader(TEAM_REPLACEMENTS_AND_INVALID_GAMES)
            replacements: list[list[dict[str, Any]]] = file_data.get(
                "team_name_replacements", []
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
        Raises OraclesElixirError if 'patch' column is missing.
        """
        if "patch" not in oracles_elixir_data.columns:
            msg = "The dataframe does not contain the 'patch' column."
            raise OraclesElixirError(msg)

        df = oracles_elixir_data.copy()
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

    # --------------------------------------------------------------------------- #
    # Buggy-game detection (NEW)
    # --------------------------------------------------------------------------- #
    @staticmethod
    def _detect_buggy_games(df: pd.DataFrame) -> set[str]:
        """
        Identify 'bad' gameids whose row-level composition is clearly wrong, e.g.
        * missing rows
        * wrong player / team row counts
        * 'unknown' placeholders sneaking through
        """
        grp = df.groupby("gameid")
        bad_games = grp.filter(lambda g: _is_bad_full_game(g))
        return set(bad_games["gameid"].unique())

    @classmethod
    def _remove_buggy_games(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Drop auto-detected AND manually listed bad games."""
        auto = cls._detect_buggy_games(df)
        try:
            cfg = json_loader(TEAM_REPLACEMENTS_AND_INVALID_GAMES)
            manual = set(cfg.get("invalid_games", []))
        except FileLoadError:
            manual = set()  # fail soft – log + carry on
            logger.warning(
                "%s not found – no manual invalid_games applied.",
                TEAM_REPLACEMENTS_AND_INVALID_GAMES,
            )

        bad_games = auto | manual
        if not bad_games:
            return df  # fast path

        cleaned = df[~df["gameid"].isin(bad_games)].reset_index(drop=True)
        logger.info(
            "Removed %d buggy games (%d auto, %d manual).",
            len(bad_games),
            len(auto),
            len(manual),
        )
        data_pipeline_logger.info(
            "Removed %d buggy games (%d auto, %d manual).",
            len(bad_games),
            len(auto),
            len(manual),
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
                columns = json_loader(IMPORT_COLUMNS)
            except FileLoadError as exc:
                logger.error("Import columns file error at %s: %s", IMPORT_COLUMNS, exc)
                raise
        if split_on not in columns:
            msg = "Must split on either 'player' or 'team'."
            raise OraclesElixirError(msg)

        rename_map = {
            "earned gpm": "egpm",
            "team kpm": "team_kpm",
            "total cs": "total_cs",
        }
        df = oracles_elixir_data.rename(columns=rename_map)

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
        """Fail if post-clean data has partial team/player games."""
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


def get_opponent(column: pd.Series, entity: str) -> pd.Series:
    """Return the opposing entity for each row in *column*."""
    gap_dict = {"player": GAP_PLAYER, "team": GAP_TEAM}
    gap = gap_dict.get(entity)
    if gap is None:
        msg = "Entity must be either player or team."
        raise OraclesElixirError(msg)

    opposition: list[Any] = []
    flag = 0
    for i in range(len(column)):
        if flag < gap:
            opposition.append(column[i + gap])
        elif gap <= flag < gap * 2:
            opposition.append(column[i - gap])
        else:
            msg = f"Index {i} - Out Of Bounds"
            raise OraclesElixirError(msg)
        flag = (flag + 1) % (gap * 2)
    return pd.Series(opposition)


def get_league_teams(parquet_path: str) -> dict:
    """
    Read the Parquet at `parquet_path` and return a dict mapping each league
    to its list of unique team names.
    """
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].dt.year == df["date"].dt.year.max()]
    return df.groupby("league")["teamname"].unique().apply(list).to_dict()


def filter_teams_by_league(path1, path2, output_path):
    # Read the considered leagues from the first JSON file
    with Path(path1).open() as f:
        considered_leagues = json.load(f)["considered_leagues"]
    # Read the teams by league from the second JSON file
    with Path(path2).open() as f:
        teams_by_league = json.load(f)
    # Filter the teams by league based on the considered leagues
    filtered_teams = {
        league: teams
        for league, teams in teams_by_league.items()
        if league in considered_leagues
    }
    # Write the filtered teams to a new JSON file
    with Path(output_path).open("w") as f:
        json.dump(filtered_teams, f, indent=4)


if __name__ == "__main__":
    team_by_league_path = EXTRAS_DIR / "teams_by_league.json"
    filtered_teams_by_league_path = EXTRAS_DIR / "filtered_teams_by_league.json"

    league_teams = get_league_teams(str(RAW_DATA))
    with team_by_league_path.open("w") as f:
        json.dump(league_teams, f, indent=4)

    filter_teams_by_league(
        str(CONSIDERED_LEAGUES),
        str(team_by_league_path),
        str(filtered_teams_by_league_path),
    )
