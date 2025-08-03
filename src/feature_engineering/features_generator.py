"""
Features Generator

This script contains the `FeatureGenerator` class, which is used to generate new features for player and team data.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.logger import instantiate_logger, logger

data_pipeline_logger = instantiate_logger("data_pipeline")


@dataclass
class FeatureGenerator:
    """A class to generate new features for player and team data."""

    @staticmethod
    def compute_key_stats(data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute key statistics for the player data efficiently.

        Args:
            data (pd.DataFrame): The player data.

        Returns:
            pd.DataFrame: The player data with key statistics added.

        """
        logger.info("Computing key statistics...")
        data_pipeline_logger.info("Computing key statistics...")
        required_columns = {
            "gameid",
            "side",
            "kills",
            "assists",
            "deaths",
            "damagetochampions",
            "damagetakenperminute",
            "damagemitigatedperminute",
            "gamelength",
            "totalgold",
            "total_cs",
            "wpm",
            "wcpm",
        }
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            data_pipeline_logger.error(f"Missing required columns: {missing_columns}")
            msg = f"Missing required columns: {missing_columns}"
            raise ValueError(msg)

        data = data.copy()
        enemy_team_stats = FeatureGenerator._aggregate_enemy_team_stats(data)

        data = data.merge(enemy_team_stats, on=["gameid", "side"], how="left")

        data = FeatureGenerator._calculate_ratios(data)

        logger.info("Key statistics computation completed.")
        data_pipeline_logger.info("Key statistics computation completed.")
        return data

    @staticmethod
    def _aggregate_enemy_team_stats(data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate enemy team statistics.

        Args:
            data (pd.DataFrame): The player data.

        Returns:
            pd.DataFrame: Aggregated enemy team statistics.

        """
        logger.info("Aggregating enemy team statistics...")
        data_pipeline_logger.info("Aggregating enemy team statistics...")
        enemy_team_stats = (
            data.groupby(["gameid", "side"])
            .agg(
                enemyTeamKills=("kills", "sum"),
                enemyTeamDeaths=("deaths", "sum"),
                enemyTeamDamages=("damagetochampions", "sum"),
                enemyTeamGolds=("totalgold", "sum"),
                enemyTeamTotalCS=("total_cs", "sum"),
                enemyTeamWardPlaced=("wpm", "sum"),
                enemyTeamWardKilled=("wcpm", "sum"),
            )
            .reset_index()
        )

        side_mapping = {"Blue": "Red", "Red": "Blue"}
        enemy_team_stats["side"] = enemy_team_stats["side"].map(side_mapping)

        if enemy_team_stats["side"].isnull().any():
            logger.error("Duplicate rows found after mapping side values.")
            data_pipeline_logger.error(
                "Duplicate rows found after mapping side values."
            )
            enemy_team_stats = enemy_team_stats.dropna(subset=["side"])

        return enemy_team_stats

    @staticmethod
    def _calculate_ratios(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various ratio features.

        Args:
            data (pd.DataFrame): The player data with enemy team statistics merged.

        Returns:
            pd.DataFrame: DataFrame with ratio features added.

        """
        logger.info("Calculating ratio features...")
        data_pipeline_logger.info("Calculating ratio features...")
        zero_columns = [
            "enemyTeamKills",
            "enemyTeamDeaths",
            "enemyTeamDamages",
            "enemyTeamGolds",
            "enemyTeamWardPlaced",
        ]
        data[zero_columns] = data[zero_columns].replace(0, np.nan)

        data["ka_ratio"] = (data["kills"] + data["assists"]) / (
            data["enemyTeamKills"] + data["kills"] + data["assists"]
        )
        data["d_ratio"] = data["deaths"] / data["enemyTeamDeaths"]
        data["damages_ratio"] = data["damagetochampions"] / data["enemyTeamDamages"]
        data["damage_tanked_ratio"] = (
            data["damagetakenperminute"] * data["gamelength"]
        ) / data["enemyTeamDamages"]
        data["damage_mitigated_ratio"] = (
            data["damagemitigatedperminute"] * data["gamelength"]
        ) / data["enemyTeamDamages"]
        data["gold_ratio"] = data["totalgold"] / data["enemyTeamGolds"]
        data["cs_to_gold_ratio"] = data["total_cs"] / data["enemyTeamGolds"]
        data["wards_placed_ratio"] = (data["wpm"] * data["gamelength"]) / data[
            "enemyTeamWardPlaced"
        ]
        data["wards_killed_ratio"] = (data["wcpm"] * data["gamelength"]) / data[
            "enemyTeamWardPlaced"
        ]

        logger.info("Ratio features computed.")
        data_pipeline_logger.info("Ratio features computed.")
        return data

    @staticmethod
    def compute_win_loss_metrics(data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute average kills, deaths, and game length for wins and losses, grouped by playerid, season, and patch.

        Args:
            data (pd.DataFrame): The player data with a 'result' column indicating 1 (Win) or 0 (Lose).

        Returns:
            pd.DataFrame: DataFrame with win/loss metrics added.

        """
        logger.info("Computing win/loss metrics...")
        data_pipeline_logger.info("Computing win/loss metrics...")

        if "result" not in data.columns:
            logger.error("'result' column is required to compute win/loss metrics.")
            data_pipeline_logger.error(
                "'result' column is required to compute win/loss metrics."
            )
            msg = "'result' column is missing."
            raise ValueError(msg)

        data = data.copy()
        win_metrics = (
            data.loc[data["result"] == 1]
            .groupby(["playerid", "season"], sort=True)
            .agg(
                kills_per_season_win=("kills", "mean"),
                deaths_per_season_win=("deaths", "mean"),
                avg_gamelength_season_win=("gamelength", "mean"),
            )
        )

        loss_metrics = (
            data.loc[data["result"] == 0]
            .groupby(["playerid", "season"], sort=True)
            .agg(
                kills_per_season_loss=("kills", "mean"),
                deaths_per_season_loss=("deaths", "mean"),
                avg_gamelength_season_loss=("gamelength", "mean"),
            )
        )

        combined_metrics = win_metrics.join(loss_metrics, how="outer").reset_index()

        data = data.set_index(["playerid", "season"], drop=False)
        data = data.join(combined_metrics.set_index(["playerid", "season"]), how="left")
        data = data.reset_index(drop=True)

        # Group wins by (playerid, patch)
        win_patch = (
            data.loc[data["result"] == 1]
            .groupby(["playerid", "patch"], sort=True)
            .agg(
                kills_per_patch_win=("kills", "mean"),
                deaths_per_patch_win=("deaths", "mean"),
                avg_gamelength_patch_win=("gamelength", "mean"),
            )
        )

        # Group losses by (playerid, patch)
        loss_patch = (
            data.loc[data["result"] == 0]
            .groupby(["playerid", "patch"], sort=True)
            .agg(
                kills_per_patch_loss=("kills", "mean"),
                deaths_per_patch_loss=("deaths", "mean"),
                avg_gamelength_patch_loss=("gamelength", "mean"),
            )
        )

        patch_metrics = win_patch.join(loss_patch, how="outer")

        # Match original indexing approach for patch
        patch_metrics = patch_metrics.reset_index()

        data = data.set_index(["playerid", "patch"], drop=False)
        data = data.join(patch_metrics.set_index(["playerid", "patch"]), how="left")
        data = data.reset_index(drop=True)

        # We'll place the new columns at the end in the same order they were added in the apply() code:
        metric_cols = [
            "kills_per_season_win",
            "deaths_per_season_win",
            "avg_gamelength_season_win",
            "kills_per_season_loss",
            "deaths_per_season_loss",
            "avg_gamelength_season_loss",
            "kills_per_patch_win",
            "deaths_per_patch_win",
            "avg_gamelength_patch_win",
            "kills_per_patch_loss",
            "deaths_per_patch_loss",
            "avg_gamelength_patch_loss",
        ]
        # Ensure only columns that actually exist remain in the list
        metric_cols = [c for c in metric_cols if c in data.columns]

        # Move metric_cols to the end in the correct order
        non_metric_cols = [c for c in data.columns if c not in metric_cols]
        final_col_order = non_metric_cols + metric_cols

        # Sort the rows by (playerid, season, patch) to mimic the apply() row order
        data = data.sort_values(by=["playerid", "season", "patch"], ascending=True)
        data = data.loc[:, final_col_order].reset_index(drop=True)

        logger.info("Win/loss metrics computation completed.")
        data_pipeline_logger.info("Win/loss metrics computation completed.")
        return data

    @staticmethod
    def generate_new_player_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate new features for the given player data.

        Enhancements include team kills calculations and various efficiency metrics.

        Args:
            data (pd.DataFrame): The player data.

        Returns:
            pd.DataFrame: The player data with new features added.

        """
        logger.info("Generating new player features...")
        data_pipeline_logger.info("Generating new player features...")

        required_columns = {
            "teamid",
            "gameid",
            "position",
            "kills",
            "assists",
            "deaths",
            "totalgold",
            "gamelength",
            "total_cs",
            "patch",
            "result",
            "playerid",
        }
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            data_pipeline_logger.error(f"Missing required columns: {missing_columns}")
            msg = f"Missing required columns: {missing_columns}"
            raise ValueError(msg)

        data = data.copy()
        data["season"] = data["patch"].astype(str).str.split(".").str[0]

        data["team_kills"] = data.groupby(["gameid", "teamid"])["kills"].transform(
            "sum"
        )

        data["kda"] = (data["kills"] + data["assists"]) / data["deaths"].replace(0, 1)
        data["gold_efficiency"] = data["totalgold"] / data["gamelength"].replace(
            0, np.nan
        )
        data["xp_efficiency"] = data["total_cs"] / data["gamelength"].replace(0, np.nan)
        data["kill_participation"] = (data["kills"] + data["assists"]) / data[
            "team_kills"
        ].replace(0, np.nan)

        data = FeatureGenerator.compute_key_stats(data)
        data = FeatureGenerator.compute_win_loss_metrics(data)

        position_dummies = pd.get_dummies(
            data["position"], prefix="position"
        ).reset_index(drop=True)
        data = data.reset_index(drop=True)
        data = pd.concat([data, position_dummies], axis=1)

        logger.info("Player features generation completed.")
        data_pipeline_logger.info("Player features generation completed.")
        return data

    @staticmethod
    def generate_new_team_features(
        data: pd.DataFrame,
        *,
        recent_window: int = 5,
        add_recent: bool = True,
    ) -> pd.DataFrame:
        """
        Enrich each team-game row with leak-free historical context **without**
        the long-horizon team mean that would overweight old matches.

        Added columns
        -------------
        season
        total_kills, total_towers
        team_season_avg_gamelength
        team_patch_avg_gamelength
        (optional) team_recent{N}_avg_gamelength
        patch_avg_gamelength
        season_avg_gamelength
        """
        _check_required(
            data,
            required={
                "date",
                "patch",
                "teamid",
                "gameid",
                "gamelength",
                "kills",
                "towers",
            },
        )

        df = data.copy()
        df["season"] = df["patch"].astype(str).str.split(".").str[0]

        # ── Game-level context ───────────────────────────────────────────────
        df = df.merge(
            df.groupby("gameid", observed=True)
            .agg(total_kills=("kills", "sum"), total_towers=("towers", "sum"))
            .reset_index(),
            on="gameid",
            how="left",
        )

        # ── Team cumulative mean, reset each *season* and *patch* ────────────
        for grp, pfx in (
            (["teamid", "season"], "team_season_avg_"),
            (["teamid", "patch"], "team_patch_avg_"),
        ):
            df = _add_expanding_mean(
                df,
                group_cols=list(grp),
                value_cols=["gamelength"],
                prefix=pfx,
            )

        # ── Optional: recent-N rolling mean ───────────────────────────────────
        if add_recent and recent_window > 0:
            df = _add_rolling_mean(
                df,
                group_cols=["teamid"],
                value_cols=["gamelength"],
                prefix=f"team_recent{recent_window}_avg_",
                window=recent_window,
            )

        # ── League-wide patch / season expanding means (deduped) ─────────────
        game_level = (
            df[["gameid", "date", "patch", "season", "gamelength"]]
            .sort_values("date")
            .drop_duplicates(subset="gameid", keep="first")
        )

        for gcol, pfx in (("patch", "patch_avg_"), ("season", "season_avg_")):
            game_level = _add_expanding_mean(
                game_level,
                group_cols=[gcol],
                value_cols=["gamelength"],
                prefix=pfx,
                sort_also_by=["date"],
            )

        return df.merge(
            game_level[["gameid", "patch_avg_gamelength", "season_avg_gamelength"]],
            on="gameid",
            how="left",
        )


# ────────────────────────────────────────────────────────────────────────────
# Internal utilities
# ────────────────────────────────────────────────────────────────────────────
def _check_required(df: pd.DataFrame, *, required: Iterable[str]) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        msg = f"Missing required columns: {', '.join(sorted(missing))}"
        logger.error(msg)
        raise ValueError(msg)


def _add_expanding_mean(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    value_cols: list[str],
    prefix: str,
    sort_also_by: list[str] | None = None,
) -> pd.DataFrame:
    sort_keys = list(group_cols) + (sort_also_by or [])
    df = df.sort_values(sort_keys, kind="mergesort")

    for col in value_cols:
        grp = df.groupby(group_cols, observed=True)[col]
        df[f"{prefix}{col}"] = (
            grp.cumsum().shift().div(grp.cumcount().replace(0, np.nan))
        )

    return df


def _add_rolling_mean(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    value_cols: list[str],
    prefix: str,
    window: int,
) -> pd.DataFrame:
    """Rolling mean of the *previous* `window` rows inside each group."""
    df = df.sort_values([*group_cols, "date"], kind="mergesort")

    for col in value_cols:
        rolled = (
            df.groupby(group_cols, observed=True)[col]
            .shift()  # → leak-free
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=group_cols, drop=True)
        )
        df[f"{prefix}{col}"] = rolled

    return df
