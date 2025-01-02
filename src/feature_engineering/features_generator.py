"""
Features Generator

This script contains the `FeatureGenerator` class, which is used to generate new features for player and team data.
"""

from dataclasses import dataclass

import fireducks.pandas as pd
import numpy as np

from src.utils.logger import logger


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

        # Validate required columns
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
            logger.error(f"Missing required columns for key stats computation: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        data = data.copy()  # Avoid modifying the original DataFrame

        # Aggregate enemy team statistics
        enemy_team_stats = FeatureGenerator._aggregate_enemy_team_stats(data)

        # Merge aggregated stats back to the original data
        data = data.merge(enemy_team_stats, on=["gameid", "side"], how="left")

        # Compute key statistics
        data = FeatureGenerator._calculate_ratios(data)

        logger.info("Key statistics computation completed.")
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
        # Aggregate stats by gameid and side
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

        # Map sides to their opposing sides
        side_mapping = {"Blue": "Red", "Red": "Blue"}
        enemy_team_stats["side"] = enemy_team_stats["side"].map(side_mapping)

        unexpected_sides = enemy_team_stats["side"].isnull()
        if unexpected_sides.any():
            missing_sides = enemy_team_stats.loc[unexpected_sides, "side"].unique()
            logger.warning(f"Unexpected side values encountered when mapping sides: {missing_sides}")
            enemy_team_stats.dropna(subset=["side"], inplace=True)

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

        # Prevent division by zero by replacing zeros with np.nan
        data["enemyTeamKills"].replace(0, np.nan, inplace=True)
        data["enemyTeamDeaths"].replace(0, np.nan, inplace=True)
        data["enemyTeamDamages"].replace(0, np.nan, inplace=True)
        data["enemyTeamGolds"].replace(0, np.nan, inplace=True)
        data["enemyTeamWardPlaced"].replace(0, np.nan, inplace=True)

        # Compute ratios
        data["ka_ratio"] = (data["kills"] + data["assists"]) / (
            data["enemyTeamKills"] + data["kills"] + data["assists"]
        )
        data["d_ratio"] = data["deaths"] / data["enemyTeamDeaths"]
        data["damages_ratio"] = data["damagetochampions"] / data["enemyTeamDamages"]
        data["damage_tanked_ratio"] = (data["damagetakenperminute"] * data["gamelength"]) / data["enemyTeamDamages"]
        data["damage_mitigated_ratio"] = (
            data["damagemitigatedperminute"] * data["gamelength"] / data["enemyTeamDamages"]
        )
        data["gold_ratio"] = data["totalgold"] / data["enemyTeamGolds"]
        data["cs_to_gold_ratio"] = data["total_cs"] / data["enemyTeamGolds"]
        data["wards_placed_ratio"] = (data["wpm"] * data["gamelength"]) / data["enemyTeamWardPlaced"]
        data["wards_killed_ratio"] = (data["wcpm"] * data["gamelength"]) / data["enemyTeamWardPlaced"]

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

        if "result" not in data.columns:
            raise ValueError("'result' column is required to compute win/loss metrics.")

        data = data.copy()

        # Apply the function per player and season
        data = data.set_index(["playerid", "season"])

        def compute_metrics(group):
            result = group["result"]
            kills = group["kills"]
            deaths = group["deaths"]
            gamelength = group["gamelength"]

            # Compute metrics for wins
            win_mask = result == 1
            group["kills_per_season_win"] = kills[win_mask].mean() if win_mask.any() else np.nan
            group["deaths_per_season_win"] = deaths[win_mask].mean() if win_mask.any() else np.nan
            group["avg_gamelength_season_win"] = gamelength[win_mask].mean() if win_mask.any() else np.nan

            # Compute metrics for losses
            loss_mask = result == 0
            group["kills_per_season_loss"] = kills[loss_mask].mean() if loss_mask.any() else np.nan
            group["deaths_per_season_loss"] = deaths[loss_mask].mean() if loss_mask.any() else np.nan
            group["avg_gamelength_season_loss"] = gamelength[loss_mask].mean() if loss_mask.any() else np.nan

            return group

        data = data.groupby(["playerid", "season"], group_keys=False).apply(compute_metrics).reset_index()

        # Repeat the process for patches
        data = data.set_index(["playerid", "patch"])

        def compute_metrics_patch(group):
            result = group["result"]
            kills = group["kills"]
            deaths = group["deaths"]
            gamelength = group["gamelength"]

            # Compute metrics for wins
            win_mask = result == 1
            group["kills_per_patch_win"] = kills[win_mask].mean() if win_mask.any() else np.nan
            group["deaths_per_patch_win"] = deaths[win_mask].mean() if win_mask.any() else np.nan
            group["avg_gamelength_patch_win"] = gamelength[win_mask].mean() if win_mask.any() else np.nan

            # Compute metrics for losses
            loss_mask = result == 0
            group["kills_per_patch_loss"] = kills[loss_mask].mean() if loss_mask.any() else np.nan
            group["deaths_per_patch_loss"] = deaths[loss_mask].mean() if loss_mask.any() else np.nan
            group["avg_gamelength_patch_loss"] = gamelength[loss_mask].mean() if loss_mask.any() else np.nan

            return group

        data = data.groupby(["playerid", "patch"], group_keys=False).apply(compute_metrics_patch).reset_index()

        logger.info("Win/loss metrics computation by season and patch completed.")
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
            logger.error(f"Missing required columns for player feature generation: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        data = data.copy()  # Avoid modifying the original DataFrame

        # Extract season from patch number
        data["season"] = data["patch"].astype(str).str.split(".").str[0]

        # Calculate team kills
        data["team_kills"] = data.groupby(["gameid", "teamid"])["kills"].transform("sum")

        # Create dummies for position
        position_dummies = pd.get_dummies(data["position"], prefix="position")

        # Calculate KDA ratio
        data["kda"] = (data["kills"] + data["assists"]) / data["deaths"].replace(0, 1)

        # Calculate Gold Efficiency
        data["gold_efficiency"] = data["totalgold"] / data["gamelength"].replace(0, np.nan)

        # Calculate XP Efficiency
        data["xp_efficiency"] = data["total_cs"] / data["gamelength"].replace(0, np.nan)

        # Calculate Kill Participation
        data["kill_participation"] = (data["kills"] + data["assists"]) / data["team_kills"].replace(0, np.nan)

        # Extract season from patch number
        data["season"] = data["patch"].astype(str).str.split(".").str[0]

        # Compute key statistics
        data = FeatureGenerator.compute_key_stats(data)

        # Compute kills and deaths per win and loss
        data = FeatureGenerator.compute_win_loss_metrics(data)

        # Concatenate position dummies
        data = pd.concat([data, position_dummies], axis=1)

        logger.info("Player features generation completed.\n")
        return data

    @staticmethod
    def generate_new_team_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate new features for the given team data.

        This function calculates additional statistics and prepares season data from patch numbers.

        Args:
            data (pd.DataFrame): The team data.

        Returns:
            pd.DataFrame: The team data with new features added.
        """
        logger.info("Generating new team features...")

        required_columns = {
            "patch",
            "kills",
            "assists",
            "deaths",
            "towers",
            "gamelength",
            "gameid",
            "result",
            "teamid",
        }
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            logger.error(f"Missing required columns for team feature generation: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        data = data.copy()  # Avoid modifying the original DataFrame

        # Extract season from patch number
        data["season"] = data["patch"].astype(str).str.split(".").str[0]

        # Calculate KDA ratio
        data["kda"] = (data["kills"] + data["assists"]) / data["deaths"].replace(0, 1)

        # Calculate total game kills and total tower kills
        game_stats = (
            data.groupby("gameid").agg(total_kills=("kills", "sum"), total_towers=("towers", "sum")).reset_index()
        )

        data = data.merge(game_stats, on="gameid", how="left")

        # Calculate season and patch average game length overall
        data["season_avg_gamelength"] = data.groupby("season")["gamelength"].transform("mean")
        data["patch_avg_gamelength"] = data.groupby("patch")["gamelength"].transform("mean")

        # Recompute season and patch average game lengths grouped by teamid as well
        data["team_season_avg_gamelength"] = data.groupby(["teamid", "season"])["gamelength"].transform("mean")
        data["team_patch_avg_gamelength"] = data.groupby(["teamid", "patch"])["gamelength"].transform("mean")

        # Compute team average game lengths by season and result
        team_season_result_avg = data.groupby(["teamid", "season", "result"])["gamelength"].mean().reset_index()
        team_season_win = team_season_result_avg[team_season_result_avg["result"] == 1][
            ["teamid", "season", "gamelength"]
        ]
        team_season_win = team_season_win.rename(columns={"gamelength": "team_win_avg_season_gamelength"})
        team_season_loss = team_season_result_avg[team_season_result_avg["result"] == 0][
            ["teamid", "season", "gamelength"]
        ]
        team_season_loss = team_season_loss.rename(columns={"gamelength": "team_lose_avg_season_gamelength"})

        data = data.merge(team_season_win, on=["teamid", "season"], how="left")
        data = data.merge(team_season_loss, on=["teamid", "season"], how="left")

        # Compute team average game lengths by patch and result
        team_patch_result_avg = data.groupby(["teamid", "patch", "result"])["gamelength"].mean().reset_index()
        team_patch_win = team_patch_result_avg[team_patch_result_avg["result"] == 1][["teamid", "patch", "gamelength"]]
        team_patch_win = team_patch_win.rename(columns={"gamelength": "team_win_avg_patch_gamelength"})
        team_patch_loss = team_patch_result_avg[team_patch_result_avg["result"] == 0][["teamid", "patch", "gamelength"]]
        team_patch_loss = team_patch_loss.rename(columns={"gamelength": "team_lose_avg_patch_gamelength"})

        data = data.merge(team_patch_win, on=["teamid", "patch"], how="left")
        data = data.merge(team_patch_loss, on=["teamid", "patch"], how="left")

        logger.info("Team features generation completed.\n")
        return data
