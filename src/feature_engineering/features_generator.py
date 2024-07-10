"""
Features Generator

This script contains the FeatureGenerator class, which is used to generate new features for player and team data.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logger import logger


@dataclass
class FeatureGenerator:
    """
    A class to generate new features for player and team data.
    """

    @staticmethod
    def compute_key_stats(data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute key statistics for the player data efficiently.

        Parameters:
            data (pd.DataFrame): The player data.

        Returns:
            pd.DataFrame: The player data with key statistics added.
        """
        # Aggregate enemy team statistics
        enemy_team_stats = (
            data.groupby(["gameid", "side"])
            .agg(
                enemyTeamKills=("kills", "sum"),
                enemyTeamDeaths=("deaths", "sum"),
                enemyTeamDamages=("damagetochampions", "sum"),
                enemyTeamGolds=("totalgold", "sum"),
                enemyTeamWardPlaced=("wpm", "sum"),
            )
            .reset_index()
        )

        # Map sides to their opposites
        enemy_team_stats["side"] = enemy_team_stats["side"].map({"Blue": "Red", "Red": "Blue"})

        # Merge aggregated stats back to the original data
        data = data.merge(enemy_team_stats, on=["gameid", "side"], how="left")

        # Compute key statistics
        data["ka_ratio"] = (data["kills"] + data["assists"]) / (
            data["enemyTeamKills"] + data["kills"] + data["assists"]
        )
        data["d_ratio"] = data["deaths"] / data["enemyTeamDeaths"]
        data["damages_ratio"] = data["damagetochampions"] / data["enemyTeamDamages"]
        data["damage_tanked_ratio"] = data["damagetakenperminute"] * data["gamelength"] / data["enemyTeamDamages"]
        data["damage_mitigated_ratio"] = (
            data["damagemitigatedperminute"] * data["gamelength"] / data["enemyTeamDamages"]
        )
        data["gold_ratio"] = data["totalgold"] / data["enemyTeamGolds"]
        data["cs_to_gold_ratio"] = data["total_cs"] / data["enemyTeamGolds"]
        data["wards_placed_ratio"] = data["wpm"] * data["gamelength"] / data["enemyTeamWardPlaced"]
        data["wards_killed_ratio"] = data["wcpm"] * data["gamelength"] / data["enemyTeamWardPlaced"]

        # Replace inf and nan values in one go to improve efficiency
        data.replace({np.inf: np.nan, np.nan: 0}, inplace=True)

        return data

    @staticmethod
    def generate_new_player_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate new features for the given player data.
        Enhancements include team kills calculations and various efficiency metrics.

        Parameters:
            data (pd.DataFrame): The player data.

        Returns:
            pd.DataFrame: The player data with new features added.
        """
        logger.info("Generating new player features...")

        required_columns = {"teamid", "gameid"}
        if not required_columns.issubset(data.columns):
            logger.error("Data must include 'teamid' and 'gameid' columns.")
            raise ValueError("Missing necessary columns in player data.")

        # Calculate team kills
        team_kills = data.groupby(["gameid", "teamid"])["kills"].transform("sum")

        # Create dummies for position
        position_dummies = pd.get_dummies(data["position"], prefix="position")

        # Calculate KDA ratio
        data["kda"] = (data["kills"] + data["assists"]) / data["deaths"].replace(0, 1)

        # Calculate Gold Efficiency
        data["gold_efficiency"] = data["totalgold"] / data["gamelength"].replace(0, np.nan)

        # Calculate XP Efficiency
        data["xp_efficiency"] = data["total_cs"] / data["gamelength"].replace(0, np.nan)

        # Calculate Kill Participation
        data["kill_participation"] = (data["kills"] + data["assists"]) / team_kills.replace(0, np.nan)

        # Extract season from patch number
        data["season"] = data["patch"].astype(str).str.split(".").str[0]

        # Compute key statistics
        data = FeatureGenerator.compute_key_stats(data)

        # Concatenate position dummies
        data = pd.concat([data, position_dummies], axis=1)

        logger.info("Player features generation completed.\n")
        return data

    @staticmethod
    def generate_new_team_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate new features for the given team data.
        Currently, this function primarily prepares season data from patch numbers.

        Parameters:
            data (pd.DataFrame): The team data.

        Returns:
            pd.DataFrame: The team data with new features added.
        """
        logger.info("Generating new team features...")

        if "patch" not in data.columns:
            logger.error("Data must include 'patch' column.")
            raise ValueError("Missing necessary 'patch' column in team data.")

        # Extract season from patch number
        data["season"] = data["patch"].astype(str).str.split(".").str[0]

        # Calculate KDA ratio
        data["kda"] = (data["kills"] + data["assists"]) / data["deaths"].replace(0, 1)

        # Calculate total game kills and total tower kills
        game_stats = (
            data.groupby("gameid").agg(total_kills=("kills", "sum"), total_towers=("towers", "sum")).reset_index()
        )[["gameid", "total_kills", "total_towers"]]

        # Merge aggregated stats back to the original data
        data = data.merge(game_stats, on="gameid", how="left")

        # Calculate patch average game length
        patch_avg_gamelength = data.groupby("patch")["gamelength"].transform("mean")
        data["patch_avg_gamelength"] = patch_avg_gamelength

        logger.info("Team features generation completed.\n")
        return data
