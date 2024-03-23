# Housekeeping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from utils.logger import logger
from utils.paths import DISPLAY_COLS, PROCESSED_DIR
from utils.utils import json_loader

display_cols = json_loader(DISPLAY_COLS)


@dataclass
class Team:
    name: Optional[str] = None
    side: str = "Blue"
    roster: Dict[str, Optional[str]] = field(
        default_factory=lambda: {
            "top": None,
            "jng": None,
            "mid": None,
            "bot": None,
            "sup": None,
        }
    )
    team_stats: Optional[pd.Series] = field(default=None, init=False)
    player_stats: Optional[pd.DataFrame] = field(default=None, init=False)

    team_data = pd.read_csv(PROCESSED_DIR / "flattened_teams.csv")
    player_data = pd.read_csv(PROCESSED_DIR / "flattened_players.csv")

    def __post_init__(self):

        self.team_stats = self._get_team_stats(self.team_data)
        self.update_roster(self._get_last_roster(self.player_data))

    def _get_team_stats(self, team_data: pd.DataFrame = None):
        try:
            filtered_team_data = team_data[
                team_data.teamname.str.lower() == (self.name or "").lower()
            ].reset_index(drop=True)
            if not filtered_team_data.empty:
                return filtered_team_data.iloc[0]
            else:
                logger.warning(
                    f"Team `{self.name}` not found in database. No team data was used."
                )
                return None
        except Exception as e:
            logger.error(f"Error occurred while fetching team data: {e}")
            return None

    def _get_last_roster(self, player_data: pd.DataFrame) -> list:
        lower_name = str(self.name).lower()
        player_data = player_data[
            player_data.teamname.str.lower().isin([lower_name])
        ].reset_index(drop=True)
        last_played = player_data.sort_values(
            ["date", "teamname", "position"]
        ).drop_duplicates(
            subset=["teamname", "position"], keep="last", ignore_index=True
        )
        return {row["position"]: row["playername"] for _, row in last_played.iterrows()}

    def update_roster(self, players: Dict[str, str]):
        self.roster.update(players)
        self._validate_roster()
        self.player_stats = self._get_player_stats()

    def _validate_roster(self):
        valid_positions = {"top", "jng", "mid", "bot", "sup"}
        if len(self.roster) != 5 or not set(self.roster.keys()).issubset(
            valid_positions
        ):
            raise ValueError("Roster must have exactly 5 players with valid positions.")
        if any(player is None for player in self.roster.values()):
            raise ValueError("Team roster cannot have None values.")

    def _get_player_stats(self):
        # Extract player names in their original case
        players = [player for player in self.roster.values() if player is not None]
        players_lowercase = [player.lower() for player in players]

        # Filter player_data case-insensitively and keep the most recent entry for each player
        filtered_player_data = self.player_data[
            self.player_data.playername.str.lower().isin(players_lowercase)
        ].reset_index(drop=True)

        filtered_player_data = (
            filtered_player_data.sort_values(["playername", "date"])
            .groupby("playername", as_index=False)
            .last()
        )

        if len(filtered_player_data) < 5:
            raise ValueError("Team cannot have less than 5 player values.")

        # Create a mapping from lowercase player names back to their original case names
        name_mapping = {name.lower(): name for name in players}
        # Apply the mapping to update the 'playername' column to its original case
        filtered_player_data["playername"] = (
            filtered_player_data["playername"].str.lower().map(name_mapping)
        )

        return filtered_player_data

    def display_team_info(self):
        logger.info(f"Team: {self.name}")
        logger.info(f"Side: {self.side}")
        logger.info("Roster:")
        for position, player in self.roster.items():
            logger.info(f"    {position.title()}: {player}")
        logger.info("Team Stats:")
        if self.team_stats is not None:
            for col in display_cols["team_display_cols"]:
                if col in self.team_stats:
                    logger.info(f"    {col.title()}: {self.team_stats[col].round(2)}")
                else:
                    logger.info(f"    {col.title()}: N/A")
        logger.info("Player Stats:")
        if self.player_stats is not None:
            for col in display_cols["player_display_cols"]:
                if col in self.player_stats.columns:
                    logger.info(
                        f"    {col.title()}: {np.mean(self.player_stats[col]).round(2)}"
                    )
                else:
                    logger.info(f"    {col.title()}: N/A")


# Example of how to use this refactored Team dataclass
if __name__ == "__main__":
    team = Team("G2 Esports")
    team.display_team_info()
