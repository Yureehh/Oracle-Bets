"""
Team dataclass

This script defines a dataclass for a team in League of Legends.
This is used to store and display information about a team, such as its roster and statistics.
This also allows for substitution of players in the roster.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from utils.logger import logger
from utils.paths import PROCESSED_DIR

# Load display columns and data once to avoid repeated disk I/O
team_data = pd.read_csv(PROCESSED_DIR / "flattened_teams.csv")
player_data = pd.read_csv(PROCESSED_DIR / "flattened_players.csv")


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

    def __post_init__(self):
        if not self.name:
            raise ValueError("Team name must be provided.")

        self.team_stats = self._get_team_stats()
        self.update_roster(self.roster if all(self.roster.values()) else self._get_last_roster())

    def _get_team_stats(self) -> Optional[pd.Series]:
        """
        Get the team statistics from the team data.
        """
        filtered_team_data = team_data[team_data.teamname.str.lower() == (self.name or "").lower()].reset_index(
            drop=True
        )
        if filtered_team_data.empty:
            logger.warning(f"Team `{self.name}` not found in database. No team data was used.")
            raise ValueError(f"Team `{self.name}` not found in database.")
        return filtered_team_data.iloc[0]

    def _get_last_roster(self) -> Dict[str, str]:
        """
        Get the last roster of the team from the player data.
        """
        roster_data = player_data[player_data.teamname.str.lower() == str(self.name).lower()].reset_index(drop=True)
        last_played = roster_data.sort_values(["date", "position"], ascending=False).drop_duplicates(
            subset=["position"], keep="first", ignore_index=True
        )
        return {row["position"]: row["playername"] for _, row in last_played.iterrows()}

    def update_roster(self, players: Dict[str, str]):
        """
        Update the team roster with the specified players.

        Parameters
        ----------
        players : Dict[str, str]
            Dictionary mapping positions to player names.
        """
        self.roster.update(players)
        self._validate_roster()
        self.player_stats = self._get_player_stats()

    def _validate_roster(self):
        """
        Validate the team roster to ensure it has exactly 5 players with valid positions.
        """
        valid_positions = {"top", "jng", "mid", "bot", "sup"}
        if len(self.roster) != 5 or not set(self.roster.keys()).issubset(valid_positions):
            raise ValueError("Roster must have exactly 5 players with valid positions.")
        if any(player is None for player in self.roster.values()):
            raise ValueError("Team roster cannot have None values.")

    def _get_player_stats(self) -> Optional[pd.DataFrame]:
        """
        Get the player statistics for the team roster.
        """
        # Extract player names in their original case
        players = [player.lower() for player in self.roster.values() if player is not None]

        # Filter player_data case-insensitively and keep the most recent entry for each player
        filtered_player_data = player_data[player_data.playername.str.lower().isin(players)].reset_index(drop=True)

        filtered_player_data = (
            filtered_player_data.sort_values(["playername", "date"]).groupby("playername", as_index=False).last()
        )

        if len(filtered_player_data) < 5:
            raise ValueError("Team cannot have less than 5 player values.")

        # Create a mapping from lowercase player names back to their original case names
        name_mapping = {name: name for name in players}
        # Apply the mapping to update the 'playername' column to its original case
        filtered_player_data["playername"] = filtered_player_data["playername"].str.lower().map(name_mapping)

        return filtered_player_data

    def get_team_info(self):
        # Basic team info
        team_info = {"Role": ["Team"], "Name": [self.name]}

        # Adding roster info
        for position, player in self.roster.items():
            team_info["Role"].append(position.title())
            team_info["Name"].append(player)

        team_info_df = pd.DataFrame(team_info)

        return team_info_df


# Example of how to use this refactored Team dataclass
if __name__ == "__main__":
    team = Team(
        "G2 Esports",
        roster={
            "top": "Wunder",
            "jng": "Jankos",
            "mid": "Caps",
            "bot": "Perkz",
            "sup": "Mikyx",
        },
    )
    logger.info(team.get_team_info())
