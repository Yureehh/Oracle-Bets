"""
Team Dataclass

This module defines the Team dataclass for managing team-related data in League of Legends.
It stores information such as the team's name, side, roster, and associated statistics.
It also provides functionalities to update the roster and retrieve team information.
"""

from dataclasses import dataclass, field

import pandas as pd

from utils.logger import logger
from utils.paths import FLATTENED_PLAYERS, FLATTENED_TEAMS


@dataclass
class Team:
    """
    A dataclass to represent a League of Legends team.

    Attributes:
        name (str): The name of the team.
        side (Optional[str]): The side of the team (e.g., 'Blue', 'Red').
        roster (Dict[str, Optional[str]]): The team's roster with roles as keys and player names as values.
        team_stats (Optional[pd.Series]): Statistical data related to the team.
        player_stats (Optional[pd.DataFrame]): Statistical data related to the players in the roster.

    """

    name: str
    side: str | None = None
    roster: dict[str, str | None] = field(
        default_factory=lambda: {
            "top": None,
            "jng": None,
            "mid": None,
            "bot": None,
            "sup": None,
        }
    )
    team_stats: pd.Series | None = field(default=None, init=False)
    player_stats: pd.DataFrame | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize team and player data upon object creation."""
        try:
            self.team_data: pd.DataFrame = self._load_team_data()
            self.player_data: pd.DataFrame = self._load_player_data()
            self.team_stats = self._get_team_stats()
            if all(self.roster.values()):
                self.update_roster(self.roster)
            else:
                last_roster = self._get_last_roster()
                self.update_roster(last_roster)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Initialization failed for team '{self.name}': {e}")
            raise
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during initialization of team '{self.name}': {e}"
            )
            raise

    def _load_team_data(self) -> pd.DataFrame:
        """
        Load team data from the flattened teams Parquet file.

        Returns:
            pd.DataFrame: The loaded team data.

        Raises:
            FileNotFoundError: If the team data file does not exist.
            ValueError: If the team data cannot be loaded properly.

        """
        try:
            return pd.read_parquet(FLATTENED_TEAMS, engine="fastparquet")
        except FileNotFoundError as e:
            logger.error(f"Team data file not found at '{FLATTENED_TEAMS}'.")
            msg = f"Team data file not found at '{FLATTENED_TEAMS}'."
            raise FileNotFoundError(msg) from e
        except Exception as e:
            logger.error(f"Error loading team data: {e}")
            msg = f"Error loading team data: {e}"
            raise ValueError(msg) from e

    def _load_player_data(self) -> pd.DataFrame:
        """
        Load player data from the flattened players Parquet file.

        Returns:
            pd.DataFrame: The loaded player data.

        Raises:
            FileNotFoundError: If the player data file does not exist.
            ValueError: If the player data cannot be loaded properly.

        """
        try:
            return pd.read_parquet(FLATTENED_PLAYERS, engine="fastparquet")
        except FileNotFoundError as e:
            logger.error(f"Player data file not found at '{FLATTENED_PLAYERS}'.")
            msg = f"Player data file not found at '{FLATTENED_PLAYERS}'."
            raise FileNotFoundError(msg) from e
        except Exception as e:
            logger.error(f"Error loading player data: {e}")
            msg = f"Error loading player data: {e}"
            raise ValueError(msg) from e

    def _get_team_stats(self) -> pd.Series:
        """
        Retrieve the team's statistics from the team data.

        Returns:
            pd.Series: Team statistics.

        Raises:
            ValueError: If the team is not found in the team data.

        """
        filtered_team_data = self.team_data[
            self.team_data.teamname.str.lower() == self.name.lower()
        ].reset_index(drop=True)

        if filtered_team_data.empty:
            logger.warning(f"Team '{self.name}' not found in the database.")
            msg = f"Team '{self.name}' not found in the database."
            raise ValueError(msg)

        return filtered_team_data.iloc[0]

    def _get_last_roster(self) -> dict[str, str]:
        """
        Retrieve the last known roster of the team from the player data.

        Returns:
            Dict[str, str]: Last known roster.

        Raises:
            ValueError: If the last roster cannot be determined.

        """
        roster_data = self.player_data[
            self.player_data.teamname.str.lower() == str(self.name).lower()
        ].reset_index(drop=True)

        if roster_data.empty:
            logger.warning(f"No player data found for team '{self.name}'.")
            msg = f"No player data found for team '{self.name}'."
            raise ValueError(msg)

        last_played = (
            roster_data.sort_values(by=["position", "date"], ascending=[True, False])
            .groupby("position")
            .first()
            .reset_index()
        )

        if last_played.empty or len(last_played) < 5:
            logger.warning(
                f"Could not determine the last roster for team '{self.name}'."
            )
            msg = f"Could not determine the last roster for team '{self.name}'."
            raise ValueError(msg)

        return {row["position"]: row["playername"] for _, row in last_played.iterrows()}

    def update_roster(self, players: dict[str, str]) -> None:
        """
        Update the team roster with the specified players.

        Args:
            players (Dict[str, str]): Players to update the roster with.

        Raises:
            ValueError: If the updated roster is invalid.

        """
        logger.info(f"Updating roster for team '{self.name}' with players: {players}")
        self.roster.update(players)
        self._validate_roster()
        self.player_stats = self._get_player_stats()

    def _validate_roster(self) -> None:
        """
        Validate the team roster to ensure it has exactly 5 players with valid positions.

        Raises:
            ValueError: If the roster is invalid.

        """
        expected_positions = {"top", "jng", "mid", "bot", "sup"}
        roster_positions = set(self.roster.keys())

        missing_positions = expected_positions - roster_positions
        if missing_positions:
            logger.error(
                f"Roster for team '{self.name}' is missing positions: {missing_positions}"
            )
            msg = f"Roster is missing required positions: {missing_positions}"
            raise ValueError(msg)

        extra_positions = roster_positions - expected_positions
        if extra_positions:
            logger.warning(
                f"Roster for team '{self.name}' has unexpected positions: {extra_positions}"
            )

        if any(
            player is None or not isinstance(player, str)
            for player in self.roster.values()
        ):
            logger.error(
                f"Roster for team '{self.name}' contains invalid player entries."
            )
            msg = "Team roster cannot have None values or non-string player names."
            raise ValueError(msg)

    def _get_player_stats(self) -> pd.DataFrame:
        """
        Retrieve the player statistics for the team roster from the player data.

        Returns:
            pd.DataFrame: Player statistics.

        Raises:
            ValueError: If player statistics are incomplete or missing.

        """
        players = [player.lower() for player in self.roster.values() if player]

        filtered_player_data = self.player_data[
            self.player_data.playername.str.lower().isin(players)
        ].reset_index(drop=True)

        if filtered_player_data.empty:
            logger.warning(
                f"No player statistics found for the roster of team '{self.name}'."
            )
            msg = f"No player statistics found for the roster of team '{self.name}'."
            raise ValueError(msg)

        # Sort by playername and date descending, then drop duplicates
        filtered_player_data = (
            filtered_player_data.sort_values(
                ["playername", "date"], ascending=[True, False]
            )
            .drop_duplicates(subset=["playername"], keep="first")
            .reset_index(drop=True)
        )

        if len(filtered_player_data) < 5:
            logger.warning(
                f"Incomplete player statistics for team '{self.name}'. Expected 5, found {len(filtered_player_data)}."
            )
            msg = "Team cannot have less than 5 player statistics."
            raise ValueError(msg)

        # Map back to original player names
        player_name_mapping = {name.lower(): name for name in players}
        filtered_player_data["playername"] = (
            filtered_player_data["playername"].str.lower().map(player_name_mapping)
        )

        # Verify that all players are present after mapping
        missing_players = set(players) - set(filtered_player_data["playername"])
        if missing_players:
            missing_players_original = [
                self.roster[pos]
                for pos in self.roster
                if self.roster[pos] in missing_players
            ]
            logger.error(f"Missing statistics for players: {missing_players_original}")
            msg = f"Missing statistics for players: {missing_players_original}"
            raise ValueError(msg)

        return filtered_player_data

    def get_team_info(self) -> pd.DataFrame:
        """
        Get the team information as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the team information.

        """
        data = [{"ROLE": "Team", "NAME": self.name}]
        for position, player in self.roster.items():
            data.append({"ROLE": position.title(), "NAME": player})

        return pd.DataFrame(data)
