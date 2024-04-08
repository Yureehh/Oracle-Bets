import pandas as pd
import pytest

from utils.paths import PROCESSED_DIR
from utils.team import Team


class TestTeam:

    @pytest.fixture(autouse=True)
    def get_player_data(self):
        return pd.read_csv(PROCESSED_DIR / "flattened_players.csv")

    @pytest.fixture(autouse=True)
    def team(self):
        return Team("G2 Esports")

    def test_get_team_stats(self, team):
        team_stats = team.team_stats
        assert isinstance(team_stats, pd.Series)
        assert len(team_stats) == 47
        assert team_stats["teamname"].lower() == "g2 esports"

    def test_get_last_roster(self, team, get_player_data):
        last_roster = team.roster
        assert isinstance(last_roster, dict)
        assert len(last_roster) == 5
        assert all([v is not None for v in last_roster.values()])
        assert all([isinstance(v, str) for v in last_roster.values()])
        assert all([v in get_player_data["playername"].values for v in last_roster.values()])

    def test_update_roster(self, team):
        team.update_roster(
            {
                "top": "Wunder",
                "jng": "Jankos",
                "mid": "Caps",
                "bot": "Perkz",
                "sup": "Mikyx",
            }
        )
        assert team.roster == {
            "top": "Wunder",
            "jng": "Jankos",
            "mid": "Caps",
            "bot": "Perkz",
            "sup": "Mikyx",
        }
        assert all([isinstance(v, str) for v in team.roster.values()])
        assert all([v in team.player_data["playername"].values for v in team.roster.values()])
        assert len(team.roster) == 5
        assert all([isinstance(v, str) for v in team.roster.values()])

    def test_get_player_stats(self, team):
        player_stats = team.player_stats
        assert isinstance(player_stats, pd.DataFrame)
        assert player_stats.shape == (5, 68)

    def test_display_team_info(self, team):
        team.display_team_info()
        assert True
