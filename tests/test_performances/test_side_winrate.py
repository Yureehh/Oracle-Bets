import pandas as pd
import pytest

from src.feature_engineering.performance_features.side_win_rate import compute_ema_side, side_win_rate_ewm_performance


class TestGamePerformances:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup common test variables"""

        self.sample_player_data = pd.DataFrame(
            {
                "date": [
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-02",
                ],
                "league": [
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                ],
                "gameid": [
                    11,
                    11,
                    11,
                    11,
                    11,
                    11,
                    11,
                    11,
                    11,
                    11,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                ],
                "teamname": [
                    "team1",
                    "team1",
                    "team1",
                    "team1",
                    "team1",
                    "team2",
                    "team2",
                    "team2",
                    "team2",
                    "team2",
                ]
                * 2,
                "teamid": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2] * 2,
                "playerid": [
                    "player1",
                    "player2",
                    "player3",
                    "player4",
                    "player5",
                    "playerA",
                    "playerB",
                    "playerC",
                    "playerD",
                    "playerE",
                ]
                * 2,
                "position": [
                    "top",
                    "jgl",
                    "mid",
                    "bot",
                    "sup",
                    "top",
                    "jgl",
                    "mid",
                    "bot",
                    "sup",
                ]
                * 2,
                "side": [
                    "Blue",
                    "Blue",
                    "Blue",
                    "Blue",
                    "Blue",
                    "Red",
                    "Red",
                    "Red",
                    "Red",
                    "Red",
                ]
                * 2,
                "result": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 2,
            }
        )
        self.sample_team_data = pd.DataFrame(
            {
                "date": [
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-03",
                    "2021-01-03",
                    "2021-01-04",
                    "2021-01-04",
                    "2021-01-03",
                    "2021-01-03",
                    "2021-01-04",
                    "2021-01-04",
                ]
                * 2,
                "league": [
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "La Liga",
                    "La Liga",
                    "La Liga",
                    "La Liga",
                ]
                * 2,
                "gameid": [
                    11,
                    11,
                    12,
                    12,
                    21,
                    21,
                    22,
                    22,
                    13,
                    13,
                    15,
                    15,
                    111,
                    111,
                    112,
                    112,
                    121,
                    121,
                    122,
                    122,
                    113,
                    113,
                    115,
                    115,
                ],
                "teamid": [1, 2, 2, 1, 1, 2, 2, 1, 3, 4, 4, 3] * 2,
                "teamname": [
                    "team1",
                    "team2",
                    "team2",
                    "team1",
                    "team1",
                    "team2",
                    "team2",
                    "team1",
                    "team3",
                    "team4",
                    "team3",
                    "team4",
                ]
                * 2,
                "position": [
                    "team",
                    "team",
                    "team",
                    "team",
                    "team",
                    "team",
                    "team",
                    "team",
                    "team",
                    "team",
                    "team",
                    "team",
                ]
                * 2,
                "side": [
                    "Blue",
                    "Red",
                    "Blue",
                    "Red",
                    "Blue",
                    "Red",
                    "Blue",
                    "Red",
                    "Blue",
                    "Red",
                    "Blue",
                    "Red",
                ]
                * 2,
                "result": [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1] * 2,
            }
        )

    def test_compute_ema_side(self):
        """
        Test the compute_ema_side function with a sample DataFrame.
        """
        # Assuming HALF_LIFE is defined within the function or globally
        red_side_df = compute_ema_side(self.sample_team_data, "Red", "teamid")
        blue_side_df = compute_ema_side(self.sample_team_data, "Blue", "teamid")

        # Basic checks
        assert not red_side_df.empty, "The DataFrame should not be empty."
        assert "ema_red_side_before" in red_side_df.columns, "EMA before column missing for Red side."
        assert "ema_red_side_after" in red_side_df.columns, "EMA after column missing for Red side."
        assert not blue_side_df.empty, "The DataFrame should not be empty."
        assert "ema_blue_side_before" in blue_side_df.columns, "EMA before column missing for Red side."
        assert "ema_blue_side_after" in blue_side_df.columns, "EMA after column missing for Red side."

    def test_side_win_rate_ewm_performance(self):
        """
        Test the side_win_rate_ewm_performance function with sample DataFrames.
        """
        # Assuming HALF_LIFE is defined within the function or globally
        team_ewm = side_win_rate_ewm_performance(self.sample_team_data, "team")
        player_ewm = side_win_rate_ewm_performance(self.sample_player_data, "player")

        ewm_columns = [
            "ema_blue_side_before",
            "ema_red_side_before",
            "ema_blue_side_after",
            "ema_red_side_after",
            "ema_side_win_perc",
        ]

        # Basic checks
        assert not team_ewm.empty, "The DataFrame should not be empty."
        assert all(column in team_ewm.columns for column in ewm_columns), "EWM columns missing in the output."
        assert not player_ewm.empty, "The DataFrame should not be empty."
        assert all(column in player_ewm.columns for column in ewm_columns), "EWM columns missing in the output."
