from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from openskill.models import PlackettLuce

from src.ratings_features.plackett_luce import (
    calculate_plackett_luce,
    rate_teams_or_players,
)


class Test_Plackett_Luce:
    @pytest.fixture
    def plackett_luce_model(self):
        return PlackettLuce()

    @pytest.fixture
    def plackett_luce_df_team(self):
        return pd.DataFrame(
            {
                "date": [
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-03",
                    "2021-01-03",
                ],
                "league": [
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "La Liga",
                    "La Liga",
                ],
                "gameid": [11, 11, 12, 12, 13, 13],
                "teamid": [1, 2, 2, 1, 3, 4],
                "position": ["team", "team", "team", "team", "team", "team"],
                "side": ["Blue", "Red", "Blue", "Red", "Blue", "Red"],
                "result": [1, 0, 0, 1, 1, 0],
            }
        )

    @pytest.fixture
    def plackett_luce_df_player(self):
        return pd.DataFrame(
            {
                "date": [
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
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
                ],
                "gameid": [11, 11, 11, 11, 12, 12, 12, 12],
                "teamid": [1, 1, 2, 2, 1, 1, 2, 2],
                "playerid": [
                    "player1",
                    "player2",
                    "playerA",
                    "playerB",
                    "player1",
                    "player2",
                    "playerA",
                    "playerB",
                ],
                "position": ["top", "mid", "top", "mid", "top", "mid", "top", "mid"],
                "side": ["Blue", "Blue", "Red", "Red", "Red", "Red", "Blue", "Blue"],
                "result": [1, 1, 0, 0, 1, 1, 0, 0],
            }
        )

    def test_rate_teams_or_players_with_default_ratings(self, plackett_luce_model):
        # Create a Plackett-Luce model instance
        model = plackett_luce_model

        # Instantiate base ratings for two entities (e.g., teams or players) using the model's default parameters
        # Note: As per the openskill library's pattern, we directly use the model's rate method for instantiation.
        # The openskill library uses lists of lists to represent teams of players. For individual players, each team is a list of one.
        team1_base_rating = [model.rating()]
        team2_base_rating = [model.rating()]

        # Simulate a match outcome where the first team wins
        ranks = [
            0,
            1,
        ]  # 0 indicates the team with a higher rank (winner), 1 indicates the loser

        # Use the rate_teams_or_players function to update the ratings based on the match outcome
        updated_ratings = rate_teams_or_players(
            model, [team1_base_rating, team2_base_rating], ranks
        )

        # Unpack the updated ratings for easier assertions
        updated_team1_rating, updated_team2_rating = updated_ratings

        # Check that ratings have been updated from their base values
        assert (
            updated_team1_rating[0].mu > team1_base_rating[0].mu
        ), "Team 1's mu rating should increase."
        assert (
            updated_team2_rating[0].mu < team2_base_rating[0].mu
        ), "Team 2's mu rating should decrease."

        # Assert the sigma values have been updated (indicating a change in uncertainty)
        assert (
            updated_team1_rating[0].sigma < team1_base_rating[0].sigma
        ), "Team 1's sigma should be updated."
        assert (
            updated_team2_rating[0].sigma < team2_base_rating[0].sigma
        ), "Team 2's sigma should be updated."

    def test_calculate_plackett_luce_for_team(self, plackett_luce_df_team):
        entity = "team"

        # Calculate Plackett-Luce ratings for teams
        plackett_luce_df_team = calculate_plackett_luce(
            plackett_luce_df_team, entity=entity
        )

        # Check every line where teamid is 1 the mu is higher than the pre_match_mu column
        assert all(
            plackett_luce_df_team[plackett_luce_df_team["teamid"] == 1]["mu"]
            > plackett_luce_df_team[plackett_luce_df_team["teamid"] == 1][
                "pre_match_mu"
            ]
        ), "Team 1's mu should increase."
        # Check every line where teamid is 1 the sigma is lower than the pre_match_sigma column
        assert all(
            plackett_luce_df_team[plackett_luce_df_team["teamid"] == 1]["sigma"]
            < plackett_luce_df_team[plackett_luce_df_team["teamid"] == 1][
                "pre_match_sigma"
            ]
        ), "Team 1's sigma should decrease."
        # Check every line where teamid is 2 the mu is lower than the pre_match_mu column
        assert all(
            plackett_luce_df_team[plackett_luce_df_team["teamid"] == 2]["mu"]
            < plackett_luce_df_team[plackett_luce_df_team["teamid"] == 2][
                "pre_match_mu"
            ]
        ), "Team 2's mu should decrease."
        # Check every line where teamid is 2 the sigma is lower than the pre_match_sigma column
        assert all(
            plackett_luce_df_team[plackett_luce_df_team["teamid"] == 2]["sigma"]
            < plackett_luce_df_team[plackett_luce_df_team["teamid"] == 2][
                "pre_match_sigma"
            ]
        ), "Team 2's sigma should decrease."
        # Check that team3 and team4 have been added to the ratings
        assert (
            len(plackett_luce_df_team[plackett_luce_df_team["teamid"] == 3]) > 0
        ), "Team 3 should be in the ratings."
        assert (
            len(plackett_luce_df_team[plackett_luce_df_team["teamid"] == 4]) > 0
        ), "Team 4 should be in the ratings."
        # Check that team3 mu is higher than team4 mu
        assert (
            plackett_luce_df_team[plackett_luce_df_team["teamid"] == 3]["mu"].values[0]
            > plackett_luce_df_team[plackett_luce_df_team["teamid"] == 4]["mu"].values[
                0
            ]
        ), "Team 3's mu should be higher than team 4's mu."

    def test_calculate_plackett_luce_for_player(self, plackett_luce_df_player):
        entity = "player"

        # Calculate Plackett-Luce ratings for teams
        plackett_luce_df_player = calculate_plackett_luce(
            plackett_luce_df_player, entity=entity
        )

        # Check every line where playerid is 1 the mu is higher than the pre_match_mu column
        assert all(
            plackett_luce_df_player[plackett_luce_df_player["playerid"] == 1]["mu"]
            > plackett_luce_df_player[plackett_luce_df_player["playerid"] == 1][
                "pre_match_mu"
            ]
        ), "Team 1's mu should increase."
        # Check every line where playerid is 1 the sigma is lower than the pre_match_sigma column
        assert all(
            plackett_luce_df_player[plackett_luce_df_player["playerid"] == 1]["sigma"]
            < plackett_luce_df_player[plackett_luce_df_player["playerid"] == 1][
                "pre_match_sigma"
            ]
        ), "Team 1's sigma should decrease."
        # Check every line where playerid is 2 the mu is lower than the pre_match_mu column
        assert all(
            plackett_luce_df_player[plackett_luce_df_player["playerid"] == 2]["mu"]
            < plackett_luce_df_player[plackett_luce_df_player["playerid"] == 2][
                "pre_match_mu"
            ]
        ), "Team 2's mu should decrease."
        # Check every line where playerid is 2 the sigma is lower than the pre_match_sigma column
        assert all(
            plackett_luce_df_player[plackett_luce_df_player["playerid"] == 2]["sigma"]
            < plackett_luce_df_player[plackett_luce_df_player["playerid"] == 2][
                "pre_match_sigma"
            ]
        ), "Team 2's sigma should decrease."
        # Check that team3 and team4 have been added to the ratings
