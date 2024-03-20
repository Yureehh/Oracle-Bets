import math
from io import StringIO

import pandas as pd
import pytest

from src.ratings_features.elo import calculate_elo, expected_outcome, update_elo_rating


class TestEloRating:
    @pytest.mark.parametrize(
        "elo_a, elo_b, expected_result",
        [(1200, 1000, 0.76), (1000, 1200, 0.24), (1000, 1000, 0.5)],
    )
    def test_expected_outcome(self, elo_a, elo_b, expected_result):
        result = expected_outcome(elo_a, elo_b)
        assert round(result, 2) == expected_result

    @pytest.mark.parametrize(
        "old_elo, expected, actual_result, k_factor, expected_new_elo",
        [
            (1200, 0.76, 1, 30, 1207.2),
            (1000, 0.24, 0, 30, 992.8),
            (1000, 0.5, 1, 10, 1005),
        ],
    )
    def test_update_elo_rating(
        self, old_elo, expected, actual_result, k_factor, expected_new_elo
    ):
        new_elo = update_elo_rating(old_elo, expected, actual_result, k_factor)
        assert round(new_elo, 1) == expected_new_elo

    def test_calculate_elo_for_team_entity(self):
        # Simplified test data
        data = StringIO(
            """date,league,gameid,side,teamid,opponentteamid,result
                            2022-01-01,A,1,Blue,Team1,Team2,1
                            2022-01-02,A,1,Red,Team2,Team1,0"""
        )
        df = pd.read_csv(data)
        result_df = calculate_elo(df, "team", initial_elo=1000, k=30)

        # Expected results after two matches with initial Elo of 1000 for both teams
        # After match 1: Team1 wins, expected 1000 vs 1000 -> Team1: 1015, Team2: 985
        # After match 2: Team1 loses, expected 1015 vs 985 -> Team1: 1000, Team2: 1000
        expected_elos = [1000, 1000]  # Elo before each game
        expected_updated_elos = [1015, 985]  # Elo after each game

        assert all(result_df["elo_pre_match"] == expected_elos)
        assert all(result_df["elo"] == expected_updated_elos)

    def test_calculate_elo_for_player_entity(self):
        # Simplified test data for player entities
        data = StringIO(
            """date,league,gameid,teamid,playerid,opponentplayerid,side,position,result
                        2022-01-01,A,1,Team1,Player1,Player2,Blue,Forward,1
                        2022-01-01,A,1,Team1,Player3,Player4,Blue,top,1
                        2022-01-01,A,1,Team2,Player2,Player1,Red,Forward,0
                        2022-01-02,A,1,Team2,Player4,Player3,Red,top,0
                        2022-01-05,B,3,Team10,PlayerA,PlayerB,Blue,sup,0
                        2022-01-05,B,3,Team20,PlayerB,PlayerA,Red,sup,1
                        2024-01-01,A,10,Team1,Player1,Player2,Blue,Forward,0
                        2024-01-01,A,10,Team2,Player2,Player1,Red,Forward,1
                        """
        )
        df = pd.read_csv(data)
        result_df = calculate_elo(df, "player", initial_elo=1000, k=30)

        expected_elos = [1000, 1000, 1000, 1000, 1000, 1000, 1015, 985]
        expected_updated_elos = [
            1015,
            1015,
            985,
            985,
            985,
            1015,
            998.708005,
            1001.291995,
        ]

        # Convert this to a more detailed comparison using math.isclose for floating-point comparison
        for actual_elo, expected_elo in zip(result_df["elo_pre_match"], expected_elos):
            assert math.isclose(actual_elo, expected_elo, rel_tol=1e-9)

        for actual_elo, expected_elo in zip(result_df["elo"], expected_updated_elos):
            assert math.isclose(actual_elo, expected_elo, rel_tol=1e-9)
