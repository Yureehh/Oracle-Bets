"""
Tests for the data generator class
"""

import pandas as pd
import pytest

from src.data_generator import DataGenerator


class TestDataGenerator:
    @pytest.fixture
    def data_generator(self):
        return DataGenerator()

    def test_create_s3_session(self, data_generator):
        session = data_generator.create_s3_session()
        assert session is not None

    def test_get_years_to_process(self, data_generator):
        years = data_generator._get_years_to_process()
        # Assert the years are current year and two previous years
        current_year = pd.Timestamp.now().year
        assert years == [
            str(current_year),
            str(current_year - 1),
        ]

    def test_remove_buggy_games(self, data_generator):
        data = pd.DataFrame(
            {
                "gameid": [1, 2, 3, 4, 5, "8479-8479_game_1", "ESPORTSTMNT02/1890835"],
                "teamid": ["A", "B", "C", "D", "E", "F", "G"],
                "playerid": ["A1", "B1", "C1", "D1", "E1", "F1", "G1"],
            }
        )
        data = data_generator._remove_buggy_games(data)
        assert data.shape[0] == 0

    def test_ingest_data_from_s3(self, data_generator):
        team_data, player_data = data_generator.ingest_data_from_s3()
        assert team_data is not None
        assert player_data is not None

    def test_enrich_datasets(self, data_generator):
        team_data, player_data = data_generator.enrich_datasets()
        assert team_data is not None
        assert player_data is not None

        assert team_data is not None
        assert player_data is not None

        # Verify every elo column is present, every plackett_luce column is present, and every trueskill column is present
        expected_player_trueskill_columns = [
            "trueskill_mu",
            "trueskill_sigma",
            "trueskill_opponent_mu",
            "trueskill_opponent_sigma",
        ]
        expected_team_trueskill_columns = [
            "trueskill_sum_mu",
            "trueskill_sigma_squared",
            "trueskill_opponent_sum_mu",
            "trueskill_opponent_sigma_squared",
            "trueskill_diff",
        ]
        expected_plackett_luce_columns = [
            "pl_pre_match_mu",
            "pl_pre_match_sigma",
            "pl_win_likelihood",
            "pl_mu",
            "pl_sigma",
            "pl_pre_match_mu_opponent",
        ]
        expexted_elo_columns = [
            "elo_pre_match",
            "elo",
            "elo_pre_match_opponent",
            "elo_opponent",
            "elo_win_likelihood",
        ]
        assert all(col in player_data.columns for col in expected_player_trueskill_columns)
        assert all(col in team_data.columns for col in expected_team_trueskill_columns)
        assert all(col in player_data.columns for col in expected_plackett_luce_columns)
        assert all(col in player_data.columns for col in expexted_elo_columns)
        assert all(col in team_data.columns for col in expexted_elo_columns)

    def test_run(self, data_generator):
        data_generator.run()
        assert True
