import json
from io import StringIO

import pandas as pd
import pytest
import requests_mock

from src.feature_engineering.impute_early_game_metrics import EarlyGameStatsImputer
from utils.paths import INTERIM_DIR


class TestEarlyGameStatsImputer:
    @pytest.fixture
    def imputer(self):
        return EarlyGameStatsImputer()

    @pytest.fixture
    def data(self):
        return pd.read_csv(INTERIM_DIR / "player_data.csv")

    def test_generate_features(self, imputer, data):
        # Expected columns after feature generation
        expected_columns = [
            "gameid",
            "teamid",
            "kills",
            "assists",
            "deaths",
            "gamelength",
            "totalgold",
            "total cs",
            "KDA",
            "gold_efficiency",
            "xp_efficiency",
            "kill_participation",
            "kills_volatility",
            "deaths_volatility",
            "kills_growth",
            "deaths_growth",
        ]

        # Generate features
        result = imputer._generate_features(data)

        # Check if all expected columns are present
        for col in expected_columns:
            assert col in result.columns

    def test_train_stacked_model(self):
        pass  # TODO: too complex to test, will include in integration tests

    def test_impute_missing_values(self):
        pass  # TODO: too complex to test, will include in integration tests

    def test_log_performance_metrics(self):
        pass  # TODO: too complex to test, will include in integration tests

    def test_prepare_data_for_modeling(self, imputer, data):
        # Expected columns after encoding
        expected_columns = [
            "position_top",
            "position_jng",
            "position_mid",
            "position_bot",
            "position_sup",
        ]

        # Prepare data
        prepared_data, _ = imputer._prepare_data_for_modeling(data)

        # Check if all expected columns are present
        for col in expected_columns:
            assert col in prepared_data.columns

    def test_train_models(self):
        pass  # TODO: too complex to test, will include in integration tests

    def test_process_data(self, imputer, data):
        data = imputer.process_data(data)
        for col in ["goldat15", "xpat15", "csat15"]:
            assert data[col].isnull().sum() == 0
