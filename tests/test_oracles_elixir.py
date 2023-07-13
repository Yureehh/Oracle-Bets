import boto3
import pandas as pd
import pytest
from moto import mock_s3
import unittest

from src.oracles_elixir import OraclesElixir


@pytest.fixture
def test_bucket():
    with mock_s3():
        conn = boto3.resource("s3", region_name="us-east-1")
        conn.create_bucket(Bucket="test_bucket")
        yield "test_bucket"


@pytest.fixture
def test_data(test_bucket):
    df = pd.DataFrame({
        "date": ["2022-01-01", "2022-01-02"],
        "gameid": ["game1", "game2"],
        "side": ["blue", "red"],
        "league": ["league1", "league2"],
        "playername": ["player1", "player2"],
        "playerid": ["player1", "player2"],
        "teamname": ["team1", "team2"],
        "teamid": ["team1", "team2"],
        "result": [1, 0],
        "kills": [10, 5],
        "deaths": [5, 10],
        "assists": [15, 20],
        "earned gpm": [400, 350],
    })
    df.to_csv("2022_LoL_esports_match_data_from_OraclesElixir.csv", index=False)
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.Object(test_bucket, "2022_LoL_esports_match_data_from_OraclesElixir.csv").put(
        Body=open("2022_LoL_esports_match_data_from_OraclesElixir.csv", "rb"))
    yield df


def test_oracles_elixir(test_data):
    oe = OraclesElixir("test_bucket")
    oe.ingest_data([2022])

    assert not oe.oe_data.empty, "Dataframe should not be empty after ingesting data"
    assert oe.oe_data.equals(test_data), "Ingested data should match test data"

    # Test for ingest_data
    assert not oe.oe_data.empty, "Dataframe should not be empty after ingesting data"
    assert oe.oe_data.equals(test_data), "Ingested data should match test data"

    # Test for format_data_types
    oe.format_data_types()
    assert pd.api.types.is_datetime64_any_dtype(
        oe.oe_data['date']), "'date' column should be datetime type after format_data_types"

    # Test for format_ids
    oe.format_ids()
    assert oe.oe_data['gameid'].notna().all(), "'gameid' should not contain null values after format_ids"

    # Test for remove_null_games
    oe.remove_null_games()
    assert oe.oe_data['gameid'].notna().all(), "'gameid' should not contain null values after remove_null_games"

    # Test for drop_unknown_entities
    oe.drop_unknown_entities()
    assert 'unknown player' not in oe.oe_data[
        'playername'], "'playername' should not contain 'unknown player' after drop_unknown_entities"
    assert 'unknown team' not in oe.oe_data[
        'teamname'], "'teamname' should not contain 'unknown team' after drop_unknown_entities"

    # Test for drop_negative_earned_gpm
    oe.drop_negative_earned_gpm()
    assert (oe.oe_data[
                'earned gpm'] >= 0).all(), "'earned gpm' should not contain negative values after drop_negative_earned_gpm"

    # Test for normalize_names
    # Suppose we have replacement dictionaries as {'player1': 'player_one', 'team1': 'team_one'}
    oe.normalize_names({'player1': 'player_one'}, {'team1': 'team_one'})
    assert 'player1' not in oe.oe_data[
        'playername'], "'playername' should not contain 'player1' after normalize_names"
    assert 'team1' not in oe.oe_data['teamname'], "'teamname' should not contain 'team1' after normalize_names"

    # Test for subset_data
    player_data = oe.subset_data('player')
    assert 'playername' in player_data.columns, "Player data should contain 'playername' column"
    assert 'teamname' not in player_data.columns, "Player data should not contain 'teamname' column"

    team_data = oe.subset_data('team')
    assert 'teamname' in team_data.columns, "Team data should contain 'teamname' column"
    assert 'playername' not in team_data.columns, "Team data should not contain 'playername' column"

    # Test for remove_inconsistent_games
    oe.remove_inconsistent_games()
    # You'll need to define what an "inconsistent game" is to be able to write this test

    # Test for sort_data
    sorted_data = oe.sort_data()
    assert sorted_data.equals(sorted_data.sort_values(by=['league', 'date', 'gameid', 'side',
                                                          'position'])), "Data should be sorted by league, date, gameid, side, and position"

    # Test for fill_null_team_ids
    oe.fill_null_team_ids()
    assert oe.oe_data['teamid'].notna().all(), "'teamid' should not contain null values after fill_null_team_ids"

    # Test for enrich_opponent_metrics
    oe.enrich_opponent_metrics('team')

    # Check if the new columns were created
    for column in ['opponentname', 'opponentid', 'opponent_egpm']:
        assert column in oe.oe_data.columns, f"'{column}' should exist in the data after enrich_opponent_metrics with split on 'team'"

    # Test for clean_data
    cleaned_data = oe.clean_data('player')
    assert not cleaned_data.empty, "Cleaned data should not be empty"

oe_test = unittest.FunctionTestCase()