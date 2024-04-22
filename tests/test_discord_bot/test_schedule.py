"""
Tests for the schedule class
"""

import datetime as dt
import json
from io import StringIO
from os import getenv

import pandas as pd
import pytest
import requests_mock

from src.data_ingest.schedule import PandaScoreSchedule


class TestPandascoreSchedule:
    # Fixture for the PandaScoreSchedule instance
    @pytest.fixture
    def panda_schedule(self):
        return PandaScoreSchedule(api_key="test_api_key")

    # Fixture for mock API data
    @pytest.fixture
    def mock_raw_data(self):
        raw_data = """{
          "slug": "mouz-vs-nno-prime-2024-03-06",
          "results": [
            { "score": 0, "team_id": 16 },
            { "score": 0, "team_id": 132418 }
          ],
          "status": "not_started",
          "detailed_stats": true,
          "begin_at": "2024-03-06T19:00:00Z",
          "league_id": 4302,
          "match_type": "best_of",
          "id": 896712,
          "forfeit": false,
          "original_scheduled_at": "2024-03-06T19:00:00Z",
          "videogame_title": null,
          "modified_at": "2024-01-03T16:22:22Z",
          "number_of_games": 1,
          "name": "MOUZ vs NNO.P",
          "games": [
            {
              "begin_at": null,
              "complete": false,
              "detailed_stats": true,
              "end_at": null,
              "finished": false,
              "forfeit": false,
              "id": 251569,
              "length": null,
              "match_id": 896712,
              "position": 1,
              "status": "not_started",
              "winner": { "id": null, "type": "Team" },
              "winner_type": "Team"
            }
          ],
          "rescheduled": false,
          "game_advantage": null,
          "winner_type": "Team",
          "winner_id": null,
          "end_at": null,
          "scheduled_at": "2024-03-06T19:00:00Z",
          "league": {
            "id": 4302,
            "image_url": "https://cdn.pandascore.co/images/league/image/4302/440px-prime_league_lightmode-png",
            "modified_at": "2023-12-26T22:31:16Z",
            "name": "Prime League 1st Division",
            "slug": "league-of-legends-prime-league-pro-division",
            "url": "https://www.primeleague.gg/de/start"
          },
          "serie": {
            "begin_at": "2024-01-17T17:00:00Z",
            "end_at": "2024-03-13T21:00:00Z",
            "full_name": "Spring 2024",
            "id": 7034,
            "league_id": 4302,
            "modified_at": "2024-01-03T16:22:21Z",
            "name": null,
            "season": "Spring",
            "slug": "league-of-legends-prime-league-pro-division-spring-2024",
            "winner_id": null,
            "winner_type": null,
            "year": 2024
          },
          "tournament": {
            "begin_at": "2024-01-17T17:00:00Z",
            "detailed_stats": true,
            "end_at": "2024-03-13T22:00:00Z",
            "has_bracket": false,
            "id": 12624,
            "league_id": 4302,
            "live_supported": false,
            "modified_at": "2024-02-22T16:27:13Z",
            "name": "Regular Season",
            "prizepool": null,
            "serie_id": 7034,
            "slug": "league-of-legends-prime-league-pro-division-spring-2024-group-stage",
            "tier": "c",
            "winner_id": null,
            "winner_type": "Team"
          },
          "videogame_version": null,
          "draw": false,
          "opponents": [
            {
              "opponent": {
                "acronym": "MOUZ",
                "id": 16,
                "image_url": "https://cdn.pandascore.co/images/team/image/16/208px_mouz_2021_allmode.png",
                "location": "DE",
                "modified_at": "2024-01-31T21:11:26Z",
                "name": "MOUZ NXT",
                "slug": "mousesports"
              },
              "type": "Team"
            },
            {
              "opponent": {
                "acronym": "NNO.P",
                "id": 132418,
                "image_url": "https://cdn.pandascore.co/images/team/image/132418/no_need_orgalogo_square.png",
                "location": "DE",
                "modified_at": "2024-01-16T12:28:06Z",
                "name": "NNO Prime",
                "slug": "nno-prime"
              },
              "type": "Team"
            }
          ],
          "serie_id": 7034,
          "live": { "opens_at": null, "supported": false, "url": null },
          "winner": null,
          "videogame": { "id": 1, "name": "LoL", "slug": "league-of-legends" },
          "tournament_id": 12624,
          "streams_list": [
            {
              "embed_url": "https://player.twitch.tv/?channel=primeleague",
              "language": "de",
              "main": true,
              "official": true,
              "raw_url": "https://www.twitch.tv/primeleague"
            }
          ]
        }"""
        return [json.loads(raw_data)]

    @pytest.fixture
    def mock_parsed_data(self):
        parsed_data = """{
            "league": "Prime League 1st Division",
            "Blue": "MOUZ NXT",
            "Red": "NNO Prime",
            "Start (UTC)": "2024-03-06T19:00:00Z",
            "Best Of": 1
        }"""
        return [json.loads(parsed_data)]

    @pytest.fixture
    def mock_full_schedule(self):
        schedule_data = """league,Blue,Red,Start (UTC),Best Of
CBLOL,INTZ e-Sports,LOS,2024-03-02 16:00:00+00:00,1
LVP SL 2nd Division,Xoldiers,Stormbringers,2024-03-02 17:00:00+00:00,1"""
        return pd.read_csv(StringIO(schedule_data))

    # Test _fetch_matches with mocked API response
    def test_fetch_matches(self, panda_schedule, mock_raw_data):
        with requests_mock.Mocker() as m:
            m.get(panda_schedule.base_url, json=mock_raw_data)
            result = panda_schedule._fetch_matches(1)
            assert result == mock_raw_data

    def test_fetch_matches_http_error(self, panda_schedule):
        with requests_mock.Mocker() as m:
            m.get(panda_schedule.base_url, status_code=404)
            result = panda_schedule._fetch_matches(1)
            assert result is None

    def test_fetch_matches_general_exception(self, panda_schedule):
        with requests_mock.Mocker() as m:
            m.get(panda_schedule.base_url, exc=Exception)
            result = panda_schedule._fetch_matches(1)
            assert result is None

    def test_process_response(self, panda_schedule, mock_raw_data, mock_parsed_data):
        """
        Test the process_response method.
        """
        # Directly use the mock_json_response for testing the method
        processed_result = panda_schedule._parse_matches_response(mock_raw_data)
        # Convert the expected JSON response to the format produced by process_response
        expected_output = mock_parsed_data
        assert processed_result == expected_output

    def test_filter_by_league(self, panda_schedule, mock_full_schedule):
        """
        Test the filter_by_league method.
        """
        # Filter the schedule by the LEC and LCS leagues
        filtered_schedule = panda_schedule.filter_by_league(mock_full_schedule, ["CBLOL"])
        assert filtered_schedule["league"].unique() == ["CBLOL"]

    def test_get_schedule(self):
        """
        Test the get_schedule method.
        """
        TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
        start = dt.datetime.now().strftime(TIME_FORMAT)
        end = (dt.datetime.now() + dt.timedelta(days=3)).strftime(TIME_FORMAT)

        # Fetch
        panda_schedule = PandaScoreSchedule(api_key=getenv("PANDASCORE_API_KEY"))
        schedule = panda_schedule.get_schedule(start_datetime=start, end_datetime=end)

        assert schedule is not None
        assert isinstance(schedule, pd.DataFrame)
        assert not schedule.empty
