import json
from unittest.mock import patch

from src.data_ingest.schedule import PandascoreSchedule


@patch("src.schedule_generator.requests.get")
def test_fetch_data(mock_get):
    # Mock the API response
    mock_get.return_value.status_code = 200
    mock_get.return_value.text = json.dumps(
        [
            {
                "league": {"name": "Test League"},
                "opponents": [
                    {"opponent": {"name": "Team Blue"}},
                    {"opponent": {"name": "Team Red"}},
                ],
                "scheduled_at": "2023-01-01T00:00:00Z",
                "number_of_games": 3,
            }
        ]
    )

    schedule = PandascoreSchedule("dummy_api_key")
    response = schedule.fetch_data(1)

    assert len(response) == 1
    assert response[0]["league"]["name"] == "Test League"
    assert response[0]["opponents"][0]["opponent"]["name"] == "Team Blue"
    assert response[0]["opponents"][1]["opponent"]["name"] == "Team Red"
    assert response[0]["scheduled_at"] == "2023-01-01T00:00:00Z"
    assert response[0]["number_of_games"] == 3


def test_process_response():
    schedule = PandascoreSchedule("dummy_api_key")
    response = [
        {
            "league": {"name": "Test League"},
            "opponents": [
                {"opponent": {"name": "Team Blue"}},
                {"opponent": {"name": "Team Red"}},
            ],
            "scheduled_at": "2023-01-01T00:00:00Z",
            "number_of_games": 3,
        }
    ]

    processed_data = schedule.process_response(response)

    assert len(processed_data) == 1
    assert processed_data[0]["league"] == "Test League"
    assert processed_data[0]["Blue"] == "Team Blue"
    assert processed_data[0]["Red"] == "Team Red"
    assert processed_data[0]["Start (UTC)"] == "2023-01-01T00:00:00Z"
    assert processed_data[0]["Best Of"] == 3
