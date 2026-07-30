import datetime as dt
import json

import pytest
import requests
from lol_bets.data_generation.ingestion.schedule import (
    SCHEDULE_COLUMNS,
    DataValidationError,
    PandaScoreLineupRefresher,
    PandaScoreSchedule,
    normalize_schedule_frame,
    validate_duplicate_fixture_agreement,
)
from oracle_bets_core.pd import pd
from oracle_bets_discord.predictions.lol import format_schedule_message

EXPECTED_STARTERS = 5
_FIXTURE_AUTH = "fixture-token"


def _pandascore_match(match_id: int = 42) -> dict:
    return {
        "id": match_id,
        "scheduled_at": "2026-05-23T14:00:00Z",
        "number_of_games": 3,
        "status": "not_started",
        "league": {"name": "LCK"},
        "serie": {"full_name": "LCK 2026 Spring"},
        "tournament": {"id": 99, "name": "Playoffs"},
        "opponents": [
            {
                "opponent": {
                    "id": 1,
                    "name": "T1",
                    "players": [
                        {"id": 10, "name": "Zeus", "role": "top"},
                        {"id": 11, "name": "Oner", "role": "jun"},
                    ],
                }
            },
            {"opponent": {"id": 2, "name": "Gen.G"}},
        ],
    }


def test_schedule_parser_returns_market_friendly_columns():
    df = PandaScoreSchedule._parse_matches_response([_pandascore_match()])

    assert list(df.columns) == list(SCHEDULE_COLUMNS)
    row = df.iloc[0]
    assert row["provider"] == "pandascore"
    assert row["provider_match_id"] == "42"
    assert row["match_key"] == "pandascore:42"
    assert row["fixture_version"].startswith("fixture-")
    assert row["tournament_id"] == "99"
    assert row["team_a"] == "T1"
    assert row["team_b"] == "Gen.G"
    assert json.loads(row["team_a_lineup_json"]) == [
        {"provider_player_id": "11", "name": "Oner", "role": "jun"},
        {"provider_player_id": "10", "name": "Zeus", "role": "top"},
    ]
    assert row["market_query"] == "LCK T1 Gen.G"
    assert row["discord_label"] == "LCK | T1 vs Gen.G | BO3"


def test_schedule_filtering_and_deduping_use_stable_match_key():
    schedule = PandaScoreSchedule(api_key="test", sleep_fn=lambda _: None)
    start = dt.datetime(2026, 5, 23, tzinfo=dt.UTC)
    end = dt.datetime(2026, 5, 24, tzinfo=dt.UTC)

    df = schedule._parse_and_filter_matches(
        [_pandascore_match(), _pandascore_match()],
        start,
        end,
        time_format=None,
    )
    out = schedule._append_to_schedule(pd.DataFrame(), df)

    assert len(out) == 1
    assert PandaScoreSchedule.filter_by_league(out, "lck").iloc[0]["team_a"] == "T1"


def test_legacy_schedule_columns_are_normalized():
    legacy = pd.DataFrame(
        [
            {
                "match_id": 7,
                "league": "LEC",
                "Blue": "G2 Esports",
                "Red": "Fnatic",
                "Start (UTC)": "2026-05-24T18:00:00Z",
                "Best Of": 5,
            }
        ]
    )

    out = normalize_schedule_frame(legacy)

    assert list(out.columns) == list(SCHEDULE_COLUMNS)
    assert out.iloc[0]["match_key"] == "pandascore:7"
    assert out.iloc[0]["market_query"] == "LEC G2 Esports Fnatic"


def test_discord_schedule_formatter_uses_normalized_columns():
    df = PandaScoreSchedule._parse_matches_response([_pandascore_match()])

    message = format_schedule_message(df)

    assert "Upcoming LCK Games" in message
    assert "Team A" in message
    assert "T1" in message
    assert "LCK T1 Gen.G" in message


def test_schedule_parser_skips_all_tbd_pages_instead_of_raising():
    # A page where every match lacks scheduled_at (TBD matches) is normal near
    # the end of the upcoming feed and must not abort the whole fetch.
    tbd_match = _pandascore_match()
    tbd_match["scheduled_at"] = None

    df = PandaScoreSchedule._parse_matches_response([tbd_match])

    assert df.empty


def test_schedule_parser_keeps_partial_pages():
    tbd_match = _pandascore_match(match_id=43)
    tbd_match["scheduled_at"] = None

    df = PandaScoreSchedule._parse_matches_response([_pandascore_match(), tbd_match])

    assert len(df) == 1
    assert df.iloc[0]["provider_match_id"] == "42"


def test_duplicate_schedule_rows_require_all_fixture_facts_to_agree():
    rows = PandaScoreSchedule._parse_matches_response(
        [_pandascore_match(), _pandascore_match()]
    )

    validate_duplicate_fixture_agreement(rows)

    conflicting = rows.copy()
    conflicting.loc[1, "start_utc"] = pd.Timestamp("2026-05-23T14:05:00Z")
    with pytest.raises(DataValidationError, match="conflicting duplicate"):
        validate_duplicate_fixture_agreement(conflicting)


def test_match_detail_refresh_updates_lineup_and_fixture_version() -> None:
    class _Response:
        content = b"{}"

        def raise_for_status(self):
            return None

        def json(self):
            payload = _pandascore_match()
            payload["scheduled_at"] = "2026-05-23T14:15:00Z"
            payload["opponents"][1]["opponent"]["players"] = [
                {"id": 20, "name": "Kiin", "role": "top"},
                {"id": 21, "name": "Canyon", "role": "jun"},
                {"id": 22, "name": "Chovy", "role": "mid"},
                {"id": 23, "name": "Ruler", "role": "adc"},
                {"id": 24, "name": "Duro", "role": "sup"},
            ]
            return payload

    class _Session:
        def get(self, url, **kwargs):
            assert url == "https://api.pandascore.co/matches/42"
            assert kwargs["headers"]["Authorization"] == f"Bearer {_FIXTURE_AUTH}"
            return _Response()

    initial = PandaScoreSchedule._parse_matches_response([_pandascore_match()])
    previous_version = initial.iloc[0]["fixture_version"]

    refreshed = PandaScoreLineupRefresher(
        api_key=_FIXTURE_AUTH,
        session=_Session(),
    ).refresh(initial, observed_at=dt.datetime(2026, 5, 23, 12, tzinfo=dt.UTC))

    row = refreshed.iloc[0]
    assert row["lineup_source"] == "pandascore_match_detail"
    assert row["lineup_refresh_error"] == ""
    assert len(json.loads(row["team_b_lineup_json"])) == EXPECTED_STARTERS
    assert row["start_utc"] == pd.Timestamp("2026-05-23T14:15:00Z")
    assert row["fixture_version"] != previous_version


def test_match_detail_refresh_skips_missing_provider_id() -> None:
    initial = PandaScoreSchedule._parse_matches_response([_pandascore_match()])
    initial.loc[0, "provider_match_id"] = pd.NA

    class _Session:
        def get(self, *_args, **_kwargs):
            raise AssertionError("missing provider IDs must not make HTTP requests")

    refreshed = PandaScoreLineupRefresher(
        api_key=_FIXTURE_AUTH,
        session=_Session(),
    ).refresh(initial)

    assert refreshed.iloc[0]["lineup_refresh_error"] == "provider_match_id_missing"


def test_match_detail_refresh_stops_after_forbidden_response() -> None:
    initial = PandaScoreSchedule._parse_matches_response([_pandascore_match()])
    schedule = pd.concat([initial, initial], ignore_index=True)
    schedule["provider_match_id"] = ["42", "43"]

    class _Response:
        status_code = 403

        def raise_for_status(self):
            response = requests.Response()
            response.status_code = self.status_code
            raise requests.HTTPError("403 Client Error", response=response)

    class _Session:
        calls = 0

        def get(self, *_args, **_kwargs):
            self.calls += 1
            return _Response()

    session = _Session()
    refreshed = PandaScoreLineupRefresher(
        api_key=_FIXTURE_AUTH,
        session=session,
    ).refresh(schedule)

    assert session.calls == 1
    assert refreshed["lineup_refresh_error"].tolist() == [
        "match_detail_http_403",
        "match_detail_http_403",
    ]
