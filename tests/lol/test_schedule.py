import datetime as dt

from lol_bets.data_generation.ingestion.schedule import (
    SCHEDULE_COLUMNS,
    PandaScoreSchedule,
    normalize_schedule_frame,
)
from oracle_bets_core.pd import pd
from oracle_bets_discord.predictions.lol import format_schedule_message


def _pandascore_match(match_id: int = 42) -> dict:
    return {
        "id": match_id,
        "scheduled_at": "2026-05-23T14:00:00Z",
        "number_of_games": 3,
        "status": "not_started",
        "league": {"name": "LCK"},
        "serie": {"full_name": "LCK 2026 Spring"},
        "tournament": {"name": "Playoffs"},
        "opponents": [
            {"opponent": {"id": 1, "name": "T1"}},
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
    assert row["team_a"] == "T1"
    assert row["team_b"] == "Gen.G"
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
