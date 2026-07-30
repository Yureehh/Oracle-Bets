from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from oracle_bets_core.evidence.contracts import (
    ContractValidationError,
    DecimalOdds,
    DecisionMode,
    FixtureRef,
    PredictionRecord,
    ProbabilityEstimate,
    StakeUnits,
    WarningCode,
)


@pytest.mark.parametrize(
    ("point", "lower", "upper"),
    [
        (-0.01, 0.0, 0.1),
        (1.01, 0.9, 1.0),
        (float("nan"), 0.1, 0.9),
        (float("inf"), 0.1, 0.9),
        (0.5, 0.6, 0.7),
        (0.5, 0.3, 0.4),
    ],
)
def test_probability_estimate_rejects_invalid_values(point, lower, upper):
    with pytest.raises(ContractValidationError):
        ProbabilityEstimate(point=point, lower=lower, upper=upper)


def test_probability_estimate_round_trip_is_stable():
    estimate = ProbabilityEstimate(point=0.61, lower=0.54, upper=0.68)

    assert ProbabilityEstimate.from_dict(estimate.to_dict()) == estimate
    assert json.dumps(estimate.to_dict(), sort_keys=True) == (
        '{"lower": 0.54, "point": 0.61, "upper": 0.68}'
    )


@pytest.mark.parametrize("value", ["1", "0", "-1", "NaN", "Infinity"])
def test_decimal_odds_reject_invalid_values(value):
    with pytest.raises(ContractValidationError):
        DecimalOdds(value)


def test_money_contracts_use_exact_decimal_values():
    odds = DecimalOdds("1.95")
    stake = StakeUnits("0.50")

    assert odds.value == Decimal("1.95")
    assert stake.value == Decimal("0.50")
    assert odds.to_json_value() == "1.95"
    assert stake.to_json_value() == "0.50"


@pytest.mark.parametrize("value", ["-0.01", "NaN", "Infinity"])
def test_stake_units_reject_invalid_values(value):
    with pytest.raises(ContractValidationError):
        StakeUnits(value)


def test_fixture_requires_distinct_teams_and_utc_start_time():
    with pytest.raises(ContractValidationError, match="distinct"):
        FixtureRef(
            fixture_id="fixture-1",
            sport="lol",
            competition_id="LCK",
            team_a_id="team-a",
            team_b_id="team-a",
            start_time=datetime(2026, 7, 26, 10, tzinfo=UTC),
            best_of=3,
        )

    with pytest.raises(ContractValidationError, match="UTC"):
        FixtureRef(
            fixture_id="fixture-1",
            sport="lol",
            competition_id="LCK",
            team_a_id="team-a",
            team_b_id="team-b",
            start_time=datetime(2026, 7, 26, 10),
            best_of=3,
        )

    non_utc = timezone(timedelta(hours=2))
    with pytest.raises(ContractValidationError, match="UTC"):
        FixtureRef(
            fixture_id="fixture-1",
            sport="lol",
            competition_id="LCK",
            team_a_id="team-a",
            team_b_id="team-b",
            start_time=datetime(2026, 7, 26, 10, tzinfo=non_utc),
            best_of=3,
        )


@pytest.mark.parametrize("best_of", [0, 4, 7])
def test_fixture_rejects_unsupported_series_formats(best_of):
    with pytest.raises(ContractValidationError, match="best_of"):
        FixtureRef(
            fixture_id="fixture-1",
            sport="lol",
            competition_id="LCK",
            team_a_id="team-a",
            team_b_id="team-b",
            start_time=datetime(2026, 7, 26, 10, tzinfo=UTC),
            best_of=best_of,
        )


def test_prediction_record_round_trip_preserves_domain_types():
    record = PredictionRecord(
        prediction_id="pred-1",
        fixture_id="fixture-1",
        selection_id="team-a",
        model_version="champion-2026-07",
        created_at=datetime(2026, 7, 26, 8, 15, tzinfo=UTC),
        mode=DecisionMode.PREMATCH,
        probability=ProbabilityEstimate(point=0.61, lower=0.54, upper=0.68),
        warnings=(WarningCode.ROSTER_UNCERTAIN,),
        schema_version=1,
    )

    payload = record.to_dict()

    assert payload["created_at"] == "2026-07-26T08:15:00Z"
    assert payload["mode"] == "prematch"
    assert payload["warnings"] == ["roster_uncertain"]
    assert PredictionRecord.from_dict(payload) == record


def test_prediction_record_rejects_naive_time_and_blank_identifiers():
    with pytest.raises(ContractValidationError):
        PredictionRecord(
            prediction_id="",
            fixture_id="fixture-1",
            selection_id="team-a",
            model_version="champion",
            created_at=datetime(2026, 7, 26, 8, 15),
            mode=DecisionMode.PREMATCH,
            probability=ProbabilityEstimate(point=0.5, lower=0.4, upper=0.6),
        )
