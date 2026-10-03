from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lol_bets.data_generation.close_series import (
    RATING_COLUMNS,
    rolling_close_series_oof,
)
from oracle_bets_core.pd import pd


def _rating_rows(series_count=24):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for index in range(series_count):
        for side, team_id in enumerate(("a", "b")):
            row = {
                "gameid": f"series-{index}",
                "date": start + timedelta(days=index),
                "teamid": team_id,
                "result": int(side == index % 2),
            }
            row.update(
                {
                    column: float(index + 1 + side) / (offset + 2)
                    for offset, column in enumerate(RATING_COLUMNS)
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_close_series_membership_is_timestamped_out_of_fold():
    predictions = rolling_close_series_oof(
        _rating_rows(), minimum_training_series=6, blocks=4
    )

    assert not predictions.empty
    assert predictions["close_series_oof"].all()
    assert (
        predictions["close_series_training_cutoff"]
        < predictions["close_series_forecast_at"]
    ).all()
    assert (predictions["close_series_forecast_at"] < predictions["fixture_at"]).all()
    assert (
        predictions["close_series_source_model"]
        .eq("direct_series_rating_logistic_oof_v1")
        .all()
    )


def test_first_block_never_receives_in_sample_close_label():
    source = _rating_rows()
    predictions = rolling_close_series_oof(source, minimum_training_series=6, blocks=4)
    first_block_end = source["date"].drop_duplicates().sort_values().iloc[5]

    assert predictions["fixture_at"].min() > first_block_end
