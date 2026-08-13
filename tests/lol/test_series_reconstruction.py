from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from lol_bets.data_generation.series import (
    _next_map_training_tables,
    _series_winner_training_tables,
    reconstruct_series,
)
from oracle_bets_core.pd import pd

BO1_PHASE_SERIES = 20


def _series_rows(
    *,
    prefix: str,
    winners: list[str],
    start: datetime,
    league: str = "LCK",
    split: str = "Summer",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, winner in enumerate(winners, start=1):
        for team_id, team_name in (("a", "Alpha"), ("b", "Beta")):
            rows.append(
                {
                    "date": start + timedelta(hours=index - 1),
                    "gameid": f"{prefix}-map-{index}",
                    "league": league,
                    "season": "2026",
                    "split": split,
                    "playoffs": False,
                    "game": index,
                    "teamid": team_id,
                    "teamname": team_name,
                    "result": int(team_id == winner),
                }
            )
    return rows


def test_reconstructs_complete_bo3_and_bo5_series() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = _series_rows(prefix="bo3", winners=["a", "b", "a"], start=start)
    rows += _series_rows(
        prefix="bo5",
        winners=["b", "a", "b", "b"],
        start=start + timedelta(days=1),
    )

    result = reconstruct_series(pd.DataFrame(rows))

    assert sorted(result.series["best_of"].tolist()) == [3, 5]
    assert sorted(result.series["map_count"].tolist()) == [3, 4]
    assert set(result.series["winner_teamid"]) == {"a", "b"}
    assert result.rejections.empty


def test_incomplete_series_is_quarantined_with_reason() -> None:
    rows = _series_rows(
        prefix="incomplete",
        winners=["a", "b"],
        start=datetime(2026, 1, 1, tzinfo=UTC),
    )

    result = reconstruct_series(pd.DataFrame(rows))

    assert result.series.empty
    assert result.rejections.iloc[0]["reason"] == "incomplete_or_ambiguous_series"


def test_single_maps_require_verified_bo1_phase() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for index in range(BO1_PHASE_SERIES):
        rows += _series_rows(
            prefix=f"bo1-{index}",
            winners=["a" if index % 2 else "b"],
            start=start + timedelta(days=index),
            league="LFL",
            split="Regular Season",
        )

    result = reconstruct_series(pd.DataFrame(rows))

    assert len(result.series) == BO1_PHASE_SERIES
    assert set(result.series["best_of"]) == {1}
    assert result.rejections.empty


def test_series_must_start_at_map_one_and_remain_sequential() -> None:
    rows = _series_rows(
        prefix="bad-start",
        winners=["a", "a"],
        start=datetime(2026, 1, 1, tzinfo=UTC),
    )
    for row in rows:
        row["game"] = int(row["game"]) + 1

    result = reconstruct_series(pd.DataFrame(rows))

    assert result.series.empty
    assert set(result.rejections["reason"]) == {"series_does_not_start_at_map_1"}


def test_map_with_duplicate_team_rows_is_quarantined() -> None:
    rows = _series_rows(
        prefix="duplicate",
        winners=["a", "a"],
        start=datetime(2026, 1, 1, tzinfo=UTC),
    )
    rows.append(dict(rows[0]))

    result = reconstruct_series(pd.DataFrame(rows))

    assert result.series.empty
    assert "invalid_map_identity_or_result" in set(result.rejections["reason"])


def test_maps_after_a_series_clinch_are_quarantined() -> None:
    rows = _series_rows(
        prefix="post-clinch",
        winners=["a", "a", "a", "b"],
        start=datetime(2026, 1, 1, tzinfo=UTC),
    )

    result = reconstruct_series(pd.DataFrame(rows))

    assert result.series.empty
    assert set(result.rejections["reason"]) == {"map_after_series_clinch"}


def test_series_winner_tables_remove_all_live_series_state() -> None:
    rows = _series_rows(
        prefix="prematch",
        winners=["a", "a"],
        start=datetime(2026, 1, 1, tzinfo=UTC),
    )
    teams = pd.DataFrame(rows)
    teams["side"] = ["Blue", "Red"] * 2
    teams["maps_completed"] = 99
    teams["next_map_number"] = 99
    teams["series_wins_before"] = 99
    teams["series_losses_before"] = 99
    teams["series_score_delta"] = 99
    teams["game_in_series"] = 99
    teams["is_deciding_game"] = True
    players = pd.concat(
        [
            pd.DataFrame(
                {
                    **{column: [row[column]] * 5 for column in teams.columns},
                    "position": ["top", "jng", "mid", "bot", "sup"],
                }
            )
            for row in teams.to_dict(orient="records")
        ],
        ignore_index=True,
    )
    manifest = reconstruct_series(teams).series

    winner_teams, winner_players = _series_winner_training_tables(
        manifest, teams, players
    )

    forbidden = {
        "maps_completed",
        "next_map_number",
        "series_wins_before",
        "series_losses_before",
        "series_score_delta",
        "game_in_series",
        "is_deciding_game",
    }
    assert forbidden.isdisjoint(winner_teams.columns)
    assert forbidden.isdisjoint(winner_players.columns)
    assert set(winner_teams["best_of"]) == {3}


def test_next_map_tables_retain_only_explicit_score_state() -> None:
    rows = _series_rows(
        prefix="reactive",
        winners=["a", "b", "a"],
        start=datetime(2026, 1, 1, tzinfo=UTC),
    )
    teams = pd.DataFrame(rows)
    teams["side"] = ["Blue", "Red"] * 3
    players = pd.concat(
        [
            pd.DataFrame(
                {
                    **{column: [row[column]] * 5 for column in teams.columns},
                    "position": ["top", "jng", "mid", "bot", "sup"],
                }
            )
            for row in teams.to_dict(orient="records")
        ],
        ignore_index=True,
    )
    manifest = reconstruct_series(teams).series

    next_teams, _ = _next_map_training_tables(manifest, teams, players)

    assert {
        "maps_completed",
        "next_map_number",
        "series_wins_before",
        "series_losses_before",
        "series_score_delta",
    }.issubset(next_teams.columns)
    assert sorted(next_teams["next_map_number"].unique()) == [2, 3]


def test_accepted_series_cannot_silently_lose_map_one_features() -> None:
    teams = pd.DataFrame(
        _series_rows(
            prefix="missing-features",
            winners=["a", "a"],
            start=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    manifest = reconstruct_series(teams).series
    players = teams.iloc[0:0].copy()

    with pytest.raises(RuntimeError, match="complete pre-Map-1"):
        _series_winner_training_tables(manifest, teams, players)
