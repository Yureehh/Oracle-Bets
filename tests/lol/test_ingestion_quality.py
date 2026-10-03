from __future__ import annotations

import json

import pytest
from lol_bets import pipeline
from lol_bets.data_generation.ingestion.quality import (
    SourceSchemaError,
    quarantine_oracles_elixir_data,
    schema_fingerprint,
    source_column_reconciliation,
    write_quality_report,
)
from lol_bets.pipeline import DataGenerator
from oracle_bets_core.pd import pd

POSITIONS = ["top", "jng", "mid", "bot", "sup"]
EXPECTED_INPUT_ROWS = 24
ROWS_PER_GAME = 12


def _game(game_id: str) -> list[dict]:
    rows = [
        {
            "date": "2026-07-01",
            "gameid": game_id,
            "league": "LCK",
            "patch": "16.13",
            "side": side,
            "position": position,
            "result": result,
            "teamname": team,
            "teamid": team_id,
            "playername": f"{team}-{position}",
            "playerid": f"{team_id}-{position}",
        }
        for side, team, team_id, result in (
            ("Blue", "T1", "oe:t1", 1),
            ("Red", "Gen.G", "oe:geng", 0),
        )
        for position in POSITIONS
    ]
    for side, team, team_id, result in (
        ("Blue", "T1", "oe:t1", 1),
        ("Red", "Gen.G", "oe:geng", 0),
    ):
        rows.append(
            {
                "date": "2026-07-01",
                "gameid": game_id,
                "league": "LCK",
                "patch": "16.13",
                "side": side,
                "position": "team",
                "result": result,
                "teamname": team,
                "teamid": team_id,
                "playername": team,
                "playerid": None,
            }
        )
    return rows


def test_quality_pass_merges_exact_duplicates_and_quarantines_bad_games():
    good = _game("good")
    bad = _game("bad")[:-1]
    raw = pd.DataFrame([*good, good[0], *bad])

    accepted, quarantined, report = quarantine_oracles_elixir_data(raw)

    assert set(accepted["gameid"]) == {"good"}
    assert set(quarantined["gameid"]) == {"bad"}
    assert len(accepted) == ROWS_PER_GAME
    assert report.input_rows == EXPECTED_INPUT_ROWS
    assert report.exact_duplicate_rows == 1
    assert report.accepted_games == 1
    assert report.quarantined_games == 1
    assert "wrong_row_count" in report.game_issues["bad"]


def test_manual_invalid_game_is_quarantined_with_explicit_reason():
    raw = pd.DataFrame(_game("manual"))

    accepted, quarantined, report = quarantine_oracles_elixir_data(
        raw, manual_invalid_games={"manual"}
    )

    assert accepted.empty
    assert set(quarantined["gameid"]) == {"manual"}
    assert report.game_issues["manual"] == ("manual_invalid_game",)


def test_schema_change_fails_visibly():
    raw = pd.DataFrame(_game("g1")).drop(columns=["playerid"])

    with pytest.raises(SourceSchemaError, match="playerid"):
        quarantine_oracles_elixir_data(raw)


def test_schema_fingerprint_changes_for_column_or_dtype_change():
    raw = pd.DataFrame(_game("g1"))
    original = schema_fingerprint(raw)

    with_column = raw.assign(new_upstream_field="value")
    with_dtype = raw.copy()
    with_dtype["result"] = with_dtype["result"].astype(str)

    assert schema_fingerprint(with_column) != original
    assert schema_fingerprint(with_dtype) != original


def test_source_column_reconciliation_reports_present_missing_and_new_columns():
    raw = pd.DataFrame(_game("g1")).assign(
        datacompleteness="complete",
        split="Summer",
        upstream_new_metric=1,
    )

    rows = {row["column"]: row for row in source_column_reconciliation(raw)}

    assert rows["gameid"] == {
        "column": "gameid",
        "source_status": "present",
        "disposition": "required",
    }
    assert rows["datacompleteness"]["disposition"] == "metadata"
    assert rows["split"]["disposition"] == "metadata"
    assert rows["upstream_new_metric"] == {
        "column": "upstream_new_metric",
        "source_status": "new",
        "disposition": "candidate",
    }
    assert rows["goldat15"]["source_status"] == "missing"
    assert rows["goldat15"]["disposition"] == "feature"

    assert {row["source_status"] for row in rows.values()} <= {
        "present",
        "missing",
        "new",
    }
    assert {row["disposition"] for row in rows.values()} <= {
        "required",
        "metadata",
        "feature",
        "ignored",
        "prohibited",
        "candidate",
    }


def test_source_column_reconciliation_understands_upstream_column_spellings():
    raw = pd.DataFrame(_game("g1")).assign(
        **{
            "earned gpm": 1000,
            "team kpm": 0.8,
            "total cs": 250,
            "firstPick": 1,
        }
    )

    rows = {row["column"]: row for row in source_column_reconciliation(raw)}

    for canonical in ("egpm", "team_kpm", "total_cs", "first_pick"):
        assert rows[canonical]["source_status"] == "present"


def test_quality_report_is_machine_readable_and_atomic(tmp_path):
    _, _, report = quarantine_oracles_elixir_data(pd.DataFrame(_game("g1")))
    destination = tmp_path / "quality.json"

    write_quality_report(report, destination)

    payload = json.loads(destination.read_text())
    assert payload["source"] == "oracles_elixir"
    assert payload["accepted_games"] == 1
    assert payload["schema_fingerprint"] == report.schema_fingerprint
    assert payload["column_reconciliation"]
    assert not list(tmp_path.glob("*.tmp"))


def test_data_generator_applies_quality_gate_before_entity_cleaning(
    tmp_path, monkeypatch
):
    good = _game("good")
    bad = _game("bad")[:-1]
    stored = {}

    def fake_store(frame, path, _loggers):
        stored[path] = frame.copy()

    monkeypatch.setattr(pipeline, "DATA_QUALITY_REPORT", tmp_path / "quality.json")
    monkeypatch.setattr(
        pipeline, "QUARANTINED_RAW_DATA", tmp_path / "quarantined.parquet"
    )
    monkeypatch.setattr(pipeline, "safe_store_df_as_parquet", fake_store)
    generator = DataGenerator()

    accepted = generator._quality_gate_raw(pd.DataFrame(good + bad))

    assert set(accepted["gameid"]) == {"good"}
    assert set(stored[tmp_path / "quarantined.parquet"]["gameid"]) == {"bad"}
    assert json.loads((tmp_path / "quality.json").read_text())["quarantined_games"] == 1
