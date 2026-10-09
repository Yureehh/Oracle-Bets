from __future__ import annotations

import pickle
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from lol_bets.inference import serving_replay
from lol_bets.inference.roster import EXPECTED_ROLES
from lol_bets.inference.serving_replay import (
    _verified_candidate_paths,
    compare_feature_values,
    replay_sealed_features,
)
from lol_bets.inference.team import (
    InsufficientRosterHistoryError,
    TeamStateUnavailableError,
)
from lol_bets.operations.models import _paths_fingerprint
from oracle_bets_core.pd import pd


def test_value_replay_detects_values_and_missing_features():
    expected = pd.Series({"rating": 0.75, "pool": "major", "missing": None})
    actual = pd.Series({"rating": 0.76, "pool": "minor", "missing": None})
    assert compare_feature_values(expected, actual) == ["rating", "pool"]
    assert compare_feature_values(expected, actual.drop("pool")) == [
        "rating",
        "pool",
    ]


def test_value_replay_rejects_unpaired_generation_before_loading_models():
    inputs = SimpleNamespace(
        manifests={
            "map": {
                "generation_id": "map-generation",
                "code_sha256": "code",
                "source": {"snapshot_id": "source"},
            }
        }
    )
    snapshot = SimpleNamespace(
        manifest={
            "training_generation_id": "other-generation",
            "code_sha256": "code",
            "source_manifest": {"snapshot_id": "source"},
        }
    )
    with pytest.raises(ValueError, match="different lineage"):
        replay_sealed_features("candidate", inputs=inputs, snapshot=snapshot)


def test_value_replay_rejects_candidate_from_another_training_table(
    tmp_path, monkeypatch
):
    candidate = tmp_path / "candidates" / "candidate"
    candidate.mkdir(parents=True)
    candidate.joinpath("manifest.json").write_text('{"data_manifest":"old"}')
    map_table = tmp_path / "map.parquet"
    map_table.write_bytes(b"current")
    generation = {
        "generation_id": "generation",
        "code_sha256": "code",
        "source": {"snapshot_id": "source"},
        "files": {
            "map_teams": {"path": str(map_table)},
            "map_players": {"path": str(map_table)},
        },
    }
    inputs = SimpleNamespace(
        manifests={"map": generation}, assert_unchanged=lambda: None
    )
    snapshot = SimpleNamespace(
        manifest={
            "training_generation_id": "generation",
            "code_sha256": "code",
            "source_manifest": {"snapshot_id": "source"},
        }
    )
    registry = SimpleNamespace(
        candidates=tmp_path / "candidates",
        verified_artifact_paths=lambda **_kwargs: {},
    )
    monkeypatch.setattr(serving_replay, "ModelRegistry", lambda _root: registry)
    monkeypatch.setattr(
        serving_replay, "_candidate_training_paths", lambda: (map_table,)
    )
    with pytest.raises(ValueError, match="different data generation"):
        replay_sealed_features("candidate", inputs=inputs, snapshot=snapshot)


def test_candidate_generation_uses_complete_registered_training_bundle(
    tmp_path, monkeypatch
):
    candidate = tmp_path / "candidates" / "candidate"
    candidate.mkdir(parents=True)
    training_paths = tuple(tmp_path / f"table-{index}.parquet" for index in range(7))
    for table in training_paths:
        table.write_bytes(b"current")
    candidate.joinpath("manifest.json").write_text(
        '{"data_manifest":"' + _paths_fingerprint(training_paths) + '"}'
    )
    monkeypatch.setattr(
        serving_replay, "_candidate_training_paths", lambda: training_paths
    )
    registry = SimpleNamespace(
        candidates=tmp_path / "candidates",
        verified_artifact_paths=lambda **_kwargs: {"artifact": training_paths[1]},
    )
    assert _verified_candidate_paths(registry, "candidate") == {
        "artifact": training_paths[1]
    }
    training_paths[-1].write_bytes(b"changed")
    with pytest.raises(ValueError, match="different data generation"):
        _verified_candidate_paths(registry, "candidate")


def test_value_replay_runs_historical_team_path_and_reports_mismatch(
    tmp_path, monkeypatch
):
    match_at = datetime(2026, 8, 24, tzinfo=UTC)
    history_at = datetime(2026, 8, 23, tzinfo=UTC)
    teams = pd.DataFrame(
        [
            {"gameid": "match-1", "teamname": "A", "side": "Blue", "best_of": 3},
            {"gameid": "match-1", "teamname": "B", "side": "Red", "best_of": 3},
        ]
    )
    team_history = pd.DataFrame(
        [
            {"teamname": "A", "date": history_at, "elo": 1500},
            {"teamname": "B", "date": history_at, "elo": 1400},
        ]
    )
    player_history = pd.DataFrame(
        [
            {
                "gameid": "match-1",
                "teamname": team,
                "side": side,
                "position": role,
                "playername": f"{team}-{role}",
                "date": history_at,
            }
            for team, side in (("A", "Blue"), ("B", "Red"))
            for role in EXPECTED_ROLES
        ]
    )
    history_path = tmp_path / "history.parquet"
    player_history.to_parquet(history_path)
    snapshot = SimpleNamespace(
        snapshot_id="serving-1",
        manifest={
            "training_generation_id": "generation-1",
            "code_sha256": "code-1",
            "source_manifest": {"snapshot_id": "source-1"},
            "observed_at": match_at.isoformat(),
        },
        read=lambda kind: team_history if kind == "teams" else player_history,
    )
    inputs = SimpleNamespace(
        manifests={
            "map": {
                "generation_id": "generation-1",
                "code_sha256": "code-1",
                "source": {"snapshot_id": "source-1"},
            }
        },
        frames={"map_teams": teams},
        assert_unchanged=lambda: None,
    )
    prefix = "_evaluation/Winner_LightGBM"
    feature_path = tmp_path / "features.parquet"
    label_path = tmp_path / "labels.parquet"
    pd.DataFrame({"elo_diff": [100], "unused": [999]}).to_parquet(feature_path)
    pd.DataFrame(
        {"gameid": ["match-1"], "source_gameid": ["match-1"], "date": [match_at]}
    ).to_parquet(label_path)
    paths = {
        f"{prefix}/features.parquet": feature_path,
        f"{prefix}/labels.parquet": label_path,
    }
    selected_path = tmp_path / "selected.pkl"
    with selected_path.open("wb") as output:
        pickle.dump(["elo_diff"], output)
    paths["Winner_LightGBM/Winner_LightGBM_final_features.pkl"] = selected_path

    class Predictor:
        def __init__(self, **_kwargs):
            pass

        def _current_matchup_features(self, blue, red, **_kwargs):
            return (
                pd.DataFrame(
                    {"elo_diff": [blue.team_stats["elo"] - red.team_stats["elo"]]}
                ),
                None,
                None,
            )

    monkeypatch.setattr(serving_replay, "ModelRegistry", lambda _root: object())
    monkeypatch.setattr(
        serving_replay, "_verified_candidate_paths", lambda *_args: paths
    )
    monkeypatch.setattr(serving_replay, "MatchPredictor", Predictor)
    monkeypatch.setattr(
        serving_replay,
        "read_history_snapshot",
        lambda **_kwargs: (history_path, {"snapshot_id": "source-1"}),
    )
    monkeypatch.setattr(
        serving_replay,
        "ALL_MODEL_CONFIGS",
        (
            SimpleNamespace(
                model_name="Winner",
                target_name="winner",
                dataset="map",
                problem_type="classification",
            ),
        ),
    )

    matched = replay_sealed_features("candidate", inputs=inputs, snapshot=snapshot)
    assert matched["value_parity_passed"] is True
    assert matched["targets"]["winner"]["compared"] == 1

    pd.DataFrame({"elo_diff": [101]}).to_parquet(feature_path)
    mismatched = replay_sealed_features("candidate", inputs=inputs, snapshot=snapshot)
    assert mismatched["value_parity_passed"] is False
    assert mismatched["targets"]["winner"]["failures"][0]["different_columns"] == [
        "elo_diff"
    ]

    for exception in (TeamStateUnavailableError, InsufficientRosterHistoryError):

        def no_prior_state(*_args, error=exception):
            raise error("no prior state")

        monkeypatch.setattr(serving_replay, "_historical_team", no_prior_state)
        unavailable = replay_sealed_features(
            "candidate", inputs=inputs, snapshot=snapshot
        )
        assert unavailable["value_parity_passed"] is False
        assert unavailable["targets"]["winner"]["compared"] == 0
        assert unavailable["targets"]["winner"]["failures"] == []
        assert unavailable["targets"]["winner"]["unavailable"] == [
            {"gameid": "match-1", "reason": "no prior state"}
        ]
