import pickle
from pathlib import Path

import numpy as np
from lol_bets import module as lol_module
from lol_bets.module import LoLBetsModule, _check_calibrator_schema
from lol_bets.prediction_models.gbdt_model import (
    MetadataAwareProbabilityCalibrator,
    ProbabilityCalibrator,
    ProbabilityUncertaintyModel,
    PropDistributionCalibrator,
)
from oracle_bets_core.pd import pd


def _valid_flattened_players() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "teamname": ["T1"],
            "playername": ["Faker"],
            "position": ["mid"],
            "date": ["2026-01-01"],
            "gameid": ["g1"],
            "side": ["Blue"],
            "elo": [1500],
            "glicko2_mu": [1500],
            "glicko2_phi": [60],
            "pl_mu": [25],
            "pl_sigma": [1],
            "trueskill_mu": [25],
            "trueskill_sigma": [1],
        }
    )


def test_lol_health_reports_missing_training_player_artifact():
    health = LoLBetsModule().training_artifact_health()
    by_name = {check.name: check for check in health.checks}

    assert "training players" in by_name
    assert by_name["training players"].path.endswith("training_players.parquet")


def test_normal_health_does_not_require_experimental_next_map_artifacts():
    names = {check.name for check in LoLBetsModule().artifact_health().checks}

    assert not any("next-map" in name for name in names)


def test_lol_health_checks_are_file_based(tmp_path):
    path = tmp_path / "artifact.parquet"
    path.write_bytes(b"not empty")

    from lol_bets.module import _check_file

    ok = _check_file("sample", path)
    missing = _check_file("missing", Path(tmp_path / "missing.parquet"))

    assert ok.ok
    assert not missing.ok
    assert missing.reason == "missing"


def test_calibrator_health_rejects_legacy_probability_calibrator(tmp_path):
    path = tmp_path / "calibrator.pkl"
    with path.open("wb") as f:
        pickle.dump(ProbabilityCalibrator(method="raw", model=None), f)

    check = _check_calibrator_schema(
        "calibrator",
        path,
        required_attrs={"global_calibrator", "segments", "version"},
    )

    assert not check.ok
    assert "outdated calibrator schema" in check.reason


def test_calibrator_health_accepts_metadata_aware_probability_calibrator(tmp_path):
    path = tmp_path / "calibrator.pkl"
    artifact = MetadataAwareProbabilityCalibrator(
        global_calibrator=ProbabilityCalibrator(method="raw", model=None)
    )
    with path.open("wb") as f:
        pickle.dump(artifact, f)

    check = _check_calibrator_schema(
        "calibrator",
        path,
        required_attrs={"global_calibrator", "segments", "version"},
    )

    assert check.ok


def test_calibrator_health_accepts_metadata_prop_calibrator(tmp_path):
    path = tmp_path / "prop_calibrator.pkl"
    artifact = PropDistributionCalibrator(
        method="metadata_shrunk",
        global_residuals=np.array([0.0, 1.0]),
    )
    with path.open("wb") as f:
        pickle.dump(artifact, f)

    check = _check_calibrator_schema(
        "prop calibrator",
        path,
        required_attrs={"global_residuals", "segment_residuals", "version"},
    )

    assert check.ok


def test_calibrator_health_accepts_probability_uncertainty(tmp_path):
    path = tmp_path / "uncertainty.pkl"
    artifact = ProbabilityUncertaintyModel(
        global_residual_lower=-0.05,
        global_residual_upper=0.05,
        sample_count=100,
    )
    with path.open("wb") as f:
        pickle.dump(artifact, f)

    check = _check_calibrator_schema(
        "uncertainty",
        path,
        required_attrs={
            "bins",
            "confidence",
            "fit_split",
            "interval",
            "sample_count",
            "version",
        },
        expected_version=1,
    )

    assert check.ok


def test_lol_inference_health_checks_flattened_team_schema(tmp_path, monkeypatch):
    teams = tmp_path / "flattened_teams.parquet"
    players = tmp_path / "flattened_players.parquet"
    pd.DataFrame({"teamname": ["T1"]}).to_parquet(teams)
    pd.DataFrame(
        {
            "teamname": ["T1"],
            "playername": ["Faker"],
            "position": ["mid"],
            "date": ["2026-01-01"],
            "gameid": ["g1"],
            "side": ["Blue"],
            "elo": [1500],
            "glicko2_mu": [1500],
            "glicko2_phi": [60],
            "pl_mu": [25],
            "pl_sigma": [1],
            "trueskill_mu": [25],
            "trueskill_sigma": [1],
        }
    ).to_parquet(players)
    monkeypatch.setattr(lol_module, "FLATTENED_TEAMS", teams)
    monkeypatch.setattr(lol_module, "FLATTENED_PLAYERS", players)

    health = LoLBetsModule().artifact_health()
    flattened_teams = {check.name: check for check in health.checks}["flattened teams"]

    assert not flattened_teams.ok
    assert "teamid" in flattened_teams.reason


def test_lol_inference_health_rejects_duplicate_flattened_team_snapshots(
    tmp_path, monkeypatch
):
    teams = tmp_path / "flattened_teams.parquet"
    players = tmp_path / "flattened_players.parquet"
    pd.DataFrame(
        {
            "teamname": ["T1", "T1"],
            "teamid": ["t1", "t1"],
            "gameid": ["g1", "g2"],
            "date": ["2026-01-01", "2026-02-01"],
            "league": ["LCK", "LCK"],
            "elo": [1500, 1510],
            "glicko2_mu": [1500, 1510],
            "glicko2_phi": [60, 59],
            "pl_mu": [25, 26],
            "pl_sigma": [1, 1],
            "trueskill_mu": [25, 26],
            "trueskill_sigma": [1, 1],
        }
    ).to_parquet(teams)
    _valid_flattened_players().to_parquet(players)
    monkeypatch.setattr(lol_module, "FLATTENED_TEAMS", teams)
    monkeypatch.setattr(lol_module, "FLATTENED_PLAYERS", players)

    health = LoLBetsModule().artifact_health()
    flattened_teams = {check.name: check for check in health.checks}["flattened teams"]

    assert not flattened_teams.ok
    assert "duplicate flattened snapshots" in flattened_teams.reason


def test_lol_inference_health_allows_multiple_flattened_players_per_team_role(
    tmp_path, monkeypatch
):
    teams = tmp_path / "flattened_teams.parquet"
    players = tmp_path / "flattened_players.parquet"
    pd.DataFrame(
        {
            "teamname": ["T1"],
            "teamid": ["t1"],
            "gameid": ["g1"],
            "date": ["2026-01-01"],
            "league": ["LCK"],
        }
    ).to_parquet(teams)
    player_df = _valid_flattened_players()
    substitute = player_df.iloc[[0]].copy()
    substitute["playername"] = "Poby"
    substitute["date"] = "2026-01-02"
    player_df = pd.concat([player_df, substitute], ignore_index=True)
    player_df.to_parquet(players)
    monkeypatch.setattr(lol_module, "FLATTENED_TEAMS", teams)
    monkeypatch.setattr(lol_module, "FLATTENED_PLAYERS", players)

    health = LoLBetsModule().artifact_health()
    flattened_players = {check.name: check for check in health.checks}[
        "flattened players"
    ]

    assert flattened_players.ok
