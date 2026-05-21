from pathlib import Path

from lol_bets.module import LoLBetsModule


def test_lol_health_reports_missing_training_player_artifact():
    health = LoLBetsModule().training_artifact_health()
    by_name = {check.name: check for check in health.checks}

    assert "training players" in by_name
    assert by_name["training players"].path.endswith("training_players.parquet")


def test_lol_health_checks_are_file_based(tmp_path):
    path = tmp_path / "artifact.parquet"
    path.write_bytes(b"not empty")

    from lol_bets.module import _check_file

    ok = _check_file("sample", path)
    missing = _check_file("missing", Path(tmp_path / "missing.parquet"))

    assert ok.ok
    assert not missing.ok
    assert missing.reason == "missing"
