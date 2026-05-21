from oracle_bets_core import paths


def test_lol_scoped_paths_use_lol_subdirectories():
    assert paths.DATA_DIR.parts[-2:] == ("data", "lol")
    assert paths.DATA_DIR.exists()
    assert paths.CONFIG_DIR.parts[-2:] == ("config", "lol")
    assert paths.MODELS_DIR.parts[-2:] == ("models", "lol")
    assert paths.REPORTS_DIR.parts[-2:] == ("reports", "lol")


def test_suite_root_points_at_repo():
    assert (paths.SUITE_ROOT / "pyproject.toml").exists()
    assert (paths.SUITE_ROOT / "config/lol").exists()
