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


def test_mutable_rating_tables_are_outside_the_serving_model_root():
    for source in (
        paths.RATING_LEAGUE_ELO,
        paths.RATING_TEAM_LEAGUES_MAPPING,
    ):
        assert source.is_relative_to(paths.PROCESSED_DIR)
        assert not source.is_relative_to(paths.MODELS_DIR)

    assert paths.LEAGUE_ELO.is_relative_to(paths.MODELS_DIR)
    assert paths.TEAM_LEAGUES_MAPPING.is_relative_to(paths.MODELS_DIR)
