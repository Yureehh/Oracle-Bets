import json
from pathlib import Path

import pytest
from lol_bets.module import LoLBetsModule
from oracle_bets_core.pd import pd

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILES = [
    ROOT / "config/lol/data_ingestion/import_columns.json",
    ROOT / "config/lol/training/training_team_config.json",
    ROOT / "config/lol/training/training_player_config.json",
    ROOT / "config/lol/training/training_compact_team_config.json",
    ROOT / "config/lol/training/training_compact_player_config.json",
    ROOT / "config/lol/training/flattened_team_config.json",
    ROOT / "config/lol/training/flattened_player_config.json",
]
TEAM_ALIASES_CONFIG = ROOT / "config/lol/data_ingestion/team_aliases.json"
NEW_TEAM_FEATURES = {
    "first_pick",
    "strength_pool",
    "strength_pool_win_likelihood",
    "ema_kill_share",
    "ema_tower_share",
    "ema_epic_monsters",
    "ema_structure_control",
    "ema_golddiff_shareat15",
    "ema_xpdiff_shareat15",
    "ema_csdiff_shareat15",
    "ema_golddiff_shareat25",
    "ema_xpdiff_shareat25",
    "ema_csdiff_shareat25",
    "ema_win_gamelength",
    "ema_loss_gamelength",
    "days_since_last_game",
    "roster_continuity",
    "roster_uncertainty",
    "rating_uncertainty",
    "elo",
    "elo_win_likelihood",
    "glicko2_mu",
    "glicko2_phi",
    "glicko2_win_likelihood",
    "pl_mu",
    "pl_sigma",
    "pl_win_likelihood",
    "trueskill_mu",
    "trueskill_sigma",
    "trueskill_win_likelihood",
    "game",
    "split",
}
NEW_TEAM_FEATURE_PREFIXES = ("diff_ema_",)
MODERATE_OPTUNA_TRIALS = 100
TEAMS_PER_IDENTITY_MERGE = 2


def _load_cols(path: str, key: str) -> list[str]:
    return json.loads((ROOT / path).read_text())[key]


def _flatten_config_values(value):
    if isinstance(value, dict):
        for nested in value.values():
            yield from _flatten_config_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _flatten_config_values(nested)
    else:
        yield value


def test_every_tracked_config_is_valid_json():
    for path in (ROOT / "config").rglob("*.json"):
        assert json.loads(path.read_text()) is not None, path


def test_product_league_profiles_and_read_only_market_are_consistent():
    product = json.loads((ROOT / "config/product/product.json").read_text())
    league_config = json.loads(
        (ROOT / "config/lol/data_ingestion/considered_leagues.json").read_text()
    )
    profiles = league_config["profiles"]
    training_profile = product["leagues"]["training_profile"]
    prediction_profile = product["leagues"]["prediction_profile"]
    exclusions = set(product["leagues"]["actionable_exclusions"])

    assert product["market"] == {"read_only": True, "quote_ttl_seconds": 120}
    assert product["timezone"] == "Europe/Rome"
    assert training_profile in profiles
    assert prediction_profile in profiles
    assert exclusions <= set(profiles[prediction_profile])
    assert set(profiles[prediction_profile]) <= set(profiles[training_profile])


def test_training_configs_do_not_use_deprecated_feature_families():
    forbidden = ("atakhan", "_std")

    for path in CONFIG_FILES:
        values = [
            str(v).casefold()
            for v in _flatten_config_values(json.loads(path.read_text()))
        ]
        assert not [v for v in values if any(token in v for token in forbidden)], path


def test_team_training_config_matches_existing_artifact_columns():
    artifact = ROOT / "data/lol/processed/teams/training_teams.parquet"
    if not artifact.exists():
        pytest.skip("team training artifact is not available")

    df = pd.read_parquet(artifact)
    cols = _load_cols("config/lol/training/training_team_config.json", "team_features")
    renamed = {c.replace("_before", "") for c in cols}
    missing = renamed - set(df.columns)
    if missing == {"first_pick"}:
        pytest.skip("team training artifact predates first-pick feature import")
    if missing and (
        missing <= NEW_TEAM_FEATURES
        or all(
            item in NEW_TEAM_FEATURES or item.startswith(NEW_TEAM_FEATURE_PREFIXES)
            for item in missing
        )
    ):
        pytest.skip("team training artifact predates configured feature additions")

    assert missing == set()


def test_first_pick_is_team_level_ingestion_feature():
    import_config = json.loads(
        (ROOT / "config/lol/data_ingestion/import_columns.json").read_text()
    )
    team_config = json.loads(
        (ROOT / "config/lol/training/training_team_config.json").read_text()
    )

    assert "first_pick" in import_config["team"]
    assert "first_pick" not in import_config["player"]
    assert "first_pick" not in team_config["team_features"]
    assert "side_win_likelihood" not in team_config["team_features"]


def test_team_configs_use_strength_pool_and_drop_noisy_economy_columns():
    import_config = json.loads(
        (ROOT / "config/lol/data_ingestion/import_columns.json").read_text()
    )
    team_config = json.loads(
        (ROOT / "config/lol/training/training_team_config.json").read_text()
    )
    flattened_config = json.loads(
        (ROOT / "config/lol/training/flattened_team_config.json").read_text()
    )
    values = json.dumps([import_config, team_config, flattened_config])

    assert "strength_pool" in team_config["team_features"]
    assert "strength_pool_win_likelihood" in team_config["team_features"]
    assert "gspd" not in values
    assert "gpr" not in values


def test_checkpoint_configs_cover_10_15_20_25_minutes():
    import_config = json.loads(
        (ROOT / "config/lol/data_ingestion/import_columns.json").read_text()
    )
    team_flat = json.loads(
        (ROOT / "config/lol/training/flattened_team_config.json").read_text()
    )["flattened_cols"]
    player_flat = json.loads(
        (ROOT / "config/lol/training/flattened_player_config.json").read_text()
    )["flattened_cols"]

    for minute in (10, 15, 20, 25):
        assert f"goldat{minute}" in import_config["team"]
        assert f"goldat{minute}" in import_config["player"]
        assert f"ema_goldat{minute}_after" in team_flat
        assert f"ema_goldat{minute}_after" in player_flat


def test_compact_configs_prefer_explicit_diff_ema_features():
    compact_team = json.loads(
        (ROOT / "config/lol/training/training_compact_team_config.json").read_text()
    )["team_features"]
    compact_player = json.loads(
        (ROOT / "config/lol/training/training_compact_player_config.json").read_text()
    )["player_features"]

    assert "diff_ema_golddiffat15_before" in compact_team
    assert "diff_ema_kda_before" in compact_player
    assert "ema_golddiffat15_before" not in compact_team
    assert "ema_kda_before" not in compact_player


def test_winner_configs_exclude_unavailable_and_constant_matchup_features():
    team = json.loads(
        (ROOT / "config/lol/training/training_team_config.json").read_text()
    )["team_features"]
    compact = json.loads(
        (ROOT / "config/lol/training/training_compact_team_config.json").read_text()
    )["team_features"]
    player = json.loads(
        (ROOT / "config/lol/training/training_player_config.json").read_text()
    )["player_features"]
    prohibited = {
        "game_in_series",
        "is_bo1",
        "is_bo3",
        "is_bo5",
        "is_deciding_game",
        "h2h_games_before",
        "h2h_wins_before",
        "h2h_win_rate_before",
        "season_avg_gamelength",
    }

    assert not prohibited & set(team)
    assert not prohibited & set(compact)
    assert "patch" not in player


def test_final_team_style_features_are_configured_for_ema_and_training():
    team_flat = json.loads(
        (ROOT / "config/lol/training/flattened_team_config.json").read_text()
    )["flattened_cols"]
    team_train = json.loads(
        (ROOT / "config/lol/training/training_team_config.json").read_text()
    )["team_features"]
    compact_team = json.loads(
        (ROOT / "config/lol/training/training_compact_team_config.json").read_text()
    )["team_features"]

    assert "ema_golddiff_growth_10_15_after" in team_flat
    assert "ema_team_vspm_after" in team_flat
    assert "ema_win_gamelength_after" in team_flat
    assert "diff_ema_golddiff_growth_10_15_before" in team_train
    assert "diff_ema_team_vspm_before" in team_train
    assert "ema_win_gamelength_before" in team_train
    assert "diff_ema_golddiff_growth_10_15_before" in compact_team
    assert "diff_ema_team_vspm_before" in compact_team
    assert "diff_ema_win_gamelength_before" not in compact_team


def test_reviewed_rating_hyperparameters_are_tracked_inputs():
    gitignore = (ROOT / ".gitignore").read_text()
    defaults = json.loads(
        (ROOT / "config/lol/hyperparameters/default_models_parameters.json").read_text()
    )
    tuned_dir = ROOT / "config/lol/hyperparameters/tuned/ratings"
    expected = {
        "leagues_elo_hyperparameters.json",
        "player_elo_hyperparameters.json",
        "player_glicko_hyperparameters.json",
        "player_pl_hyperparameters.json",
        "player_trueskill_hyperparameters.json",
        "team_elo_hyperparameters.json",
        "team_glicko_hyperparameters.json",
        "team_pl_hyperparameters.json",
        "team_trueskill_hyperparameters.json",
    }

    assert "/config/lol/hyperparameters/tuned/ratings/*.json" not in gitignore
    assert expected == {path.name for path in tuned_dir.glob("*.json")}
    assert defaults["optuna"]["trials"] == MODERATE_OPTUNA_TRIALS


def test_league_strength_artifacts_have_current_schema_when_present():
    league_elo = ROOT / "data/lol/processed/ratings/league_elo.parquet"
    team_mapping = ROOT / "data/lol/processed/ratings/team_league_mapping.parquet"
    if not league_elo.exists() or not team_mapping.exists():
        pytest.skip("league strength artifacts are not available")

    league_cols = set(pd.read_parquet(league_elo).columns)
    mapping_cols = set(pd.read_parquet(team_mapping).columns)
    expected_league_cols = {
        "league",
        "elo",
        "strength_pool",
        "strength_pool_elo",
        "strength_pool_cross_games",
    }
    expected_mapping_cols = {"teamid", "league", "strength_pool"}

    if expected_league_cols <= league_cols and expected_mapping_cols <= mapping_cols:
        return

    failed = {
        check.name: check.reason for check in LoLBetsModule().artifact_health().checks
    }
    assert "outdated schema" in failed.get("league elo", "") or "outdated schema" in (
        failed.get("team league mapping", "")
    )


def test_player_artifact_gap_is_explicit():
    player_paths = [
        ROOT / "data/lol/processed/players/training_players.parquet",
        ROOT / "data/lol/processed/players/flattened_players.parquet",
    ]
    if all(path.exists() for path in player_paths):
        pytest.skip("player artifacts are available")

    failed = {
        check.name
        for health in (
            LoLBetsModule().artifact_health(),
            LoLBetsModule().training_artifact_health(),
        )
        for check in health.checks
        if not check.ok
    }

    assert {"flattened players", "training players"} <= failed


def test_team_name_replacements_have_current_raw_evidence():
    artifact = ROOT / "data/lol/raw/raw_data.parquet"
    if not artifact.exists():
        pytest.skip("raw data artifact is not available")

    alias_config = json.loads(TEAM_ALIASES_CONFIG.read_text())
    replacements = alias_config.get("historical_identity_merges", [])
    assert replacements

    teams = pd.read_parquet(
        artifact,
        columns=["position", "teamid", "teamname"],
    )
    teams = teams.loc[
        teams["position"] == "team",
        ["teamid", "teamname"],
    ].drop_duplicates()
    observed = set(
        zip(teams["teamid"].astype(str), teams["teamname"].astype(str), strict=False)
    )

    missing = [
        team
        for replacement in replacements
        for team in replacement
        if (team["teamid"], team["name"]) not in observed
    ]

    assert missing == []


def test_team_alias_sections_have_distinct_valid_semantics():
    config = json.loads(TEAM_ALIASES_CONFIG.read_text())
    external = config["external_aliases"]
    merges = config["historical_identity_merges"]

    assert external
    assert all(source.strip() and target.strip() for source, target in external.items())
    assert all(len(pair) == TEAMS_PER_IDENTITY_MERGE for pair in merges)
    old_ids = [pair[0]["teamid"] for pair in merges]
    new_ids = [pair[1]["teamid"] for pair in merges]
    assert len(old_ids) == len(set(old_ids))
    assert not set(old_ids) & set(new_ids)
