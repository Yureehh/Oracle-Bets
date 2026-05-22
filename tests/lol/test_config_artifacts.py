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
TEAM_CLEANUP_CONFIG = (
    ROOT / "config/lol/data_ingestion/team_name_replacements_and_invalid_games.json"
)
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
}


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


def test_training_configs_do_not_use_deprecated_feature_families():
    forbidden = ("atakhan", "_std", "at10", "at20")

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
    if missing <= NEW_TEAM_FEATURES:
        pytest.skip("team training artifact predates strength-pool feature revamp")

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
    assert "first_pick" in team_config["team_features"]


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


def test_league_strength_artifacts_have_current_schema_when_present():
    league_elo = ROOT / "models/lol/league_elo.parquet"
    team_mapping = ROOT / "models/lol/team_league_mapping.parquet"
    if not league_elo.exists() or not team_mapping.exists():
        pytest.skip("league strength artifacts are not available")

    league_cols = set(pd.read_parquet(league_elo).columns)
    mapping_cols = set(pd.read_parquet(team_mapping).columns)
    expected_league_cols = {"league", "elo", "strength_pool", "strength_pool_elo"}
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


def test_manual_invalid_games_are_current_if_configured():
    artifact = ROOT / "data/lol/raw/raw_data.parquet"
    if not artifact.exists():
        pytest.skip("raw data artifact is not available")

    cleanup_config = json.loads(TEAM_CLEANUP_CONFIG.read_text())
    invalid_games = set(cleanup_config.get("invalid_games", []))
    if not invalid_games:
        return

    df = pd.read_parquet(artifact, columns=["gameid"])
    raw_games = set(df["gameid"].dropna().astype(str))

    assert invalid_games <= raw_games


def test_team_name_replacements_have_current_raw_evidence():
    artifact = ROOT / "data/lol/raw/raw_data.parquet"
    if not artifact.exists():
        pytest.skip("raw data artifact is not available")

    cleanup_config = json.loads(TEAM_CLEANUP_CONFIG.read_text())
    replacements = cleanup_config.get("team_name_replacements", [])
    if not replacements:
        return

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
