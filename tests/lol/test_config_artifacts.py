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

    assert renamed <= set(df.columns)


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
