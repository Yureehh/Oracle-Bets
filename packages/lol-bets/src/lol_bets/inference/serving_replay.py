"""Retrospective value-level comparison of sealed training and serving features."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from oracle_bets_core.io_utils import load_model
from oracle_bets_core.paths import MODEL_REGISTRY_DIR, RAW_CURRENT_POINTER
from oracle_bets_core.pd import pd

from lol_bets.data_generation.ingestion.history import read_history_snapshot
from lol_bets.inference.match_predictor import MatchPredictor
from lol_bets.inference.roster import EXPECTED_ROLES
from lol_bets.inference.team import Team, TeamStateUnavailableError
from lol_bets.operations.models import ModelRegistry, _paths_fingerprint
from lol_bets.training import ALL_MODEL_CONFIGS, _candidate_training_paths

if TYPE_CHECKING:
    from pathlib import Path

    from lol_bets.inference.snapshots import FeatureSnapshot
    from lol_bets.operations.training_inputs import TrainingInputs

TEAMS_PER_GAME = 2


def compare_feature_values(
    expected: pd.Series, actual: pd.Series, *, atol: float = 1e-8
) -> list[str]:
    """List absent or different canonical values; NaN matches only NaN."""
    mismatches = []
    for column, left in expected.items():
        if column not in actual.index:
            mismatches.append(str(column))
            continue
        right = actual[column]
        if pd.isna(left) and pd.isna(right):
            continue
        if pd.isna(left) or pd.isna(right):
            mismatches.append(str(column))
            continue
        try:
            numeric = bool(np.isclose(float(left), float(right), rtol=0, atol=atol))
        except (TypeError, ValueError):
            numeric = str(left) == str(right)
        if not numeric:
            mismatches.append(str(column))
    return mismatches


def _selected_model_features(
    paths: dict[str, Path], model_name: str, expected: pd.DataFrame
) -> pd.DataFrame:
    selected = load_model(paths[f"{model_name}/{model_name}_final_features.pkl"])
    if (
        not isinstance(selected, list)
        or not selected
        or not all(isinstance(name, str) for name in selected)
        or not set(selected).issubset(expected.columns)
    ):
        raise ValueError(f"Sealed {model_name} model features are invalid")
    return expected.loc[:, selected]


def replay_sealed_features(  # noqa: PLR0915
    candidate_id: str,
    *,
    inputs: TrainingInputs,
    snapshot: FeatureSnapshot,
    registry_root: Path = MODEL_REGISTRY_DIR,
    sample_per_target: int = 8,
) -> dict[str, Any]:
    """
    Rebuild sampled historical rows with the real Team/MatchPredictor path.

    Historical roster and a later-published snapshot establish calculation parity
    only. They do not prove the source or lineup was available before the match.
    """
    if sample_per_target < 1:
        raise ValueError("sample_per_target must be positive")
    map_generation = inputs.manifests["map"]
    if (
        snapshot.manifest["training_generation_id"] != map_generation["generation_id"]
        or snapshot.manifest["code_sha256"] != map_generation["code_sha256"]
        or snapshot.manifest["source_manifest"] != map_generation["source"]
    ):
        raise ValueError("Serving snapshot and training inputs have different lineage")
    inputs.assert_unchanged()
    registry = ModelRegistry(registry_root)
    paths = _verified_candidate_paths(registry, candidate_id)
    predictor = MatchPredictor(model_id=candidate_id, registry_root=registry_root)
    history_path, history_manifest = read_history_snapshot(
        pointer_path=RAW_CURRENT_POINTER
    )
    if history_manifest["snapshot_id"] != map_generation["source"]["snapshot_id"]:
        raise ValueError("Historical roster source changed after training")
    history = pd.read_parquet(
        history_path,
        columns=["gameid", "side", "position", "playername", "teamname"],
    )
    results: dict[str, Any] = {}
    for cfg in ALL_MODEL_CONFIGS:
        model_name = f"{cfg.model_name}_LightGBM"
        prefix = f"_evaluation/{model_name}"
        expected = _selected_model_features(
            paths, model_name, pd.read_parquet(paths[f"{prefix}/features.parquet"])
        )
        labels = pd.read_parquet(paths[f"{prefix}/labels.parquet"])
        if len(expected) != len(labels) or expected.empty:
            raise ValueError(f"Sealed {cfg.target_name} evaluation is misaligned")
        dataset = "map" if cfg.dataset == "map" else "series"
        team_rows = inputs.frames[f"{dataset}_teams"]
        sampled = np.linspace(
            0, len(expected) - 1, min(sample_per_target, len(expected)), dtype=int
        )
        failures: list[dict[str, Any]] = []
        unavailable: list[dict[str, str]] = []
        compared = 0
        for index in sampled:
            label = labels.iloc[int(index)]
            gameid = str(label["gameid"])
            game = team_rows.loc[team_rows["gameid"].astype(str).eq(gameid)]
            source_id = label.get("source_gameid")
            source_gameid = str(source_id) if pd.notna(source_id) else gameid
            roster_rows = history.loc[history["gameid"].astype(str).eq(source_gameid)]
            try:
                _require_pair(game)
                teams = [
                    _historical_team(row, roster_rows, label["date"], snapshot)
                    for _, row in game.iterrows()
                ]
                if cfg.problem_type == "classification":
                    best_of = int(game.iloc[0].get("best_of", 1) or 1)
                    actual, _, _ = predictor._current_matchup_features(
                        teams[0], teams[1], match_type=f"bo{best_of}", gameid=gameid
                    )
                else:
                    ordered = sorted(teams, key=lambda team: team.side != "Blue")
                    actual = predictor.calculate_prop_features(
                        ordered[0], ordered[1], account_for_side=False
                    )
                mismatches = compare_feature_values(
                    expected.iloc[int(index)], actual.iloc[0]
                )
                compared += 1
                if mismatches:
                    failures.append(
                        {
                            "gameid": gameid,
                            "different_columns": mismatches[:30],
                            "different_count": len(mismatches),
                        }
                    )
            except TeamStateUnavailableError as error:
                unavailable.append({"gameid": gameid, "reason": str(error)})
            except (KeyError, TypeError, ValueError) as error:
                failures.append({"gameid": gameid, "error": str(error)})
        results[cfg.target_name] = {
            "sampled": len(sampled),
            "compared": compared,
            "matching": compared
            - sum("different_columns" in item for item in failures),
            "unavailable": unavailable,
            "failures": failures,
        }
    inputs.assert_unchanged()
    return {
        "candidate_id": candidate_id,
        "feature_snapshot_id": snapshot.snapshot_id,
        "training_generation_id": map_generation["generation_id"],
        "source_snapshot_id": history_manifest["snapshot_id"],
        "code_sha256": map_generation["code_sha256"],
        "observed_at": datetime.now(UTC).isoformat(),
        "availability_status": "retrospective_calculation_only",
        "value_parity_passed": all(
            row["compared"] > 0 and not row["failures"] for row in results.values()
        ),
        "targets": results,
    }


def _verified_candidate_paths(
    registry: ModelRegistry, candidate_id: str
) -> dict[str, Path]:
    paths = registry.verified_artifact_paths(model_id=candidate_id)
    candidate_manifest = json.loads(
        (registry.candidates / candidate_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if candidate_manifest["data_manifest"] != _paths_fingerprint(
        _candidate_training_paths()
    ):
        raise ValueError("Candidate was trained from a different data generation")
    return paths


def _historical_team(
    row: pd.Series,
    roster_rows: pd.DataFrame,
    match_at: Any,
    snapshot: FeatureSnapshot,
) -> Team:
    matching = roster_rows.loc[
        roster_rows["teamname"]
        .astype(str)
        .str.casefold()
        .eq(str(row["teamname"]).casefold())
    ]
    roster: dict[str, str | None] = {
        str(player["position"]).casefold(): str(player["playername"])
        for _, player in matching.iterrows()
        if str(player["position"]).casefold() != "team"
    }
    if set(roster) != set(EXPECTED_ROLES) or not all(roster.values()):
        raise ValueError("historical starting roster is incomplete")
    return Team(
        name=str(row["teamname"]),
        side=str(row["side"]),
        league=str(row["league"]) if pd.notna(row.get("league")) else None,
        as_of_date=match_at,
        decision_at=datetime.now(UTC),
        feature_snapshot=snapshot,
        roster=roster,
    )


def _require_pair(game: pd.DataFrame) -> None:
    if len(game) != TEAMS_PER_GAME:
        raise ValueError("expected two historical teams")


def replay_digest(result: dict[str, Any]) -> str:
    """Stable receipt for a report; excludes only the report's wall-clock time."""
    payload = {key: value for key, value in result.items() if key != "observed_at"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
