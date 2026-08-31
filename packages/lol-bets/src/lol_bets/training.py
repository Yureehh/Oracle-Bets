"""
Models Training

Initializes and trains the project's models using team & player training tables.
Outputs serialized models to MODELS_DIR.

Models covered:
- Outcome prediction (classification)
- Gamelength prediction (regression)
- Total kills prediction (regression)
- Total towers prediction (regression)

Usage:
    oracle-bets lol train --targets all --feature-selection report
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

from oracle_bets_core.io_utils import (
    atomic_write_text,
    load_model,
    load_training_data,
    store_model,
)
from oracle_bets_core.logger import logger
from oracle_bets_core.paths import (
    DEFAULT_MODELS_PARAMETERS,
    MODELS_DIR,
    NEXT_MAP_PLAYER_DATA,
    NEXT_MAP_TEAM_DATA,
    RATING_HYPERPARAMETER_PROVENANCE,
    RATING_LEAGUE_ELO,
    RATING_TEAM_LEAGUES_MAPPING,
    RAW_CURRENT_POINTER,
    RAW_DATA,
    REPORTS_DIR,
    SERIES_MANIFEST,
    SERIES_REJECTIONS,
    SERIES_WINNER_PLAYER_DATA,
    SERIES_WINNER_TEAM_DATA,
    TRAINING_PLAYER_DATA,
    TRAINING_TEAM_DATA,
    TUNED_LIGHTGBM_HYPERPARAMETERS,
    TUNED_RATING_HYPERPARAMETERS,
)
from oracle_bets_core.pd import pd

from lol_bets.data_generation.ingestion.history import (
    current_history_data_path,
    read_history_snapshot,
)
from lol_bets.data_generation.ingestion.quality import normalize_result
from lol_bets.module import LoLBetsModule
from lol_bets.operations.provenance import (
    RepositoryProvenance,
    require_clean_repository,
)
from lol_bets.prediction_models.gbdt_model import (
    DEFAULT_SELECTED_MAX_FEATURES,
    valid_probability_calibration_artifacts,
)
from lol_bets.prediction_models.lightgbm_model import LightGBMModel
from lol_bets.prediction_models.prop_features import PROP_TARGETS
from lol_bets.prediction_models.winner_model import WinnerLightGBMModel

# ───────────────────────────────  types / config  ─────────────────────────────

ProblemType = Literal["classification", "regression"]
FeatureSelectionMethod = Literal["none", "importance", "cumulative", "report"]
TrainingFeatureSet = Literal["full", "compact", "selected", "auto"]
CalibrationMode = Literal["auto", "none"]
CalibrationMethod = Literal["raw", "sigmoid", "isotonic", "auto"]
MODEL_FILE_EXTENSION = "pkl"
MIN_CALIBRATION_REVIEW_SLOPE = 0.8
MAX_CALIBRATION_REVIEW_SLOPE = 1.2
MAX_CALIBRATION_REVIEW_INTERCEPT = 0.10
MIN_DIAGNOSTIC_COHORT_SIZE = 30
MAX_DIAGNOSTIC_COHORT_REGRESSION = 0.02


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    target_name: str
    target_column: str
    problem_type: ProblemType
    dataset: Literal["map", "series", "next_map"] = "map"
    validate: bool = True


ALL_MODEL_CONFIGS: tuple[ModelConfig, ...] = (
    ModelConfig(
        model_name="OutcomePrediction",
        target_name="map_winner",
        target_column="result",
        problem_type="classification",
    ),
    ModelConfig(
        "SeriesWinnerPrediction",
        "series_winner",
        "result",
        "classification",
        "series",
    ),
    ModelConfig("GamelengthPrediction", "gamelength", "gamelength", "regression"),
    ModelConfig("TotalKillsPrediction", "total_kills", "total_kills", "regression"),
    ModelConfig("TotalTowersPrediction", "total_towers", "total_towers", "regression"),
)
EXPERIMENTAL_MODEL_CONFIGS: tuple[ModelConfig, ...] = (
    ModelConfig(
        "NextMapWinnerPrediction",
        "next_map_winner",
        "result",
        "classification",
        "next_map",
    ),
)
MODEL_CONFIGS = (*ALL_MODEL_CONFIGS, *EXPERIMENTAL_MODEL_CONFIGS)
WINNER_TUNING_MODELS = {
    "series_winner": "SeriesWinnerPrediction_LightGBM",
    "next_map_winner": "NextMapWinnerPrediction_LightGBM",
}

TARGET_ALIASES = {
    "outcome": "map_winner",
    "map_winner": "map_winner",
    "result": "map_winner",
    "winner": "series_winner",
    "match_winner": "series_winner",
    "series_winner": "series_winner",
    "next_map_winner": "next_map_winner",
    "gamelength": "gamelength",
    "game_length": "gamelength",
    "length": "gamelength",
    "total_kills": "total_kills",
    "kills": "total_kills",
    "total_towers": "total_towers",
    "towers": "total_towers",
}
EXPECTED_TEAM_ROWS_PER_GAME = 2
EXPECTED_PLAYER_ROWS_PER_GAME = 10
EXPECTED_PLAYERS_PER_SIDE = 5
EXPECTED_SIDES = {"Blue", "Red"}
EXPECTED_POSITIONS = {"top", "jng", "mid", "bot", "sup"}
EXPECTED_RESULT_SUM = 1.0
TRAINING_REPORT_RETENTION = 12
RANDOM_CLASSIFIER_LOG_LOSS = 0.693147
RANDOM_CLASSIFIER_BRIER = 0.25
WEAK_REGRESSION_R2 = 0.1
TUNING_REVIEW_SCHEMA_VERSION = 4


# ───────────────────────────────  helpers  ───────────────────────────────────


def _model_path(
    name: str,
    ext: str = MODEL_FILE_EXTENSION,
    *,
    root: Path = MODELS_DIR,
) -> Path:
    return root / name / f"{name}.{ext}"


def _check_target_presence(team_df: pd.DataFrame, target: str) -> None:
    """
    Minimal guard: team targets (result / gamelength / totals) must exist
    in the team training table before we hand off to the model pipeline.
    """
    if target not in team_df.columns:
        msg = (
            f"Target column '{target}' not found in team training data "
            f"(available: {len(team_df.columns)} columns)."
        )
        raise ValueError(msg)


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        msg = f"{label} training data missing required columns: {sorted(missing)}"
        raise ValueError(msg)


def _validate_team_training_table(team_df: pd.DataFrame) -> None:
    team_counts = team_df.groupby("gameid", observed=True).size()
    bad_team_counts = team_counts[team_counts != EXPECTED_TEAM_ROWS_PER_GAME]
    if not bad_team_counts.empty:
        sample = bad_team_counts.head(5).to_dict()
        msg = (
            "Team training data must have exactly two side rows per game; "
            f"sample={sample}"
        )
        raise ValueError(msg)

    side_sets = team_df.groupby("gameid", observed=True)["side"].agg(
        lambda values: set(values.astype(str))
    )
    bad_sides = side_sets[side_sets != EXPECTED_SIDES]
    if not bad_sides.empty:
        sample = {gameid: sorted(sides) for gameid, sides in bad_sides.head(5).items()}
        msg = f"Team training data must contain Blue and Red for each game; sample={sample}"
        raise ValueError(msg)

    result = normalize_result(team_df["result"])
    if result.isna().any() or not result.isin([0, 1]).all():
        msg = "Team training data result must be binary 0/1 or win/loss labels."
        raise ValueError(msg)
    result_sums = result.groupby(team_df["gameid"], observed=True).sum()
    bad_results = result_sums[result_sums != EXPECTED_RESULT_SUM]
    if not bad_results.empty:
        msg = (
            "Team training data must have exactly one winner per game; "
            f"sample={bad_results.head(5).to_dict()}"
        )
        raise ValueError(msg)

    for target in PROP_TARGETS:
        if target not in team_df.columns:
            continue
        target_values = pd.to_numeric(team_df[target], errors="coerce")
        disagreements = target_values.groupby(team_df["gameid"], observed=True).nunique(
            dropna=True
        )
        disagreements = disagreements[disagreements > 1]
        if not disagreements.empty:
            msg = (
                f"Team training data target '{target}' must agree across both sides; "
                f"sample={disagreements.head(5).to_dict()}"
            )
            raise ValueError(msg)


def _validate_player_training_table(
    team_df: pd.DataFrame, player_df: pd.DataFrame
) -> None:
    team_games = set(team_df["gameid"])
    player_games = set(player_df["gameid"])
    if team_games != player_games:
        missing_players = sorted(team_games - player_games)[:5]
        extra_players = sorted(player_games - team_games)[:5]
        msg = (
            "Team/player training gameids must match; "
            f"missing_player_games={missing_players}, extra_player_games={extra_players}"
        )
        raise ValueError(msg)

    player_counts = player_df.groupby("gameid", observed=True).size()
    bad_player_counts = player_counts[player_counts != EXPECTED_PLAYER_ROWS_PER_GAME]
    if not bad_player_counts.empty:
        msg = (
            "Player training data must have ten player rows per game; "
            f"sample={bad_player_counts.head(5).to_dict()}"
        )
        raise ValueError(msg)

    side_position_counts = player_df.groupby(["gameid", "side"], observed=True).size()
    bad_side_counts = side_position_counts[
        side_position_counts != EXPECTED_PLAYERS_PER_SIDE
    ]
    if not bad_side_counts.empty:
        msg = (
            "Player training data must have five players per game side; "
            f"sample={bad_side_counts.head(5).to_dict()}"
        )
        raise ValueError(msg)

    position_sets = player_df.groupby(["gameid", "side"], observed=True)[
        "position"
    ].agg(lambda values: set(values.astype(str).str.casefold()))
    bad_positions = position_sets[position_sets != EXPECTED_POSITIONS]
    if not bad_positions.empty:
        sample = {
            str(key): sorted(positions)
            for key, positions in bad_positions.head(5).items()
        }
        msg = (
            "Player training data must contain one top/jng/mid/bot/sup per side; "
            f"sample={sample}"
        )
        raise ValueError(msg)


def validate_training_tables(team_df: pd.DataFrame, player_df: pd.DataFrame) -> None:
    """Fail fast on malformed supervised tables before model training."""
    _require_columns(team_df, {"gameid", "side", "result"}, "team")
    _require_columns(player_df, {"gameid", "side", "position"}, "player")
    _validate_team_training_table(team_df)
    _validate_player_training_table(team_df, player_df)


def parse_training_targets(targets: str) -> tuple[ModelConfig, ...]:
    """Resolve CLI target selectors to concrete model configs."""
    raw = targets.strip().casefold()
    by_target = {cfg.target_name: cfg for cfg in MODEL_CONFIGS}
    if not raw or raw == "all":
        return ALL_MODEL_CONFIGS
    if raw == "props":
        return tuple(by_target[target] for target in PROP_TARGETS)

    selected: list[ModelConfig] = []
    unknown: list[str] = []
    for token in (part.strip().casefold() for part in raw.split(",")):
        if not token:
            continue
        resolved = TARGET_ALIASES.get(token)
        if resolved is None:
            unknown.append(token)
            continue
        cfg = by_target[resolved]
        if cfg not in selected:
            selected.append(cfg)

    if unknown:
        valid = ", ".join(sorted([*TARGET_ALIASES, "all", "props"]))
        msg = (
            f"Unknown training target(s): {', '.join(unknown)}. Valid values: {valid}."
        )
        raise ValueError(msg)
    if not selected:
        msg = "No training targets selected."
        raise ValueError(msg)
    return tuple(selected)


# ───────────────────────────────  core training  ─────────────────────────────


def _lightgbm_model_name(base_name: str) -> str:
    return f"{base_name}_LightGBM"


def initialize_and_train_model(
    cfg: ModelConfig,
    training_team_data: pd.DataFrame,
    training_player_data: pd.DataFrame,
    feature_selection: FeatureSelectionMethod = "none",
    force_retune: bool = False,
    feature_set: TrainingFeatureSet = "compact",
    max_features: int = DEFAULT_SELECTED_MAX_FEATURES,
    calibration: CalibrationMode = "auto",
    calibration_method: CalibrationMethod = "auto",
    calibration_size: float = 0.15,
    tune_size: float = 0.10,
    test_size: float = 0.15,
    artifact_root: Path = MODELS_DIR,
    report_root: Path | None = None,
    run_id: str | None = None,
    dataset_fingerprint: str | None = None,
) -> Path | None:
    """
    Initialize, train (and optionally validate) a model defined by `cfg`.
    Returns the stored model path on success.
    """
    _check_target_presence(training_team_data, cfg.target_column)

    # Add model type suffix to prevent overwriting different model types
    full_model_name = _lightgbm_model_name(cfg.model_name)

    logger.info(
        f"Initializing '{full_model_name}' "
        f"({cfg.problem_type}, model=lightgbm) with target='{cfg.target_column}'…"
    )

    if cfg.problem_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported problem type: {cfg.problem_type}")
    model_class = WinnerLightGBMModel if cfg.dataset != "map" else LightGBMModel
    model = model_class(
        model_name=full_model_name,
        problem_type=cfg.problem_type,
        team_data=training_team_data,
        player_data=training_player_data,
        force_retune=force_retune,
        feature_set=feature_set,
        max_features=max_features,
        allow_hparam_schema_drift=cfg.dataset == "map",
        calibration=calibration,
        calibration_method=calibration_method,
        calibration_size=calibration_size,
        tune_size=tune_size,
        test_size=test_size,
        artifact_root=artifact_root,
        report_root=report_root,
        run_id=run_id or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S"),
        dataset_fingerprint=dataset_fingerprint,
    )

    start = dt.datetime.now()
    # Allow model to do its own splits/joins/feature selection internally
    model.preprocess_data(target_col=cfg.target_column)

    logger.info(
        f"Training '{full_model_name}' (validate={cfg.validate}, "
        f"feature_selection={feature_selection}, feature_set={feature_set}, "
        f"max_features={max_features})…"
    )
    trained_model = model.train_and_validate_model(
        target_col=cfg.target_column,
        validate=cfg.validate,
        feature_selection=feature_selection,
    )
    elapsed = (dt.datetime.now() - start).total_seconds()

    logger.info(f"'{full_model_name}' training complete in {elapsed:.2f}s.")

    path = _model_path(full_model_name, root=artifact_root)
    store_model(path, trained_model, full_model_name, logger)
    logger.info(f"Stored trained model: {path}\n")

    return path


def _training_frames_for_config(
    cfg: ModelConfig,
    map_teams: pd.DataFrame,
    map_players: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cfg.dataset == "map":
        return map_teams, map_players
    paths = (
        (SERIES_WINNER_TEAM_DATA, SERIES_WINNER_PLAYER_DATA)
        if cfg.dataset == "series"
        else (NEXT_MAP_TEAM_DATA, NEXT_MAP_PLAYER_DATA)
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Winner V2 datasets are missing; run `oracle-bets lol build-series`: "
            f"{missing}"
        )
    return pd.read_parquet(paths[0]), pd.read_parquet(paths[1])


def _feature_set_for_config(
    cfg: ModelConfig,
    requested: TrainingFeatureSet,
) -> TrainingFeatureSet:
    """Freeze the reviewed Winner research schema for routine training."""
    if cfg.dataset not in {"series", "next_map"} or requested == "auto":
        return requested
    path = (
        TUNED_LIGHTGBM_HYPERPARAMETERS / f"{_lightgbm_model_name(cfg.model_name)}.json"
    )
    try:
        feature_set = json.loads(path.read_text(encoding="utf-8"))["metadata"][
            "feature_set"
        ]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return "full"
    return feature_set if feature_set in {"full", "compact"} else "full"


def train_models(  # noqa: PLR0915
    feature_selection: FeatureSelectionMethod = "none",
    targets: str = "all",
    force_retune: bool = False,
    feature_set: TrainingFeatureSet = "compact",
    max_features: int = DEFAULT_SELECTED_MAX_FEATURES,
    calibration: CalibrationMode = "auto",
    calibration_method: CalibrationMethod = "auto",
    calibration_size: float = 0.15,
    tune_size: float = 0.10,
    test_size: float = 0.15,
) -> Path:
    """Train all configured models using shared training tables."""
    repository = _training_preflight()
    selected_models = parse_training_targets(targets)
    _validate_reviewed_winner_parameters(selected_models, force_retune=force_retune)

    try:
        logger.info("Loading training data…")
        team_df, player_df = load_training_data(
            TRAINING_TEAM_DATA, TRAINING_PLAYER_DATA, logger
        )
        validate_training_tables(team_df, player_df)
    except Exception as e:
        logger.exception(f"Failed to load training data: {e}")
        raise

    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    report_root = REPORTS_DIR / "training" / "runs" / run_id
    promotable = tuple(selected_models) == ALL_MODEL_CONFIGS and not force_retune
    artifact_root = (
        MODELS_DIR / ".staging" / run_id if promotable else report_root / "artifacts"
    )
    trained: list[str] = []
    failed: list[str] = []
    if RAW_CURRENT_POINTER.is_file():
        history_path, history_manifest = read_history_snapshot(
            pointer_path=RAW_CURRENT_POINTER
        )
        dataset_fingerprint = str(history_manifest["data_sha256"])
    else:
        history_path = RAW_DATA
        history_manifest = {}
        dataset_fingerprint = _sha256_file(history_path)
    parameter_source, tuning_run_id = _parameter_provenance()
    base_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "targets_requested": [cfg.target_name for cfg in selected_models],
        "retune": force_retune,
        "optuna_allowed": force_retune,
        "promotable_full_bundle": promotable,
        "workspace_promoted": False,
        "artifact_root": str(artifact_root),
        "parameter_source": parameter_source,
        "tuning_run_id": tuning_run_id,
        "parameter_provenance": _parameter_provenance_by_target(),
        "shadow_hparameter_policy": (
            "reviewed map and prop parameters may cross feature-schema drift; "
            "these targets remain non-actionable"
        ),
        "feature_set_by_target": {
            cfg.target_name: _feature_set_for_config(cfg, feature_set)
            for cfg in selected_models
        },
        "code_version": repository.revision,
        "worktree_clean": repository.clean,
        "history_snapshot_id": history_manifest.get("snapshot_id"),
        "source_snapshot_id": history_manifest.get("source_snapshot_id"),
    }

    logger.info(
        "Selected training targets: %s",
        ", ".join(cfg.target_name for cfg in selected_models),
    )

    for cfg in selected_models:
        try:
            cfg_team, cfg_player = _training_frames_for_config(cfg, team_df, player_df)
            validate_training_tables(cfg_team, cfg_player)
            initialize_and_train_model(
                cfg=cfg,
                training_team_data=cfg_team,
                training_player_data=cfg_player,
                feature_selection=feature_selection,
                force_retune=force_retune,
                feature_set=_feature_set_for_config(cfg, feature_set),
                max_features=max_features,
                calibration=calibration,
                calibration_method=calibration_method,
                calibration_size=calibration_size,
                tune_size=tune_size,
                test_size=test_size,
                artifact_root=artifact_root,
                report_root=report_root,
                run_id=run_id,
                dataset_fingerprint=dataset_fingerprint,
            )
            trained.append(cfg.model_name)
        except KeyboardInterrupt:
            logger.info(f"Training interrupted by user during '{cfg.model_name}'")
            _write_training_manifest(
                report_root,
                base_manifest
                | {
                    "status": "interrupted",
                    "targets_trained": trained,
                    "targets_failed": failed,
                    "target_interrupted": cfg.model_name,
                },
            )
            raise
        except Exception as e:
            failed.append(cfg.model_name)
            logger.exception(f"Training failed for '{cfg.model_name}': {e}")

    manifest = base_manifest | {
        "targets_trained": trained,
        "targets_failed": failed,
    }

    if trained:
        logger.info(f"Successfully trained: {', '.join(trained)}")
    if failed:
        logger.warning(f"Failed: {', '.join(failed)}")
        _write_training_manifest(report_root, manifest | {"status": "failed"})
        msg = f"Training failed for: {', '.join(failed)}"
        raise RuntimeError(msg)

    _write_training_summary(report_root, selected_models, manifest)
    if promotable:
        candidate_id = _register_training_candidate(
            artifact_root,
            report_root=report_root,
            run_id=run_id,
        )
        manifest["candidate_id"] = candidate_id
        review = _review_training_candidate(candidate_id, parameter_source)
        manifest["promotion_status"] = review.status
        manifest["promotion_reasons"] = list(review.reasons)
    _write_training_manifest(report_root, manifest | {"status": "completed"})
    _publish_latest_training_report(report_root, run_id)
    _prune_training_reports(report_root.parent)
    logger.info("All model training tasks finished.\n")
    return report_root


def run_research_studies(
    *,
    targets: str = "all",
    include_next_map: bool = False,
) -> dict[str, Path]:
    """Run one isolated Optuna study per target; never review or promote it."""
    selected = parse_training_targets(targets)
    if targets.strip().casefold() == "all" and include_next_map:
        selected = (*selected, *EXPERIMENTAL_MODEL_CONFIGS)
    if not include_next_map and any(
        config.target_name == "next_map_winner" for config in selected
    ):
        raise ValueError("next_map_winner requires --include-next-map")
    if any(config.target_name == "next_map_winner" for config in selected):
        from lol_bets.data_generation.close_series import (
            build_close_series_oof_artifacts,
        )

        build_close_series_oof_artifacts()
    reports: dict[str, Path] = {}
    for config in selected:
        reports[config.target_name] = train_models(
            targets=config.target_name,
            force_retune=True,
            feature_set=(
                "auto" if config.dataset in {"series", "next_map"} else "compact"
            ),
        )
    return reports


def _training_preflight() -> RepositoryProvenance:
    repository = require_clean_repository()
    try:
        rating_provenance = json.loads(
            RATING_HYPERPARAMETER_PROVENANCE.read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Rating hyperparameter provenance is missing or invalid."
        ) from exc
    if (
        rating_provenance.get("source") != "predeclared_defaults"
        or rating_provenance.get("target_fitted") is not False
    ):
        raise RuntimeError(
            "Winner V2 requires predeclared, target-independent rating parameters."
        )
    _validate_predeclared_rating_parameters()
    LoLBetsModule().training_artifact_health().raise_if_unhealthy()
    return repository


def _validate_reviewed_winner_parameters(
    selected_models: tuple[ModelConfig, ...],
    *,
    force_retune: bool,
) -> None:
    """Fail routine training before any target runs without reviewed V2 params."""
    if force_retune:
        winner_targets = [
            config.target_name
            for config in selected_models
            if config.target_name in WINNER_TUNING_MODELS
        ]
        if winner_targets and len(selected_models) != 1:
            raise ValueError("Winner V2 targets require independent tuning runs.")
        return
    missing: list[str] = []
    for config in selected_models:
        if config.target_name not in WINNER_TUNING_MODELS:
            continue
        model_name = WINNER_TUNING_MODELS[config.target_name]
        path = TUNED_LIGHTGBM_HYPERPARAMETERS / f"{model_name}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            missing.append(config.target_name)
            continue
        metadata = payload.get("metadata") or {}
        if (
            metadata.get("model_name") != model_name
            or metadata.get("source") != "optuna_reviewed"
            or not metadata.get("promoted_tuning_run_id")
            or not isinstance(payload.get("params"), dict)
            or not payload["params"]
        ):
            missing.append(config.target_name)
    if missing:
        targets = ", ".join(missing)
        raise RuntimeError(
            "Routine Winner V2 training requires reviewed fixed parameters for: "
            f"{targets}. Review and promote independent research runs first."
        )


def _validate_predeclared_rating_parameters() -> None:
    defaults = json.loads(DEFAULT_MODELS_PARAMETERS.read_text(encoding="utf-8"))
    shared = defaults["shared"]
    expected = {
        "leagues_elo_hyperparameters.json": defaults["leagues_elo"],
    }
    families = {
        "elo": {
            "k_factor": defaults["elo"]["k_factor"],
            "initial_elo": defaults["elo"]["initial"],
            "elo_divisor": defaults["elo"]["elo_divisor"],
        },
        "glicko": {key: defaults["glicko2"][key] for key in ("mu", "phi", "sigma")},
        "pl": {key: defaults["plackett_luce"][key] for key in ("mu", "sigma")},
        "trueskill": {
            key: defaults["trueskill"][key] for key in ("mu", "sigma", "beta")
        },
    }
    shared_keys = (
        "decay_factor",
        "transfer_factor",
        "initial_elo_adjustment_factor",
        "position_reset_factor",
    )
    for entity in ("player", "team"):
        for family, base in families.items():
            expected[f"{entity}_{family}_hyperparameters.json"] = base | {
                key: shared[key] for key in shared_keys
            }
    for filename, expected_values in expected.items():
        path = TUNED_RATING_HYPERPARAMETERS / filename
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Rating parameter file is invalid: {path}") from exc
        if actual != expected_values:
            raise RuntimeError(
                "Rating parameters differ from their predeclared defaults: "
                f"{path}. Rating retuning must be isolated inside the Winner V2 "
                "development window before these values may change."
            )


def _sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _tuning_review_input_fingerprint(run_root: Path) -> str:
    """Bind a tuning approval to every model/evaluation file it reviewed."""
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested = manifest.get("targets_requested")
    winner_targets = (
        [str(target) for target in requested if str(target) in WINNER_TUNING_MODELS]
        if isinstance(requested, list)
        else []
    )
    if len(requested or []) != 1 or len(winner_targets) != 1:
        raise ValueError("Winner V2 targets require independent tuning runs.")
    model_name = WINNER_TUNING_MODELS[winner_targets[0]]
    artifact_root = run_root / "artifacts"
    paths = (
        manifest_path,
        run_root / model_name / "tuned_hyperparameters.json",
        artifact_root / "_evaluation" / model_name / "features.parquet",
        artifact_root / "_evaluation" / model_name / "labels.parquet",
        artifact_root / model_name / f"{model_name}.pkl",
        artifact_root / model_name / f"{model_name}_feature_pipeline.pkl",
        artifact_root / model_name / f"{model_name}_probability_calibrator.pkl",
        artifact_root / model_name / f"{model_name}_probability_uncertainty.pkl",
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Tuning review input is missing: {path}")
        digest.update(path.relative_to(run_root).as_posix().encode())
        digest.update(_sha256_file(path).encode())
    return digest.hexdigest()


def _same_tuned_parameter(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        if not isinstance(actual, (int, float, str)):
            return False
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _winner_parameter_failures(model: object, payload: dict) -> list[str]:
    """Bind the reviewed Winner ensemble to the parameter payload being promoted."""
    expected = payload.get("params")
    members = getattr(model, "members", ())
    if not isinstance(expected, dict) or not expected:
        return ["winner_tuned_parameters_missing"]
    if not members:
        return ["winner_ensemble_members_missing"]
    failures: list[str] = []
    for index, member in enumerate(members):
        get_params = getattr(member, "get_params", None)
        if not callable(get_params):
            failures.append(f"winner_member_parameters_unavailable:{index}")
            continue
        actual = get_params()
        mismatched = sorted(
            key
            for key, value in expected.items()
            if key not in actual or not _same_tuned_parameter(actual[key], value)
        )
        if mismatched:
            failures.append(
                f"winner_member_parameter_mismatch:{index}:{','.join(mismatched)}"
            )
    return failures


def promote_tuning_run(run_id: str) -> tuple[Path, ...]:  # noqa: PLR0912, PLR0915
    """Promote every target completed by one reviewed tuning run."""
    run_root = REPORTS_DIR / "training" / "runs" / run_id
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Tuning manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_target = {config.target_name: config for config in MODEL_CONFIGS}
    requested = manifest.get("targets_requested")
    if requested is None:
        selected = ALL_MODEL_CONFIGS
    elif not isinstance(requested, list) or not requested:
        raise ValueError(f"Tuning run has no requested targets: {manifest_path}")
    else:
        requested_targets = [str(target) for target in requested]
        unknown = sorted(set(requested_targets) - set(by_target))
        if unknown:
            raise ValueError(
                f"Tuning run has unknown requested targets {unknown}: {manifest_path}"
            )
        selected = tuple(by_target[target] for target in requested_targets)
    expected_trained = [config.model_name for config in selected]
    if (
        manifest.get("status") != "completed"
        or manifest.get("retune") is not True
        or manifest.get("targets_trained") != expected_trained
        or manifest.get("targets_failed")
    ):
        raise ValueError(
            f"Tuning run is not complete for its requested targets: {manifest_path}"
        )
    winner_targets = [
        config.target_name
        for config in selected
        if config.target_name in WINNER_TUNING_MODELS
    ]
    if len(winner_targets) > 1:
        raise ValueError("Winner V2 targets require independent tuning runs.")
    review: dict[str, object] = {}
    if winner_targets:
        winner_target = winner_targets[0]
        review_path = run_root / "tuning_review.json"
        try:
            review = json.loads(review_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValueError(
                "Winner tuning must pass `oracle-bets lol review-tuning` before "
                "parameter promotion."
            ) from exc
        reviewed_fingerprint = review.get("review_input_sha256")
        accepted_statuses = {"approved", "approved_with_warnings"}
        if (
            review.get("schema_version") != TUNING_REVIEW_SCHEMA_VERSION
            or review.get("status") not in accepted_statuses
            or review.get("run_id") != run_id
            or review.get("target") != winner_target
            or reviewed_fingerprint != _tuning_review_input_fingerprint(run_root)
        ):
            raise ValueError("Winner tuning review is missing, stale, or blocked.")
    else:
        reviewed_fingerprint = None

    names = tuple(_lightgbm_model_name(cfg.model_name) for cfg in selected)
    payloads: dict[str, dict] = {}
    required_provenance_fields = (
        "feature_set",
        "max_features",
        "feature_schema_fingerprint",
        "dataset_fingerprint",
        "code_version",
        "random_seed",
    )
    shared_provenance_fields = (
        "feature_set",
        "max_features",
        "dataset_fingerprint",
        "code_version",
        "random_seed",
    )
    shared_provenance: dict[str, object] | None = None
    for name in names:
        candidate = run_root / name / "tuned_hyperparameters.json"
        if not candidate.is_file():
            raise FileNotFoundError(f"Tuning candidate is missing: {candidate}")
        content = candidate.read_bytes()
        payload = json.loads(content)
        metadata = payload.get("metadata", {})
        params = payload.get("params")
        if (
            metadata.get("model_name") != name
            or not isinstance(params, dict)
            or not params
        ):
            raise ValueError(f"Invalid tuning candidate: {candidate}")
        if metadata.get("validation_score") is None:
            raise ValueError(f"Tuning candidate has no validation score: {candidate}")
        required = {field: metadata.get(field) for field in required_provenance_fields}
        if any(value is None for value in required.values()):
            raise ValueError(f"Tuning candidate has incomplete provenance: {candidate}")
        provenance = {field: metadata.get(field) for field in shared_provenance_fields}
        if shared_provenance is None:
            shared_provenance = provenance
        elif provenance != shared_provenance:
            raise ValueError(
                f"Tuning candidates have inconsistent provenance: {candidate}"
            )
        payloads[name] = payload

    if (
        reviewed_fingerprint is not None
        and reviewed_fingerprint != _tuning_review_input_fingerprint(run_root)
    ):
        raise ValueError("Winner tuning review changed during promotion.")

    review_warnings = review.get("warnings")
    if not isinstance(review_warnings, list):
        review_warnings = []
    for payload in payloads.values():
        payload["metadata"] |= {
            "source": "optuna_reviewed",
            "promoted_tuning_run_id": run_id,
            "review_warnings": review_warnings,
        }

    TUNED_LIGHTGBM_HYPERPARAMETERS.mkdir(parents=True, exist_ok=True)
    previous: dict[Path, bytes | None] = {}
    promoted: list[Path] = []
    try:
        for name, payload in payloads.items():
            destination = TUNED_LIGHTGBM_HYPERPARAMETERS / f"{name}.json"
            previous[destination] = (
                destination.read_bytes() if destination.exists() else None
            )
            atomic_write_text(
                destination,
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
            )
            promoted.append(destination)
    except Exception:
        for destination, content in previous.items():
            if content is None:
                destination.unlink(missing_ok=True)
            else:
                destination.write_bytes(content)
        raise
    return tuple(promoted)


def review_tuning_run(run_id: str) -> dict[str, object]:
    """Review one Winner V2 Optuna run against its sealed rating baseline."""
    run_root = REPORTS_DIR / "training" / "runs" / run_id
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    target = _validated_winner_tuning_target(manifest)
    review_fingerprint = _tuning_review_input_fingerprint(run_root)
    review_path = run_root / "tuning_review.json"
    if review_path.is_file():
        existing = json.loads(review_path.read_text(encoding="utf-8"))
        if (
            existing.get("schema_version") == TUNING_REVIEW_SCHEMA_VERSION
            and existing.get("review_input_sha256") == review_fingerprint
        ):
            return existing
    holdout = _sealed_tuning_holdout(run_root, target=target)
    evidence = _winner_tuning_review_evidence(run_root, target=target)
    if not holdout["fresh_for_promotion"]:
        existing_reasons = evidence.get("reasons")
        if not isinstance(existing_reasons, list):
            raise ValueError("Winner tuning evidence has malformed reasons.")
        reasons = [str(reason) for reason in existing_reasons]
        reasons.append("holdout_not_strictly_later_than_exposed_window")
        evidence["reasons"] = list(dict.fromkeys(reasons))
        evidence["status"] = "blocked"
    result: dict[str, object] = {
        "schema_version": TUNING_REVIEW_SCHEMA_VERSION,
        "run_id": run_id,
        "target": target,
        "review_input_sha256": review_fingerprint,
        "sealed_holdout": holdout,
    } | evidence
    if _tuning_review_input_fingerprint(run_root) != review_fingerprint:
        raise ValueError("Winner tuning artifacts changed during review.")
    atomic_write_text(
        run_root / "tuning_review.json",
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    return result


def _validated_winner_tuning_target(manifest: dict[str, object]) -> str:
    requested = manifest.get("targets_requested")
    if (
        manifest.get("status") != "completed"
        or manifest.get("retune") is not True
        or not isinstance(requested, list)
    ):
        raise ValueError("Run is not a completed Winner V2 tuning study.")
    winner_targets = [
        str(target) for target in requested if str(target) in WINNER_TUNING_MODELS
    ]
    if len(requested) != 1 or len(winner_targets) != 1:
        raise ValueError("Winner V2 targets require independent tuning runs.")
    target = winner_targets[0]
    expected_trained = WINNER_TUNING_MODELS[target].removesuffix("_LightGBM")
    if manifest.get("targets_trained") != [expected_trained] or manifest.get(
        "targets_failed"
    ):
        raise ValueError("Run is not a completed Winner V2 tuning study.")
    return target


def _sealed_tuning_holdout(run_root: Path, *, target: str) -> dict[str, object]:
    """Identify a holdout and reject temporal overlap with earlier reviews."""
    model_name = WINNER_TUNING_MODELS[target]
    split_path = run_root / model_name / "split_report.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))["test"]
    date_min = str(split.get("date_min") or "")
    date_max = str(split.get("date_max") or "")
    if not date_min or not date_max:
        raise ValueError("Winner tuning review requires sealed test date bounds.")
    current_min = _parse_utc_holdout_date(date_min)
    current_max = _parse_utc_holdout_date(date_max)
    prior: list[tuple[dt.datetime, str]] = []
    runs_root = run_root.parent
    for review_path in sorted(runs_root.glob("*/tuning_review.json")):
        previous_root = review_path.parent
        try:
            review = json.loads(review_path.read_text(encoding="utf-8"))
            if review.get("target") != target:
                continue
            sealed = review["sealed_holdout"]
            previous_max = _parse_utc_holdout_date(str(sealed["date_max"]))
            prior.append((previous_max, previous_root.name))
        except (
            FileNotFoundError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            continue
    latest_prior = max(prior, default=None)
    labels_path = run_root / "artifacts" / "_evaluation" / model_name / "labels.parquet"
    fresh = latest_prior is None or current_min > latest_prior[0]
    return {
        "date_min": current_min.isoformat(),
        "date_max": current_max.isoformat(),
        "labels_sha256": _sha256_file(labels_path),
        "fresh_for_promotion": fresh,
        "latest_prior_exposed_date_max": (
            latest_prior[0].isoformat() if latest_prior is not None else None
        ),
        "latest_prior_run_id": latest_prior[1] if latest_prior is not None else None,
    }


def _parse_utc_holdout_date(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def _winner_tuning_review_evidence(
    run_root: Path,
    *,
    target: str,
) -> dict[str, object]:
    """Build sealed, target-specific Optuna evidence against the rating baseline."""
    from oracle_bets_core.evidence.performance import prediction_quality

    from lol_bets.operations.models import (
        PromotionEvidence,
        PromotionPolicy,
        _binary_log_losses,
        _clustered_binary_log_losses,
        _cohort_replay_losses,
        evaluate_promotion,
    )
    from lol_bets.prediction_models.gbdt_model import (
        GradientBoostingModel,
        conservative_probability_report,
    )

    model_name = WINNER_TUNING_MODELS[target]
    artifact_root = run_root / "artifacts"
    evaluation = artifact_root / "_evaluation" / model_name
    raw = pd.read_parquet(evaluation / "features.parquet")
    labels = pd.read_parquet(evaluation / "labels.parquet")
    actual = pd.to_numeric(labels["actual"], errors="raise").to_numpy(dtype=float)
    metadata = labels.drop(columns=["actual"])
    model_root = artifact_root / model_name
    pipeline = load_model(model_root / f"{model_name}_feature_pipeline.pkl")
    model = load_model(model_root / f"{model_name}.pkl")
    calibrator = load_model(model_root / f"{model_name}_probability_calibrator.pkl")
    uncertainty = load_model(model_root / f"{model_name}_probability_uncertainty.pkl")
    tuning_payload = json.loads(
        (run_root / model_name / "tuned_hyperparameters.json").read_text(
            encoding="utf-8"
        )
    )
    transformed = pipeline.transform(raw)
    baseline = np.asarray(model.rating_baseline_probability(transformed), dtype=float)
    candidate = np.asarray(
        calibrator.predict(model.predict_proba(transformed)[:, 1], metadata=metadata),
        dtype=float,
    )
    baseline_quality = prediction_quality(
        actual.astype(int).tolist(), baseline.tolist()
    )
    candidate_quality = prediction_quality(
        actual.astype(int).tolist(), candidate.tolist()
    )
    operational_failures = _winner_parameter_failures(model, tuning_payload)
    probability_artifacts_valid = valid_probability_calibration_artifacts(
        calibrator, uncertainty
    )
    if not probability_artifacts_valid:
        operational_failures.append("invalid_probability_calibration_artifacts")
    cluster_col = "series_id" if target == "next_map_winner" else None
    cluster_identity_source = None
    if cluster_col is not None:
        labels, cluster_identity_source = _next_map_review_labels(labels)
    if cluster_col is None:
        baseline_losses = _binary_log_losses(actual, baseline)
        candidate_losses = _binary_log_losses(actual, candidate)
    else:
        baseline_losses = _clustered_binary_log_losses(
            labels, actual, baseline, cluster_col=cluster_col
        )
        candidate_losses = _clustered_binary_log_losses(
            labels, actual, candidate, cluster_col=cluster_col
        )
    evidence = PromotionEvidence(
        champion_log_losses=tuple(baseline_losses),
        candidate_log_losses=tuple(candidate_losses),
        champion_brier=baseline_quality.brier,
        candidate_brier=candidate_quality.brier,
        champion_ece=baseline_quality.calibration_error,
        candidate_ece=candidate_quality.calibration_error,
        cohort_log_loss=_cohort_replay_losses(
            labels,
            actual,
            baseline,
            candidate,
            cluster_col=cluster_col,
        ),
        operational_failures=tuple(operational_failures),
    )
    decision = evaluate_promotion(evidence, policy=PromotionPolicy.OPTUNA)
    calibration = GradientBoostingModel.compute_probability_calibration_metrics(
        pd.Series(actual), candidate
    )
    reasons = list(decision.reasons)
    warnings = _calibration_review_warnings(calibration)
    conservative_report: dict[str, object] | None = None
    if target == "series_winner" and probability_artifacts_valid:
        conservative_report = conservative_probability_report(
            model,
            transformed,
            pd.Series(actual),
            calibrator,
            uncertainty,
            metadata,
        )
        coverage = conservative_report["coverage"]
        if not isinstance(coverage, dict) or coverage.get("passed") is not True:
            reasons.append("conservative_probability_coverage_failed")
    for cohort, (baseline_loss, candidate_loss, count) in sorted(
        evidence.cohort_log_loss.items()
    ):
        if (
            not cohort.startswith("actionable")
            and count >= MIN_DIAGNOSTIC_COHORT_SIZE
            and baseline_loss > 0
            and (candidate_loss - baseline_loss) / baseline_loss
            > MAX_DIAGNOSTIC_COHORT_REGRESSION
        ):
            warnings.append(f"cohort_regression:{cohort}")
    status = (
        "blocked" if reasons else ("approved_with_warnings" if warnings else "approved")
    )
    return {
        "bootstrap_unit": cluster_col or "series",
        "bootstrap_units": len(baseline_losses),
        "cluster_identity_source": cluster_identity_source,
        "status": status,
        "reasons": list(dict.fromkeys(reasons)),
        "warnings": list(dict.fromkeys(warnings)),
        "rating_baseline": {
            "log_loss": baseline_quality.log_loss,
            "brier": baseline_quality.brier,
            "ece": baseline_quality.calibration_error,
        },
        "candidate": {
            "log_loss": candidate_quality.log_loss,
            "brier": candidate_quality.brier,
            "ece": candidate_quality.calibration_error,
        },
        "calibration": calibration,
        "conservative_probability": conservative_report,
        "relative_improvement": decision.relative_improvement,
        "confidence_lower_bound": decision.confidence_lower_bound,
        "cohorts": {
            name: {"baseline": values[0], "candidate": values[1], "count": values[2]}
            for name, values in evidence.cohort_log_loss.items()
        },
    }


def _calibration_review_warnings(calibration: dict[str, float]) -> list[str]:
    warnings: list[str] = []
    slope = calibration.get("calibration_slope")
    intercept = calibration.get("calibration_intercept")
    if slope is None or not (
        MIN_CALIBRATION_REVIEW_SLOPE <= slope <= MAX_CALIBRATION_REVIEW_SLOPE
    ):
        warnings.append("calibration_slope_outside_0.8_1.2")
    if intercept is None or abs(intercept) > MAX_CALIBRATION_REVIEW_INTERCEPT:
        warnings.append("calibration_intercept_above_0.10")
    return warnings


_SYNTHETIC_NEXT_MAP_ID = re.compile(r"^(series-[0-9a-f]{24}):map-([2-9][0-9]*)$")


def _next_map_review_labels(labels: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Return complete next-map cluster labels without changing sealed outcomes."""
    if "series_id" in labels and labels["series_id"].notna().all():
        values = labels["series_id"].astype(str).str.strip()
        if values.ne("").all():
            output = labels.copy()
            output["series_id"] = values
            return output, "sealed_series_id"

    if "gameid" not in labels or labels["gameid"].isna().any():
        raise ValueError("sealed labels require complete series_id clusters")
    matches = [
        _SYNTHETIC_NEXT_MAP_ID.fullmatch(str(value).strip())
        for value in labels["gameid"]
    ]
    if not matches or any(match is None for match in matches):
        raise ValueError("sealed labels require complete series_id clusters")
    output = labels.copy()
    output["series_id"] = [match.group(1) for match in matches if match is not None]
    return output, "derived_from_synthetic_gameid"


def _write_training_manifest(report_root: Path, payload: dict) -> Path:
    report_root.mkdir(parents=True, exist_ok=True)
    destination = report_root / "manifest.json"
    atomic_write_text(
        destination,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return destination


def _write_training_summary(
    report_root: Path,
    configs: tuple[ModelConfig, ...],
    manifest: dict,
) -> tuple[Path, Path]:
    """Build one reviewable JSON/Markdown summary from per-model artifacts."""
    models: list[dict] = []
    for config in configs:
        name = _lightgbm_model_name(config.model_name)
        model_root = report_root / name
        metrics_path = model_root / "metrics.json"
        cards = sorted(model_root.glob("model_card_*.json"))
        if not metrics_path.is_file() or not cards:
            msg = f"Required training evidence is missing for {name}: {model_root}"
            raise RuntimeError(msg)
        metrics = json.loads(metrics_path.read_text())
        card = json.loads(cards[-1].read_text())
        calibration_path = model_root / "calibration_report.json"
        calibration = (
            json.loads(calibration_path.read_text())
            if calibration_path.is_file()
            else {}
        )
        if config.problem_type == "classification":
            pairwise = calibration.get("pairwise_test_metrics", {})
            evidence_status = (
                "meets_basic_sanity"
                if metrics.get("log_loss", 1.0) < RANDOM_CLASSIFIER_LOG_LOSS
                and metrics.get("brier", 1.0) < RANDOM_CLASSIFIER_BRIER
                and pairwise.get("max_probability_sum_error") == 0.0
                else "review_required"
            )
        else:
            r2 = float(metrics.get("r2", float("-inf")))
            evidence_status = (
                "below_constant_baseline"
                if r2 <= 0.0
                else "weak_signal"
                if r2 < WEAK_REGRESSION_R2
                else "meets_basic_sanity"
            )
        top_features: list[dict] = []
        importance_path = model_root / "feature_importances.parquet"
        if importance_path.is_file():
            importance = pd.read_parquet(importance_path)
            top_features = importance.head(20).to_dict(orient="records")
        models.append(
            {
                "model_name": name,
                "target": config.target_name,
                "problem_type": config.problem_type,
                "metrics": metrics,
                "model_card": card,
                "calibration": calibration,
                "evidence_status": evidence_status,
                "top_features": top_features,
            }
        )

    payload = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "created_at_utc": manifest["created_at_utc"],
        "retune": manifest["retune"],
        "optuna_allowed": manifest["optuna_allowed"],
        "models": models,
    }
    json_path = report_root / "summary.json"
    markdown_path = report_root / "summary.md"
    atomic_write_text(json_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        f"# LoL training run {manifest['run_id']}",
        "",
        f"- Retune: `{manifest['retune']}`",
        f"- Optuna allowed: `{manifest['optuna_allowed']}`",
        f"- Created: `{manifest['created_at_utc']}`",
        "",
    ]
    for model in models:
        lines.extend(
            [
                f"## {model['model_name']}",
                "",
                f"Target: `{model['target']}`; problem: `{model['problem_type']}`.",
                f"Evidence status: `{model['evidence_status']}`.",
                "",
                "```json",
                json.dumps(model["metrics"], indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
        top_features = model["top_features"][:10]
        if top_features:
            lines.extend(["Top features by LightGBM gain:", ""])
            lines.extend(
                f"- `{row['feature']}` — gain `{float(row['gain']):.3f}`"
                for row in top_features
            )
            lines.append("")
    atomic_write_text(markdown_path, "\n".join(lines))
    return json_path, markdown_path


def _publish_latest_training_report(report_root: Path, run_id: str) -> Path:
    latest = report_root.parent.parent / "latest.json"
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "report_directory": str(report_root),
    }
    atomic_write_text(latest, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return latest


def _prune_training_reports(runs_root: Path) -> None:
    completed = sorted(
        (
            path
            for path in runs_root.iterdir()
            if path.is_dir() and (path / "manifest.json").is_file()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    for obsolete in completed[TRAINING_REPORT_RETENTION:]:
        shutil.rmtree(obsolete)


def _register_training_candidate(
    staging_root: Path,
    *,
    report_root: Path,
    run_id: str,
) -> str:
    """Freeze a complete staged run without changing the serving champion."""
    from oracle_bets_core.paths import MODEL_REGISTRY_DIR

    from lol_bets.operations.models import ModelRegistry, register_current_candidate

    history_path = current_history_data_path(
        RAW_DATA,
        pointer_path=RAW_CURRENT_POINTER,
    )
    summary = json.loads((report_root / "summary.json").read_text(encoding="utf-8"))
    metrics = {
        f"{model['target']}.{name}": float(value)
        for model in summary["models"]
        for name, value in model["metrics"].items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    code_versions = {
        str(model["model_card"].get("code_version") or "unknown")
        for model in summary["models"]
    }
    if len(code_versions) != 1 or "unknown" in code_versions:
        raise RuntimeError(
            "Registered candidates require one exact model-card code version."
        )
    code_version = "+".join(sorted(code_versions))
    candidate_id = f"lol-{run_id}"
    _stage_shared_inference_artifacts(staging_root)
    evaluation_root = staging_root / "_evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(report_root / "summary.json", evaluation_root / "summary.json")
    for model in summary["models"]:
        model_name = str(model["model_name"])
        for filename in (
            "split_report.json",
            "calibration_report.json",
            "winner_model_report.json",
            "prop_evaluation_report.json",
        ):
            source = report_root / model_name / filename
            if source.is_file():
                destination = evaluation_root / model_name / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
    register_current_candidate(
        registry=ModelRegistry(MODEL_REGISTRY_DIR),
        model_id=candidate_id,
        target="complete_lol_bundle",
        code_version=code_version,
        metrics=metrics,
        created_at=dt.datetime.now(dt.UTC),
        model_root=staging_root,
        training_paths=(
            history_path,
            TRAINING_TEAM_DATA,
            TRAINING_PLAYER_DATA,
            SERIES_MANIFEST,
            SERIES_REJECTIONS,
            SERIES_WINNER_TEAM_DATA,
            SERIES_WINNER_PLAYER_DATA,
        ),
    )
    shutil.rmtree(staging_root)
    return candidate_id


def _stage_shared_inference_artifacts(staging_root: Path) -> None:
    """Include rating lookup tables required by registry-based inference."""
    for source in (RATING_TEAM_LEAGUES_MAPPING, RATING_LEAGUE_ELO):
        if not source.is_file():
            raise RuntimeError(f"Required inference artifact is missing: {source}")
        shutil.copyfile(source, staging_root / source.name)


def _parameter_provenance_by_target() -> dict[str, dict[str, str | None]]:
    """Return the fixed-parameter origin used by each model target."""
    provenance: dict[str, dict[str, str | None]] = {}
    for cfg in ALL_MODEL_CONFIGS:
        path = (
            TUNED_LIGHTGBM_HYPERPARAMETERS
            / f"{_lightgbm_model_name(cfg.model_name)}.json"
        )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            provenance[cfg.target_name] = {"source": "defaults", "run_id": None}
            continue
        metadata = payload.get("metadata") or {}
        source = str(metadata.get("source") or "reviewed_tuned")
        run_id = metadata.get("promoted_tuning_run_id")
        provenance[cfg.target_name] = {
            "source": source,
            "run_id": str(run_id) if run_id is not None else None,
        }
    return provenance


def _parameter_provenance() -> tuple[str, str | None]:
    """Identify whether any fixed parameters came from reviewed Optuna output."""
    provenance = _parameter_provenance_by_target()
    optuna_run_ids = sorted(
        {
            str(item["run_id"])
            for item in provenance.values()
            if item["source"] == "optuna_reviewed" and item["run_id"] is not None
        }
    )
    if optuna_run_ids:
        return "optuna_reviewed", ",".join(optuna_run_ids)
    if any(item["source"] != "defaults" for item in provenance.values()):
        return "reviewed_tuned", None
    return "defaults", None


def _review_training_candidate(candidate_id: str, parameter_source: str):
    from oracle_bets_core.paths import MODEL_REGISTRY_DIR

    from lol_bets.operations.models import (
        ModelRegistry,
        PromotionPolicy,
        review_candidate_on_sealed_rows,
    )

    policy = (
        PromotionPolicy.OPTUNA
        if parameter_source == "optuna_reviewed"
        else PromotionPolicy.ROUTINE
    )
    return review_candidate_on_sealed_rows(
        ModelRegistry(MODEL_REGISTRY_DIR),
        candidate_id,
        policy=policy,
        automatic=policy is PromotionPolicy.ROUTINE,
        reviewed_at=dt.datetime.now(dt.UTC),
    )
