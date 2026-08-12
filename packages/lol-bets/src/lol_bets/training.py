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
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

from oracle_bets_core.io_utils import load_training_data, store_model
from oracle_bets_core.logger import logger
from oracle_bets_core.paths import (
    LEAGUE_ELO,
    MODELS_DIR,
    NEXT_MAP_PLAYER_DATA,
    NEXT_MAP_TEAM_DATA,
    RAW_DATA,
    REPORTS_DIR,
    SERIES_WINNER_PLAYER_DATA,
    SERIES_WINNER_TEAM_DATA,
    TEAM_LEAGUES_MAPPING,
    TRAINING_PLAYER_DATA,
    TRAINING_TEAM_DATA,
    TUNED_LIGHTGBM_HYPERPARAMETERS,
)
from oracle_bets_core.pd import pd

from lol_bets.data_generation.ingestion.quality import normalize_result
from lol_bets.module import LoLBetsModule
from lol_bets.operations.provenance import (
    RepositoryProvenance,
    require_clean_repository,
)
from lol_bets.prediction_models.gbdt_model import DEFAULT_SELECTED_MAX_FEATURES
from lol_bets.prediction_models.lightgbm_model import LightGBMModel
from lol_bets.prediction_models.prop_features import PROP_TARGETS
from lol_bets.prediction_models.winner_model import WinnerLightGBMModel

# ───────────────────────────────  types / config  ─────────────────────────────

ProblemType = Literal["classification", "regression"]
FeatureSelectionMethod = Literal["none", "importance", "cumulative", "report"]
TrainingFeatureSet = Literal["full", "compact", "selected"]
CalibrationMode = Literal["auto", "none"]
CalibrationMethod = Literal["raw", "sigmoid", "isotonic", "auto"]
MODEL_FILE_EXTENSION = "pkl"


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
    ModelConfig(
        "NextMapWinnerPrediction",
        "next_map_winner",
        "result",
        "classification",
        "next_map",
    ),
    ModelConfig("GamelengthPrediction", "gamelength", "gamelength", "regression"),
    ModelConfig("TotalKillsPrediction", "total_kills", "total_kills", "regression"),
    ModelConfig("TotalTowersPrediction", "total_towers", "total_towers", "regression"),
)

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
    by_target = {cfg.target_name: cfg for cfg in ALL_MODEL_CONFIGS}
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
    selected_models = parse_training_targets(targets)
    promotable = tuple(selected_models) == ALL_MODEL_CONFIGS and not force_retune
    artifact_root = (
        MODELS_DIR / ".staging" / run_id if promotable else report_root / "artifacts"
    )
    trained: list[str] = []
    failed: list[str] = []
    dataset_fingerprint = _sha256_file(RAW_DATA)
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
        "code_version": repository.revision,
        "worktree_clean": repository.clean,
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
                feature_set=feature_set,
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


def _training_preflight() -> RepositoryProvenance:
    repository = require_clean_repository()
    LoLBetsModule().training_artifact_health().raise_if_unhealthy()
    return repository


def _sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def promote_tuning_run(run_id: str) -> tuple[Path, ...]:  # noqa: PLR0912, PLR0915
    """Promote every target completed by one reviewed tuning run."""
    run_root = REPORTS_DIR / "training" / "runs" / run_id
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Tuning manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_target = {config.target_name: config for config in ALL_MODEL_CONFIGS}
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
        payload = json.loads(candidate.read_text(encoding="utf-8"))
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

    for payload in payloads.values():
        payload["metadata"] |= {
            "source": "optuna_reviewed",
            "promoted_tuning_run_id": run_id,
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
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(destination)
            promoted.append(destination)
    except Exception:
        for destination, content in previous.items():
            if content is None:
                destination.unlink(missing_ok=True)
            else:
                destination.write_bytes(content)
        raise
    return tuple(promoted)


def _write_training_manifest(report_root: Path, payload: dict) -> Path:
    report_root.mkdir(parents=True, exist_ok=True)
    destination = report_root / "manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
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
    _atomic_write_text(json_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

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
    _atomic_write_text(markdown_path, "\n".join(lines))
    return json_path, markdown_path


def _publish_latest_training_report(report_root: Path, run_id: str) -> Path:
    latest = report_root.parent.parent / "latest.json"
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "report_directory": str(report_root),
    }
    _atomic_write_text(latest, json.dumps(payload, indent=2, sort_keys=True) + "\n")
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


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _register_training_candidate(
    staging_root: Path,
    *,
    report_root: Path,
    run_id: str,
) -> str:
    """Freeze a complete staged run without changing the serving champion."""
    from oracle_bets_core.paths import MODEL_REGISTRY_DIR

    from lol_bets.operations.models import ModelRegistry, register_current_candidate

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
    )
    shutil.rmtree(staging_root)
    return candidate_id


def _stage_shared_inference_artifacts(staging_root: Path) -> None:
    """Include rating lookup tables required by registry-based inference."""
    for source in (TEAM_LEAGUES_MAPPING, LEAGUE_ELO):
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


def _raise_missing_staged_model(path: Path) -> None:
    raise RuntimeError(f"Staged model directory is missing: {path}")
