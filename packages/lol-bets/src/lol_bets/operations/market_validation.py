"""Read-only structural validation for exact-link market comparisons."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from oracle_bets_core.io_utils import load_model
from oracle_bets_core.paths import (
    MODEL_REGISTRY_DIR,
    MODELS_DIR,
    OUTCOME_PREDICTION_FEATURE_PIPELINE,
    OUTCOME_PREDICTION_MATCHUP_SCHEMA,
    OUTCOME_PREDICTION_MODEL_PATH,
    REPORTS_DIR,
    SERIES_MANIFEST,
)
from oracle_bets_core.pd import pd

from lol_bets.operations.market_strategies import enumerate_series_paths
from lol_bets.operations.models import ModelRegistry, resolve_serving_artifact
from lol_bets.operations.winner_validation import (
    FORBIDDEN_WINNER_FEATURE_FRAGMENTS,
    validate_winner_model,
)

if TYPE_CHECKING:
    from pathlib import Path

CLASSIFICATION_THRESHOLD = 0.5


@dataclass(frozen=True)
class MarketStrategyValidationReport:
    ok: bool
    checks: dict[str, bool]
    failures: tuple[str, ...]
    map_holdout_metrics: dict[str, float]
    map_cohort_metrics: dict[str, dict[str, float]]
    series_counts: dict[str, int]
    derived_backtest_metrics: dict[str, dict[str, float]]
    derived_backtest_counts: dict[str, int]
    experimental_strategies: tuple[str, ...]
    strategy_readiness: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_market_strategies(
    *,
    outcome_model_path: Path = OUTCOME_PREDICTION_MODEL_PATH,
    outcome_pipeline_path: Path = OUTCOME_PREDICTION_FEATURE_PIPELINE,
    outcome_schema_path: Path = OUTCOME_PREDICTION_MATCHUP_SCHEMA,
    series_manifest_path: Path = SERIES_MANIFEST,
    outcome_predictions_path: Path | None = None,
) -> MarketStrategyValidationReport:
    """Validate artifacts, sealed map metrics, legal BO paths, and series data."""
    checks: dict[str, bool] = {}
    failures: list[str] = []
    readiness = _strategy_readiness_checks(checks, failures)
    checks["direct_series_model"] = validate_winner_model().ok
    try:
        resolved_model = _serving_path(outcome_model_path)
        model = load_model(resolved_model)
        pipeline = load_model(_serving_path(outcome_pipeline_path))
        schema = load_model(_serving_path(outcome_schema_path))
        columns = tuple(str(value) for value in getattr(pipeline, "train_columns", ()))
        checks["map_artifacts_readable"] = callable(
            getattr(model, "predict_proba", None)
        )
        checks["map_matchup_schema"] = (
            isinstance(schema, dict)
            and schema.get("canonical_key") == "teamid_then_teamname"
        )
        checks["map_train_serve_columns"] = bool(columns)
        checks["map_has_no_forbidden_features"] = not any(
            fragment in column.casefold()
            for column in columns
            for fragment in FORBIDDEN_WINNER_FEATURE_FRAGMENTS
        )
        report_path = resolved_model.parent / (
            f"{resolved_model.stem}_calibration_report.json"
        )
        calibration = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = {
            key: float(value)
            for key, value in calibration["test_metrics"]["calibrated"].items()
            if isinstance(value, (int, float))
        }
        checks["sealed_map_metrics_finite"] = bool(metrics) and all(
            math.isfinite(value) for value in metrics.values()
        )
    except Exception as error:
        metrics = {}
        checks["map_artifacts_readable"] = False
        failures.append(f"map_artifact_validation:{type(error).__name__}")

    checks["bo_path_algebra"] = _path_algebra_is_valid()
    try:
        manifest = pd.read_parquet(series_manifest_path)
        series_counts = {
            f"bo{best_of}": int(manifest["best_of"].eq(best_of).sum())
            for best_of in (3, 5)
        }
        checks["reconstructed_bo3_bo5_history"] = all(series_counts.values())
        checks["terminal_series_are_legal"] = _manifest_is_legal(manifest)
        predictions_path = outcome_predictions_path or _champion_map_predictions_path()
        map_predictions = pd.read_parquet(predictions_path)
        map_cohorts = _map_cohort_metrics(manifest, map_predictions)
        derived_metrics, derived_counts = _backtest_derived_markets(
            manifest, map_predictions
        )
        checks["map_required_cohorts"] = all(
            map_cohorts.get(name, {}).get("count", 0) > 0
            for name in ("map_1", "later_maps", "actionable")
        ) and all(
            math.isfinite(value)
            for values in map_cohorts.values()
            for value in values.values()
        )
        checks["bo3_bo5_derived_backtest"] = all(
            derived_counts.get(f"bo{best_of}", 0) > 0 for best_of in (3, 5)
        ) and all(
            math.isfinite(value)
            for values in derived_metrics.values()
            for value in values.values()
        )
    except Exception as error:
        series_counts = {}
        derived_metrics = {}
        derived_counts = {}
        map_cohorts = {}
        checks["reconstructed_bo3_bo5_history"] = False
        checks["terminal_series_are_legal"] = False
        failures.append(f"series_manifest_validation:{type(error).__name__}")
    failures.extend(name for name, passed in checks.items() if not passed)
    return MarketStrategyValidationReport(
        ok=not failures,
        checks=checks,
        failures=tuple(dict.fromkeys(failures)),
        map_holdout_metrics=metrics,
        map_cohort_metrics=map_cohorts,
        series_counts=series_counts,
        derived_backtest_metrics=derived_metrics,
        derived_backtest_counts=derived_counts,
        experimental_strategies=(
            "map_prematch_v1",
            "series_totals_map_path_v1",
            "series_handicap_map_path_v1",
        ),
        strategy_readiness=readiness or {},
    )


def _strategy_readiness_checks(
    checks: dict[str, bool], failures: list[str]
) -> dict[str, Any]:
    try:
        readiness = ModelRegistry(MODEL_REGISTRY_DIR).strategy_readiness()
        cells = readiness.get("cells") if readiness else None
        checks["strategy_readiness_complete"] = bool(cells) and all(
            isinstance(cell, dict)
            and cell.get("state")
            in {"recommendation_active", "exploration_only", "display_only"}
            for cell in cells
        )
        checks["recommendations_are_direct_series_only"] = bool(cells) and all(
            cell.get("target") == "series_winner"
            for cell in cells
            if cell.get("state") == "recommendation_active"
        )
        return readiness or {}
    except Exception as error:
        checks["strategy_readiness_complete"] = False
        checks["recommendations_are_direct_series_only"] = False
        failures.append(f"strategy_readiness:{type(error).__name__}")
        return {}


def _serving_path(path: Path) -> Path:
    if not path.resolve().is_relative_to(MODELS_DIR.resolve()):
        return path
    return resolve_serving_artifact(
        path,
        registry_root=MODEL_REGISTRY_DIR,
        legacy_root=MODELS_DIR,
    )


def _map_cohort_metrics(
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Report sealed Map-1/later, actionable, and probability-band evidence."""
    required = {"gameid", "actual", "proba", "actionable"}
    if required - set(predictions.columns):
        raise ValueError("map prediction evidence is missing cohort columns")
    map_numbers = {
        str(map_id): index
        for row in manifest.to_dict(orient="records")
        for index, map_id in enumerate(json.loads(str(row["source_map_ids"])), start=1)
    }
    rows = predictions.copy()
    rows["map_number"] = rows["gameid"].astype(str).map(map_numbers)
    cohorts = {
        "map_1": rows[rows["map_number"].eq(1)],
        "later_maps": rows[rows["map_number"].gt(1)],
        "actionable": rows[rows["actionable"].astype(bool)],
        "probability_0_00_0_40": rows[rows["proba"].lt(0.4)],
        "probability_0_40_0_60": rows[rows["proba"].ge(0.4) & rows["proba"].le(0.6)],
        "probability_0_60_1_00": rows[rows["proba"].gt(0.6)],
    }
    return {
        name: _binary_metrics(frame)
        for name, frame in cohorts.items()
        if not frame.empty
    }


def _binary_metrics(frame: pd.DataFrame) -> dict[str, float]:
    actual = [float(value) for value in frame["actual"].tolist()]
    probabilities = [float(value) for value in frame["proba"].tolist()]
    clipped = [min(max(value, 1e-15), 1 - 1e-15) for value in probabilities]
    count = len(actual)
    return {
        "count": float(count),
        "log_loss": -sum(
            target * math.log(probability) + (1 - target) * math.log(1 - probability)
            for target, probability in zip(actual, clipped, strict=True)
        )
        / count,
        "brier": sum(
            (probability - target) ** 2
            for target, probability in zip(actual, probabilities, strict=True)
        )
        / count,
        "accuracy": sum(
            (probability >= CLASSIFICATION_THRESHOLD) == bool(target)
            for target, probability in zip(actual, probabilities, strict=True)
        )
        / count,
        "calibration_gap": abs(sum(probabilities) / count - sum(actual) / count),
    }


def _champion_map_predictions_path() -> Path:
    champion = ModelRegistry(MODEL_REGISTRY_DIR).champion_id()
    if not champion:
        raise ValueError("market strategy validation requires a promoted champion")
    path = (
        REPORTS_DIR
        / "training"
        / "runs"
        / champion.removeprefix("lol-")
        / "OutcomePrediction_LightGBM"
        / "predictions.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(f"champion sealed map predictions are missing: {path}")
    return path


def _backtest_derived_markets(
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Score total-map and final-differential paths from held-out Map-1 prices."""
    required_predictions = {"gameid", "proba"}
    if required_predictions - set(predictions.columns):
        raise ValueError("map prediction evidence is missing gameid/proba")
    probabilities = {
        str(row["gameid"]): float(row["proba"])
        for row in predictions.to_dict(orient="records")
    }
    scores = {
        "series_totals_map_path_v1": {"log_loss": [], "brier": [], "correct": []},
        "series_handicap_map_path_v1": {
            "log_loss": [],
            "brier": [],
            "correct": [],
        },
    }
    counts = {"bo3": 0, "bo5": 0}
    for row in manifest.to_dict(orient="records"):
        best_of = int(row["best_of"])
        if best_of not in {3, 5}:
            continue
        map_ids = json.loads(str(row["source_map_ids"]))
        map_winners = json.loads(str(row["map_winner_ids"]))
        if not map_ids or map_ids[0] not in probabilities:
            continue
        probability = probabilities[map_ids[0]]
        if not 0 <= probability <= 1:
            raise ValueError("held-out map probability is outside [0,1]")
        distribution = enumerate_series_paths(probability, best_of=best_of)
        actual_total = int(row["map_count"])
        a_wins = sum(winner == str(row["team_a_id"]) for winner in map_winners)
        actual_difference = a_wins - (len(map_winners) - a_wins)
        _append_multiclass_score(
            scores["series_totals_map_path_v1"],
            distribution.total_maps,
            actual_total,
        )
        _append_multiclass_score(
            scores["series_handicap_map_path_v1"],
            distribution.map_differential,
            actual_difference,
        )
        counts[f"bo{best_of}"] += 1
    return (
        {
            strategy: {
                "log_loss": sum(values["log_loss"]) / len(values["log_loss"]),
                "brier": sum(values["brier"]) / len(values["brier"]),
                "accuracy": sum(values["correct"]) / len(values["correct"]),
            }
            for strategy, values in scores.items()
            if values["log_loss"]
        },
        counts,
    )


def _append_multiclass_score(
    scores: dict[str, list[float]],
    probabilities: dict[int, float],
    actual: int,
) -> None:
    assigned = max(float(probabilities.get(actual, 0.0)), 1e-15)
    predicted = max(probabilities, key=probabilities.__getitem__)
    scores["log_loss"].append(-math.log(assigned))
    scores["brier"].append(
        sum(
            (probability - float(outcome == actual)) ** 2
            for outcome, probability in probabilities.items()
        )
    )
    scores["correct"].append(float(predicted == actual))


def _path_algebra_is_valid() -> bool:
    for best_of in (3, 5):
        for index in range(101):
            probability = index / 100
            direct = enumerate_series_paths(probability, best_of=best_of)
            swapped = enumerate_series_paths(1 - probability, best_of=best_of)
            if not math.isclose(sum(direct.exact_score.values()), 1.0, abs_tol=1e-12):
                return False
            if not math.isclose(
                direct.team_a_win, 1 - swapped.team_a_win, abs_tol=1e-12
            ):
                return False
            if any(
                not math.isclose(value, swapped.total_maps[total], abs_tol=1e-12)
                for total, value in direct.total_maps.items()
            ):
                return False
    return True


def _manifest_is_legal(manifest: pd.DataFrame) -> bool:
    required = {
        "best_of",
        "map_count",
        "map_winner_ids",
        "team_a_id",
        "team_b_id",
    }
    if required - set(manifest.columns):
        return False
    for row in manifest.to_dict(orient="records"):
        if int(row["best_of"]) not in {1, 3, 5}:
            continue
        winners = json.loads(str(row["map_winner_ids"]))
        wins_needed = int(row["best_of"]) // 2 + 1
        counts = {str(row["team_a_id"]): 0, str(row["team_b_id"]): 0}
        for index, winner in enumerate(winners):
            if winner not in counts or max(counts.values()) == wins_needed:
                return False
            counts[winner] += 1
            if max(counts.values()) == wins_needed and index != len(winners) - 1:
                return False
        if max(counts.values()) != wins_needed or len(winners) != int(row["map_count"]):
            return False
    return True
