"""Structural acceptance checks for the independent direct-series winner model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from oracle_bets_core.io_utils import load_model
from oracle_bets_core.paths import (
    MODEL_REGISTRY_DIR,
    SERIES_WINNER_FEATURE_LINEAGE,
    SERIES_WINNER_FEATURE_PIPELINE,
    SERIES_WINNER_MATCHUP_SCHEMA,
    SERIES_WINNER_MODEL_PATH,
)
from oracle_bets_core.pd import pd

from lol_bets.operations.models import ModelRegistry
from lol_bets.prediction_models.winner_model import (
    ENSEMBLE_MEMBERS,
    has_complete_direct_rating_contract,
)

FORBIDDEN_WINNER_FEATURE_FRAGMENTS = (
    "polymarket",
    "bookmaker",
    "market_odds",
    "decimal_odds",
    "market_price",
    "first_pick",
    "side_win_likelihood",
    "current_series",
    "game_in_series",
    "is_deciding_game",
    "maps_completed",
    "next_map_number",
    "series_wins_before",
    "series_losses_before",
    "series_score",
)
PROBABILITY_SUM_TOLERANCE = 1e-12


@dataclass(frozen=True)
class WinnerValidationReport:
    ok: bool
    champion_id: str | None
    actionable: bool
    checks: dict[str, bool]
    failures: tuple[str, ...]
    feature_count: int
    rating_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_winner_model(
    *,
    model_path=SERIES_WINNER_MODEL_PATH,
    pipeline_path=SERIES_WINNER_FEATURE_PIPELINE,
    schema_path=SERIES_WINNER_MATCHUP_SCHEMA,
    lineage_path=SERIES_WINNER_FEATURE_LINEAGE,
    registry_root=MODEL_REGISTRY_DIR,
) -> WinnerValidationReport:
    """Fail closed when the serving bundle violates Winner V2's contract."""
    failures: list[str] = []
    registry = ModelRegistry(registry_root)
    champion_id = registry.champion_id()
    actionable = bool(champion_id and registry.is_actionable(champion_id))
    if not actionable:
        failures.append("champion_not_actionable")

    try:
        model = load_model(model_path)
        pipeline = load_model(pipeline_path)
        schema = load_model(schema_path)
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    except Exception as error:
        failures.append(f"winner_artifact_unreadable:{type(error).__name__}")
        return WinnerValidationReport(
            ok=False,
            champion_id=champion_id,
            actionable=actionable,
            checks={},
            failures=tuple(failures),
            feature_count=0,
            rating_columns=(),
        )

    train_columns = tuple(str(item) for item in getattr(pipeline, "train_columns", ()))
    rating_columns = tuple(str(item) for item in getattr(model, "rating_columns", ()))
    members = tuple(getattr(model, "members", ()))
    member_columns = [
        tuple(str(item) for item in member.feature_name_) for member in members
    ]
    lineage_by_feature = {
        str(item.get("feature")): item for item in lineage if isinstance(item, dict)
    }
    lineage_fields = {
        "source",
        "availability_timestamp",
        "family",
        "swap_behavior",
        "model_eligible",
    }
    probability_contract = _probability_contract_ok(model, pipeline, train_columns)

    checks = {
        "direct_series_model": bool(members and rating_columns),
        "ten_week_block_members": len(members) == ENSEMBLE_MEMBERS,
        "train_serve_feature_parity": bool(train_columns)
        and all(columns == train_columns for columns in member_columns),
        "all_direct_rating_families": has_complete_direct_rating_contract(
            rating_columns
        ),
        "rating_columns_in_pipeline": set(rating_columns).issubset(train_columns),
        "complete_feature_lineage": set(lineage_by_feature) == set(train_columns)
        and all(
            lineage_fields.issubset(item)
            and item.get("availability_timestamp") == "strictly_before_fixture_start"
            and item.get("model_eligible") is True
            for item in lineage_by_feature.values()
        ),
        "no_forbidden_features": not any(
            fragment in column.casefold()
            for column in train_columns
            for fragment in FORBIDDEN_WINNER_FEATURE_FRAGMENTS
        ),
        "canonical_swap_contract": isinstance(schema, dict)
        and schema.get("canonical_key") == "teamid_then_teamname"
        and schema.get("version") == 1,
        "probability_components": callable(
            getattr(model, "component_probabilities", None)
        )
        and callable(getattr(model, "conservative_probability", None)),
        "calibrated_rating_baseline": callable(
            getattr(model, "rating_baseline_probability", None)
        ),
        "exact_probability_complement": probability_contract,
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    return WinnerValidationReport(
        ok=not failures,
        champion_id=champion_id,
        actionable=actionable,
        checks=checks,
        failures=tuple(failures),
        feature_count=len(train_columns),
        rating_columns=rating_columns,
    )


def _probability_contract_ok(
    model: Any, pipeline: Any, columns: tuple[str, ...]
) -> bool:
    """Smoke-test the fitted binary probability surface on one valid feature row."""
    if not columns or not callable(getattr(model, "predict_proba", None)):
        return False
    categorical_levels = getattr(pipeline, "categorical_levels", {}) or {}
    numeric_medians = getattr(pipeline, "numeric_medians", {}) or {}
    values: dict[str, Any] = {}
    for column in columns:
        if column in categorical_levels:
            levels = list(categorical_levels[column]) or ["Unknown"]
            values[column] = pd.Categorical([levels[0]], categories=levels)
        else:
            values[column] = [float(numeric_medians.get(column, 0.0))]
    try:
        probability = np.asarray(model.predict_proba(pd.DataFrame(values)), dtype=float)
    except Exception:
        return False
    return bool(
        probability.shape == (1, 2)
        and np.isfinite(probability).all()
        and np.all((probability >= 0.0) & (probability <= 1.0))
        and abs(float(probability.sum()) - 1.0) <= PROBABILITY_SUM_TOLERANCE
    )
