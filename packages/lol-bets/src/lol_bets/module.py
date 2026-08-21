"""Oracle Bets module contract implementation for League of Legends."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from oracle_bets_core.interfaces import ArtifactCheck, ArtifactHealth
from oracle_bets_core.paths import (
    FLATTENED_PLAYERS,
    FLATTENED_TEAMS,
    GAMELENGTH_PREDICTION_FEATURE_PIPELINE,
    GAMELENGTH_PREDICTION_MODEL_PATH,
    GAMELENGTH_PREDICTION_PROP_CALIBRATOR,
    GAMELENGTH_PREDICTION_RESIDUAL_SUMMARY,
    LEAGUE_ELO,
    MODEL_REGISTRY_DIR,
    MODELS_DIR,
    OUTCOME_PREDICTION_FEATURE_PIPELINE,
    OUTCOME_PREDICTION_MATCHUP_SCHEMA,
    OUTCOME_PREDICTION_MODEL_PATH,
    OUTCOME_PREDICTION_PROBABILITY_CALIBRATOR,
    OUTCOME_PREDICTION_PROBABILITY_UNCERTAINTY,
    SERIES_WINNER_FEATURE_PIPELINE,
    SERIES_WINNER_MATCHUP_SCHEMA,
    SERIES_WINNER_MODEL_PATH,
    SERIES_WINNER_PROBABILITY_CALIBRATOR,
    SERIES_WINNER_PROBABILITY_UNCERTAINTY,
    TEAM_LEAGUES_MAPPING,
    TOTAL_KILLS_PREDICTION_FEATURE_PIPELINE,
    TOTAL_KILLS_PREDICTION_MODEL_PATH,
    TOTAL_KILLS_PREDICTION_PROP_CALIBRATOR,
    TOTAL_KILLS_PREDICTION_RESIDUAL_SUMMARY,
    TOTAL_TOWERS_PREDICTION_FEATURE_PIPELINE,
    TOTAL_TOWERS_PREDICTION_MODEL_PATH,
    TOTAL_TOWERS_PREDICTION_PROP_CALIBRATOR,
    TOTAL_TOWERS_PREDICTION_RESIDUAL_SUMMARY,
    TRAINING_PLAYER_DATA,
    TRAINING_TEAM_DATA,
)
from oracle_bets_core.pd import pd

from lol_bets.operations.models import ModelRegistryError, resolve_serving_artifact

MODULE_ID = "lol-bets"
TEAM_LEAGUE_COLUMNS = {"teamid", "league", "strength_pool"}
LEAGUE_ELO_COLUMNS = {
    "league",
    "elo",
    "strength_pool",
    "strength_pool_elo",
    "strength_pool_cross_games",
}
FLATTENED_TEAM_COLUMNS = {"teamname", "teamid", "gameid", "date", "league"}
FLATTENED_TEAM_COLUMNS |= {
    "elo",
    "glicko2_mu",
    "glicko2_phi",
    "pl_mu",
    "pl_sigma",
    "trueskill_mu",
    "trueskill_sigma",
}
FLATTENED_PLAYER_COLUMNS = {
    "teamname",
    "playername",
    "position",
    "date",
    "gameid",
    "side",
    "elo",
    "glicko2_mu",
    "glicko2_phi",
    "pl_mu",
    "pl_sigma",
    "trueskill_mu",
    "trueskill_sigma",
}
CALIBRATOR_VERSION = 3


def _resolve_model_path(path) -> Path:
    candidate = Path(path)
    if not candidate.resolve().is_relative_to(MODELS_DIR.resolve()):
        return candidate
    return resolve_serving_artifact(
        candidate,
        registry_root=MODEL_REGISTRY_DIR,
        legacy_root=MODELS_DIR,
    )


def _resolve_file_check(name: str, path) -> tuple[ArtifactCheck, Path | None]:
    try:
        resolved = _resolve_model_path(path)
    except ModelRegistryError as error:
        return (
            ArtifactCheck(
                name=name,
                path=str(path),
                ok=False,
                reason=f"serving bundle invalid: {error}",
            ),
            None,
        )
    if not resolved.exists():
        return ArtifactCheck(
            name=name, path=str(resolved), ok=False, reason="missing"
        ), None
    if not resolved.is_file():
        return (
            ArtifactCheck(name=name, path=str(resolved), ok=False, reason="not a file"),
            None,
        )
    if resolved.stat().st_size <= 0:
        return ArtifactCheck(
            name=name, path=str(resolved), ok=False, reason="empty"
        ), None
    return ArtifactCheck(name=name, path=str(resolved), ok=True), resolved


def _check_file(name: str, path) -> ArtifactCheck:
    return _resolve_file_check(name, path)[0]


def _check_parquet_schema(name: str, path, required: set[str]) -> ArtifactCheck:
    file_check, resolved = _resolve_file_check(name, path)
    if not file_check.ok or resolved is None:
        return file_check
    try:
        columns = set(pd.read_parquet(resolved).columns)
    except Exception as exc:
        return ArtifactCheck(
            name=name,
            path=str(resolved),
            ok=False,
            reason=f"unreadable parquet: {exc}",
        )
    missing = required - columns
    if missing:
        return ArtifactCheck(
            name=name,
            path=str(resolved),
            ok=False,
            reason=f"outdated schema; missing {', '.join(sorted(missing))}",
        )
    return file_check


def _check_flattened_parquet(
    name: str,
    path,
    required: set[str],
    *,
    unique_keys: list[str],
) -> ArtifactCheck:
    file_check, resolved = _resolve_file_check(name, path)
    if not file_check.ok or resolved is None:
        return file_check
    try:
        df = pd.read_parquet(resolved)
    except Exception as exc:
        return ArtifactCheck(
            name=name,
            path=str(resolved),
            ok=False,
            reason=f"unreadable parquet: {exc}",
        )
    missing = required - set(df.columns)
    if missing:
        return ArtifactCheck(
            name=name,
            path=str(resolved),
            ok=False,
            reason=f"outdated schema; missing {', '.join(sorted(missing))}",
        )
    duplicate_count = int(df.duplicated(unique_keys, keep=False).sum())
    if duplicate_count:
        return ArtifactCheck(
            name=name,
            path=str(resolved),
            ok=False,
            reason=(
                "duplicate flattened snapshots; "
                f"{duplicate_count} rows share {', '.join(unique_keys)}"
            ),
        )
    return file_check


def _check_calibrator_schema(
    name: str,
    path,
    *,
    required_attrs: set[str],
    expected_version: int = CALIBRATOR_VERSION,
) -> ArtifactCheck:
    file_check, resolved = _resolve_file_check(name, path)
    if not file_check.ok or resolved is None:
        return file_check
    try:
        with resolved.open("rb") as f:
            artifact = pickle.load(f)  # noqa: S301
    except Exception as exc:
        return ArtifactCheck(
            name=name,
            path=str(resolved),
            ok=False,
            reason=f"unreadable calibrator: {exc}",
        )

    missing = [attr for attr in sorted(required_attrs) if not hasattr(artifact, attr)]
    version = getattr(artifact, "version", None)
    if missing or version != expected_version:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if version != expected_version:
            details.append(f"version {version!r} != {expected_version}")
        return ArtifactCheck(
            name=name,
            path=str(resolved),
            ok=False,
            reason=f"outdated calibrator schema; {'; '.join(details)}",
        )
    return file_check


def _check_winner_contract() -> ArtifactCheck:
    from lol_bets.operations.winner_validation import validate_winner_model

    report = validate_winner_model()
    return ArtifactCheck(
        name="direct series winner contract",
        path=str(SERIES_WINNER_MODEL_PATH),
        ok=report.ok,
        reason="" if report.ok else ", ".join(report.failures),
    )


@dataclass(frozen=True)
class LoLBetsModule:
    """LoL prediction module metadata and health checks."""

    id: str = MODULE_ID

    def artifact_health(self) -> ArtifactHealth:
        """Artifacts required for Discord/inference."""
        checks = (
            _check_flattened_parquet(
                "flattened teams",
                FLATTENED_TEAMS,
                FLATTENED_TEAM_COLUMNS,
                unique_keys=["teamname"],
            ),
            _check_parquet_schema(
                "flattened players",
                FLATTENED_PLAYERS,
                FLATTENED_PLAYER_COLUMNS,
            ),
            _check_file("outcome model", OUTCOME_PREDICTION_MODEL_PATH),
            _check_file(
                "outcome feature pipeline", OUTCOME_PREDICTION_FEATURE_PIPELINE
            ),
            _check_file("outcome matchup schema", OUTCOME_PREDICTION_MATCHUP_SCHEMA),
            _check_calibrator_schema(
                "outcome probability calibrator",
                OUTCOME_PREDICTION_PROBABILITY_CALIBRATOR,
                required_attrs={"global_calibrator", "segments", "version"},
            ),
            _check_calibrator_schema(
                "outcome probability uncertainty",
                OUTCOME_PREDICTION_PROBABILITY_UNCERTAINTY,
                required_attrs={
                    "bins",
                    "confidence",
                    "fit_split",
                    "interval",
                    "sample_count",
                    "version",
                },
                expected_version=1,
            ),
            _check_file("direct series winner model", SERIES_WINNER_MODEL_PATH),
            _check_file(
                "direct series winner feature pipeline", SERIES_WINNER_FEATURE_PIPELINE
            ),
            _check_file(
                "direct series winner matchup schema", SERIES_WINNER_MATCHUP_SCHEMA
            ),
            _check_calibrator_schema(
                "direct series winner calibrator",
                SERIES_WINNER_PROBABILITY_CALIBRATOR,
                required_attrs={"global_calibrator", "segments", "version"},
            ),
            _check_calibrator_schema(
                "direct series winner uncertainty",
                SERIES_WINNER_PROBABILITY_UNCERTAINTY,
                required_attrs={
                    "bins",
                    "confidence",
                    "fit_split",
                    "interval",
                    "sample_count",
                    "version",
                },
                expected_version=1,
            ),
            _check_winner_contract(),
            _check_file("gamelength model", GAMELENGTH_PREDICTION_MODEL_PATH),
            _check_file(
                "gamelength feature pipeline", GAMELENGTH_PREDICTION_FEATURE_PIPELINE
            ),
            _check_file(
                "gamelength residual summary", GAMELENGTH_PREDICTION_RESIDUAL_SUMMARY
            ),
            _check_calibrator_schema(
                "gamelength prop calibrator",
                GAMELENGTH_PREDICTION_PROP_CALIBRATOR,
                required_attrs={"global_residuals", "segment_residuals", "version"},
            ),
            _check_file("total kills model", TOTAL_KILLS_PREDICTION_MODEL_PATH),
            _check_file(
                "total kills feature pipeline", TOTAL_KILLS_PREDICTION_FEATURE_PIPELINE
            ),
            _check_file(
                "total kills residual summary", TOTAL_KILLS_PREDICTION_RESIDUAL_SUMMARY
            ),
            _check_calibrator_schema(
                "total kills prop calibrator",
                TOTAL_KILLS_PREDICTION_PROP_CALIBRATOR,
                required_attrs={"global_residuals", "segment_residuals", "version"},
            ),
            _check_file("total towers model", TOTAL_TOWERS_PREDICTION_MODEL_PATH),
            _check_file(
                "total towers feature pipeline",
                TOTAL_TOWERS_PREDICTION_FEATURE_PIPELINE,
            ),
            _check_file(
                "total towers residual summary",
                TOTAL_TOWERS_PREDICTION_RESIDUAL_SUMMARY,
            ),
            _check_calibrator_schema(
                "total towers prop calibrator",
                TOTAL_TOWERS_PREDICTION_PROP_CALIBRATOR,
                required_attrs={"global_residuals", "segment_residuals", "version"},
            ),
            _check_parquet_schema(
                "team league mapping", TEAM_LEAGUES_MAPPING, TEAM_LEAGUE_COLUMNS
            ),
            _check_parquet_schema("league elo", LEAGUE_ELO, LEAGUE_ELO_COLUMNS),
        )
        return ArtifactHealth(module_id=self.id, checks=checks)

    def training_artifact_health(self) -> ArtifactHealth:
        """Artifacts required before supervised model training."""
        checks = (
            _check_file("training teams", TRAINING_TEAM_DATA),
            _check_file("training players", TRAINING_PLAYER_DATA),
        )
        return ArtifactHealth(module_id=self.id, checks=checks)

    def predict_match(self, *args, **kwargs):
        from lol_bets.inference.match_predictor import MatchPredictor

        return MatchPredictor().predict_match(*args, **kwargs)
