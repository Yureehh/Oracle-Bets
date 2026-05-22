"""Oracle Bets module contract implementation for League of Legends."""

from __future__ import annotations

from dataclasses import dataclass

from oracle_bets_core.interfaces import ArtifactCheck, ArtifactHealth
from oracle_bets_core.paths import (
    FLATTENED_PLAYERS,
    FLATTENED_TEAMS,
    LEAGUE_ELO,
    OUTCOME_PREDICTION_FEATURE_PIPELINE,
    OUTCOME_PREDICTION_MODEL_PATH,
    TEAM_LEAGUES_MAPPING,
    TRAINING_PLAYER_DATA,
    TRAINING_TEAM_DATA,
)
from oracle_bets_core.pd import pd

MODULE_ID = "lol-bets"
TEAM_LEAGUE_COLUMNS = {"teamid", "league", "strength_pool"}
LEAGUE_ELO_COLUMNS = {"league", "elo", "strength_pool", "strength_pool_elo"}


def _check_file(name: str, path) -> ArtifactCheck:
    if not path.exists():
        return ArtifactCheck(name=name, path=str(path), ok=False, reason="missing")
    if not path.is_file():
        return ArtifactCheck(name=name, path=str(path), ok=False, reason="not a file")
    if path.stat().st_size <= 0:
        return ArtifactCheck(name=name, path=str(path), ok=False, reason="empty")
    return ArtifactCheck(name=name, path=str(path), ok=True)


def _check_parquet_schema(name: str, path, required: set[str]) -> ArtifactCheck:
    file_check = _check_file(name, path)
    if not file_check.ok:
        return file_check
    try:
        columns = set(pd.read_parquet(path).columns)
    except Exception as exc:
        return ArtifactCheck(
            name=name,
            path=str(path),
            ok=False,
            reason=f"unreadable parquet: {exc}",
        )
    missing = required - columns
    if missing:
        return ArtifactCheck(
            name=name,
            path=str(path),
            ok=False,
            reason=f"outdated schema; missing {', '.join(sorted(missing))}",
        )
    return file_check


@dataclass(frozen=True)
class LoLBetsModule:
    """LoL prediction module metadata and health checks."""

    id: str = MODULE_ID

    def artifact_health(self) -> ArtifactHealth:
        """Artifacts required for Discord/inference."""
        checks = (
            _check_file("flattened teams", FLATTENED_TEAMS),
            _check_file("flattened players", FLATTENED_PLAYERS),
            _check_file("outcome model", OUTCOME_PREDICTION_MODEL_PATH),
            _check_file(
                "outcome feature pipeline", OUTCOME_PREDICTION_FEATURE_PIPELINE
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

    def predict_props(self, *args, **kwargs):
        from lol_bets.inference.match_predictor import MatchPredictor

        predictor = MatchPredictor()
        return {
            "gamelength": predictor.predict_gamelength(*args, **kwargs),
            "total_kills": predictor.predict_total_kills(*args, **kwargs),
            "total_towers": predictor.predict_total_towers(*args, **kwargs),
        }
