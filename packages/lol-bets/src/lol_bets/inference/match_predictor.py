# lol_bets/inference/match_predictor.py
"""
LoL Oracle – Match Predictor

- Loads artifacts once (LRU-cached parquet reads + lazy model loading).
- Clean rating helpers (Elo, Glicko-2, Plackett–Luce, TrueSkill, League Elo).
- Safe probability math (clipping + strict validation).
- Robust team + player feature assembly (mirrors training-time transforms).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from oracle_bets_core.io_utils import load_model
from oracle_bets_core.league_taxonomy import get_league_taxonomy
from oracle_bets_core.paths import (
    GAMELENGTH_PREDICTION_CATEGORICAL_FEATURES,
    GAMELENGTH_PREDICTION_FEATURE_PIPELINE,
    GAMELENGTH_PREDICTION_FINAL_FEATURES,
    GAMELENGTH_PREDICTION_MODEL_PATH,
    GAMELENGTH_PREDICTION_PROP_CALIBRATOR,
    GAMELENGTH_PREDICTION_RESIDUAL_SUMMARY,
    LEAGUE_ELO,
    MODEL_REGISTRY_DIR,
    MODELS_DIR,
    OUTCOME_PREDICTION_CATEGORICAL_FEATURES,
    OUTCOME_PREDICTION_FEATURE_PIPELINE,
    OUTCOME_PREDICTION_FINAL_FEATURES,
    OUTCOME_PREDICTION_MATCHUP_SCHEMA,
    OUTCOME_PREDICTION_MODEL_PATH,
    OUTCOME_PREDICTION_PROBABILITY_CALIBRATOR,
    OUTCOME_PREDICTION_PROBABILITY_UNCERTAINTY,
    TEAM_LEAGUES_MAPPING,
    TOTAL_KILLS_PREDICTION_CATEGORICAL_FEATURES,
    TOTAL_KILLS_PREDICTION_FEATURE_PIPELINE,
    TOTAL_KILLS_PREDICTION_FINAL_FEATURES,
    TOTAL_KILLS_PREDICTION_MODEL_PATH,
    TOTAL_KILLS_PREDICTION_PROP_CALIBRATOR,
    TOTAL_KILLS_PREDICTION_RESIDUAL_SUMMARY,
    TOTAL_TOWERS_PREDICTION_CATEGORICAL_FEATURES,
    TOTAL_TOWERS_PREDICTION_FEATURE_PIPELINE,
    TOTAL_TOWERS_PREDICTION_FINAL_FEATURES,
    TOTAL_TOWERS_PREDICTION_MODEL_PATH,
    TOTAL_TOWERS_PREDICTION_PROP_CALIBRATOR,
    TOTAL_TOWERS_PREDICTION_RESIDUAL_SUMMARY,
)
from oracle_bets_core.pd import pd

from lol_bets.data_generation.feature_engineering.ratings_features.glicko import (
    DEFAULT_MU,
    DEFAULT_PHI,
    DEFAULT_SIGMA,
    Glicko2,
    calculate_mean_rating,
)
from lol_bets.data_generation.feature_engineering.ratings_features.glicko import (
    Rating as GlickoRating,
)
from lol_bets.data_generation.feature_engineering.ratings_features.plackett_luce import (
    PlackettLuce,
)
from lol_bets.data_generation.feature_engineering.ratings_features.plackett_luce import (
    predict_win_probability as pl_win_probability,
)
from lol_bets.data_generation.feature_engineering.ratings_features.trueskill import (
    Rating as TrueskillRating,
)
from lol_bets.data_generation.feature_engineering.ratings_features.trueskill import (
    expected_win_probability as trueskill_win_probability,
)
from lol_bets.operations.models import resolve_serving_artifact
from lol_bets.prediction_models.gbdt_model import FeaturePipeline, GradientBoostingModel
from lol_bets.prediction_models.prop_features import (
    build_game_level_outcome_features,
    build_game_level_prop_features,
)

if TYPE_CHECKING:
    from lol_bets.inference.team import Team

# ── constants ────────────────────────────────────────────────────────────── #

ELO_FACTOR = 400.0
RATING_DECIMALS = 3  # for human-facing rounded rating-based probs
_PROB_EPS = 1e-12  # small epsilon for clipping
_ROLES = ("top", "jng", "mid", "bot", "sup")
TARGET_COLUMNS = {"result", "gamelength", "total_kills", "total_towers"}
TEAM_LEAGUE_COLUMNS = {"teamid", "league", "strength_pool"}
LEAGUE_ELO_COLUMNS = {
    "league",
    "elo",
    "strength_pool",
    "strength_pool_elo",
    "strength_pool_cross_games",
}
POOL_SHRINKAGE_GAMES = 30
CONTRIBUTION_ARRAY_DIMENSIONS = 2
MAX_PREDICTION_DRIVERS = 3


# ── cached parquet reads ─────────────────────────────────────────────────── #


@lru_cache(maxsize=4)
def _read_parquet_cached(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="fastparquet")
    except (ImportError, ValueError):
        # fallback to pyarrow if available
        return pd.read_parquet(path)


def _serving_path(path) -> Any:
    """Resolve a checksum-verified champion while supporting bootstrap installs."""
    return resolve_serving_artifact(
        path,
        registry_root=MODEL_REGISTRY_DIR,
        legacy_root=MODELS_DIR,
    )


def _require_columns(df: pd.DataFrame, required: set[str], artifact_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        msg = f"{artifact_name} has outdated schema; missing columns: {missing_cols}"
        raise RuntimeError(msg)


# ── probability helpers ──────────────────────────────────────────────────── #


def _clip01(x: float | np.ndarray | pd.Series) -> Any:
    return np.clip(x, 0.0 + _PROB_EPS, 1.0 - _PROB_EPS)


def _to_float(x: Any) -> float:
    return float(x)  # raises if NaN/None; good for surfacing issues early


# ── rating models (team-level scalar + player-level vector support) ──────── #


def _elo_prob(t1, t2) -> Any:
    """
    Logistic Elo win prob for Team 1.
    Accepts scalars, Series, or ndarrays (vectorized as needed).
    """
    t1v = np.asarray(t1)
    t2v = np.asarray(t2)
    return _clip01(1.0 / (1.0 + 10.0 ** ((t2v - t1v) / ELO_FACTOR)))


def _glicko2_prob(team1_mus, team1_phis, team2_mus, team2_phis) -> float:
    """
    Glicko-2 team vs team (scalar). Aggregates by team average with impact reduction.
    """
    # normalize to list of floats
    if isinstance(team1_mus, Iterable) and not isinstance(team1_mus, float | int):
        t1_mus = list(team1_mus)
        t1_phis = list(team1_phis)
        t2_mus = list(team2_mus)
        t2_phis = list(team2_phis)
    else:
        t1_mus = [team1_mus]
        t1_phis = [team1_phis]
        t2_mus = [team2_mus]
        t2_phis = [team2_phis]

    blue = [
        GlickoRating(_to_float(mu), _to_float(phi))
        for mu, phi in zip(t1_mus, t1_phis, strict=False)
    ]
    red = [
        GlickoRating(_to_float(mu), _to_float(phi))
        for mu, phi in zip(t2_mus, t2_phis, strict=False)
    ]

    model = Glicko2(mu=DEFAULT_MU, phi=DEFAULT_PHI, sigma=DEFAULT_SIGMA)
    mean_blue = calculate_mean_rating(blue)
    mean_red = calculate_mean_rating(red)
    mean_impact = sum(model.reduce_impact(r) for r in blue) / max(1, len(blue))
    return float(_clip01(model.expect_score(mean_blue, mean_red, mean_impact)))


def _pl_prob(team1_mus, team1_sigmas, team2_mus, team2_sigmas) -> float:
    """
    Plackett–Luce (scalar, team vs team). Aggregates players as a lineup of ratings.
    """
    if isinstance(team1_mus, Iterable) and not isinstance(team1_mus, float | int):
        t1_mus = list(team1_mus)
        t1_sig = list(team1_sigmas)
        t2_mus = list(team2_mus)
        t2_sig = list(team2_sigmas)
    else:
        t1_mus = [team1_mus]
        t1_sig = [team1_sigmas]
        t2_mus = [team2_mus]
        t2_sig = [team2_sigmas]

    model = PlackettLuce()
    t1 = [
        model.rating(_to_float(mu), _to_float(s))
        for mu, s in zip(t1_mus, t1_sig, strict=False)
    ]
    t2 = [
        model.rating(_to_float(mu), _to_float(s))
        for mu, s in zip(t2_mus, t2_sig, strict=False)
    ]
    prob = pl_win_probability(model, t1, t2)
    return float(_clip01(prob))


def _ts_prob(team1_mus, team1_sigmas, team2_mus, team2_sigmas) -> float:
    """
    TrueSkill (scalar, team vs team). Aggregates players as lineup of ratings.
    """
    if isinstance(team1_mus, Iterable) and not isinstance(team1_mus, float | int):
        t1_mus = list(team1_mus)
        t1_sig = list(team1_sigmas)
        t2_mus = list(team2_mus)
        t2_sig = list(team2_sigmas)
    else:
        t1_mus = [team1_mus]
        t1_sig = [team1_sigmas]
        t2_mus = [team2_mus]
        t2_sig = [team2_sigmas]

    t1 = [
        TrueskillRating(_to_float(mu), _to_float(s))
        for mu, s in zip(t1_mus, t1_sig, strict=False)
    ]
    t2 = [
        TrueskillRating(_to_float(mu), _to_float(s))
        for mu, s in zip(t2_mus, t2_sig, strict=False)
    ]
    return float(_clip01(trueskill_win_probability(t1, t2)))


# ── predictor ────────────────────────────────────────────────────────────── #


@dataclass
class MatchPredictor:
    """
    Predicts match outcomes using:
      - team/player ratings (Elo, Glicko-2, PL, TrueSkill, League Elo)
      - trained GBDT outcome model (lightgbm)
    """

    outcome_model: Any = field(default=None, init=False, repr=False)
    outcome_calibrator: Any = field(default=None, init=False, repr=False)
    outcome_uncertainty: Any = field(default=None, init=False, repr=False)
    gamelength_model: Any = field(default=None, init=False, repr=False)
    total_kills_model: Any = field(default=None, init=False, repr=False)
    total_towers_model: Any = field(default=None, init=False, repr=False)
    gamelength_residual_summary: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )
    total_kills_residual_summary: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )
    total_towers_residual_summary: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )
    gamelength_prop_calibrator: Any = field(default=None, init=False, repr=False)
    total_kills_prop_calibrator: Any = field(default=None, init=False, repr=False)
    total_towers_prop_calibrator: Any = field(default=None, init=False, repr=False)
    team_to_league: pd.DataFrame = field(init=False, repr=False)
    league_to_elo: pd.DataFrame = field(init=False, repr=False)
    # Feature pipelines for inference parity with training
    outcome_pipeline: FeaturePipeline | None = field(
        default=None, init=False, repr=False
    )
    outcome_matchup_schema: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )
    gamelength_pipeline: FeaturePipeline | None = field(
        default=None, init=False, repr=False
    )
    total_kills_pipeline: FeaturePipeline | None = field(
        default=None, init=False, repr=False
    )
    total_towers_pipeline: FeaturePipeline | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        # artifacts: load lazily but fail fast if missing
        self._load_artifacts()

    # ── artifacts ───────────────────────────────────────────────────────── #

    def _load_artifacts(self) -> None:  # noqa: PLR0912
        # sourcery skip: remove-redundant-exception, simplify-single-exception-tuple
        try:
            self.outcome_model = load_model(
                _serving_path(OUTCOME_PREDICTION_MODEL_PATH)
            )
        except Exception as e:
            msg = f"Failed to load outcome model: {e}"
            raise RuntimeError(msg) from e

        try:
            self.outcome_calibrator = load_model(
                _serving_path(OUTCOME_PREDICTION_PROBABILITY_CALIBRATOR)
            )
        except Exception:
            self.outcome_calibrator = None

        try:
            self.outcome_uncertainty = load_model(
                _serving_path(OUTCOME_PREDICTION_PROBABILITY_UNCERTAINTY)
            )
        except Exception:
            self.outcome_uncertainty = None

        for model_name, path in (
            ("gamelength", GAMELENGTH_PREDICTION_MODEL_PATH),
            ("total_kills", TOTAL_KILLS_PREDICTION_MODEL_PATH),
            ("total_towers", TOTAL_TOWERS_PREDICTION_MODEL_PATH),
        ):
            try:
                setattr(self, f"{model_name}_model", load_model(_serving_path(path)))
            except Exception:
                setattr(self, f"{model_name}_model", None)

        for name, path in (
            ("gamelength_residual_summary", GAMELENGTH_PREDICTION_RESIDUAL_SUMMARY),
            ("total_kills_residual_summary", TOTAL_KILLS_PREDICTION_RESIDUAL_SUMMARY),
            ("total_towers_residual_summary", TOTAL_TOWERS_PREDICTION_RESIDUAL_SUMMARY),
        ):
            try:
                setattr(self, name, load_model(_serving_path(path)))
            except Exception:
                setattr(self, name, None)

        for name, path in (
            ("gamelength_prop_calibrator", GAMELENGTH_PREDICTION_PROP_CALIBRATOR),
            ("total_kills_prop_calibrator", TOTAL_KILLS_PREDICTION_PROP_CALIBRATOR),
            ("total_towers_prop_calibrator", TOTAL_TOWERS_PREDICTION_PROP_CALIBRATOR),
        ):
            try:
                setattr(self, name, load_model(_serving_path(path)))
            except Exception:
                setattr(self, name, None)

        try:
            self.team_to_league = _read_parquet_cached(
                str(_serving_path(TEAM_LEAGUES_MAPPING))
            )
            self.league_to_elo = _read_parquet_cached(str(_serving_path(LEAGUE_ELO)))
            _require_columns(
                self.team_to_league,
                TEAM_LEAGUE_COLUMNS,
                "team_league_mapping.parquet",
            )
            _require_columns(
                self.league_to_elo, LEAGUE_ELO_COLUMNS, "league_elo.parquet"
            )
        except Exception as e:
            msg = f"Failed to load mapping/elo parquet: {e}"
            raise RuntimeError(msg) from e

        # Load feature pipelines for training-inference parity
        try:
            self.outcome_pipeline = load_model(
                _serving_path(OUTCOME_PREDICTION_FEATURE_PIPELINE)
            )
        except Exception:
            self.outcome_pipeline = None
        try:
            self.outcome_matchup_schema = load_model(
                _serving_path(OUTCOME_PREDICTION_MATCHUP_SCHEMA)
            )
        except Exception as exc:
            msg = (
                "Outcome model uses the retired team-row schema. Retrain the outcome "
                "model with `oracle-bets lol retune`, review and promote its "
                "parameters, then run routine training before matchup predictions."
            )
            raise RuntimeError(msg) from exc

        for pipeline_name, path in (
            ("gamelength_pipeline", GAMELENGTH_PREDICTION_FEATURE_PIPELINE),
            ("total_kills_pipeline", TOTAL_KILLS_PREDICTION_FEATURE_PIPELINE),
            ("total_towers_pipeline", TOTAL_TOWERS_PREDICTION_FEATURE_PIPELINE),
        ):
            try:
                setattr(self, pipeline_name, load_model(_serving_path(path)))
            except Exception:
                setattr(self, pipeline_name, None)

    def _resolve_team_league(self, team_id: Any) -> str:
        t2l = self.team_to_league
        row = t2l.loc[t2l["teamid"].astype(str) == str(team_id)]
        if row.empty:
            msg = "Team ID not found in team-to-league mapping."
            raise ValueError(msg)
        return str(row["league"].iloc[0])

    def _resolve_league_elo(self, league: str) -> float:
        l2e = self.league_to_elo
        e = l2e.loc[l2e["league"] == league, "elo"]
        if e.empty:
            msg = "League not found in ELO ratings."
            raise ValueError(msg)
        return float(e.iloc[0])

    def _resolve_strength_pool_elo(self, league: str) -> float:
        l2e = self.league_to_elo
        pool = get_league_taxonomy(league)["strength_pool"]
        e = l2e.loc[l2e["strength_pool"] == pool, "strength_pool_elo"]
        if e.empty:
            msg = "Strength pool not found in league ELO ratings."
            raise ValueError(msg)
        return float(e.iloc[0])

    def league_elo_prediction(self, team1_id: int, team2_id: int) -> float:
        """
        League-level Elo logistic probability (Team 1 vs Team 2), using team->league mapping.
        """
        t1_league = self._resolve_team_league(team1_id)
        t2_league = self._resolve_team_league(team2_id)
        e1 = self._resolve_league_elo(t1_league)
        e2 = self._resolve_league_elo(t2_league)
        prob = _elo_prob(e1, e2)
        return round(float(prob), RATING_DECIMALS)

    def strength_pool_prediction(self, team1_id: int, team2_id: int) -> float:
        """Macro league-pool Elo probability for Team 1 vs Team 2."""
        t1_league = self._resolve_team_league(team1_id)
        t2_league = self._resolve_team_league(team2_id)
        e1 = self._resolve_strength_pool_elo(t1_league)
        e2 = self._resolve_strength_pool_elo(t2_league)
        prob = _elo_prob(e1, e2)
        pool1 = get_league_taxonomy(t1_league)["strength_pool"]
        pool2 = get_league_taxonomy(t2_league)["strength_pool"]
        support = self.league_to_elo.loc[
            self.league_to_elo["strength_pool"].isin([pool1, pool2]),
            "strength_pool_cross_games",
        ]
        support_games = int(support.min()) if not support.empty else 0
        weight = support_games / (support_games + POOL_SHRINKAGE_GAMES)
        shrunk = 0.5 + ((float(prob) - 0.5) * weight)
        return round(float(shrunk), RATING_DECIMALS)

    # ── simple WR-based heuristics ──────────────────────────────────────── #

    @staticmethod
    def patch_season_wr_prediction(team1_wr: float, team2_wr: float) -> float:
        total = team1_wr + team2_wr
        return 0.5 if total <= 0.0 else round(team1_wr / total, RATING_DECIMALS)

    # ── feature assembly (teams) ────────────────────────────────────────── #

    def apply_stat_modifications(
        self,
        team1_stats: pd.Series,
        team2_stats: pd.Series,
        account_for_side: bool,
        match_type: str | None = None,
    ) -> pd.Series:
        """
        Build derived likelihood features for team rows.
        Returns copy of team1_stats with derived features added.

        Only computes features that are in the training config:
        - league_elo_win_likelihood (inter-league calibration)
        - season_win_likelihood (season-specific)

        Note: elo_win_likelihood, glicko2_win_likelihood, pl_win_likelihood,
        trueskill_win_likelihood were removed as they are redundant
        (deterministic transforms of base ratings).
        """
        t1 = team1_stats.copy()
        t2 = team2_stats.copy()

        # Add league metadata
        t1_league = t1.get("league") or self._resolve_team_league(t1.get("teamid"))
        t2.get("league") or self._resolve_team_league(t2.get("teamid"))
        t1_tax = get_league_taxonomy(t1_league)
        t1["league_region"] = t1_tax["region"]
        t1["league_tier"] = t1_tax["tier"]
        t1["strength_pool"] = t1_tax["strength_pool"]

        # League Elo win likelihood (inter-league calibration)
        t1["league_elo_win_likelihood"] = self.league_elo_prediction(
            t1["teamid"], t2["teamid"]
        )
        t1["strength_pool_win_likelihood"] = self.strength_pool_prediction(
            t1["teamid"], t2["teamid"]
        )

        # Side and First Selection are retained as source context, not model
        # signals, until separately timestamped data earns promotion.
        del account_for_side

        # Season win likelihood
        t1["season_win_likelihood"] = self.patch_season_wr_prediction(
            float(t1.get("ema_season_win_rate", 0.0)),
            float(t2.get("ema_season_win_rate", 0.0)),
        )

        # Future-match context must not inherit the last historical map's
        # series state from flattened team snapshots.
        series = self.series_context(match_type)
        for key, value in series.items():
            t1[key] = value

        # Drop raw ratings/ids from features (keep meta like gameid/teamname/side for merges)
        base_drop = [
            "league_elo",
            "strength_pool_elo",
            "elo",
            "glicko2_mu",
            "glicko2_phi",
            "pl_mu",
            "pl_sigma",
            "trueskill_mu",
            "trueskill_sigma",
            "ema_red_side",
            "ema_blue_side",
            # leave teamid, gameid, teamname, side on t1; on t2 we drop id/name/date below
        ]
        t1 = t1.drop(labels=base_drop, errors="ignore")
        t2 = t2.drop(labels=[*base_drop, "teamname", "gameid", "date"], errors="ignore")

        return t1

    @staticmethod
    def series_context(match_type: str | None = None) -> dict[str, int]:
        normalized = (match_type or "").strip().casefold()
        return {
            "game_in_series": 1,
            "is_bo1": int(normalized == "bo1"),
            "is_bo3": int(normalized == "bo3"),
            "is_bo5": int(normalized == "bo5"),
            "is_deciding_game": 0,
        }

    def calculate_team_stats(
        self,
        team1: Team,
        team2: Team,
        account_for_side: bool,
        match_type: str | None = None,
    ) -> pd.DataFrame:
        # Ensure 'side' is present (needed for side-based WR lookups + later merge)
        t1_stats = team1.team_stats.copy()
        t2_stats = team2.team_stats.copy()
        if "side" not in t1_stats.index:
            t1_stats["side"] = team1.side or ""
        if "side" not in t2_stats.index:
            t2_stats["side"] = team2.side or ""

        # 1) Team1 vs Team2 -> own features
        t1_fwd = self.apply_stat_modifications(
            t1_stats, t2_stats, account_for_side, match_type
        )

        # 2) Team2 vs Team1 -> source for opponent features
        t2_mirr = self.apply_stat_modifications(
            t2_stats, t1_stats, account_for_side, match_type
        )

        # 3) Left row (own features) — ensure merge keys exist
        gid = t1_stats.get("gameid", np.nan)
        sde = (team1.side or str(t1_stats.get("side", ""))).strip()

        left = t1_fwd.to_frame().T
        left["gameid"] = gid
        left["side"] = sde

        # 4) Right row (opponent features) — keep all *_win_likelihood + any ema_* used in training
        opp_keep = [c for c in t2_mirr.index if c.endswith("_win_likelihood")]
        opp_keep += [c for c in t2_mirr.index if c.startswith("ema_")]

        opp_payload = {f"opp_{c}": t2_mirr.get(c, np.nan) for c in opp_keep}
        right = pd.DataFrame([{**opp_payload, "gameid": gid, "side": sde}])

        # 5) Proper one-to-one merge on keys -> single wide row
        return left.merge(
            right, on=["gameid", "side"], how="left", validate="one_to_one"
        )

    # ── feature assembly (players) ──────────────────────────────────────── #

    def apply_player_stat_modifications(
        self, p1: pd.DataFrame, p2: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Row-wise modifications for players; returns copies.
        """
        a = p1.copy()
        b = p2.copy()

        if not a.empty and not b.empty:
            a["elo_win_likelihood"] = _elo_prob(
                a["elo"].mean(),
                b["elo"].mean(),
            )
            a["glicko2_win_likelihood"] = _glicko2_prob(
                a["glicko2_mu"],
                a["glicko2_phi"],
                b["glicko2_mu"],
                b["glicko2_phi"],
            )
            a["pl_win_likelihood"] = _pl_prob(
                a["pl_mu"],
                a["pl_sigma"],
                b["pl_mu"],
                b["pl_sigma"],
            )
            a["trueskill_win_likelihood"] = _ts_prob(
                a["trueskill_mu"],
                a["trueskill_sigma"],
                b["trueskill_mu"],
                b["trueskill_sigma"],
            )

        # Drop raw rating columns (keep id/meta like gameid/side/teamname/position on 'a')
        drop_cols = [
            "date",
            "playername",
            "elo",
            "glicko2_mu",
            "glicko2_phi",
            "pl_mu",
            "pl_sigma",
            "trueskill_mu",
            "trueskill_sigma",
        ]
        a = a.drop(columns=drop_cols, errors="ignore")

        # On opponent side keep ONLY EMA-based numeric stats for parity; drop meta/ids entirely
        b = b.drop(columns=[*drop_cols, "gameid", "teamname", "side"], errors="ignore")
        ema_cols = [c for c in b.columns if c.startswith("ema_")]
        b = b[["position", *ema_cols]]

        # Prefix opponent columns (except 'position') and merge by role
        b = b.rename(columns=lambda c: f"opp_{c}" if c != "position" else c)

        return a, b

    def calculate_player_stats(self, team1: Team, team2: Team) -> pd.DataFrame:
        a, b = self.apply_player_stat_modifications(
            team1.player_stats, team2.player_stats
        )

        # Nuke any stale 'side' from parquet & stamp the current orientation
        a = a.drop(columns=["side"], errors="ignore")
        b = b.drop(columns=["side"], errors="ignore")

        gid = team1.team_stats.get("gameid", np.nan)
        sde = (team1.side or str(team1.team_stats.get("side", ""))).strip().title()
        a["gameid"] = gid
        a["side"] = sde

        return a.merge(b, on="position", how="inner", validate="many_to_many")

    # ── preprocessing / model IO ────────────────────────────────────────── #

    def pivot_player_data(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot per-role players into wide columns, matching training naming:
        - own:        <pos>_<field>      (e.g., top_ema_kda)
        - opponent:   opp_<pos>_<field>  (e.g., opp_top_ema_kda)
        Pivot index matches training parity: (gameid, side).
        """
        player_data = player_data.drop(columns=list(TARGET_COLUMNS), errors="ignore")
        numeric = player_data.select_dtypes(include=["number"]).columns
        non_numeric = player_data.columns.difference(numeric)

        agg = dict.fromkeys(numeric, "mean")
        # Don't aggregate 'position' or 'side' as values; they are pivot column / index
        agg.update(
            {col: "first" for col in non_numeric if col not in {"position", "side"}}
        )

        pivot = player_data.pivot_table(
            index=["gameid", "side"],
            columns="position",
            aggfunc=cast("Any", agg),
            fill_value=0,
        )

        # Columns are MultiIndex like (field, position). We want:
        #   - if field starts with 'opp_', name -> 'opp_<pos>_<field[4:]>',
        #   - else name -> '<pos>_<field>'
        new_cols = []
        for field, pos in pivot.columns.to_flat_index():  # noqa: F402
            field = str(field)  # noqa: PLW2901
            pos = str(pos)  # noqa: PLW2901
            if field.startswith("opp_"):
                new_cols.append(f"opp_{pos}_{field[4:]}")
            else:
                new_cols.append(f"{pos}_{field}")
        pivot.columns = new_cols

        return pivot.reset_index()

    def merge_datasets(
        self, team_df: pd.DataFrame, player_pivot: pd.DataFrame
    ) -> pd.DataFrame:
        merged = team_df.merge(
            player_pivot,
            on=["gameid", "side"],
            how="inner",
            validate="many_to_many",
        )
        # Drop meta keys and role-propagated meta
        drop = ["gameid", "side", "teamname"]
        drop += [f"{r}_gameid" for r in _ROLES]
        drop += [f"{r}_side" for r in _ROLES]
        drop += [f"{r}_teamname" for r in _ROLES]
        return merged.drop(columns=drop, errors="ignore")

    def preprocess_data(
        self, team_df: pd.DataFrame, player_df: pd.DataFrame
    ) -> pd.DataFrame:
        team_df = GradientBoostingModel.add_explicit_ema_diffs(
            team_df, drop_opponents=False
        )
        pivot = self.pivot_player_data(player_df)
        pivot = GradientBoostingModel.add_explicit_ema_diffs(
            pivot, drop_opponents=False
        )
        return self.merge_datasets(team_df, pivot)

    def _feature_paths_for(self, model_name: str) -> tuple[Any, Any]:
        mapping = {
            "outcome": (
                OUTCOME_PREDICTION_FINAL_FEATURES,
                OUTCOME_PREDICTION_CATEGORICAL_FEATURES,
            ),
            "gamelength": (
                GAMELENGTH_PREDICTION_FINAL_FEATURES,
                GAMELENGTH_PREDICTION_CATEGORICAL_FEATURES,
            ),
            "total_kills": (
                TOTAL_KILLS_PREDICTION_FINAL_FEATURES,
                TOTAL_KILLS_PREDICTION_CATEGORICAL_FEATURES,
            ),
            "total_towers": (
                TOTAL_TOWERS_PREDICTION_FINAL_FEATURES,
                TOTAL_TOWERS_PREDICTION_CATEGORICAL_FEATURES,
            ),
        }
        if model_name not in mapping:
            raise ValueError(f"Unknown model name: {model_name}")
        return mapping[model_name]

    def _pipeline_for(self, model_name: str) -> FeaturePipeline | None:
        """Get the FeaturePipeline for a given model (if loaded)."""
        mapping = {
            "outcome": self.outcome_pipeline,
            "gamelength": self.gamelength_pipeline,
            "total_kills": self.total_kills_pipeline,
            "total_towers": self.total_towers_pipeline,
        }
        return mapping.get(model_name)

    def keep_necessary_columns(
        self, X: pd.DataFrame, *, model_name: str
    ) -> pd.DataFrame:
        """
        Prepare features for prediction, ensuring training-inference parity.

        Uses FeaturePipeline.transform() when available (preferred) to apply:
        - Feature drops (high missing, low variance, high correlation)
        - Categorical encoding with proper levels
        - Numeric imputation with training medians
        - Column alignment to match training order

        Falls back to legacy behavior (reindex + dtype conversion) when
        FeaturePipeline is not available.
        """
        pipeline = self._pipeline_for(model_name)

        if pipeline is not None:
            # Preferred path: use the full FeaturePipeline for parity
            return pipeline.transform(X)

        # Legacy fallback when pipeline not available
        features_path, cats_path = self._feature_paths_for(model_name)
        try:
            final_features: list[str] = load_model(_serving_path(features_path))
        except Exception as e:
            msg = f"Failed to load final features list ({model_name}): {e}"
            raise RuntimeError(msg) from e

        X = X.reindex(columns=final_features)
        return self.convert_data_types(X, cats_path)

    def convert_data_types(self, X: pd.DataFrame, cats_path) -> pd.DataFrame:
        try:
            cat_features: list[str] = load_model(_serving_path(cats_path))
        except Exception as e:
            msg = f"Failed to load categorical features list: {e}"
            raise RuntimeError(msg) from e

        X = X.copy()
        for col in X.columns:
            if col in cat_features:
                X[col] = X[col].astype("category")
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        return X

    def predict_outcomes(self, X: pd.DataFrame) -> np.ndarray:
        """
        Mirror training-time transforms + predict_proba.
        Returns full-precision shape (n, 2) probabilities. Display rounding
        belongs in Discord formatting, after market math and best-of transforms.
        """
        calibration_metadata = X.copy()
        X = GradientBoostingModel.fuse_opposing_team_features(X)

        # players_* aggregation
        X = GradientBoostingModel.process_players_likelihood_columns(X)
        X = GradientBoostingModel.add_rating_consensus_features(X)
        X = self.keep_necessary_columns(X, model_name="outcome")

        proba = self.outcome_model.predict_proba(X)
        if self.outcome_calibrator is not None:
            team1 = self.outcome_calibrator.predict(
                proba[:, 1], metadata=calibration_metadata
            )
            proba = np.column_stack([1.0 - team1, team1])
        return proba

    def _predict_regression(self, X: pd.DataFrame, *, model_name: str) -> float:
        model = getattr(self, f"{model_name}_model", None)
        if model is None:
            msg = (
                f"{model_name} model not loaded. Train it first to enable predictions."
            )
            raise RuntimeError(msg)

        X = self.keep_necessary_columns(X, model_name=model_name)

        pred = model.predict(X)
        return float(pred[0]) if len(pred) else float("nan")

    def prop_residual_summary(self, model_name: str) -> dict[str, Any]:
        summary = getattr(self, f"{model_name}_residual_summary", None)
        if not summary:
            msg = (
                f"{model_name} residual summary not loaded. Train that prop model "
                "with validation before pricing over/under lines."
            )
            raise RuntimeError(msg)
        return summary

    # ── end-to-end API ──────────────────────────────────────────────────── #

    def calculate_team_and_player_stats(
        self,
        team1: Team,
        team2: Team,
        account_for_side: bool,
        match_type: str | None = None,
    ) -> pd.DataFrame:
        team_stats = self.calculate_team_stats(
            team1, team2, account_for_side, match_type
        )
        player_stats = self.calculate_player_stats(team1, team2)
        return self.preprocess_data(team_stats, player_stats)

    def predict_match(
        self,
        team1: Team,
        team2: Team,
        account_for_side: bool = True,
        match_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Returns:
            {
                "team1_win_probability": P(team1 wins),
                "team2_win_probability": P(team2 wins)
            }

        """
        del account_for_side, match_type
        X_team = pd.concat(
            [
                self._outcome_team_features(team1, team2),
                self._outcome_team_features(team2, team1),
            ],
            ignore_index=True,
        )
        metadata = pd.DataFrame(
            {
                "gameid": ["live", "live"],
                "teamid": [
                    str(team1.team_stats.get("teamid", team1.name)),
                    str(team2.team_stats.get("teamid", team2.name)),
                ],
                "teamname": [team1.name, team2.name],
                "date": [team1.team_stats.get("date"), team2.team_stats.get("date")],
                "league": [
                    team1.team_stats.get("league"),
                    team2.team_stats.get("league"),
                ],
                "season": [
                    team1.team_stats.get("season"),
                    team2.team_stats.get("season"),
                ],
                "patch": [team1.team_stats.get("patch"), team2.team_stats.get("patch")],
            }
        )
        X_matchup, _, matchup_meta = build_game_level_outcome_features(X_team, metadata)
        if self.outcome_pipeline is None:
            msg = "Outcome matchup feature pipeline is missing. Retrain the outcome model."
            raise RuntimeError(msg)
        X_matchup = self.outcome_pipeline.transform(X_matchup)
        proba = self.outcome_model.predict_proba(X_matchup)[:, 1]
        if self.outcome_calibrator is not None:
            proba = self.outcome_calibrator.predict(proba, metadata=matchup_meta)
        canonical_teamid = str(matchup_meta.loc[0, "canonical_teamid"])
        team1_is_canonical = (
            str(team1.team_stats.get("teamid", team1.name)) == canonical_teamid
        )
        canonical_probability = float(proba[0])
        complement = 1.0 - canonical_probability
        canonical_lower = canonical_probability
        canonical_upper = canonical_probability
        outcome_uncertainty = getattr(self, "outcome_uncertainty", None)
        if outcome_uncertainty is not None:
            lower, upper = outcome_uncertainty.interval(proba)
            canonical_lower = float(lower[0])
            canonical_upper = float(upper[0])
        team1_probability, team2_probability = (
            (canonical_probability, complement)
            if team1_is_canonical
            else (complement, canonical_probability)
        )
        team1_lower, team1_upper, team2_lower, team2_upper = (
            (
                canonical_lower,
                canonical_upper,
                1.0 - canonical_upper,
                1.0 - canonical_lower,
            )
            if team1_is_canonical
            else (
                1.0 - canonical_upper,
                1.0 - canonical_lower,
                canonical_lower,
                canonical_upper,
            )
        )
        drivers = self._outcome_prediction_drivers(
            X_matchup,
            team1_name=team1.name,
            team2_name=team2.name,
            team1_is_canonical=team1_is_canonical,
        )
        return {
            "team1_win_probability": team1_probability,
            "team2_win_probability": team2_probability,
            "team1_probability_lower": team1_lower,
            "team1_probability_upper": team1_upper,
            "team2_probability_lower": team2_lower,
            "team2_probability_upper": team2_upper,
            "uncertainty_method": getattr(outcome_uncertainty, "method", None),
            "uncertainty_confidence": getattr(outcome_uncertainty, "confidence", None),
            "uncertainty_sample_count": getattr(
                outcome_uncertainty, "sample_count", None
            ),
            "drivers": drivers,
        }

    def _outcome_prediction_drivers(
        self,
        X_matchup: pd.DataFrame,
        *,
        team1_name: str,
        team2_name: str,
        team1_is_canonical: bool,
    ) -> list[str]:
        """Return local model contributions with an explicit non-causal label."""
        raw_model = getattr(self.outcome_model, "raw_model", self.outcome_model)
        predict = getattr(raw_model, "predict", None)
        if not callable(predict):
            return []
        try:
            raw_contributions = np.asarray(
                predict(X_matchup, pred_contrib=True), dtype=float
            )
        except (TypeError, ValueError, AttributeError):
            return []
        if (
            raw_contributions.ndim != CONTRIBUTION_ARRAY_DIMENSIONS
            or raw_contributions.shape[0] != 1
        ):
            return []

        contributions = raw_contributions[0]
        if len(contributions) == len(X_matchup.columns) + 1:
            contributions = contributions[:-1]
        if len(contributions) != len(X_matchup.columns):
            return []

        drivers: list[str] = []
        seen_labels: set[str] = set()
        order = np.argsort(np.abs(contributions))[::-1]
        for position in order:
            contribution = float(contributions[position])
            if not np.isfinite(contribution) or contribution == 0:
                continue
            label = self._human_feature_label(str(X_matchup.columns[position]))
            if label in seen_labels:
                continue
            seen_labels.add(label)
            favors_canonical = contribution > 0
            favors_team1 = (
                favors_canonical if team1_is_canonical else not favors_canonical
            )
            favored_team = team1_name if favors_team1 else team2_name
            drivers.append(
                f"{label} pushed the model toward {favored_team} "
                "(local model contribution; not causal proof)"
            )
            if len(drivers) == MAX_PREDICTION_DRIVERS:
                break
        return drivers

    @staticmethod
    def _human_feature_label(feature: str) -> str:
        lowered = feature.casefold()
        labels = (
            ("roster_continuity", "Roster continuity"),
            ("days_since", "Recent activity"),
            ("h2h", "Head-to-head history"),
            ("strength_pool", "League strength"),
            ("league_elo", "League strength"),
            ("player", "Player strength"),
            ("rating", "Team rating strength"),
            ("elo", "Team rating strength"),
            ("glicko", "Team rating strength"),
            ("trueskill", "Team rating strength"),
            ("diff_ema", "Recent performance"),
            ("patch", "Patch context"),
        )
        for fragment, label in labels:
            if fragment in lowered:
                return label
        return feature.replace("_", " ").strip().capitalize()

    def _outcome_team_features(self, team1: Team, team2: Team) -> pd.DataFrame:
        """Prepare one team view using the same pre-match transforms as training."""
        X = self.calculate_team_and_player_stats(
            team1, team2, account_for_side=False, match_type=None
        )
        X = GradientBoostingModel.fuse_opposing_team_features(X)
        X = GradientBoostingModel.process_players_likelihood_columns(X)
        X = GradientBoostingModel.add_rating_consensus_features(X)
        return X.drop(columns=GradientBoostingModel._meta_columns(), errors="ignore")

    def calculate_prop_features(
        self, blue_team: Team, red_team: Team, account_for_side: bool
    ) -> pd.DataFrame:
        blue_row = self.calculate_team_and_player_stats(
            blue_team, red_team, account_for_side
        )
        red_row = self.calculate_team_and_player_stats(
            red_team, blue_team, account_for_side
        )
        X_side = pd.concat([blue_row, red_row], ignore_index=True)
        meta_side = pd.DataFrame(
            {
                "gameid": ["live", "live"],
                "side": ["Blue", "Red"],
                "league": [
                    blue_team.team_stats.get("league", np.nan),
                    red_team.team_stats.get("league", np.nan),
                ],
            }
        )
        X_game, _, _ = build_game_level_prop_features(X_side, meta_side)
        return X_game

    def predict_gamelength(
        self, team1: Team, team2: Team, account_for_side: bool = True
    ) -> float:
        X = self.calculate_prop_features(team1, team2, account_for_side)
        return self._predict_regression(X, model_name="gamelength")

    def predict_total_kills(
        self, team1: Team, team2: Team, account_for_side: bool = True
    ) -> float:
        X = self.calculate_prop_features(team1, team2, account_for_side)
        return self._predict_regression(X, model_name="total_kills")

    def predict_total_towers(
        self, team1: Team, team2: Team, account_for_side: bool = True
    ) -> float:
        X = self.calculate_prop_features(team1, team2, account_for_side)
        return self._predict_regression(X, model_name="total_towers")
