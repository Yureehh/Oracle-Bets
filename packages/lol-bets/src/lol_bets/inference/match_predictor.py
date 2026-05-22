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
from typing import TYPE_CHECKING, Any

import numpy as np
from oracle_bets_core.io_utils import load_model
from oracle_bets_core.league_taxonomy import get_league_taxonomy
from oracle_bets_core.paths import (
    ACTIVE_MODEL_TYPE,
    GAMELENGTH_PREDICTION_CATEGORICAL_ENCODINGS,
    GAMELENGTH_PREDICTION_CATEGORICAL_FEATURES,
    GAMELENGTH_PREDICTION_FEATURE_PIPELINE,
    GAMELENGTH_PREDICTION_FINAL_FEATURES,
    GAMELENGTH_PREDICTION_MODEL_PATH,
    LEAGUE_ELO,
    OUTCOME_PREDICTION_CATEGORICAL_ENCODINGS,
    OUTCOME_PREDICTION_CATEGORICAL_FEATURES,
    OUTCOME_PREDICTION_FEATURE_PIPELINE,
    OUTCOME_PREDICTION_FINAL_FEATURES,
    OUTCOME_PREDICTION_MODEL_PATH,
    TEAM_LEAGUES_MAPPING,
    TOTAL_KILLS_PREDICTION_CATEGORICAL_ENCODINGS,
    TOTAL_KILLS_PREDICTION_CATEGORICAL_FEATURES,
    TOTAL_KILLS_PREDICTION_FEATURE_PIPELINE,
    TOTAL_KILLS_PREDICTION_FINAL_FEATURES,
    TOTAL_KILLS_PREDICTION_MODEL_PATH,
    TOTAL_TOWERS_PREDICTION_CATEGORICAL_ENCODINGS,
    TOTAL_TOWERS_PREDICTION_CATEGORICAL_FEATURES,
    TOTAL_TOWERS_PREDICTION_FEATURE_PIPELINE,
    TOTAL_TOWERS_PREDICTION_FINAL_FEATURES,
    TOTAL_TOWERS_PREDICTION_MODEL_PATH,
    WHOLE_HISTORY_RATING_PATH,
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
from lol_bets.prediction_models.gbdt_model import FeaturePipeline, GradientBoostingModel

if TYPE_CHECKING:
    from lol_bets.inference.team import Team

# ── constants ────────────────────────────────────────────────────────────── #

PREDICTION_PRECISION = 3  # for final np.round
ELO_FACTOR = 400.0
RATING_DECIMALS = 3  # for human-facing rounded rating-based probs
_PROB_EPS = 1e-12  # small epsilon for clipping
_ROLES = ("top", "jng", "mid", "bot", "sup")


# ── cached parquet reads ─────────────────────────────────────────────────── #


@lru_cache(maxsize=4)
def _read_parquet_cached(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="fastparquet")
    except (ImportError, ValueError):
        # fallback to pyarrow if available
        return pd.read_parquet(path)


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
      - optional Whole-History Rating (WHR) for raw ID-based head-to-head
    """

    outcome_model: Any = field(default=None, init=False, repr=False)
    gamelength_model: Any = field(default=None, init=False, repr=False)
    total_kills_model: Any = field(default=None, init=False, repr=False)
    total_towers_model: Any = field(default=None, init=False, repr=False)
    whr_model: Any = field(default=None, init=False, repr=False)
    team_to_league: pd.DataFrame = field(default=None, init=False, repr=False)
    league_to_elo: pd.DataFrame = field(default=None, init=False, repr=False)
    # Feature pipelines for inference parity with training
    outcome_pipeline: FeaturePipeline | None = field(
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

    def _load_artifacts(self) -> None:
        # sourcery skip: remove-redundant-exception, simplify-single-exception-tuple
        try:
            self.outcome_model = load_model(OUTCOME_PREDICTION_MODEL_PATH)
        except Exception as e:
            msg = f"Failed to load outcome model: {e}"
            raise RuntimeError(msg) from e

        try:
            self.whr_model = load_model(WHOLE_HISTORY_RATING_PATH)
        except (Exception, ImportError):
            # WHR is optional; only used if whr_prediction() is called explicitly
            self.whr_model = None

        for model_name, path in (
            ("gamelength", GAMELENGTH_PREDICTION_MODEL_PATH),
            ("total_kills", TOTAL_KILLS_PREDICTION_MODEL_PATH),
            ("total_towers", TOTAL_TOWERS_PREDICTION_MODEL_PATH),
        ):
            try:
                setattr(self, f"{model_name}_model", load_model(path))
            except Exception:
                setattr(self, f"{model_name}_model", None)

        try:
            self.team_to_league = _read_parquet_cached(str(TEAM_LEAGUES_MAPPING))
            self.league_to_elo = _read_parquet_cached(str(LEAGUE_ELO))
        except Exception as e:
            msg = f"Failed to load mapping/elo parquet: {e}"
            raise RuntimeError(msg) from e

        # Load feature pipelines for training-inference parity
        try:
            self.outcome_pipeline = load_model(OUTCOME_PREDICTION_FEATURE_PIPELINE)
        except Exception:
            self.outcome_pipeline = None

        for pipeline_name, path in (
            ("gamelength_pipeline", GAMELENGTH_PREDICTION_FEATURE_PIPELINE),
            ("total_kills_pipeline", TOTAL_KILLS_PREDICTION_FEATURE_PIPELINE),
            ("total_towers_pipeline", TOTAL_TOWERS_PREDICTION_FEATURE_PIPELINE),
        ):
            try:
                setattr(self, pipeline_name, load_model(path))
            except Exception:
                setattr(self, pipeline_name, None)

    # ── rating-based predictions (scalar) ───────────────────────────────── #

    @staticmethod
    def elo_prediction(
        team1_elo: float | Iterable[float], team2_elo: float | Iterable[float]
    ) -> Any:
        prob = _elo_prob(team1_elo, team2_elo)
        # keep vectorization for Series; round only scalars
        return (
            np.round(prob, RATING_DECIMALS)
            if np.ndim(prob)
            else round(float(prob), RATING_DECIMALS)
        )

    @staticmethod
    def glicko2_prediction(team1_mus, team1_phis, team2_mus, team2_phis) -> float:
        return round(
            _glicko2_prob(team1_mus, team1_phis, team2_mus, team2_phis), RATING_DECIMALS
        )

    @staticmethod
    def pl_prediction(team1_mus, team1_sigmas, team2_mus, team2_sigmas) -> float:
        return round(
            _pl_prob(team1_mus, team1_sigmas, team2_mus, team2_sigmas), RATING_DECIMALS
        )

    @staticmethod
    def trueskill_prediction(team1_mus, team1_sigmas, team2_mus, team2_sigmas) -> float:
        return round(
            _ts_prob(team1_mus, team1_sigmas, team2_mus, team2_sigmas), RATING_DECIMALS
        )

    def whr_prediction(self, blue_team_id: int, red_team_id: int) -> float:
        """
        WHR-based win probability for (blue_team_id vs red_team_id).
        Requires a loaded WHR model with .probability_future_match(b, r) -> [p_blue, p_red]
        """
        if self.whr_model is None:
            msg = "WHR model not loaded; cannot compute WHR prediction."
            raise RuntimeError(msg)
        try:
            p = self.whr_model.probability_future_match(blue_team_id, red_team_id)
            return round(float(_clip01(p[0])), RATING_DECIMALS)
        except Exception as e:
            msg = f"Error during WHR prediction: {e}"
            raise RuntimeError(msg) from e

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

    # ── simple WR-based heuristics ──────────────────────────────────────── #

    @staticmethod
    def side_wr_prediction(team1_side_wr: float, team2_side_wr: float) -> float:
        # sourcery skip: remove-unnecessary-cast
        total = float(team1_side_wr) + float(team2_side_wr)
        if total <= 0.0:
            return 0.5
        return round(float(team1_side_wr) / total, RATING_DECIMALS)

    @staticmethod
    def patch_season_wr_prediction(team1_wr: float, team2_wr: float) -> float:
        total = team1_wr + team2_wr
        return 0.5 if total <= 0.0 else round(team1_wr / total, RATING_DECIMALS)

    # ── feature assembly (teams) ────────────────────────────────────────── #

    @staticmethod
    def _drop_columns(s: pd.Series, cols: list[str]) -> pd.Series:
        return s.drop(labels=cols, errors="ignore")

    def apply_stat_modifications(
        self,
        team1_stats: pd.Series,
        team2_stats: pd.Series,
        account_for_side: bool,
    ) -> pd.Series:
        """
        Build derived likelihood features for team rows.
        Returns copy of team1_stats with derived features added.

        Only computes features that are in the training config:
        - league_elo_win_likelihood (inter-league calibration)
        - side_win_likelihood (side-specific)
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

        # League Elo win likelihood (inter-league calibration)
        t1["league_elo_win_likelihood"] = self.league_elo_prediction(
            t1["teamid"], t2["teamid"]
        )

        # Side win likelihood
        if account_for_side:
            s1_key = f"ema_{str(t1.get('side', '')).casefold()}_side"
            s2_key = f"ema_{str(t2.get('side', '')).casefold()}_side"
            t1_side_wr = float(t1.get(s1_key, 0.0))
            t2_side_wr = float(t2.get(s2_key, 0.0))
            t1["side_win_likelihood"] = self.side_wr_prediction(t1_side_wr, t2_side_wr)
        else:
            t1["side_win_likelihood"] = 0.5

        # Season win likelihood
        t1["season_win_likelihood"] = self.patch_season_wr_prediction(
            float(t1.get("ema_season_win_rate", 0.0)),
            float(t2.get("ema_season_win_rate", 0.0)),
        )

        # Drop raw ratings/ids from features (keep meta like gameid/teamname/side for merges)
        base_drop = [
            "league_elo",
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

    def calculate_team_stats(
        self, team1: Team, team2: Team, account_for_side: bool
    ) -> pd.DataFrame:
        # Ensure 'side' is present (needed for side-based WR lookups + later merge)
        t1_stats = team1.team_stats.copy()
        t2_stats = team2.team_stats.copy()
        if "side" not in t1_stats.index:
            t1_stats["side"] = team1.side or ""
        if "side" not in t2_stats.index:
            t2_stats["side"] = team2.side or ""

        # 1) Team1 vs Team2 -> own features
        t1_fwd = self.apply_stat_modifications(t1_stats, t2_stats, account_for_side)

        # 2) Team2 vs Team1 -> source for opponent features
        t2_mirr = self.apply_stat_modifications(t2_stats, t1_stats, account_for_side)

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

        Note: Player-level rating likelihoods (elo_win_likelihood, etc.) were
        removed as they are redundant (deterministic transforms of base ratings).
        The model now uses team-level features without player-level likelihoods.
        """
        a = p1.copy()
        b = p2.copy()

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
            aggfunc=agg,
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
        pivot = self.pivot_player_data(player_df)
        # Print team_df gameid and side for debugging
        # DO the same for pivot
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

    def _is_tabnet(self) -> bool:
        """Check if using TabNet model type."""
        return ACTIVE_MODEL_TYPE == "TabNet"

    def _encodings_path_for(self, model_name: str):
        """Get the categorical encodings path for a given model."""
        mapping = {
            "outcome": OUTCOME_PREDICTION_CATEGORICAL_ENCODINGS,
            "gamelength": GAMELENGTH_PREDICTION_CATEGORICAL_ENCODINGS,
            "total_kills": TOTAL_KILLS_PREDICTION_CATEGORICAL_ENCODINGS,
            "total_towers": TOTAL_TOWERS_PREDICTION_CATEGORICAL_ENCODINGS,
        }
        return mapping.get(model_name)

    def _load_categorical_encodings(self, model_name: str) -> dict:
        """Load categorical encodings for TabNet inference."""
        path = self._encodings_path_for(model_name)
        if path is None:
            return {}
        try:
            return load_model(path)
        except Exception:
            return {}

    def _prepare_for_tabnet(self, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """Convert DataFrame to float32 numpy array for TabNet inference."""
        encodings = self._load_categorical_encodings(model_name)
        X = X.copy()

        for col in X.columns:
            if col in encodings:
                # Use stored encoding from training
                X[col] = X[col].map(encodings[col]).fillna(-1)
            elif X[col].dtype == "category" or X[col].dtype == "object":
                # Fallback: encode to integers using pandas category codes
                X[col] = X[col].astype("category").cat.codes

        arr = X.values.astype(np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

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
            final_features: list[str] = load_model(features_path)
        except Exception as e:
            msg = f"Failed to load final features list ({model_name}): {e}"
            raise RuntimeError(msg) from e

        X = X.reindex(columns=final_features)
        return self.convert_data_types(X, cats_path)

    def convert_data_types(self, X: pd.DataFrame, cats_path) -> pd.DataFrame:
        try:
            cat_features: list[str] = load_model(cats_path)
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
        Returns shape (n, 2) array.
        """
        # If we already built team-level opp_* columns, don't re-fuse them.
        if not any(c.startswith("opp_") for c in X.columns):
            X = GradientBoostingModel.fuse_opposing_team_features(X)

        # players_* aggregation
        X = GradientBoostingModel.process_players_likelihood_columns(X)
        X = self.keep_necessary_columns(X, model_name="outcome")

        # TabNet needs numpy float32 with encoded categoricals
        if self._is_tabnet():
            X = self._prepare_for_tabnet(X, model_name="outcome")

        proba = self.outcome_model.predict_proba(X)
        return np.round(proba, PREDICTION_PRECISION)

    def _predict_regression(self, X: pd.DataFrame, *, model_name: str) -> float:
        model = getattr(self, f"{model_name}_model", None)
        if model is None:
            msg = (
                f"{model_name} model not loaded. Train it first to enable predictions."
            )
            raise RuntimeError(msg)

        if not any(c.startswith("opp_") for c in X.columns):
            X = GradientBoostingModel.fuse_opposing_team_features(X)
        X = GradientBoostingModel.process_players_likelihood_columns(X)
        X = self.keep_necessary_columns(X, model_name=model_name)

        # TabNet needs numpy float32 with encoded categoricals
        if self._is_tabnet():
            X = self._prepare_for_tabnet(X, model_name=model_name)

        pred = model.predict(X)
        return float(pred[0]) if len(pred) else float("nan")

    # ── end-to-end API ──────────────────────────────────────────────────── #

    def calculate_team_and_player_stats(
        self, team1: Team, team2: Team, account_for_side: bool
    ) -> pd.DataFrame:
        team_stats = self.calculate_team_stats(team1, team2, account_for_side)
        player_stats = self.calculate_player_stats(team1, team2)
        return self.preprocess_data(team_stats, player_stats)

    def predict_match(
        self, team1: Team, team2: Team, account_for_side: bool = True
    ) -> dict[str, float]:
        """
        Returns:
            {
                "team1_win_probability": P(team1 wins),
                "team2_win_probability": P(team2 wins)
            }

        """
        X = self.calculate_team_and_player_stats(team1, team2, account_for_side)
        proba = self.predict_outcomes(X)
        # class 1 assumed "win" for team1 (same as training)
        return {
            "team1_win_probability": float(proba[0, 1]),
            "team2_win_probability": float(proba[0, 0]),
        }

    def predict_gamelength(
        self, team1: Team, team2: Team, account_for_side: bool = True
    ) -> float:
        X = self.calculate_team_and_player_stats(team1, team2, account_for_side)
        return self._predict_regression(X, model_name="gamelength")

    def predict_total_kills(
        self, team1: Team, team2: Team, account_for_side: bool = True
    ) -> float:
        X = self.calculate_team_and_player_stats(team1, team2, account_for_side)
        return self._predict_regression(X, model_name="total_kills")

    def predict_total_towers(
        self, team1: Team, team2: Team, account_for_side: bool = True
    ) -> float:
        X = self.calculate_team_and_player_stats(team1, team2, account_for_side)
        return self._predict_regression(X, model_name="total_towers")
