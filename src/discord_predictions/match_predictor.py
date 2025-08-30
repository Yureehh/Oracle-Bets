# discord_predictions/match_predictor.py
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
import pandas as pd

from feature_engineering.ratings_features.glicko import (
    DEFAULT_MU,
    DEFAULT_PHI,
    DEFAULT_SIGMA,
    Glicko2,
    calculate_mean_rating,
)
from feature_engineering.ratings_features.glicko import Rating as GlickoRating
from feature_engineering.ratings_features.plackett_luce import PlackettLuce
from feature_engineering.ratings_features.plackett_luce import (
    predict_win_probability as pl_win_probability,
)
from feature_engineering.ratings_features.trueskill import Rating as TrueskillRating
from feature_engineering.ratings_features.trueskill import (
    expected_win_probability as trueskill_win_probability,
)
from prediction_models.gbdt_model import GradientBoostingModel
from utils.io_utils import load_model
from utils.paths import (
    LEAGUE_ELO,
    OUTCOME_PREDICTION_CATEGORICAL_FEATURES,
    OUTCOME_PREDICTION_FINAL_FEATURES,
    OUTCOME_PREDICTION_MODEL_PATH,
    TEAM_LEAGUES_MAPPING,
    WHOLE_HISTORY_RATING_PATH,
)

if TYPE_CHECKING:
    from discord_predictions.team import Team

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
    if isinstance(team1_mus, Iterable) and not isinstance(team1_mus, (float, int)):
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
    if isinstance(team1_mus, Iterable) and not isinstance(team1_mus, (float, int)):
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
    if isinstance(team1_mus, Iterable) and not isinstance(team1_mus, (float, int)):
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
    whr_model: Any = field(default=None, init=False, repr=False)
    team_to_league: pd.DataFrame = field(default=None, init=False, repr=False)
    league_to_elo: pd.DataFrame = field(default=None, init=False, repr=False)

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

        try:
            self.team_to_league = _read_parquet_cached(str(TEAM_LEAGUES_MAPPING))
            self.league_to_elo = _read_parquet_cached(str(LEAGUE_ELO))
        except Exception as e:
            msg = f"Failed to load mapping/elo parquet: {e}"
            raise RuntimeError(msg) from e

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

    def league_elo_prediction(self, team1_id: int, team2_id: int) -> float:
        """
        League-level Elo logistic probability (Team 1 vs Team 2), using team->league mapping.
        """
        t2l = self.team_to_league
        l2e = self.league_to_elo

        t1_row = t2l.loc[t2l["teamid"] == team1_id]
        t2_row = t2l.loc[t2l["teamid"] == team2_id]
        if t1_row.empty or t2_row.empty:
            msg = "Team ID not found in team-to-league mapping."
            raise ValueError(msg)

        t1_league = t1_row["league"].iloc[0]
        t2_league = t2_row["league"].iloc[0]

        e1 = l2e.loc[l2e["league"] == t1_league, "elo"]
        e2 = l2e.loc[l2e["league"] == t2_league, "elo"]
        if e1.empty or e2.empty:
            msg = "League not found in ELO ratings."
            raise ValueError(msg)

        prob = _elo_prob(float(e1.iloc[0]), float(e2.iloc[0]))
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
    ) -> tuple[pd.Series, pd.Series]:
        """
        Build derived likelihood features for team rows.
        Returns copies (originals untouched).
        """
        t1 = team1_stats.copy()
        t2 = team2_stats.copy()
        # Rating-based likelihoods
        t1["elo_win_likelihood"] = self.elo_prediction(t1["elo"], t2["elo"])
        t1["glicko2_win_likelihood"] = self.glicko2_prediction(
            t1["glicko2_mu"], t1["glicko2_phi"], t2["glicko2_mu"], t2["glicko2_phi"]
        )
        t1["pl_win_likelihood"] = self.pl_prediction(
            t1["pl_mu"], t1["pl_sigma"], t2["pl_mu"], t2["pl_sigma"]
        )
        t1["trueskill_win_likelihood"] = self.trueskill_prediction(
            t1["trueskill_mu"],
            t1["trueskill_sigma"],
            t2["trueskill_mu"],
            t2["trueskill_sigma"],
        )
        t1["league_elo_win_likelihood"] = self.league_elo_prediction(
            t1["teamid"], t2["teamid"]
        )
        # Side / patch / season likelihoods
        if account_for_side:
            s1_key = f"ema_{str(t1['side']).casefold()}_side"
            s2_key = f"ema_{str(t2['side']).casefold()}_side"
            t1_side_wr = float(t1.get(s1_key, 0.0))
            t2_side_wr = float(t2.get(s2_key, 0.0))
            t1["side_win_likelihood"] = self.side_wr_prediction(t1_side_wr, t2_side_wr)
        else:
            t1["side_win_likelihood"] = 0.5

        t1["patch_win_likelihood"] = self.patch_season_wr_prediction(
            t1["ema_patch_win_rate"], t2["ema_patch_win_rate"]
        )
        t1["season_win_likelihood"] = self.patch_season_wr_prediction(
            t1["ema_season_win_rate"], t2["ema_season_win_rate"]
        )
        # Slim columns down (keep only usable features)
        base_drop = [
            "teamid",
            "date",
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
        ]
        t1 = self._drop_columns(t1, base_drop)
        t2 = self._drop_columns(t2, [*base_drop, "teamname", "gameid"])
        return t1, t2

    def calculate_team_stats(
        self, team1: Team, team2: Team, account_for_side: bool
    ) -> pd.DataFrame:
        t1, t2 = self.apply_stat_modifications(
            team1.team_stats, team2.team_stats, account_for_side
        )
        t2 = t2.add_prefix("opp_")
        return pd.concat([t1.to_frame().T, t2.to_frame().T], axis=1)

    # ── feature assembly (players) ──────────────────────────────────────── #

    def apply_player_stat_modifications(
        self, p1: pd.DataFrame, p2: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Row-wise (per role) likelihoods for players; returns copies.
        """
        a = p1.copy()
        b = p2.copy()

        # Elo is vectorizable
        a["elo_win_likelihood"] = self.elo_prediction(a["elo"], b["elo"])

        # Glicko-2 / PL / TrueSkill computed row-wise (each role vs role)
        a["glicko2_win_likelihood"] = [
            self.glicko2_prediction(mu1, phi1, mu2, phi2)
            for mu1, phi1, mu2, phi2 in zip(
                a["glicko2_mu"],
                a["glicko2_phi"],
                b["glicko2_mu"],
                b["glicko2_phi"],
                strict=False,
            )
        ]
        a["pl_win_likelihood"] = [
            self.pl_prediction(mu1, s1, mu2, s2)
            for mu1, s1, mu2, s2 in zip(
                a["pl_mu"], a["pl_sigma"], b["pl_mu"], b["pl_sigma"], strict=False
            )
        ]
        a["trueskill_win_likelihood"] = [
            self.trueskill_prediction(mu1, s1, mu2, s2)
            for mu1, s1, mu2, s2 in zip(
                a["trueskill_mu"],
                a["trueskill_sigma"],
                b["trueskill_mu"],
                b["trueskill_sigma"],
                strict=False,
            )
        ]

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
        b = b.drop(columns=[*drop_cols, "gameid", "teamname"], errors="ignore")
        return a, b

    def calculate_player_stats(self, team1: Team, team2: Team) -> pd.DataFrame:
        a, b = self.apply_player_stat_modifications(
            team1.player_stats, team2.player_stats
        )
        b = b.rename(columns=lambda c: f"opp_{c}" if c != "position" else c)
        # safe merge on 'position' (roles)
        return a.merge(b, on="position", how="inner", validate="many_to_many")

    # ── preprocessing / model IO ────────────────────────────────────────── #

    @staticmethod
    def pivot_player_data(player_data: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot per-role players into wide columns like 'top_kda', 'jng_dpm', ...
        """
        numeric = player_data.select_dtypes(include=["number"]).columns
        non_numeric = player_data.columns.difference(numeric)

        agg = dict.fromkeys(numeric, "mean")
        agg.update(
            {col: "first" for col in non_numeric if col not in {"position", "side"}}
        )

        pivot = player_data.pivot_table(
            index=["gameid", "teamname"], columns="position", aggfunc=agg, fill_value=0
        )

        # ⬇️ IMPORTANT: MultiIndex is (field, position), so unpack as (field, pos)
        pivot.columns = [f"{pos}_{field}" for field, pos in pivot.columns]
        return pivot.reset_index()

    @staticmethod
    def merge_datasets(
        team_df: pd.DataFrame, player_pivot: pd.DataFrame
    ) -> pd.DataFrame:
        merged = team_df.merge(
            player_pivot,
            on=["gameid", "teamname"],
            how="inner",
            validate="many_to_many",
        )
        drop = (
            ["gameid", "teamname"]
            + [f"{r}_gameid" for r in _ROLES]
            + [f"{r}_teamname" for r in _ROLES]
        )
        return merged.drop(columns=drop, errors="ignore")

    def preprocess_data(
        self, team_df: pd.DataFrame, player_df: pd.DataFrame
    ) -> pd.DataFrame:
        pivot = self.pivot_player_data(player_df)
        return self.merge_datasets(team_df, pivot)

    def keep_necessary_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            final_features: list[str] = load_model(OUTCOME_PREDICTION_FINAL_FEATURES)
        except Exception as e:
            msg = f"Failed to load final features list: {e}"
            raise RuntimeError(msg) from e

        X = X.reindex(columns=final_features)
        return self.convert_data_types(X)

    def convert_data_types(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            cat_features: list[str] = load_model(
                OUTCOME_PREDICTION_CATEGORICAL_FEATURES
            )
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
        # players_* aggregation and opposing-feature fusion (same as training)
        X = GradientBoostingModel.process_players_likelihood_columns(X)
        X = GradientBoostingModel.fuse_opposing_team_features(X)
        X = self.keep_necessary_columns(X)

        proba = self.outcome_model.predict_proba(X)
        return np.round(proba, PREDICTION_PRECISION)

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
