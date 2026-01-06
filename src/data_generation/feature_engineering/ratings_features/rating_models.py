"""
Wrapper Class for Rating Models

This module contains a class that serves as a wrapper for various rating models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from data_generation.feature_engineering.ratings_features.elo import calculate_elo
from data_generation.feature_engineering.ratings_features.glicko import (
    calculate_glicko2,
)
from data_generation.feature_engineering.ratings_features.leagues_elo import (
    calculate_leagues_elo,
)
from data_generation.feature_engineering.ratings_features.plackett_luce import (
    calculate_plackett_luce,
)
from data_generation.feature_engineering.ratings_features.trueskill import (
    calculate_trueskill,
)
from utils.pd import pd

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

# from data_generation.feature_engineering.ratings_features.wh import calculate_whr  # noqa: ERA001

Entity = Literal["team", "player"]
ModelName = Literal["elo", "glicko2", "plackett_luce", "trueskill", "leagues_elo"]


@dataclass(slots=True)
class Ratings:
    """
    A wrapper around rating model calculators with convenience methods.

    Notes
    -----
    - Each calculator is assumed to return the *same* row granularity as the input
      (i.e., one row per team/player-side per game) plus model-specific columns.
    - When combining outputs, we merge on robust keys found in both frames.

    """

    # --------------------------
    # Public single-model calls
    # --------------------------
    def compute_elo(self, df: pd.DataFrame, entity: Entity) -> pd.DataFrame:
        return calculate_elo(df, entity)

    def compute_glicko2(self, df: pd.DataFrame, entity: Entity) -> pd.DataFrame:
        return calculate_glicko2(df, entity)

    def compute_plackett_luce(
        self,
        df: pd.DataFrame,
        entity: Entity,
        league_elo_dict: Mapping[str, float] | None = None,
    ) -> pd.DataFrame:
        # calculate_plackett_luce accepts league_elo_dict=None and will load if needed.
        return calculate_plackett_luce(df, entity, league_elo_dict=league_elo_dict)

    def compute_trueskill(
        self,
        df: pd.DataFrame,
        entity: Entity,
        league_elo_dict: Mapping[str, float] | None = None,
    ) -> pd.DataFrame:
        # calculate_trueskill accepts league_elo_dict=None and will load if needed.
        return calculate_trueskill(df, entity, league_elo_dict=league_elo_dict)

    def compute_whr(self, df: pd.DataFrame, entity: Entity) -> pd.DataFrame:
        msg = "Whole History Rating is currently disabled."
        raise NotImplementedError(msg)

    def compute_leagues_elo(self, df: pd.DataFrame) -> pd.DataFrame:
        # leagues-elo is defined for team context; it pivots and merges back to tall.
        return calculate_leagues_elo(df, entity="team")

    # -------------------------------------------
    # Multi-model orchestrator (glues everything)
    # -------------------------------------------
    def compute_bundle(
        self,
        df: pd.DataFrame,
        entity: Entity,
        models: Sequence[ModelName] = ("elo", "glicko2", "plackett_luce", "trueskill"),
        *,
        # If you’ve already computed leagues_elo externally and want to pass its final
        # league->elo mapping to TS/PL initialisation, provide it here. Otherwise
        # those calculators will load from disk if configured to do so.
        league_elo_dict: Mapping[str, float] | None = None,
    ) -> pd.DataFrame:
        """
        Compute multiple rating models and merge the *new* columns each adds,
        without duplicating base columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input match-level frame (one row per side per game).
        entity : {'team','player'}
            Granularity of the rating (team or player).
        models : sequence of model names
            Which models to run and merge.
        league_elo_dict : optional mapping
            Optional league->elo map to seed PL/TrueSkill.

        Returns
        -------
        pd.DataFrame
            The input DataFrame augmented with all selected models' columns.

        """
        if not isinstance(df, pd.DataFrame):
            msg = "df must be a pandas DataFrame"
            raise TypeError(msg)

        base = df  # avoid copy; underlying calculators copy as needed
        base_cols = set(base.columns)

        # Run models in a sensible order. If caller requested leagues_elo explicitly,
        # do it first so its parquet side effects (and potential dict) are ready.
        model_order = self._order_models(models)

        for m in model_order:
            out = self._run_model(m, base, entity, league_elo_dict)
            # Merge only the columns that were added by the model
            new_cols = [c for c in out.columns if c not in base_cols]
            if not new_cols:
                continue

            merge_keys = self._infer_merge_keys(base, out, entity)
            # Left-merge preserves base row order/length
            base = base.merge(
                out[merge_keys + new_cols],
                on=merge_keys,
                how="left",
                # We avoid 'validate' to reduce fragility across leagues/players.
            )
            # Drop temp index-merge key if it was used
            if "_row_ix" in base.columns:
                base = base.drop(columns="_row_ix")

            # Update the known column set so downstream models only add genuinely new cols
            base_cols.update(new_cols)

        return base

    # --------------------------
    # Internal helpers
    # --------------------------
    @staticmethod
    def _order_models(models: Iterable[ModelName]) -> list[ModelName]:
        """
        Order models so dependencies run first (e.g., leagues_elo before PL/TS if present).
        """
        priority = {
            "leagues_elo": 0,
            "elo": 1,
            "glicko2": 2,
            "plackett_luce": 3,
            "trueskill": 4,
        }
        return sorted(models, key=lambda m: priority.get(m, 99))

    @staticmethod
    def _infer_merge_keys(
        left: pd.DataFrame, right: pd.DataFrame, entity: Entity
    ) -> list[str]:
        """
        Infer robust merge keys present in both frames.
        Prefers a strict key set but gracefully falls back if some are missing.
        """
        MIN_DISCRIMINATIVE_KEY_LEN = 3
        candidates_by_priority = [
            [
                "date",
                "gameid",
                "season",
                "side",
                "teamid" if entity == "team" else "playerid",
            ],
            ["date", "gameid", "side", "teamid" if entity == "team" else "playerid"],
            ["date", "gameid", "side"],
            ["date", "gameid"],
        ]
        lcols = set(left.columns)
        rcols = set(right.columns)
        for keys in candidates_by_priority:
            k = [c for c in keys if c in lcols and c in rcols]
            if (
                len(k) >= MIN_DISCRIMINATIVE_KEY_LEN
            ):  # require minimally discriminative key
                return k
        # Last resort: intersect on whatever common keys exist (not ideal, but prevents crashes)
        common = [
            c
            for c in ("date", "gameid", "side", "season", "teamid", "playerid")
            if c in lcols & rcols
        ]
        if common:
            return common
        # If nothing reasonable exists, merge on index (assumes same order/length)
        left["_row_ix"] = range(len(left))
        # IMPORTANT: mutate the *same* right frame (no copy) so the column is available for merge
        right["_row_ix"] = range(len(right))
        return ["_row_ix"]
        # Clean up will be the caller's responsibility; but we keep it internal by returning the key only

    def _run_model(
        self,
        model: ModelName,
        df: pd.DataFrame,
        entity: Entity,
        league_elo_dict: Mapping[str, float] | None,
    ) -> pd.DataFrame:
        if model == "elo":
            return self.compute_elo(df, entity)
        if model == "glicko2":
            return self.compute_glicko2(df, entity)
        if model == "plackett_luce":
            return self.compute_plackett_luce(
                df, entity, league_elo_dict=league_elo_dict
            )
        if model == "trueskill":
            return self.compute_trueskill(df, entity, league_elo_dict=league_elo_dict)
        if model == "leagues_elo":
            # leagues elo ignores `entity` internally, but we keep the signature uniform
            return self.compute_leagues_elo(df)
        msg = f"Unknown model: {model!r}"
        raise ValueError(msg)
