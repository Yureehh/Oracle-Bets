"""
Features Generator

This script contains the `FeatureGenerator` class, which is used to generate
new features for player and team data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from utils.logger import instantiate_logger, logger

if TYPE_CHECKING:
    from collections.abc import Iterable

data_pipeline_logger = instantiate_logger("data_pipeline")

# ────────────────────────────────────────────────────────────────────────────
# Helpers/constants shared by multiple methods
# ────────────────────────────────────────────────────────────────────────────
_OPPOSITE_SIDE = {"Blue": "Red", "Red": "Blue"}  # quick side-flip

# ---------------------------------------------------------------------------


@dataclass
class FeatureGenerator:
    """Generate new features for player and team data."""

    # ── 1. Key in-game opponent context ───────────────────────────────────

    @staticmethod
    def _aggregate_enemy_team_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Return one row per (gameid, **opposite** side) with enemy totals."""
        enemy = (
            df.groupby(["gameid", "side"], observed=True)
            .agg(
                enemyTeamKills=("kills", "sum"),
                enemyTeamDeaths=("deaths", "sum"),
                enemyTeamDamages=("damagetochampions", "sum"),
                enemyTeamGolds=("totalgold", "sum"),
                enemyTeamTotalCS=("total_cs", "sum"),
                enemyTeamWardPlaced=("wpm", "sum"),
                enemyTeamWardKilled=("wcpm", "sum"),
            )
            .reset_index()
        )
        enemy["side"] = enemy["side"].map(_OPPOSITE_SIDE)  # flip to opponent
        return enemy  # no duplicate rows

    @staticmethod
    def compute_key_stats(data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute key statistics for each player in a game.
        This includes aggregating enemy team stats and calculating ratios
        such as kill/assist ratios, damage ratios, and gold ratios.
        """
        logger.info("Computing key statistics...")
        _check_required(
            data,
            required={
                "gameid",
                "side",
                "kills",
                "assists",
                "deaths",
                "damagetochampions",
                "damagetakenperminute",
                "damagemitigatedperminute",
                "gamelength",
                "totalgold",
                "total_cs",
                "wpm",
                "wcpm",
            },
        )

        df = data.copy()
        enemy = FeatureGenerator._aggregate_enemy_team_stats(df)
        df = df.merge(enemy, on=["gameid", "side"], how="left")
        df = FeatureGenerator._calculate_ratios(df)

        data_pipeline_logger.info("Key statistics computation completed.")
        return df

    @staticmethod
    def _calculate_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """Add efficiency / share ratios with safe divide-by-zero handling."""
        denom_cols = [
            "enemyTeamKills",
            "enemyTeamDeaths",
            "enemyTeamDamages",
            "enemyTeamGolds",
            "enemyTeamWardPlaced",
        ]
        df[denom_cols] = df[denom_cols].replace(0, np.nan)  # avoid /0

        df["ka_ratio"] = np.divide(
            df["kills"] + df["assists"],
            df["enemyTeamKills"] + df["kills"] + df["assists"],
        )
        df["d_ratio"] = np.divide(df["deaths"], df["enemyTeamDeaths"])
        df["damages_ratio"] = np.divide(df["damagetochampions"], df["enemyTeamDamages"])
        df["damage_tanked_ratio"] = np.divide(
            df["damagetakenperminute"] * df["gamelength"], df["enemyTeamDamages"]
        )
        df["damage_mitigated_ratio"] = np.divide(
            df["damagemitigatedperminute"] * df["gamelength"], df["enemyTeamDamages"]
        )
        df["gold_ratio"] = np.divide(df["totalgold"], df["enemyTeamGolds"])
        df["cs_to_gold_ratio"] = np.divide(df["total_cs"], df["enemyTeamGolds"])
        df["wards_placed_ratio"] = np.divide(
            df["wpm"] * df["gamelength"], df["enemyTeamWardPlaced"]
        )
        df["wards_killed_ratio"] = np.divide(
            df["wcpm"] * df["gamelength"], df["enemyTeamWardPlaced"]
        )
        return df

        # ── 2. Win / loss historical means – kills & deaths only, no leakage ──

    @staticmethod
    def compute_win_loss_metrics(data: pd.DataFrame) -> pd.DataFrame:
        """
        For each row, add the player's expanding-mean **kills** and **deaths**
        in wins and losses, scoped by season *and* patch, using only games
        strictly prior to the current one.

        Added columns
        -------------
        kills_prev_avg_season_win / loss
        deaths_prev_avg_season_win / loss
        kills_prev_avg_patch_win  / loss
        deaths_prev_avg_patch_win / loss
        """
        logger.info("Computing win/loss metrics (kills & deaths)…")

        req = {"playerid", "season", "patch", "result", "kills", "deaths", "date"}
        _check_required(data, required=req)

        df = data.copy().sort_values(["playerid", "date"], kind="mergesort")

        def _add_expanding(by: list[str], scope: str) -> None:
            for metric in ("kills", "deaths"):
                grp = df.groupby([*by, "result"], observed=True)[metric]
                shifted_mean = (
                    grp.cumsum().shift().div(grp.cumcount().replace(0, np.nan))
                )

                win_col = f"{metric}_prev_avg_{scope}_win"
                loss_col = f"{metric}_prev_avg_{scope}_loss"

                df[win_col] = np.where(df["result"] == 1, shifted_mean, df.get(win_col))
                df[loss_col] = np.where(
                    df["result"] == 0, shifted_mean, df.get(loss_col)
                )

        _add_expanding(["playerid", "season"], "season")
        _add_expanding(["playerid", "patch"], "patch")

        # Propagate NaNs so every row has both win & loss histories where possible
        df = df.fillna(method="ffill").fillna(method="bfill")

        return df

    # ── 3. Public player-feature pipeline ────────────────────────────────
    @staticmethod
    def generate_new_player_features(data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Generating new player features...")
        _check_required(
            data,
            required={
                "teamid",
                "gameid",
                "position",
                "kills",
                "assists",
                "deaths",
                "totalgold",
                "gamelength",
                "total_cs",
                "patch",
                "result",
                "playerid",
            },
        )

        df = data.copy()
        df["season"] = df["patch"].astype(str).str.split(".").str[0]

        # Base per-game stats
        df["team_kills"] = df.groupby(["gameid", "teamid"], observed=True)[
            "kills"
        ].transform("sum")
        df["kda"] = np.divide(
            df["kills"] + df["assists"], df["deaths"].replace(0, np.nan)
        )
        df["gold_efficiency"] = np.divide(
            df["totalgold"], df["gamelength"].replace(0, np.nan)
        )
        df["xp_efficiency"] = np.divide(
            df["total_cs"], df["gamelength"].replace(0, np.nan)
        )
        df["kill_participation"] = np.divide(
            df["kills"] + df["assists"], df["team_kills"].replace(0, np.nan)
        )

        # Opponent context + historical means
        df = FeatureGenerator.compute_key_stats(df)
        df = FeatureGenerator.compute_win_loss_metrics(df)

        # One-hot position
        pos_dummies = pd.get_dummies(df["position"], prefix="position", dtype=np.uint8)
        df = pd.concat(
            [df.reset_index(drop=True), pos_dummies.reset_index(drop=True)], axis=1
        )

        data_pipeline_logger.info("Player features generation completed.")
        return df

    @staticmethod
    def generate_new_team_features(
        data: pd.DataFrame,
        *,
        recent_window: int = 5,
        add_recent: bool = True,
    ) -> pd.DataFrame:
        """
        Enrich each team-game row with leak-free historical context **without**
        the long-horizon team mean that would overweight old matches.

        Added columns
        -------------
        season
        total_kills, total_towers
        team_season_avg_gamelength
        team_patch_avg_gamelength
        (optional) team_recent{N}_avg_gamelength
        patch_avg_gamelength
        season_avg_gamelength
        """
        _check_required(
            data,
            required={
                "date",
                "patch",
                "teamid",
                "gameid",
                "gamelength",
                "kills",
                "towers",
            },
        )

        df = data.copy()
        df["season"] = df["patch"].astype(str).str.split(".").str[0]

        # ── Game-level context ───────────────────────────────────────────────
        df = df.merge(
            df.groupby("gameid", observed=True)
            .agg(total_kills=("kills", "sum"), total_towers=("towers", "sum"))
            .reset_index(),
            on="gameid",
            how="left",
        )

        # ── Team cumulative mean, reset each *season* and *patch* ────────────
        for grp, pfx in (
            (["teamid", "season"], "team_season_avg_"),
            (["teamid", "patch"], "team_patch_avg_"),
        ):
            df = _add_expanding_mean(
                df,
                group_cols=list(grp),
                value_cols=["gamelength"],
                prefix=pfx,
            )

        # ── Optional: recent-N rolling mean ───────────────────────────────────
        if add_recent and recent_window > 0:
            df = _add_rolling_mean(
                df,
                group_cols=["teamid"],
                value_cols=["gamelength"],
                prefix=f"team_recent{recent_window}_avg_",
                window=recent_window,
            )

        # ── League-wide patch / season expanding means (deduped) ─────────────
        game_level = (
            df[["gameid", "date", "patch", "season", "gamelength"]]
            .sort_values("date")
            .drop_duplicates(subset="gameid", keep="first")
        )

        for gcol, pfx in (("patch", "patch_avg_"), ("season", "season_avg_")):
            game_level = _add_expanding_mean(
                game_level,
                group_cols=[gcol],
                value_cols=["gamelength"],
                prefix=pfx,
                sort_also_by=["date"],
            )

        return df.merge(
            game_level[["gameid", "patch_avg_gamelength", "season_avg_gamelength"]],
            on="gameid",
            how="left",
        )


# ────────────────────────────────────────────────────────────────────────────
# Internal utilities
# ────────────────────────────────────────────────────────────────────────────
def _check_required(df: pd.DataFrame, *, required: Iterable[str]) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        msg = f"Missing required columns: {', '.join(sorted(missing))}"
        logger.error(msg)
        raise ValueError(msg)


def _add_expanding_mean(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    value_cols: list[str],
    prefix: str,
    sort_also_by: list[str] | None = None,
) -> pd.DataFrame:
    sort_keys = list(group_cols) + (sort_also_by or [])
    df = df.sort_values(sort_keys, kind="mergesort")

    for col in value_cols:
        grp = df.groupby(group_cols, observed=True)[col]
        df[f"{prefix}{col}"] = (
            grp.cumsum().shift().div(grp.cumcount().replace(0, np.nan))
        )

    return df


def _add_rolling_mean(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    value_cols: list[str],
    prefix: str,
    window: int,
) -> pd.DataFrame:
    """Rolling mean of the *previous* `window` rows inside each group."""
    df = df.sort_values([*group_cols, "date"], kind="mergesort")

    for col in value_cols:
        rolled = (
            df.groupby(group_cols, observed=True)[col]
            .shift()  # → leak-free
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=group_cols, drop=True)
        )
        df[f"{prefix}{col}"] = rolled

    return df
