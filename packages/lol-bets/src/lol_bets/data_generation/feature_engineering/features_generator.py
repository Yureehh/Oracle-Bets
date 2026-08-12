"""
Features Generator

This script contains the `FeatureGenerator` class, which is used to generate
new features for player and team data.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
from oracle_bets_core.io_utils import get_sorting_keys
from oracle_bets_core.league_taxonomy import add_league_taxonomy_columns
from oracle_bets_core.logger import LOG_TOPIC, instantiate_logger, logger
from oracle_bets_core.pd import pd

from lol_bets.data_generation.feature_engineering.performance_features.opponent import (
    add_opponent_columns,
)
from lol_bets.data_generation.ingestion.quality import normalize_result

if TYPE_CHECKING:
    from collections.abc import Iterable

data_pipeline_logger = instantiate_logger(LOG_TOPIC.DATA_PIPELINE)

# ────────────────────────────────────────────────────────────────────────────
# Helpers/constants shared by multiple methods
# ────────────────────────────────────────────────────────────────────────────
_OPPOSITE_SIDE = {"Blue": "Red", "Red": "Blue"}  # quick side-flip
GAMES_IN_BO3 = 3
GAMES_IN_BO5 = 5
BREAK_THRESHOLD_DAYS = 45  # ~1.5 months, indicates split break
H2H_MIN_GAMES = 2  # minimum games to compute head-to-head
EARLY_GAME_MARKERS = (10, 15, 20, 25)
_EXPECTED_ROSTER = ("top", "jng", "mid", "bot", "sup")
DEFAULT_GLICKO_PHI = 350.0
DEFAULT_SKILL_SIGMA = 8.333


# Replace zeros with NaN without using pandas' deprecated downcasting in replace
def _zero_to_nan(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.mask(s == 0)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.Series(
        np.divide(
            pd.to_numeric(numerator, errors="coerce"),
            _zero_to_nan(denominator).abs(),
        ),
        index=numerator.index,
    )


def _days_since_previous_date(df: pd.DataFrame, entity_col: str) -> pd.Series:
    """Days since an entity's prior distinct match date, aligned to every row."""
    dates = df[[entity_col, "date"]].copy()
    dates["date"] = pd.to_datetime(dates["date"], errors="coerce")
    distinct = dates.drop_duplicates().sort_values([entity_col, "date"])
    distinct["_days_since"] = (
        distinct.groupby(entity_col, observed=True)["date"].diff().dt.days
    )
    merged = dates.merge(distinct, on=[entity_col, "date"], how="left", validate="m:1")
    return pd.Series(merged["_days_since"].to_numpy(), index=df.index, dtype="float64")


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
                "damagetakenperminute",
                "damagemitigatedperminute",
                "gamelength",
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
        denom_cols = ["enemyTeamKills", "enemyTeamDeaths", "enemyTeamWardPlaced"]
        df[denom_cols] = df[denom_cols].apply(_zero_to_nan)  # avoid /0, keep numeric

        df["ka_ratio"] = np.divide(
            df["kills"] + df["assists"],
            df["enemyTeamKills"] + df["kills"] + df["assists"],
        )
        df["d_ratio"] = np.divide(df["deaths"], df["enemyTeamDeaths"])
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

        # Normalize inputs to numeric to avoid None/object arithmetic surprises.
        # Unknown/missing results stay NaN and are excluded from win/loss
        # histories instead of being silently counted as losses.
        df = data.copy()
        df["result"] = normalize_result(df["result"])
        df["kills"] = pd.to_numeric(df["kills"], errors="coerce")
        df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce")
        df = df.sort_values(["playerid", "date"], kind="mergesort").reset_index(
            drop=True
        )

        def _prev_avg(
            by: list[str], result_value: int, metric_values: pd.Series
        ) -> pd.Series:
            """
            Expanding mean of metric for rows matching result_value, grouped by
            `by`, using only games from strictly earlier dates. Oracle's Elixir
            dates do not guarantee intra-day ordering, so same-date maps must
            not feed each other (mirrors the head-to-head handling).
            """
            mask = df["result"].eq(float(result_value)) & metric_values.notna()
            date_keys = [*by, "date"]
            tmp = df[date_keys].copy()
            tmp["_val"] = metric_values.where(mask, 0.0)
            tmp["_cnt"] = mask.astype("float64")
            daily = (
                tmp.groupby(date_keys, observed=True, as_index=False)
                .agg(_val=("_val", "sum"), _cnt=("_cnt", "sum"))
                .sort_values(date_keys, kind="mergesort")
            )
            groupers = [daily[key] for key in by]
            prev_sum = (
                daily.groupby(by, observed=True)["_val"]
                .cumsum()
                .groupby(groupers, sort=False)
                .shift()
            )
            prev_count = (
                daily.groupby(by, observed=True)["_cnt"]
                .cumsum()
                .groupby(groupers, sort=False)
                .shift()
            )
            daily["_prev_avg"] = prev_sum.div(prev_count.mask(prev_count == 0))
            merged = df[date_keys].merge(
                daily[[*date_keys, "_prev_avg"]],
                on=date_keys,
                how="left",
                validate="m:1",
            )
            return pd.Series(merged["_prev_avg"].to_numpy(), index=df.index)

        for scope, group_cols in (
            ("season", ["playerid", "season"]),
            ("patch", ["playerid", "patch"]),
        ):
            for metric in ("kills", "deaths"):
                metric_values = df[metric]
                df[f"{metric}_prev_avg_{scope}_win"] = _prev_avg(
                    group_cols, 1, metric_values
                )
                df[f"{metric}_prev_avg_{scope}_loss"] = _prev_avg(
                    group_cols, 0, metric_values
                )

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
                "league",
                "kills",
                "assists",
                "deaths",
                "gamelength",
                "total_cs",
                "patch",
                "result",
                "playerid",
            },
        )

        df = data.copy()
        df = add_league_taxonomy_columns(df, league_col="league")
        df["season"] = df["patch"].astype(str).str.split(".").str[0]
        df["days_since_last_game"] = _days_since_previous_date(df, "playerid")

        # Base per-game stats
        df["team_kills"] = df.groupby(["gameid", "teamid"], observed=True)[
            "kills"
        ].transform("sum")
        deaths = pd.to_numeric(df["deaths"], errors="coerce")
        kill_assist_total = pd.to_numeric(df["kills"], errors="coerce") + pd.to_numeric(
            df["assists"], errors="coerce"
        )
        df["kda"] = np.where(
            deaths.eq(0), kill_assist_total, np.divide(kill_assist_total, deaths)
        )
        gamelength = _zero_to_nan(df["gamelength"])
        df["xp_efficiency"] = np.divide(df["total_cs"], gamelength)
        df["kill_participation"] = np.divide(
            df["kills"] + df["assists"], _zero_to_nan(df["team_kills"])
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
        player_data: pd.DataFrame | None = None,
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
                "league",
                "patch",
                "teamid",
                "gameid",
                "gamelength",
                "kills",
                "towers",
            },
        )

        df = data.copy()
        df = df.sort_values(get_sorting_keys("team")).reset_index(drop=True)
        df = add_league_taxonomy_columns(df, league_col="league")
        df["season"] = df["patch"].astype(str).str.split(".").str[0]

        # ── Game-level context ───────────────────────────────────────────────
        df = df.merge(
            df.groupby("gameid", observed=True)
            .agg(total_kills=("kills", "sum"), total_towers=("towers", "sum"))
            .reset_index(),
            on="gameid",
            how="left",
        )
        df = FeatureGenerator.add_team_control_features(df)
        df = FeatureGenerator.add_team_vision_features(df, player_data)
        df = FeatureGenerator.add_roster_features(df, player_data)

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
                sort_also_by=["date"],
                exclude_same_date=True,
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
                exclude_same_date=True,
            )

        df = df.merge(
            game_level[["gameid", "patch_avg_gamelength", "season_avg_gamelength"]],
            on="gameid",
            how="left",
        )

        # Add series context (BO format, game number, deciding game)
        df = FeatureGenerator.add_series_context(df)

        # Add break indicator (first game after split break)
        df = FeatureGenerator.add_break_indicator(df)

        # Add head-to-head history against opponent
        return FeatureGenerator.add_head_to_head_history(df)

    @staticmethod
    def add_team_control_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add bounded objective and early-game control features."""
        out = df.copy()

        if {"kills", "total_kills"} <= set(out.columns):
            out["kill_share"] = _safe_divide(out["kills"], out["total_kills"])
        if {"towers", "total_towers"} <= set(out.columns):
            out["tower_share"] = _safe_divide(out["towers"], out["total_towers"])

        objective_cols = [
            col
            for col in ("dragons", "barons", "heralds", "elders", "void_grubs")
            if col in out.columns
        ]
        if objective_cols:
            out["epic_monsters"] = (
                out[objective_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            )

        structure_cols = [col for col in ("towers", "inhibitors") if col in out.columns]
        if structure_cols:
            out["structure_control"] = (
                out[structure_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            )

        out = FeatureGenerator.add_closing_speed_features(out)
        out = FeatureGenerator.add_objective_conversion_features(out)
        out = FeatureGenerator.add_checkpoint_growth_features(out)
        return FeatureGenerator.add_lead_conversion_features(out)

    @staticmethod
    def add_closing_speed_features(df: pd.DataFrame) -> pd.DataFrame:
        """Split game length by prior win/loss context for own-team EMAs."""
        out = df.copy()
        result_num = normalize_result(out["result"]) if "result" in out else None
        if result_num is not None and "gamelength" in out:
            gamelength = pd.to_numeric(out["gamelength"], errors="coerce")
            out["win_gamelength"] = gamelength.where(result_num == 1)
            out["loss_gamelength"] = gamelength.where(result_num == 0)
        return out

    @staticmethod
    def add_objective_conversion_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add objective-to-structure and first-objective conversion signals."""
        out = df.copy()
        result_num = normalize_result(out["result"]) if "result" in out else None
        if {"epic_monsters", "towers"} <= set(out.columns):
            out["towers_per_epic_monster"] = _safe_divide(
                out["towers"], out["epic_monsters"]
            )
        if {"epic_monsters", "structure_control"} <= set(out.columns):
            out["structure_per_epic_monster"] = _safe_divide(
                out["structure_control"], out["epic_monsters"]
            )
        if result_num is not None:
            if "firsttower" in out:
                first_tower = pd.to_numeric(out["firsttower"], errors="coerce").fillna(
                    0
                )
                out["first_tower_to_win"] = first_tower.mul(result_num)
            if "firstdragon" in out:
                first_dragon = pd.to_numeric(
                    out["firstdragon"], errors="coerce"
                ).fillna(0)
                out["first_dragon_to_win"] = first_dragon.mul(result_num)
        return out

    @staticmethod
    def add_checkpoint_growth_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add lead-growth deltas between adjacent game checkpoints."""
        out = df.copy()
        for minute in EARLY_GAME_MARKERS:
            for metric, diff in (
                ("gold", "golddiff"),
                ("xp", "xpdiff"),
                ("cs", "csdiff"),
            ):
                value_col = f"{metric}at{minute}"
                diff_col = f"{diff}at{minute}"
                if {value_col, diff_col} <= set(out.columns):
                    out[f"{diff}_shareat{minute}"] = _safe_divide(
                        out[diff_col], out[value_col]
                    )

        for start, end in pairwise(EARLY_GAME_MARKERS):
            for diff in ("golddiff", "xpdiff", "csdiff"):
                start_col = f"{diff}at{start}"
                end_col = f"{diff}at{end}"
                if {start_col, end_col} <= set(out.columns):
                    out[f"{diff}_growth_{start}_{end}"] = pd.to_numeric(
                        out[end_col], errors="coerce"
                    ) - pd.to_numeric(out[start_col], errors="coerce")
        return out

    @staticmethod
    def add_lead_conversion_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add win/loss conversion rates for teams ahead or behind by gold."""
        out = df.copy()
        result_num = normalize_result(out["result"]) if "result" in out else None
        if result_num is not None:
            for minute in (15, 25):
                gold_col = f"golddiffat{minute}"
                if gold_col not in out:
                    continue
                ahead = pd.to_numeric(out[gold_col], errors="coerce") > 0
                behind = pd.to_numeric(out[gold_col], errors="coerce") < 0
                out[f"ahead_goldat{minute}"] = ahead.astype("uint8")
                out[f"won_when_ahead_goldat{minute}"] = (
                    ahead & result_num.eq(1)
                ).astype("uint8")
                out[f"lost_when_ahead_goldat{minute}"] = (
                    ahead & result_num.eq(0)
                ).astype("uint8")
                out[f"won_when_behind_goldat{minute}"] = (
                    behind & result_num.eq(1)
                ).astype("uint8")
                out[f"lost_when_behind_goldat{minute}"] = (
                    behind & result_num.eq(0)
                ).astype("uint8")

        return out

    @staticmethod
    def add_team_vision_features(
        team_df: pd.DataFrame, player_df: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Aggregate player vision stats to team-game rows when player data is available."""
        if player_df is None:
            return team_df

        required = {"gameid", "teamid", "wpm", "wcpm", "vspm", "controlwardsbought"}
        if not required <= set(player_df.columns):
            return team_df

        vision = (
            player_df[list(required)]
            .copy()
            .assign(
                wpm=lambda x: pd.to_numeric(x["wpm"], errors="coerce"),
                wcpm=lambda x: pd.to_numeric(x["wcpm"], errors="coerce"),
                vspm=lambda x: pd.to_numeric(x["vspm"], errors="coerce"),
                controlwardsbought=lambda x: pd.to_numeric(
                    x["controlwardsbought"], errors="coerce"
                ),
            )
            .groupby(["gameid", "teamid"], observed=True, as_index=False)
            .agg(
                team_wpm=("wpm", "sum"),
                team_wcpm=("wcpm", "sum"),
                team_vspm=("vspm", "sum"),
                team_controlwardsbought=("controlwardsbought", "sum"),
            )
        )
        return team_df.merge(vision, on=["gameid", "teamid"], how="left")

    @staticmethod
    def add_roster_features(
        team_df: pd.DataFrame, player_df: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Add pre-match lineup continuity relative to the prior distinct match date."""
        if player_df is None:
            return team_df
        player_key = "playerid" if "playerid" in player_df else "playername"
        required = {"teamid", "date", player_key}
        if not required <= set(player_df.columns):
            return team_df

        lineups = player_df[list(required)].dropna(subset=[player_key]).copy()
        lineups["date"] = pd.to_datetime(lineups["date"], errors="coerce")
        lineups = (
            lineups.groupby(["teamid", "date"], observed=True)[player_key]
            .agg(frozenset)
            .reset_index(name="_roster")
            .sort_values(["teamid", "date"])
        )
        lineups["_previous_roster"] = lineups.groupby("teamid", observed=True)[
            "_roster"
        ].shift()
        lineups["roster_continuity"] = [
            len(current & previous) / len(_EXPECTED_ROSTER)
            if len(current) == len(_EXPECTED_ROSTER)
            and isinstance(previous, frozenset)
            and len(previous) == len(_EXPECTED_ROSTER)
            else np.nan
            for current, previous in zip(
                lineups["_roster"], lineups["_previous_roster"], strict=True
            )
        ]
        lineups["roster_uncertainty"] = (1.0 - lineups["roster_continuity"]).fillna(1.0)

        out = team_df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        return out.merge(
            lineups[["teamid", "date", "roster_continuity", "roster_uncertainty"]],
            on=["teamid", "date"],
            how="left",
            validate="m:1",
        )

    @staticmethod
    def add_rating_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
        """Expose normalized uncertainty already estimated by rating systems."""
        out = df.copy()
        sources: list[pd.Series] = []
        for column, baseline in (
            ("glicko2_phi_before", DEFAULT_GLICKO_PHI),
            ("pl_sigma_before", DEFAULT_SKILL_SIGMA),
            ("trueskill_sigma_before", DEFAULT_SKILL_SIGMA),
        ):
            if column in out:
                sources.append(pd.to_numeric(out[column], errors="coerce") / baseline)
        if sources:
            out["rating_uncertainty"] = pd.concat(sources, axis=1).mean(axis=1)
        return out

    @staticmethod
    def add_series_context(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add series format and game context features.

        Uses explicit scheduled best-of metadata when present. It does not infer
        BO format from completed series length, because that leaks whether a
        BO3/BO5 ended early.

        Added columns
        -------------
        game_in_series : int
            Game number (1, 2, 3, etc.) - derived from 'game' column
        is_bo1 : uint8
            1 if match is Best-of-1 format
        is_bo3 : uint8
            1 if match is scheduled Best-of-3 format
        is_bo5 : uint8
            1 if match is scheduled Best-of-5 format
        is_deciding_game : uint8
            1 if this is game 3 in BO3 or game 5 in BO5, when scheduled format is known
        """
        _check_required(df, required={"gameid", "game"})

        out = df.copy()

        # game column already has game number (1, 2, 3, etc.)
        out["game_in_series"] = out["game"].astype(int)

        best_of = pd.Series(np.nan, index=out.index, dtype="float64")
        for col in ("best_of", "bestof", "bestOf", "match_type"):
            if col not in out.columns:
                continue
            raw = out[col].astype(str).str.extract(r"(\d+)", expand=False)
            best_of = pd.to_numeric(raw, errors="coerce")
            break

        # Determine BO format
        out["is_bo1"] = best_of.eq(1).astype("uint8")
        out["is_bo3"] = best_of.eq(3).astype("uint8")
        out["is_bo5"] = best_of.eq(5).astype("uint8")

        # Deciding game: game 3 in BO3, game 5 in BO5
        out["is_deciding_game"] = (
            ((out["is_bo3"] == 1) & (out["game"] == GAMES_IN_BO3))
            | ((out["is_bo5"] == 1) & (out["game"] == GAMES_IN_BO5))
        ).astype("uint8")

        data_pipeline_logger.info("Series context features added.")
        return out

    @staticmethod
    def add_break_indicator(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add indicator for first game after a long break (split break).

        Added columns
        -------------
        days_since_last_game : float
            Days since team's previous game (NaN for first game ever)
        is_after_break : uint8
            1 if this is first game after >45 days (split break)
        is_first_season_game : uint8
            1 if this is team's first game of the season
        """
        _check_required(df, required={"teamid", "date", "season"})

        out = df.sort_values(["teamid", "date"], kind="mergesort").copy()

        # Days since last game
        out["days_since_last_game"] = _days_since_previous_date(out, "teamid")

        # First game after break (>45 days gap)
        out["is_after_break"] = (
            out["days_since_last_game"] > BREAK_THRESHOLD_DAYS
        ).astype("uint8")

        # First game of season
        out["is_first_season_game"] = (
            out.groupby(["teamid", "season"], observed=True).cumcount() == 0
        ).astype("uint8")

        data_pipeline_logger.info("Break indicator features added.")
        return out

    @staticmethod
    def add_head_to_head_history(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add head-to-head win rate against current opponent using only prior games.

        Added columns
        -------------
        h2h_games_before : int
            Number of prior games against this opponent
        h2h_wins_before : int
            Number of wins against this opponent in prior games
        h2h_win_rate_before : float
            Win rate against this opponent (NaN if <2 prior games)
        """
        _check_required(df, required={"teamid", "gameid", "date", "result", "side"})

        out = df.sort_values(["date", "gameid"], kind="mergesort").copy()

        out["_result_num"] = normalize_result(out["result"])
        out = add_opponent_columns(
            out,
            entity="team",
            source_columns=["teamid"],
        ).rename(columns={"opp_teamid": "_opponent_id"})

        # Create matchup key (sorted team pair for consistency)
        out["_matchup"] = out.apply(
            lambda r: (
                tuple(sorted([r["teamid"], r["_opponent_id"]]))
                if pd.notna(r["_opponent_id"])
                else None
            ),
            axis=1,
        )

        # For each team, compute h2h stats against each opponent using only
        # prior dates. Oracle's Elixir dates do not guarantee exact start order,
        # so same-date maps must not feed each other.
        out = out.sort_values(["teamid", "_opponent_id", "date"], kind="mergesort")
        h2h_keys = ["teamid", "_opponent_id"]
        date_keys = [*h2h_keys, "date"]
        daily = (
            out[[*date_keys, "gameid", "_result_num"]]
            .groupby(date_keys, observed=True, as_index=False)
            .agg(_daily_games=("gameid", "count"), _daily_wins=("_result_num", "sum"))
            .sort_values(date_keys, kind="mergesort")
        )
        groupers = [daily[key] for key in h2h_keys]
        daily["h2h_games_before"] = (
            daily.groupby(h2h_keys, observed=True)["_daily_games"]
            .cumsum()
            .groupby(groupers, sort=False)
            .shift()
            .fillna(0)
            .astype(int)
        )
        daily["h2h_wins_before"] = (
            daily.groupby(h2h_keys, observed=True)["_daily_wins"]
            .cumsum()
            .groupby(groupers, sort=False)
            .shift()
            .fillna(0)
            .astype(int)
        )
        out = out.merge(
            daily[[*date_keys, "h2h_games_before", "h2h_wins_before"]],
            on=date_keys,
            how="left",
            validate="m:1",
        )

        # Win rate (only if >= H2H_MIN_GAMES prior games)
        out["h2h_win_rate_before"] = np.where(
            out["h2h_games_before"] >= H2H_MIN_GAMES,
            out["h2h_wins_before"] / out["h2h_games_before"],
            np.nan,
        )

        # Clean up temporary columns
        out = out.drop(columns=["_result_num", "_opponent_id", "_matchup"])

        # Re-sort to original order
        out = out.sort_values(["date", "gameid", "side"], kind="mergesort")

        data_pipeline_logger.info("Head-to-head history features added.")
        return out


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
    exclude_same_date: bool = False,
    date_col: str = "date",
) -> pd.DataFrame:
    sort_keys = list(group_cols) + (sort_also_by or [])
    df = df.sort_values(sort_keys, kind="mergesort")

    if exclude_same_date:
        _check_required(df, required={date_col})
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        date_keys = [*group_cols, date_col]
        daily = (
            df[date_keys + value_cols]
            .groupby(date_keys, observed=True, as_index=False)
            .agg({col: ["sum", "count"] for col in value_cols})
        )
        daily.columns = [
            "_".join(str(part) for part in col if part)
            if isinstance(col, tuple)
            else col
            for col in daily.columns
        ]
        daily = daily.sort_values(date_keys, kind="mergesort")
        for col in value_cols:
            groupers = [daily[group_col] for group_col in group_cols]
            sum_col = f"{col}_sum"
            count_col = f"{col}_count"
            prev_sum = daily.groupby(group_cols, observed=True)[sum_col].cumsum()
            prev_count = daily.groupby(group_cols, observed=True)[count_col].cumsum()
            daily[f"{prefix}{col}"] = (
                prev_sum.groupby(groupers, sort=False)
                .shift()
                .div(prev_count.groupby(groupers, sort=False).shift())
            )
        return df.merge(
            daily[date_keys + [f"{prefix}{col}" for col in value_cols]],
            on=date_keys,
            how="left",
            validate="m:1",
        )

    for col in value_cols:
        grp = df.groupby(group_cols, observed=True)[col]
        counts = grp.cumcount()
        running_sum = grp.cumsum()
        prev_sum = running_sum.groupby([df[c] for c in group_cols], sort=False).shift()
        df[f"{prefix}{col}"] = prev_sum.div(counts.mask(counts == 0))

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
    # Ensure chronological order within each group before rolling
    df = df.sort_values([*group_cols, "date"], kind="mergesort")

    for col in value_cols:
        df[f"{prefix}{col}"] = df.groupby(group_cols, observed=True)[col].transform(
            lambda s: s.shift().rolling(window, min_periods=1).mean()
        )

    return df
