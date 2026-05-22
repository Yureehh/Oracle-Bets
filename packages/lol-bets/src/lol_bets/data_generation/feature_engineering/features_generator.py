"""
Features Generator

This script contains the `FeatureGenerator` class, which is used to generate
new features for player and team data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from oracle_bets_core.io_utils import get_sorting_keys
from oracle_bets_core.league_taxonomy import add_league_taxonomy_columns
from oracle_bets_core.logger import LOG_TOPIC, instantiate_logger, logger
from oracle_bets_core.pd import pd

if TYPE_CHECKING:
    from collections.abc import Iterable

data_pipeline_logger = instantiate_logger(LOG_TOPIC.DATA_PIPELINE)

# ────────────────────────────────────────────────────────────────────────────
# Helpers/constants shared by multiple methods
# ────────────────────────────────────────────────────────────────────────────
_OPPOSITE_SIDE = {"Blue": "Red", "Red": "Blue"}  # quick side-flip
GAME_ID_PARTS = 2
GAMES_IN_BO3 = 3
GAMES_IN_BO5 = 5
BREAK_THRESHOLD_DAYS = 45  # ~1.5 months, indicates split break
H2H_MIN_GAMES = 2  # minimum games to compute head-to-head
EARLY_GAME_MARKERS = (15, 25)


# Replace zeros with NaN without using pandas' deprecated downcasting in replace
def _zero_to_nan(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.mask(s == 0)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return np.divide(
        pd.to_numeric(numerator, errors="coerce"),
        _zero_to_nan(denominator).abs(),
    )


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

        # Normalize inputs to numeric to avoid None/object arithmetic surprises
        df = data.copy()
        df["result"] = (
            df["result"]
            .map(
                {
                    "W": 1,
                    "Win": 1,
                    "win": 1,
                    "Won": 1,
                    "won": 1,
                    True: 1,
                    "L": 0,
                    "Loss": 0,
                    "loss": 0,
                    "Lose": 0,
                    "lose": 0,
                    False: 0,
                }
            )
            .astype("float64")
            .fillna(0.0)
        )
        df["kills"] = pd.to_numeric(df["kills"], errors="coerce").fillna(0.0)
        df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce").fillna(0.0)
        df = df.sort_values(["playerid", "date"], kind="mergesort").reset_index(
            drop=True
        )

        def _prev_avg(
            by: list[str], result_value: int, metric_values: pd.Series
        ) -> pd.Series:
            """
            Expanding mean of metric for rows matching result_value, grouped by `by`,
            using only prior games (shifted).
            """
            mask = df["result"].eq(result_value)
            groups = df[by].apply(tuple, axis=1)
            running_sum = metric_values.where(mask, 0.0).groupby(groups).cumsum()
            running_count = mask.groupby(groups).cumsum()
            prev_sum = running_sum.groupby(groups).shift()
            prev_count = running_count.groupby(groups).shift()
            return prev_sum.div(prev_count.mask(prev_count == 0))

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

        # Base per-game stats
        df["team_kills"] = df.groupby(["gameid", "teamid"], observed=True)[
            "kills"
        ].transform("sum")
        deaths = _zero_to_nan(df["deaths"])
        df["kda"] = np.divide(df["kills"] + df["assists"], deaths)
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

        return out

    @staticmethod
    def add_series_context(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add series format and game context features.

        Uses gameid pattern and game column to infer BO format.

        Added columns
        -------------
        game_in_series : int
            Game number (1, 2, 3, etc.) - derived from 'game' column
        is_bo1 : uint8
            1 if match is Best-of-1 format
        is_bo3 : uint8
            1 if match is Best-of-3 format (games 2 or 3 played)
        is_bo5 : uint8
            1 if match is Best-of-5 format (games 4 or 5 played)
        is_deciding_game : uint8
            1 if this is game 3 in BO3 or game 5 in BO5
        """
        _check_required(df, required={"gameid", "game"})

        out = df.copy()

        # game column already has game number (1, 2, 3, etc.)
        out["game_in_series"] = out["game"].astype(int)

        # Infer BO format from max game number per match
        # Extract match identifier by removing game number suffix
        # gameid format varies: sometimes ends with "_game1", sometimes different patterns
        # Group by removing trailing digits or "_gameN" pattern
        # Approach: group by (date, league, team pair) to identify unique matches

        # Use all but last underscore segment if gameid has underscore
        def _extract_match_id(gid: str) -> str:
            if "_" in gid:
                # Try to remove game suffix (e.g., "_1", "_2", "_game1")
                parts = gid.rsplit("_", 1)
                if len(parts) == GAME_ID_PARTS and (
                    parts[1].isdigit() or parts[1].startswith("game")
                ):
                    return parts[0]
            return gid

        out["_match_id"] = out["gameid"].apply(_extract_match_id)

        # Get max game number per match
        match_max_game = out.groupby("_match_id", observed=True)["game"].transform(
            "max"
        )

        # Determine BO format
        out["is_bo1"] = (match_max_game == 1).astype("uint8")
        out["is_bo3"] = (match_max_game.isin([2, 3])).astype("uint8")
        out["is_bo5"] = (match_max_game.isin([4, 5])).astype("uint8")

        # Deciding game: game 3 in BO3, game 5 in BO5
        out["is_deciding_game"] = (
            ((out["is_bo3"] == 1) & (out["game"] == GAMES_IN_BO3))
            | ((out["is_bo5"] == 1) & (out["game"] == GAMES_IN_BO5))
        ).astype("uint8")

        # Clean up temporary column
        out = out.drop(columns=["_match_id"])

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
        out["days_since_last_game"] = (
            out.groupby("teamid", observed=True)["date"].diff().dt.days
        )

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

        # Normalize result to numeric
        if not pd.api.types.is_numeric_dtype(out["result"]):
            result_map = {
                "W": 1,
                "Win": 1,
                "win": 1,
                True: 1,
                "L": 0,
                "Loss": 0,
                "loss": 0,
                False: 0,
            }
            out["_result_num"] = out["result"].map(result_map).astype("float64")
        else:
            out["_result_num"] = out["result"].astype("float64")

        # Create opponent mapping per game
        # For each row, get the opponent teamid from the same gameid but opposite side
        game_teams = (
            out.groupby("gameid", observed=True)
            .apply(
                lambda g: dict(zip(g["side"], g["teamid"], strict=False)),
                include_groups=False,
            )
            .to_dict()
        )

        def _get_opponent(row):
            teams = game_teams.get(row["gameid"], {})
            opp_side = _OPPOSITE_SIDE.get(row["side"])
            return teams.get(opp_side)

        out["_opponent_id"] = out.apply(_get_opponent, axis=1)

        # Create matchup key (sorted team pair for consistency)
        out["_matchup"] = out.apply(
            lambda r: tuple(sorted([r["teamid"], r["_opponent_id"]]))
            if pd.notna(r["_opponent_id"])
            else None,
            axis=1,
        )

        # For each team, compute expanding h2h stats against each opponent
        # Group by (teamid, opponent) and compute cumulative stats shifted
        out = out.sort_values(["teamid", "_opponent_id", "date"], kind="mergesort")

        # Cumulative games and wins vs this opponent (shifted for leak-free)
        grp = out.groupby(["teamid", "_opponent_id"], observed=True, sort=False)
        out["h2h_games_before"] = grp.cumcount()  # 0-indexed, so this is count before
        running_wins = grp["_result_num"].cumsum()
        out["h2h_wins_before"] = (
            running_wins.groupby([out["teamid"], out["_opponent_id"]], sort=False)
            .shift()
            .fillna(0)
            .astype(int)
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
) -> pd.DataFrame:
    sort_keys = list(group_cols) + (sort_also_by or [])
    df = df.sort_values(sort_keys, kind="mergesort")

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
