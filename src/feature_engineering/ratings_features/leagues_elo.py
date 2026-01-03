"""
League Elo Rating System with Hyperparameter Tuning using Optuna

This script calculates league Elo ratings and uses Optuna to optimize hyperparameters.
"""

from __future__ import annotations

import contextlib
import json
from collections import defaultdict
from pathlib import Path

import optuna
import pandas as pd
from sklearn.metrics import log_loss
from tqdm import tqdm

from utils.io_utils import get_sorting_keys, json_loader, safe_store_df_as_parquet
from utils.league_taxonomy import get_config_strength_prior, get_league_taxonomy
from utils.logger import instantiate_logger, logger
from utils.paths import (
    CONSIDERED_LEAGUES,
    LEAGUE_ELO,
    LEAGUE_PRIOR_SETTINGS,
    LEAGUE_STRENGTH_PRIORS,
    LEAGUES_ELO_HYPERPARAMETERS,
    TEAM_LEAGUES_MAPPING,
)

# ----------------------------------------------------------------------
# Global Config / Constants
# ----------------------------------------------------------------------
considered_leagues_config = json_loader(CONSIDERED_LEAGUES)
CROSS_LEAGUE_COMPETITIONS = set(considered_leagues_config["cross_league_competitions"])
data_pipeline_logger = instantiate_logger("data_pipeline")

ROWS_PER_TEAM = 2  # one per side
TRIALS_NUM = 50
MAX_EXPONENT = 8.0  # clamp for expected outcome stability


# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------
def preprocess_dataframe(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Perform all necessary preprocessing on the input DataFrame, ensuring it's clean and ready for tuning or computation.
    """
    required_columns = [
        "season",
        "date",
        "gameid",
        "teamid",
        "league",
        "side",
        "result",
    ]
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        msg = f"Input DataFrame is missing required columns: {missing_columns}"
        raise ValueError(msg)

    df = df.copy()

    # Ensure 'date' is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if df["date"].isna().any():
                n_missing_dates = df["date"].isna().sum()
                logger.warning(
                    f"{n_missing_dates} 'date' entries could not be converted and are NaT. Dropping these."
                )
                data_pipeline_logger.warning(
                    f"{n_missing_dates} 'date' entries could not be converted and are NaT. Dropping these."
                )
                df = df.dropna(subset=["date"])
        except Exception as e:
            logger.error(f"Error converting 'date' to datetime: {e}")
            data_pipeline_logger.exception(f"Error converting 'date' to datetime: {e}")
            raise

    # Drop rows missing league or result
    if df["league"].isna().any() or df["result"].isna().any():
        n_missing_leagues = df["league"].isna().sum()
        n_missing_results = df["result"].isna().sum()
        logger.warning(
            f"{n_missing_leagues} 'league' and {n_missing_results} 'result' missing. Dropping."
        )
        data_pipeline_logger.warning(
            f"{n_missing_leagues} 'league' and {n_missing_results} 'result' missing. Dropping."
        )
        df = df.dropna(subset=["league", "result"]).reset_index(drop=True)

    # Normalize result to numeric 0/1 if needed
    if not pd.api.types.is_numeric_dtype(df["result"]):
        valmap = {
            "W": 1,
            "Win": 1,
            "win": 1,
            True: 1,
            "L": 0,
            "Loss": 0,
            "loss": 0,
            False: 0,
        }
        df["result"] = df["result"].map(valmap).astype("float64")

    # Stable sort by canonical keys
    return df.sort_values(by=get_sorting_keys(entity), kind="mergesort").reset_index(
        drop=True
    )


def map_team_to_league(
    df: pd.DataFrame, team_column: str
) -> dict[str, list[tuple[pd.Timestamp, str]]]:
    """
    Map each team to the leagues it played in over time, excluding cross-league competitions.
    Returns a dict: teamid -> [(date, league), ...] in chronological order.
    """
    # Filter out cross-league competitions
    df_filtered = df[~df["league"].isin(CROSS_LEAGUE_COMPETITIONS)].copy()

    # Sort by date to get the chronological order
    df_filtered = df_filtered.sort_values(by=["date"], kind="mergesort")

    league_history: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    for team, group in df_filtered.groupby(team_column, sort=False):
        # Ensure chronological order within each team
        g = group.sort_values("date", kind="mergesort")
        league_history[team] = list(zip(g["date"], g["league"], strict=False))
    return league_history


# ----------------------------------------------------------------------
# Core Elo helpers
# ----------------------------------------------------------------------
def expected_outcome(elo_a: float, elo_b: float, elo_divisor: float) -> float:
    """
    Calculate the expected match outcome between two aggregated Elo ratings with clamped exponent.
    """
    exponent = (elo_b - elo_a) / elo_divisor
    if exponent > MAX_EXPONENT:
        exponent = MAX_EXPONENT
    elif exponent < -MAX_EXPONENT:
        exponent = -MAX_EXPONENT
    return 1.0 / (1.0 + 10.0**exponent)


def update_elo_rating(
    old_elo: float, expected: float, actual_result: float, k_factor: float
) -> float:
    """Update Elo rating based on match result."""
    return old_elo + k_factor * (actual_result - expected)


def linear_decay_reset_leagues_elo(
    elo_ratings: dict[str, dict[str, float | int]],
    baseline: float,
    current_season: int,
    decay_factor: float,
) -> dict[str, dict[str, float | int]]:
    """
    Apply linear decay reset to league Elo ratings at the beginning of a new season.
    Returns a *new* dictionary with the updated Elo ratings.
    """
    updated_elo_ratings: dict[str, dict[str, float | int]] = {}
    for league, data in elo_ratings.items():
        updated = dict(data)
        if int(data.get("season", current_season)) < current_season:
            updated["elo"] = baseline + ((float(data["elo"]) - baseline) * decay_factor)
            updated["season"] = current_season
        updated_elo_ratings[league] = updated
    return updated_elo_ratings


# ----------------------------------------------------------------------
# Wide pivot (one row per game)
# ----------------------------------------------------------------------
def pivot_games_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the original DataFrame so that each game is a single row:
      - 'Blue' columns => e.g. teamid_Blue, league_Blue, result_Blue
      - 'Red' columns  => e.g. teamid_Red,  league_Red,  result_Red
    """
    counts = df.groupby(["gameid"]).size()
    if not (counts == ROWS_PER_TEAM).all():
        logger.warning(
            "Some gameid groups do not have exactly 2 rows. Filtering to valid games only."
        )
        data_pipeline_logger.warning(
            "Some gameid groups do not have exactly 2 rows. Filtering to valid games only."
        )
        df = df[df["gameid"].isin(counts[counts == ROWS_PER_TEAM].index)]

    df_wide = df.pivot_table(
        index=["date", "gameid", "season"],
        columns="side",
        values=["teamid", "league", "result"],
        aggfunc="first",  # <- critical change
        observed=True,  # <- small speed/footprint win for categoricals
    ).reset_index()

    # Flatten columns robustly (unchanged)
    flat_cols: list[str] = []
    for col in df_wide.columns.to_flat_index():
        if isinstance(col, tuple):
            a, b = col
            flat_cols.append(f"{a}_{b}" if b else f"{a}")
        else:
            flat_cols.append(str(col))
    df_wide.columns = flat_cols
    return df_wide


# ----------------------------------------------------------------------
# Merge back to tall (two rows per game)
# ----------------------------------------------------------------------
def merge_wide_results_back(
    df: pd.DataFrame,
    df_wide: pd.DataFrame,
    columns_to_add: dict[str, str],
) -> pd.DataFrame:
    """
    After computing Elo in the wide pivot, merge those results back to the original tall shape.
    """
    df_blue = df_wide.copy()
    df_blue["side"] = "Blue"
    df_red = df_wide.copy()
    df_red["side"] = "Red"

    # Drop opposite-side columns to reduce clutter
    drop_cols_blue = [c for c in df_blue.columns if c.endswith(("_Red", "_red"))]
    drop_cols_red = [c for c in df_red.columns if c.endswith(("_Blue", "_blue"))]
    df_blue = df_blue.drop(columns=drop_cols_blue)
    df_red = df_red.drop(columns=drop_cols_red)

    # Rename Elo-related columns to final names
    for wide_col, final_col in columns_to_add.items():
        if wide_col in df_blue.columns:
            df_blue = df_blue.rename(columns={wide_col: final_col})
        if wide_col in df_red.columns:
            df_red = df_red.rename(columns={wide_col: final_col})

    # Keep only what we need for the merge
    keep_cols = [
        "date",
        "gameid",
        "season",
        "side",
        "league_elo_before",
        "opp_league_elo_before",
        "league_elo_win_likelihood",
        "league_elo_prior_win_likelihood",
        "league_elo_after",
    ]
    df_tall = pd.concat([df_blue[keep_cols], df_red[keep_cols]], ignore_index=True)

    merged = pd.merge(  # noqa: PD015
        df,
        df_tall,
        on=["date", "gameid", "season", "side"],
        how="left",
        validate="many_to_one",
    )

    required_columns = set(df.columns).union(
        {
            "league_elo_before",
            "opp_league_elo_before",
            "league_elo_win_likelihood",
            "league_elo_prior_win_likelihood",
            "league_elo_after",
        }
    )
    return merged[[c for c in merged.columns if c in required_columns]]


# ----------------------------------------------------------------------
# Hyperparameter Tuning
# ----------------------------------------------------------------------
def tune_hyperparameters(  # noqa: PLR0915
    df: pd.DataFrame,
    entity: str,
    belonging_league: dict[str, list[tuple[pd.Timestamp, str]]],
    hyperparameters_path: str,
) -> dict[str, float]:
    """
    Perform hyperparameter tuning using Optuna and return the best parameters.
    Trains on cross-league competitions, validates on the last year (Blue rows only).
    """
    # Keep only cross-league competitions for tuning
    df_cross = df[df["league"].isin(CROSS_LEAGUE_COMPETITIONS)].copy()
    if df_cross.empty:
        logger.warning("No cross-league competitions found for tuning. Aborting.")
        data_pipeline_logger.warning(
            "No cross-league competitions found for tuning. Aborting."
        )
        msg = "No cross-league competitions available for tuning."
        raise ValueError(msg)

    def resolve_league(team_id: str, match_date: pd.Timestamp) -> str | None:
        history = belonging_league.get(team_id, [])
        # find most recent league up to match_date
        for date, lg in reversed(history):
            if date <= match_date:
                return lg
        return None

    def objective(trial: optuna.trial.Trial) -> float:  # noqa: PLR0915
        hyper = {
            "k_factor": trial.suggest_float("k_factor", 16, 96, step=8),
            "initial_elo": trial.suggest_float("initial_elo", 1200, 1800, step=100),
            "elo_divisor": trial.suggest_float("elo_divisor", 100, 500, step=50),
            "decay_factor": trial.suggest_float("decay_factor", 0.5, 1.0, step=0.05),
        }

        # Sort & split
        df_sorted = df_cross.sort_values(
            by=["date", "gameid", "side"], kind="mergesort"
        ).reset_index(drop=True)
        if df_sorted.empty:
            return float("inf")

        try:
            split_year = df_sorted["date"].dt.year.max()
        except AttributeError as e:
            logger.error(f"Error accessing 'date' column with .dt accessor: {e}")
            data_pipeline_logger.exception(
                f"Error accessing 'date' column with .dt accessor: {e}"
            )
            return float("inf")
        split_date = pd.to_datetime(f"{split_year}-01-01")

        df_train = df_sorted[df_sorted["date"] < split_date].reset_index(drop=True)
        df_valid = df_sorted[df_sorted["date"] >= split_date].reset_index(drop=True)

        # each game should have exactly 2 rows
        if (
            df_train.empty
            or not (df_train.groupby("gameid").size() == ROWS_PER_TEAM).all()
        ):
            logger.warning(
                "Training data is insufficient or improperly structured. Skipping trial."
            )
            data_pipeline_logger.warning(
                "Training data is insufficient or improperly structured. Skipping trial."
            )
            return float("inf")

        # --- Train: compute league elos over training (using full pipeline) ---
        try:
            df_train_res = leagues_elo_computation(
                df=df_train.copy(),
                entity=entity,
                belonging_league=belonging_league,
                initial_elo=hyper["initial_elo"],
                k_factor=hyper["k_factor"],
                elo_divisor=hyper["elo_divisor"],
                decay_factor=hyper["decay_factor"],
                performing_tuning=True,
            )
        except Exception as e:
            logger.error(f"Elo calculation error in trial: {e}")
            data_pipeline_logger.exception(f"Elo calculation error in trial: {e}")
            return float("inf")

        # Build starting validation ratings from training tail (league->dict)
        # Ensure chronological order first (keep your existing sort)
        df_train_res = df_train_res.sort_values(
            by=["date", "gameid", "side"], kind="mergesort"
        )

        # Resolve "home league" for each row using the map built by `map_team_to_league`
        def _resolve(team_id, dt: pd.Timestamp) -> str | None:
            hist = belonging_league.get(team_id, [])
            for d, lg in reversed(hist):
                if d <= dt:
                    return lg
            return None

        df_train_res["resolved_league"] = df_train_res.apply(
            lambda r: _resolve(r["teamid"], r["date"]), axis=1
        )

        # Warm-start validation from the last *resolved league* ratings, not competition names
        last_league_elos = (
            df_train_res.dropna(subset=["resolved_league", "league_elo_after"])
            .sort_values(["date", "gameid", "side"], kind="mergesort")
            .groupby("resolved_league")["league_elo_after"]
            .last()
            .to_dict()
        )

        validation_elo_ratings = defaultdict(
            lambda: {"elo": float(hyper["initial_elo"]), "season": int(split_year)}
        )
        for lg, elo_val in last_league_elos.items():
            validation_elo_ratings[lg] = {
                "elo": float(elo_val),
                "season": int(split_year),
            }

        # --- Validate: simulate over df_valid grouped by game (Blue rows only for loss) ---
        expected_probs: list[float] = []
        true_labels: list[float] = []
        df_valid_sorted = df_valid.sort_values(
            by=["date", "gameid", "side"], kind="mergesort"
        ).reset_index(drop=True)

        for (_, _gameid), grp in df_valid_sorted.groupby(
            ["date", "gameid"], sort=False
        ):
            if len(grp) != ROWS_PER_TEAM:
                continue  # skip malformed games

            blue_row = grp[grp["side"] == "Blue"]
            red_row = grp[grp["side"] == "Red"]
            if blue_row.empty or red_row.empty:
                continue

            match_date = blue_row.iloc[0]["date"]
            season = int(blue_row.iloc[0]["season"])
            # seasonal decay once at season boundary
            validation_elo_ratings = linear_decay_reset_leagues_elo(
                validation_elo_ratings,
                float(hyper["initial_elo"]),
                season,
                float(hyper["decay_factor"]),
            )

            blue_team = blue_row.iloc[0]["teamid"]
            red_team = red_row.iloc[0]["teamid"]
            blue_league = resolve_league(blue_team, match_date)
            red_league = resolve_league(red_team, match_date)
            if blue_league is None or red_league is None:
                continue

            blue_elo = float(validation_elo_ratings[blue_league]["elo"])
            red_elo = float(validation_elo_ratings[red_league]["elo"])

            exp_blue = expected_outcome(blue_elo, red_elo, float(hyper["elo_divisor"]))
            expected_probs.append(exp_blue)

            blue_result = float(blue_row.iloc[0]["result"])
            true_labels.append(blue_result)  # keep labels aligned with predictions
            red_result = 1.0 - blue_result

            # Update both leagues
            validation_elo_ratings[blue_league]["elo"] = update_elo_rating(
                blue_elo, exp_blue, blue_result, float(hyper["k_factor"])
            )
            validation_elo_ratings[red_league]["elo"] = update_elo_rating(
                red_elo, 1.0 - exp_blue, red_result, float(hyper["k_factor"])
            )

        # Evaluate log loss on Blue rows
        y_true = pd.Series(true_labels, dtype="float64")
        y_pred = pd.Series(expected_probs, dtype="float64").clip(0.0001, 0.9999)

        if len(y_true) != len(y_pred) or len(y_true) == 0:
            logger.error("Mismatch in lengths of y_true and y_pred during validation.")
            data_pipeline_logger.error(
                "Mismatch in lengths of y_true and y_pred during validation."
            )
            return float("inf")

        return log_loss(y_true, y_pred)

    # Optimize
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=TRIALS_NUM, show_progress_bar=True)

    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")
    data_pipeline_logger.info(f"Best hyperparameters: {best_params}")

    # Store
    try:
        logger.info(f"Storing hyperparameters to {hyperparameters_path}")
        data_pipeline_logger.info(f"Storing hyperparameters to {hyperparameters_path}")
        # tune_hyperparameters(): saving best_params
        with Path(hyperparameters_path).open("w") as f:
            json.dump(best_params, f)
    except Exception as e:
        logger.error(f"Failed to save hyperparameters: {e}")
        data_pipeline_logger.exception(f"Failed to save hyperparameters: {e}")

    return best_params


# ----------------------------------------------------------------------
# League Elo computation (wide -> tall)
# ----------------------------------------------------------------------
def leagues_elo_computation(
    df: pd.DataFrame,
    entity: str,
    belonging_league: dict[str, list[tuple[pd.Timestamp, str]]],
    initial_elo: float,
    k_factor: float,
    elo_divisor: float,
    decay_factor: float,
    performing_tuning: bool = False,
) -> pd.DataFrame:
    """
    Calculate and update Elo ratings for leagues based on match results.
    """
    df_wide = pivot_games_to_wide(df)

    league_elo_ratings: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"elo": initial_elo, "season": int(df_wide["season"].min())}
    )

    wide_columns: dict[str, list] = {
        "league_elo_before_blue": [],
        "league_elo_before_red": [],
        "opp_league_elo_before_blue": [],
        "opp_league_elo_before_red": [],
        "league_elo_win_likelihood_blue": [],
        "league_elo_win_likelihood_red": [],
        "league_elo_prior_win_likelihood_blue": [],
        "league_elo_prior_win_likelihood_red": [],
        "league_elo_after_blue": [],
        "league_elo_after_red": [],
    }

    if not performing_tuning:
        logger.info("Calculating Leagues Elo")
        data_pipeline_logger.info("Calculating Leagues Elo")

    row_iter = df_wide.itertuples(index=True)
    if not performing_tuning:
        row_iter = tqdm(row_iter, total=len(df_wide), desc="League Elo")

    for row in row_iter:
        wide_columns, league_elo_ratings = process_elo_for_row(
            row=row,
            league_elo_ratings=league_elo_ratings,
            belonging_league=belonging_league,
            wide_columns=wide_columns,
            initial_elo=initial_elo,
            k_factor=k_factor,
            elo_divisor=elo_divisor,
            decay_factor=decay_factor,
        )

    update_wide_dataframe(df_wide, wide_columns)
    df_final = finalize_dataframe(df, df_wide, entity)

    if not performing_tuning:
        store_results(belonging_league, league_elo_ratings)

    return df_final


def process_elo_for_row(
    row,
    league_elo_ratings: dict[str, dict[str, float | int]],
    belonging_league: dict[str, list[tuple[pd.Timestamp, str]]],
    wide_columns: dict[str, list],
    initial_elo: float,
    k_factor: float,
    elo_divisor: float,
    decay_factor: float,
) -> tuple[dict[str, list], dict[str, dict[str, float | int]]]:
    """Process a single wide row to calculate and update Elo ratings for leagues."""
    current_season = int(row.season)
    league_elo_ratings = linear_decay_reset_leagues_elo(
        league_elo_ratings, initial_elo, current_season, decay_factor
    )

    def resolve_league(team_id: str, match_date: pd.Timestamp) -> str | None:
        history = belonging_league.get(team_id, [])
        for date, lg in reversed(history):
            if date <= match_date:
                return lg
        return None

    match_date: pd.Timestamp = row.date
    blue_league = resolve_league(row.teamid_Blue, match_date)
    red_league = resolve_league(row.teamid_Red, match_date)
    blue_result = getattr(row, "result_Blue", None)
    red_result = getattr(row, "result_Red", None)

    if (
        blue_result is None
        or red_result is None
        or blue_league is None
        or red_league is None
    ):
        # Fill Nones for all columns for this game
        for c in wide_columns:  # noqa: PLC0206
            wide_columns[c].append(None)
        return wide_columns, league_elo_ratings

    # Initialize leagues if needed
    if blue_league not in league_elo_ratings:
        league_elo_ratings[blue_league] = {
            "elo": initial_elo,
            "season": current_season,
        }
    if red_league not in league_elo_ratings:
        league_elo_ratings[red_league] = {"elo": initial_elo, "season": current_season}

    blue_before = float(league_elo_ratings[blue_league]["elo"])
    red_before = float(league_elo_ratings[red_league]["elo"])
    blue_prior = get_config_strength_prior(blue_league)
    red_prior = get_config_strength_prior(red_league)

    if blue_league != red_league:
        exp_blue = expected_outcome(blue_before, red_before, elo_divisor)
        exp_blue_prior = expected_outcome(
            blue_before + blue_prior,
            red_before + red_prior,
            elo_divisor,
        )
        exp_red = 1.0 - exp_blue
        new_blue = update_elo_rating(
            blue_before, exp_blue, float(blue_result), k_factor
        )
        new_red = update_elo_rating(red_before, exp_red, float(red_result), k_factor)
    else:
        exp_blue = 0.5
        exp_blue_prior = 0.5
        new_blue = blue_before
        new_red = red_before

    league_elo_ratings[blue_league]["elo"] = new_blue
    league_elo_ratings[red_league]["elo"] = new_red

    wide_columns["league_elo_before_blue"].append(blue_before)
    wide_columns["league_elo_before_red"].append(red_before)
    wide_columns["opp_league_elo_before_blue"].append(red_before)
    wide_columns["opp_league_elo_before_red"].append(blue_before)
    wide_columns["league_elo_win_likelihood_blue"].append(exp_blue)
    wide_columns["league_elo_win_likelihood_red"].append(1.0 - exp_blue)
    wide_columns["league_elo_prior_win_likelihood_blue"].append(exp_blue_prior)
    wide_columns["league_elo_prior_win_likelihood_red"].append(1.0 - exp_blue_prior)
    wide_columns["league_elo_after_blue"].append(new_blue)
    wide_columns["league_elo_after_red"].append(new_red)

    return wide_columns, league_elo_ratings


def update_wide_dataframe(df_wide: pd.DataFrame, wide_columns: dict[str, list]) -> None:
    """Update the wide dataframe with the computed columns."""
    for col_name, col_values in wide_columns.items():
        df_wide[col_name] = pd.Series(col_values, dtype="float64")


def finalize_dataframe(
    df: pd.DataFrame, df_wide: pd.DataFrame, entity: str
) -> pd.DataFrame:
    """Finalize the dataframe by pivoting back and sorting."""
    columns_map = {
        "league_elo_before_blue": "league_elo_before",
        "league_elo_before_red": "league_elo_before",
        "opp_league_elo_before_blue": "opp_league_elo_before",
        "opp_league_elo_before_red": "opp_league_elo_before",
        "league_elo_win_likelihood_blue": "league_elo_win_likelihood",
        "league_elo_win_likelihood_red": "league_elo_win_likelihood",
        "league_elo_prior_win_likelihood_blue": "league_elo_prior_win_likelihood",
        "league_elo_prior_win_likelihood_red": "league_elo_prior_win_likelihood",
        "league_elo_after_blue": "league_elo_after",
        "league_elo_after_red": "league_elo_after",
    }
    df_final = merge_wide_results_back(df, df_wide, columns_map)
    return df_final.sort_values(
        by=get_sorting_keys(entity), kind="mergesort"
    ).reset_index(drop=True)


# ----------------------------------------------------------------------
# Storing artifacts
# ----------------------------------------------------------------------
def store_results(
    belonging_league: dict[str, list[tuple[pd.Timestamp, str]]],
    league_elo_ratings: dict[str, dict[str, float | int]],
) -> None:
    """Store belonging leagues and league Elo ratings."""
    store_belonging_leagues(belonging_league)
    league_elo_df = store_leagues_elo(league_elo_ratings)
    store_league_strength_priors(league_elo_df)


def store_belonging_leagues(
    belonging_league: dict[str, list[tuple[pd.Timestamp, str]]],
) -> None:
    """
    Store the mapping of teams to their most recent league (latest entry per team) as a parquet.
    """
    latest_belonging_league = {
        team: history[-1][1] for team, history in belonging_league.items() if history
    }
    belonging_league_df = pd.DataFrame(
        latest_belonging_league.items(), columns=["teamid", "league"]
    )
    safe_store_df_as_parquet(
        belonging_league_df, TEAM_LEAGUES_MAPPING, [logger, data_pipeline_logger]
    )


def store_leagues_elo(
    league_elo_ratings: dict[str, dict[str, float | int]],
) -> pd.DataFrame:
    """Store the Elo ratings for leagues."""
    league_elo_df = (
        pd.DataFrame(
            [(lg, dat["elo"]) for lg, dat in league_elo_ratings.items()],
            columns=["league", "elo"],
        )
        .dropna(subset=["league"])
        .sort_values(by="elo", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    safe_store_df_as_parquet(league_elo_df, LEAGUE_ELO, [logger, data_pipeline_logger])
    return league_elo_df


def store_league_strength_priors(league_elo_df: pd.DataFrame) -> None:
    """
    Derive league strength priors from the league Elo table and persist to JSON.
    Defaults can be tuned in config/data_ingestion/leagues_handling/league_prior_settings.json
    (including `calibration_mode`: "global" or "tiered").
    """
    settings = {
        "prior_scale": 0.25,
        "max_abs_prior": 200.0,
        "calibration_mode": "global",
    }
    with contextlib.suppress(FileNotFoundError):
        cfg = json_loader(LEAGUE_PRIOR_SETTINGS)
        settings["prior_scale"] = float(cfg.get("prior_scale", settings["prior_scale"]))
        settings["max_abs_prior"] = float(
            cfg.get("max_abs_prior", settings["max_abs_prior"])
        )
        settings["calibration_mode"] = str(
            cfg.get("calibration_mode", settings["calibration_mode"])
        )
    if league_elo_df.empty:
        return

    scale = float(settings["prior_scale"])
    max_abs = float(settings["max_abs_prior"])
    mode = str(settings["calibration_mode"]).casefold()
    if mode not in {"global", "tiered"}:
        mode = "global"

    league_elo = league_elo_df.set_index("league")["elo"]

    def _clip(value: float) -> float:
        return max(-max_abs, min(max_abs, value))

    if mode == "tiered":
        league_tiers = {
            league: get_league_taxonomy(league)["tier"] for league in league_elo.index
        }
        tiers = league_elo.index.to_series().map(league_tiers.get)
        tier_medians = league_elo.groupby(tiers).median().to_dict()
        global_median = float(league_elo.median())

        def _median_for(league: str) -> float:
            tier = league_tiers.get(league)
            return float(tier_medians.get(tier, global_median))

        priors = {
            league: _clip((elo - _median_for(league)) * scale)
            for league, elo in league_elo.items()
        }
    else:
        median = float(league_elo.median())
        priors = {
            league: _clip((elo - median) * scale) for league, elo in league_elo.items()
        }
    try:
        with LEAGUE_STRENGTH_PRIORS.open("w") as f:
            json.dump(priors, f, indent=2)
        logger.info("Stored league strength priors to %s.", LEAGUE_STRENGTH_PRIORS)
        data_pipeline_logger.info(
            "Stored league strength priors to %s.", LEAGUE_STRENGTH_PRIORS
        )
    except Exception as e:
        logger.error("Failed to save league strength priors: %s", e)
        data_pipeline_logger.exception("Failed to save league strength priors.")


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def calculate_leagues_elo(
    df: pd.DataFrame,
    entity: str,
    hyperparameters_path: str = LEAGUES_ELO_HYPERPARAMETERS,
) -> pd.DataFrame:
    """Run the Leagues Elo pipeline with hyperparameter tuning."""
    df_preprocessed = preprocess_dataframe(df, entity)

    # Precompute the belonging league mapping
    belonging_league = map_team_to_league(df_preprocessed, "teamid")

    # Load or tune hyperparameters
    if Path(hyperparameters_path).exists():
        logger.info(f"Hyperparameters found at {Path(hyperparameters_path).name}")
        data_pipeline_logger.info(
            f"Hyperparameters found at {Path(hyperparameters_path).name}"
        )
        # calculate_leagues_elo(): loading best_params
        with Path(hyperparameters_path).open() as f:
            best_params = json.load(f)
    else:
        logger.info("No hyperparameters found; starting tuning.")
        data_pipeline_logger.info("No hyperparameters found; starting tuning.")
        best_params = tune_hyperparameters(
            df_preprocessed, entity, belonging_league, hyperparameters_path
        )

    return leagues_elo_computation(
        df=df_preprocessed,
        entity=entity,
        belonging_league=belonging_league,
        initial_elo=float(best_params["initial_elo"]),
        k_factor=float(best_params["k_factor"]),
        elo_divisor=float(best_params["elo_divisor"]),
        decay_factor=float(best_params["decay_factor"]),
    )
