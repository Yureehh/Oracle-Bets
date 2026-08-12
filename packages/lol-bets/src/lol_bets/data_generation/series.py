"""Deterministic, leakage-safe reconstruction of historical LoL series."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from oracle_bets_core.paths import (
    NEXT_MAP_PLAYER_DATA,
    NEXT_MAP_TEAM_DATA,
    RAW_DATA,
    SERIES_MANIFEST,
    SERIES_REJECTIONS,
    SERIES_WINNER_PLAYER_DATA,
    SERIES_WINNER_TEAM_DATA,
    TRAINING_PLAYER_DATA,
    TRAINING_TEAM_DATA,
)
from oracle_bets_core.pd import pd

from lol_bets.data_generation.ingestion.quality import normalize_result

MAX_MAP_GAP = timedelta(hours=6)
MIN_BO1_PHASE_SERIES = 20
MIN_BO1_PHASE_RATIO = 0.95
TEAM_POSITION = "team"
TEAM_ROWS_PER_MAP = 2
PLAYER_ROWS_PER_MAP = 10
MIN_MULTI_MAP_SERIES = 2
PREMATCH_SERIES_STATE_COLUMNS = frozenset(
    {
        "game_in_series",
        "is_deciding_game",
        "maps_completed",
        "next_map_number",
        "series_wins_before",
        "series_losses_before",
        "series_score_delta",
    }
)
BO3_WINS = 2
BO5_WINS = 3


@dataclass(frozen=True)
class SeriesBuildResult:
    """Accepted series and explicit reconstruction failures."""

    series: pd.DataFrame
    rejections: pd.DataFrame


def reconstruct_series(team_rows: pd.DataFrame) -> SeriesBuildResult:
    """Reconstruct complete BO1/BO3/BO5 series from two team rows per map."""
    required = {
        "date",
        "gameid",
        "league",
        "split",
        "playoffs",
        "game",
        "teamid",
        "teamname",
        "result",
    }
    missing = required - set(team_rows.columns)
    if missing:
        raise ValueError(f"series source is missing columns: {sorted(missing)}")

    maps, invalid_maps = _map_records(team_rows)
    candidates: list[dict[str, Any]] = []
    rejections = list(invalid_maps)
    phase_columns = ["league", "season", "split", "playoffs", "team_pair"]
    for _, raw_phase_maps in maps.groupby(phase_columns, dropna=False, sort=False):
        phase_maps = raw_phase_maps.sort_values(["date", "game", "gameid"])
        current: list[dict[str, Any]] = []
        for row in phase_maps.to_dict(orient="records"):
            map_number = int(row["game"])
            if map_number == 1:
                if current:
                    candidates.append(_candidate(current))
                current = [row]
                continue
            if not current:
                rejections.append(_rejection([row], "series_does_not_start_at_map_1"))
                continue
            previous = current[-1]
            gap = row["date"] - previous["date"]
            if map_number != int(previous["game"]) + 1 or gap > MAX_MAP_GAP:
                candidates.append(_candidate(current))
                current = []
                reason = (
                    "map_gap_exceeds_six_hours"
                    if gap > MAX_MAP_GAP
                    else "non_sequential_map_number"
                )
                rejections.append(_rejection([row], reason))
                continue
            current.append(row)
        if current:
            candidates.append(_candidate(current))

    phase_counts: dict[tuple[Any, ...], tuple[int, int]] = {}
    for candidate in candidates:
        phase = tuple(candidate[key] for key in phase_columns[:-1])
        total, singleton = phase_counts.get(phase, (0, 0))
        phase_counts[phase] = (
            total + 1,
            singleton + int(len(candidate["maps"]) == 1),
        )

    accepted: list[dict[str, Any]] = []
    for candidate in candidates:
        result, reason = _classify_candidate(candidate, phase_counts)
        if result is None:
            rejections.append(_rejection(candidate["maps"], reason or "ambiguous"))
        else:
            accepted.append(result)
    return SeriesBuildResult(pd.DataFrame(accepted), pd.DataFrame(rejections))


def build_series_artifacts() -> dict[str, Any]:
    """Build series manifests plus Map-1-frozen supervised tables."""
    raw = pd.read_parquet(RAW_DATA)
    if "position" in raw.columns:
        raw = raw.loc[raw["position"].astype(str).str.casefold() == TEAM_POSITION]
    result = reconstruct_series(raw)
    team = pd.read_parquet(TRAINING_TEAM_DATA)
    players = pd.read_parquet(TRAINING_PLAYER_DATA)
    series_team, series_players = _series_winner_training_tables(
        result.series, team, players
    )
    next_team, next_players = _next_map_training_tables(result.series, team, players)

    for path, frame in (
        (SERIES_MANIFEST, result.series),
        (SERIES_REJECTIONS, result.rejections),
        (SERIES_WINNER_TEAM_DATA, series_team),
        (SERIES_WINNER_PLAYER_DATA, series_players),
        (NEXT_MAP_TEAM_DATA, next_team),
        (NEXT_MAP_PLAYER_DATA, next_players),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    return {
        "accepted_series": int(len(result.series)),
        "rejected_groups": int(len(result.rejections)),
        "series_winner_team_rows": int(len(series_team)),
        "next_map_team_rows": int(len(next_team)),
        "manifest": str(SERIES_MANIFEST),
        "rejections": str(SERIES_REJECTIONS),
    }


def _map_records(team_rows: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    source = team_rows.copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce", utc=True)
    source["game"] = pd.to_numeric(source["game"], errors="coerce")
    source["result"] = normalize_result(source["result"])
    source["season"] = source.get("season", source["date"].dt.year)
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for gameid, raw_rows in source.groupby("gameid", dropna=False, sort=False):
        rows = raw_rows.drop_duplicates(subset=["teamid"], keep="last")
        ids = tuple(sorted(rows["teamid"].dropna().astype(str).unique()))
        valid = (
            len(rows) == TEAM_ROWS_PER_MAP
            and len(ids) == TEAM_ROWS_PER_MAP
            and rows["date"].notna().all()
            and rows["game"].notna().all()
            and rows["result"].isin([0, 1]).all()
            and float(rows["result"].sum()) == 1.0
        )
        if not valid:
            rejected.append(
                {
                    "series_id": None,
                    "reason": "invalid_map_identity_or_result",
                    "source_map_ids": json.dumps([str(gameid)]),
                }
            )
            continue
        winner = str(rows.loc[rows["result"] == 1, "teamid"].iloc[0])
        first = rows.iloc[0]
        records.append(
            {
                "gameid": str(gameid),
                "date": rows["date"].min(),
                "league": str(first["league"]),
                "season": str(first["season"]),
                "split": str(first["split"]),
                "playoffs": bool(first["playoffs"]),
                "game": int(first["game"]),
                "team_pair": ids,
                "team_names": tuple(
                    rows.set_index(rows["teamid"].astype(str))["teamname"]
                    .reindex(ids)
                    .astype(str)
                ),
                "winner_teamid": winner,
            }
        )
    return pd.DataFrame(records), rejected


def _candidate(maps: list[dict[str, Any]]) -> dict[str, Any]:
    first = maps[0]
    return {
        "league": first["league"],
        "season": first["season"],
        "split": first["split"],
        "playoffs": first["playoffs"],
        "team_pair": first["team_pair"],
        "team_names": first["team_names"],
        "maps": maps,
    }


def _classify_candidate(
    candidate: dict[str, Any],
    phase_counts: dict[tuple[Any, ...], tuple[int, int]],
) -> tuple[dict[str, Any] | None, str | None]:
    maps = candidate["maps"]
    map_numbers = [int(row["game"]) for row in maps]
    if map_numbers != list(range(1, len(maps) + 1)):
        return None, "non_sequential_map_number"
    wins = dict.fromkeys(candidate["team_pair"], 0)
    for row in maps:
        winner = row["winner_teamid"]
        if winner not in wins:
            return None, "team_identity_changed_within_series"
        wins[winner] += 1
    winner, win_count = max(wins.items(), key=lambda item: item[1])
    if len(maps) == 1:
        phase = tuple(
            candidate[key] for key in ("league", "season", "split", "playoffs")
        )
        total, singleton = phase_counts[phase]
        if total < MIN_BO1_PHASE_SERIES or singleton / total < MIN_BO1_PHASE_RATIO:
            return None, "single_map_phase_not_verified_as_bo1"
        best_of = 1
    elif win_count == BO3_WINS and len(maps) in {2, 3}:
        best_of = 3
    elif win_count == BO5_WINS and len(maps) in {3, 4, 5}:
        best_of = 5
    else:
        return None, "incomplete_or_ambiguous_series"

    source_ids = [str(row["gameid"]) for row in maps]
    identity = "|".join(
        [
            str(candidate["league"]),
            str(candidate["season"]),
            str(candidate["split"]),
            str(candidate["playoffs"]),
            *candidate["team_pair"],
            *source_ids,
        ]
    )
    series_id = "series-" + hashlib.sha256(identity.encode()).hexdigest()[:24]
    return (
        {
            "series_id": series_id,
            "date": maps[0]["date"],
            "league": candidate["league"],
            "season": candidate["season"],
            "split": candidate["split"],
            "playoffs": candidate["playoffs"],
            "best_of": best_of,
            "map_count": len(maps),
            "team_a_id": candidate["team_pair"][0],
            "team_b_id": candidate["team_pair"][1],
            "team_a_name": candidate["team_names"][0],
            "team_b_name": candidate["team_names"][1],
            "winner_teamid": winner,
            "source_map_ids": json.dumps(source_ids),
            "map_winner_ids": json.dumps([str(row["winner_teamid"]) for row in maps]),
            "reconstruction_evidence": json.dumps(
                {
                    "sequential_maps": True,
                    "unchanged_team_pair": True,
                    "maximum_gap_hours": MAX_MAP_GAP.total_seconds() / 3600,
                },
                sort_keys=True,
            ),
        },
        None,
    )


def _rejection(maps: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    return {
        "series_id": None,
        "reason": reason,
        "source_map_ids": json.dumps([str(row["gameid"]) for row in maps]),
    }


def _series_winner_training_tables(
    manifest: pd.DataFrame,
    teams: pd.DataFrame,
    players: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_frames: list[pd.DataFrame] = []
    player_frames: list[pd.DataFrame] = []
    for row in manifest.to_dict(orient="records"):
        first_map = json.loads(row["source_map_ids"])[0]
        team_frame = teams.loc[teams["gameid"].astype(str) == first_map].copy()
        player_frame = players.loc[players["gameid"].astype(str) == first_map].copy()
        if (
            len(team_frame) != TEAM_ROWS_PER_MAP
            or len(player_frame) != PLAYER_ROWS_PER_MAP
        ):
            continue
        for frame in (team_frame, player_frame):
            # The series winner is priced before Map 1. These columns belong to
            # the separate next-map experiment and must not survive even when a
            # source table happens to contain them.
            frame.drop(
                columns=PREMATCH_SERIES_STATE_COLUMNS,
                errors="ignore",
                inplace=True,
            )
            frame["source_gameid"] = first_map
            frame["series_id"] = row["series_id"]
            frame["gameid"] = row["series_id"]
            frame["best_of"] = int(row["best_of"])
            frame["result"] = (
                frame["teamid"].astype(str) == str(row["winner_teamid"])
            ).astype(int)
        team_frames.append(team_frame)
        player_frames.append(player_frame)
    return _concat_like(team_frames, teams), _concat_like(player_frames, players)


def _next_map_training_tables(
    manifest: pd.DataFrame,
    teams: pd.DataFrame,
    players: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_frames: list[pd.DataFrame] = []
    player_frames: list[pd.DataFrame] = []
    for row in manifest.to_dict(orient="records"):
        map_ids = json.loads(row["source_map_ids"])
        winners = json.loads(row["map_winner_ids"])
        if len(map_ids) < MIN_MULTI_MAP_SERIES:
            continue
        first_teams = teams.loc[teams["gameid"].astype(str) == map_ids[0]].copy()
        first_players = players.loc[players["gameid"].astype(str) == map_ids[0]].copy()
        if (
            len(first_teams) != TEAM_ROWS_PER_MAP
            or len(first_players) != PLAYER_ROWS_PER_MAP
        ):
            continue
        score = {str(row["team_a_id"]): 0, str(row["team_b_id"]): 0}
        score[str(winners[0])] += 1
        for index in range(1, len(map_ids)):
            map_number = index + 1
            target_winner = str(winners[index])
            synthetic_id = f"{row['series_id']}:map-{map_number}"
            for base, output in (
                (first_teams, team_frames),
                (first_players, player_frames),
            ):
                frame = base.copy()
                team_ids = frame["teamid"].astype(str)
                frame["source_gameid"] = map_ids[0]
                frame["target_gameid"] = map_ids[index]
                frame["series_id"] = row["series_id"]
                frame["gameid"] = synthetic_id
                frame["best_of"] = int(row["best_of"])
                frame["next_map_number"] = map_number
                frame["maps_completed"] = index
                frame["series_wins_before"] = team_ids.map(score).astype(int)
                total_maps = sum(score.values())
                losses = {team_id: total_maps - wins for team_id, wins in score.items()}
                frame["series_losses_before"] = team_ids.map(losses)
                frame["series_score_delta"] = (
                    frame["series_wins_before"] - frame["series_losses_before"]
                )
                frame["result"] = (team_ids == target_winner).astype(int)
                output.append(frame)
            score[target_winner] += 1
    return _concat_like(team_frames, teams), _concat_like(player_frames, players)


def _concat_like(frames: list[pd.DataFrame], source: pd.DataFrame) -> pd.DataFrame:
    if not frames:
        return source.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)
