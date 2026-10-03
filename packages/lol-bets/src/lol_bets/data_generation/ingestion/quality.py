"""Oracle's Elixir schema, duplicate, quarantine, and quality reporting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from oracle_bets_core.paths import IMPORT_COLUMNS
from oracle_bets_core.pd import pd

if TYPE_CHECKING:
    from collections.abc import Iterable


EXPECTED_RAW_COLUMNS = frozenset(
    {
        "date",
        "gameid",
        "league",
        "patch",
        "side",
        "position",
        "result",
        "teamname",
        "teamid",
        "playername",
        "playerid",
    }
)
EXPECTED_SIDES = {"Blue", "Red"}
EXPECTED_POSITIONS = {"top", "jng", "mid", "bot", "sup"}
ROWS_PER_GAME = 12
PLAYERS_PER_GAME = 10
TEAMS_PER_GAME = 2
SOURCE_METADATA_COLUMNS = frozenset({"datacompleteness", "split"})
SOURCE_COLUMN_RENAMES = {
    "earned gpm": "egpm",
    "team kpm": "team_kpm",
    "total cs": "total_cs",
    "firstPick": "first_pick",
}


class SourceSchemaError(RuntimeError):
    """Raised when an upstream frame no longer satisfies the required schema."""


@dataclass(frozen=True)
class DataQualityReport:
    source: str
    generated_at: str
    schema_fingerprint: str
    input_rows: int
    input_games: int
    exact_duplicate_rows: int
    accepted_rows: int
    accepted_games: int
    quarantined_rows: int
    quarantined_games: int
    missing_values: dict[str, int]
    column_reconciliation: tuple[dict[str, str], ...]
    game_issues: dict[str, tuple[str, ...]]
    abnormal_games: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "generated_at": self.generated_at,
            "schema_fingerprint": self.schema_fingerprint,
            "input_rows": self.input_rows,
            "input_games": self.input_games,
            "exact_duplicate_rows": self.exact_duplicate_rows,
            "accepted_rows": self.accepted_rows,
            "accepted_games": self.accepted_games,
            "quarantined_rows": self.quarantined_rows,
            "quarantined_games": self.quarantined_games,
            "missing_values": self.missing_values,
            "column_reconciliation": list(self.column_reconciliation),
            "game_issues": {
                game_id: list(reasons)
                for game_id, reasons in sorted(self.game_issues.items())
            },
            "abnormal_games": dict(sorted(self.abnormal_games.items())),
        }


def schema_fingerprint(df: pd.DataFrame) -> str:
    """Fingerprint ordered column names and pandas dtypes."""
    schema = [
        {"position": position, "name": str(column), "dtype": str(df[column].dtype)}
        for position, column in enumerate(df.columns)
    ]
    serialized = json.dumps(schema, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def source_column_reconciliation(
    df: pd.DataFrame,
) -> tuple[dict[str, str], ...]:
    """Classify configured and newly observed Oracle's Elixir columns."""
    configured = json.loads(IMPORT_COLUMNS.read_text())
    feature_columns = set(configured["team"]) | set(configured["player"])
    known_columns = (
        feature_columns | set(EXPECTED_RAW_COLUMNS) | set(SOURCE_METADATA_COLUMNS)
    )
    actual_columns = {
        SOURCE_COLUMN_RENAMES.get(column, column) for column in df.columns
    }

    rows: list[dict[str, str]] = []
    for column in sorted(known_columns | actual_columns):
        if column in actual_columns:
            status = "present" if column in known_columns else "new"
        else:
            status = "missing"

        if column in EXPECTED_RAW_COLUMNS:
            disposition = "required"
        elif column in SOURCE_METADATA_COLUMNS:
            disposition = "metadata"
        elif column in feature_columns:
            disposition = "feature"
        else:
            disposition = "candidate"
        rows.append(
            {
                "column": column,
                "source_status": status,
                "disposition": disposition,
            }
        )
    return tuple(rows)


def _validate_schema(df: pd.DataFrame) -> None:
    missing = EXPECTED_RAW_COLUMNS - set(df.columns)
    if missing:
        msg = f"Oracle's Elixir schema is missing required columns: {sorted(missing)}"
        raise SourceSchemaError(msg)


def normalize_result(values: pd.Series) -> pd.Series:
    """Normalize supported Oracle's Elixir result labels to float 0/1 values."""
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce")
    return values.map(
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
    ).astype("float64")


def _issues_for_game(game: pd.DataFrame) -> tuple[str, ...]:
    issues: list[str] = []
    positions = game["position"].fillna("").astype(str).str.casefold()
    team_rows = game[positions.eq("team")]
    player_rows = game[~positions.eq("team")]
    if len(game) != ROWS_PER_GAME:
        issues.append("wrong_row_count")
    if len(team_rows) != TEAMS_PER_GAME:
        issues.append("wrong_team_row_count")
    if len(player_rows) != PLAYERS_PER_GAME:
        issues.append("wrong_player_row_count")
    if set(game["side"].dropna().astype(str)) != EXPECTED_SIDES:
        issues.append("wrong_side_composition")
    team_ids = game["teamid"].fillna(game["teamname"])
    if team_ids.nunique(dropna=True) != TEAMS_PER_GAME:
        issues.append("wrong_team_identity_count")

    if not player_rows.empty:
        for _, side_rows in player_rows.groupby("side", observed=True):
            if (
                set(side_rows["position"].astype(str).str.casefold())
                != EXPECTED_POSITIONS
            ):
                issues.append("wrong_player_role_composition")
                break

    if len(team_rows) == TEAMS_PER_GAME:
        results = normalize_result(team_rows["result"])
        if results.isna().any() or float(results.sum()) != 1.0:
            issues.append("wrong_result_composition")

    names = pd.concat([game["teamname"], game["playername"]])
    if names.fillna("").astype(str).str.contains("unknown", case=False).any():
        issues.append("unknown_entity")
    return tuple(dict.fromkeys(issues))


def _abnormal_reason(game: pd.DataFrame) -> str | None:
    for column in ("game_status", "status", "notes"):
        if column not in game.columns:
            continue
        joined = " ".join(game[column].dropna().astype(str)).casefold()
        for token in ("forfeit", "remake", "abandoned"):
            if token in joined:
                return token
    return None


def quarantine_oracles_elixir_data(
    df: pd.DataFrame,
    *,
    manual_invalid_games: Iterable[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, DataQualityReport]:
    """Merge exact duplicates and isolate malformed or manually excluded games."""
    _validate_schema(df)
    duplicate_mask = df.duplicated(keep="first")
    deduplicated = df.loc[~duplicate_mask].copy()
    manual = {str(game_id) for game_id in manual_invalid_games}
    game_issues: dict[str, tuple[str, ...]] = {}
    abnormal_games: dict[str, str] = {}

    missing_id_mask = deduplicated["gameid"].isna()
    if missing_id_mask.any():
        game_issues["__missing_gameid__"] = ("missing_game_id",)

    for game_id, game in deduplicated.loc[~missing_id_mask].groupby(
        "gameid", observed=True
    ):
        game_key = str(game_id)
        reasons = (
            ("manual_invalid_game",) if game_key in manual else _issues_for_game(game)
        )
        if reasons:
            game_issues[game_key] = reasons
        abnormal = _abnormal_reason(game)
        if abnormal is not None:
            abnormal_games[game_key] = abnormal
            current = game_issues.get(game_key, ())
            game_issues[game_key] = (*current, f"abnormal_{abnormal}")

    invalid_ids = set(game_issues) - {"__missing_gameid__"}
    quarantine_mask = missing_id_mask | deduplicated["gameid"].astype(str).isin(
        invalid_ids
    )
    accepted = deduplicated.loc[~quarantine_mask].reset_index(drop=True)
    quarantined = deduplicated.loc[quarantine_mask].reset_index(drop=True)
    missing_values = {
        column: int(df[column].isna().sum()) for column in sorted(EXPECTED_RAW_COLUMNS)
    }
    report = DataQualityReport(
        source="oracles_elixir",
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        schema_fingerprint=schema_fingerprint(df),
        input_rows=len(df),
        input_games=int(df["gameid"].nunique(dropna=True)),
        exact_duplicate_rows=int(duplicate_mask.sum()),
        accepted_rows=len(accepted),
        accepted_games=int(accepted["gameid"].nunique(dropna=True)),
        quarantined_rows=len(quarantined),
        quarantined_games=int(quarantined["gameid"].nunique(dropna=True)),
        missing_values=missing_values,
        column_reconciliation=source_column_reconciliation(df),
        game_issues=game_issues,
        abnormal_games=abnormal_games,
    )
    return accepted, quarantined, report


def write_quality_report(
    report: DataQualityReport,
    destination: str | Path,
) -> Path:
    """Write a deterministic JSON report through an atomic replace."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n"
    )
    temporary.replace(path)
    return path
