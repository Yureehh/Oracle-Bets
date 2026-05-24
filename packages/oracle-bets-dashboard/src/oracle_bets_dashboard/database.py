"""SQLite bet ledger schema, validation, migrations, and CRUD."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar

from oracle_bets_core.paths import SUITE_ROOT

DB_PATH = SUITE_ROOT / "data" / "ledger.db"

SCHEMA_VERSION = 2

BET_STATUSES = {"pending", "settled"}
BET_RESULTS = {"win", "loss", "push", "void"}
OVER_UNDER_VALUES = {"Over", "Under"}
SIDES = {"Blue", "Red", "Home", "Away", "Over", "Under"}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),

    -- Core bet info
    event_date TEXT NOT NULL,
    sport TEXT NOT NULL,
    league TEXT,
    event_name TEXT NOT NULL,
    market_type TEXT NOT NULL,
    selection TEXT NOT NULL,
    side TEXT,

    -- Odds & stake
    odds_decimal REAL NOT NULL,
    odds_format TEXT NOT NULL DEFAULT 'decimal',
    stake REAL,
    potential_payout REAL,
    currency TEXT NOT NULL DEFAULT 'EUR',

    -- Model signals (from Oracle Bets predictions)
    model_probability REAL,
    implied_probability REAL,
    edge REAL,
    kelly_fraction REAL,
    confidence_tier TEXT,

    -- Outcome
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT,
    pnl REAL,
    settled_at TEXT,

    -- Bet classification
    is_smoke INTEGER NOT NULL DEFAULT 0,
    is_live INTEGER NOT NULL DEFAULT 0,
    is_parlay INTEGER NOT NULL DEFAULT 0,
    parlay_id INTEGER,

    -- Source & context
    bookmaker TEXT,
    bet_slip_ref TEXT,
    idempotency_key TEXT,
    source_prediction TEXT,
    notes TEXT,
    tags TEXT,

    -- Over/under specifics
    line REAL,
    over_under TEXT,

    -- Series context (BO3/BO5)
    best_of INTEGER,
    map_number INTEGER
);

CREATE TABLE IF NOT EXISTS parlays (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),
    event_date TEXT NOT NULL,
    combined_odds REAL NOT NULL,
    stake REAL,
    potential_payout REAL,
    currency TEXT NOT NULL DEFAULT 'EUR',
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT,
    pnl REAL,
    is_smoke INTEGER NOT NULL DEFAULT 0,
    bookmaker TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS bankroll_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),
    balance REAL NOT NULL,
    deposit REAL DEFAULT 0,
    withdrawal REAL DEFAULT 0,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    color TEXT
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bets_event_date ON bets(event_date);
CREATE INDEX IF NOT EXISTS idx_bets_sport ON bets(sport);
CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status);
CREATE INDEX IF NOT EXISTS idx_bets_is_smoke ON bets(is_smoke);
CREATE INDEX IF NOT EXISTS idx_bets_bookmaker ON bets(bookmaker);
CREATE INDEX IF NOT EXISTS idx_bets_market_type ON bets(market_type);
CREATE INDEX IF NOT EXISTS idx_bets_parlay_id ON bets(parlay_id);
"""


BET_COLUMNS = {
    "event_date",
    "sport",
    "league",
    "event_name",
    "market_type",
    "selection",
    "side",
    "odds_decimal",
    "odds_format",
    "stake",
    "potential_payout",
    "currency",
    "model_probability",
    "implied_probability",
    "edge",
    "kelly_fraction",
    "confidence_tier",
    "status",
    "result",
    "pnl",
    "settled_at",
    "is_smoke",
    "is_live",
    "is_parlay",
    "parlay_id",
    "bookmaker",
    "bet_slip_ref",
    "idempotency_key",
    "source_prediction",
    "notes",
    "tags",
    "line",
    "over_under",
    "best_of",
    "map_number",
}
PARLAY_COLUMNS = {
    "event_date",
    "combined_odds",
    "stake",
    "potential_payout",
    "currency",
    "status",
    "result",
    "pnl",
    "is_smoke",
    "bookmaker",
    "notes",
}


@dataclass(frozen=True)
class BetCreate:
    event_date: str
    sport: str
    event_name: str
    market_type: str
    selection: str
    odds_decimal: float
    league: str | None = None
    side: str | None = None
    stake: float | None = None
    currency: str = "EUR"
    model_probability: float | None = None
    implied_probability: float | None = None
    edge: float | None = None
    kelly_fraction: float | None = None
    confidence_tier: str | None = None
    is_smoke: bool = False
    is_live: bool = False
    is_parlay: bool = False
    parlay_id: int | None = None
    bookmaker: str | None = None
    bet_slip_ref: str | None = None
    source_prediction: str | None = None
    notes: str | None = None
    tags: str | None = None
    line: float | None = None
    over_under: str | None = None
    best_of: int | None = None
    map_number: int | None = None

    REQUIRED_TEXT: ClassVar[tuple[str, ...]] = (
        "event_date",
        "sport",
        "event_name",
        "market_type",
        "selection",
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BetCreate:
        unknown = set(data) - BET_COLUMNS
        if unknown:
            msg = f"Unknown bet field(s): {', '.join(sorted(unknown))}"
            raise ValueError(msg)
        payload = dict(data)
        payload.pop("potential_payout", None)
        payload.pop("implied_probability", None)
        payload.pop("idempotency_key", None)
        payload.pop("odds_format", None)
        payload.pop("status", None)
        return cls(**payload)

    def validated(self) -> dict[str, Any]:
        for field in self.REQUIRED_TEXT:
            if not str(getattr(self, field) or "").strip():
                msg = f"{field} is required."
                raise ValueError(msg)
        if self.odds_decimal <= 1:
            msg = "Decimal odds must be greater than 1."
            raise ValueError(msg)
        if self.stake is not None and self.stake < 0:
            msg = "Stake cannot be negative."
            raise ValueError(msg)
        for field in ("model_probability", "implied_probability"):
            value = getattr(self, field)
            if value is not None and not 0 <= value <= 1:
                msg = f"{field} must be between 0 and 1."
                raise ValueError(msg)
        if self.side is not None and self.side not in SIDES:
            msg = f"side must be one of: {', '.join(sorted(SIDES))}."
            raise ValueError(msg)
        if self.over_under is not None and self.over_under not in OVER_UNDER_VALUES:
            msg = "over_under must be Over or Under."
            raise ValueError(msg)
        if self.map_number is not None and self.map_number <= 0:
            msg = "map_number must be positive."
            raise ValueError(msg)
        if self.best_of is not None and self.best_of not in {1, 2, 3, 5}:
            msg = "best_of must be 1, 2, 3, or 5."
            raise ValueError(msg)

        data = {
            "event_date": self.event_date,
            "sport": self.sport,
            "league": self.league,
            "event_name": self.event_name,
            "market_type": self.market_type,
            "selection": self.selection,
            "side": self.side,
            "odds_decimal": self.odds_decimal,
            "odds_format": "decimal",
            "stake": self.stake,
            "potential_payout": self.stake * self.odds_decimal
            if self.stake is not None
            else None,
            "currency": self.currency,
            "model_probability": self.model_probability,
            "implied_probability": self.implied_probability
            if self.implied_probability is not None
            else 1.0 / self.odds_decimal,
            "edge": self.edge,
            "kelly_fraction": self.kelly_fraction,
            "confidence_tier": self.confidence_tier,
            "is_smoke": int(self.is_smoke),
            "is_live": int(self.is_live),
            "is_parlay": int(self.is_parlay),
            "parlay_id": self.parlay_id,
            "bookmaker": self.bookmaker,
            "bet_slip_ref": self.bet_slip_ref,
            "source_prediction": self.source_prediction,
            "notes": self.notes,
            "tags": self.tags,
            "line": self.line,
            "over_under": self.over_under,
            "best_of": self.best_of,
            "map_number": self.map_number,
        }
        data["idempotency_key"] = self.bet_slip_ref or _hash_bet(data)
        return {key: value for key, value in data.items() if value is not None}


@dataclass(frozen=True)
class BetSettle:
    result: str

    def validated(self, bet: dict[str, Any]) -> dict[str, Any]:
        result = self.result.casefold()
        if result not in BET_RESULTS:
            msg = f"result must be one of: {', '.join(sorted(BET_RESULTS))}."
            raise ValueError(msg)
        stake = float(bet.get("stake") or 0)
        odds = float(bet["odds_decimal"])
        pnl = 0.0
        if result == "win":
            pnl = stake * (odds - 1)
        elif result == "loss":
            pnl = -stake
        return {
            "status": "settled",
            "result": result,
            "pnl": pnl,
            "settled_at": utc_now_iso(),
        }


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _hash_bet(data: dict[str, Any]) -> str:
    parts = [
        str(data.get(key) or "")
        for key in (
            "event_date",
            "sport",
            "league",
            "event_name",
            "market_type",
            "selection",
            "side",
            "odds_decimal",
            "stake",
            "bookmaker",
            "line",
            "over_under",
            "best_of",
            "map_number",
        )
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DB_PATH


@contextmanager
def get_connection():
    db = _ensure_db()
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)
        _migrate(conn)
        existing = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
            )
        elif int(existing["version"]) != SCHEMA_VERSION:
            conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))


def _migrate(conn: sqlite3.Connection) -> None:
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(bets)").fetchall()
    }
    if "idempotency_key" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN idempotency_key TEXT")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_bets_idempotency_key ON bets(idempotency_key)"
    )


def insert_bet(data: dict[str, Any]) -> int:
    validated = BetCreate.from_dict(data).validated()
    columns = ", ".join(validated.keys())
    placeholders = ", ".join(["?"] * len(validated))
    with get_connection() as conn:
        try:
            cursor = conn.execute(
                f"INSERT INTO bets ({columns}) VALUES ({placeholders})",  # noqa: S608
                list(validated.values()),
            )
        except sqlite3.IntegrityError as e:
            msg = "Duplicate bet ignored: bet_slip_ref/idempotency_key already exists."
            raise ValueError(msg) from e
        return int(cursor.lastrowid)


def update_bet(bet_id: int, data: dict[str, Any]) -> None:
    unknown = set(data) - BET_COLUMNS
    if unknown:
        msg = f"Unknown bet field(s): {', '.join(sorted(unknown))}"
        raise ValueError(msg)
    if (status := data.get("status")) and status not in BET_STATUSES:
        msg = f"status must be one of: {', '.join(sorted(BET_STATUSES))}."
        raise ValueError(msg)
    if (result := data.get("result")) and result not in BET_RESULTS:
        msg = f"result must be one of: {', '.join(sorted(BET_RESULTS))}."
        raise ValueError(msg)
    payload = dict(data)
    payload["updated_at"] = utc_now_iso()
    set_clause = ", ".join(f"{key} = ?" for key in payload)
    with get_connection() as conn:
        conn.execute(
            f"UPDATE bets SET {set_clause} WHERE id = ?",  # noqa: S608
            [*list(payload.values()), bet_id],
        )


def settle_bet(bet_id: int, result: str) -> dict[str, Any]:
    bet = get_bet(bet_id)
    if bet is None:
        msg = f"Bet #{bet_id} not found."
        raise ValueError(msg)
    update = BetSettle(result=result).validated(bet)
    update_bet(bet_id, update)
    return {**bet, **update}


def delete_bet(bet_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM bets WHERE id = ?", (bet_id,))


def get_all_bets() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM bets ORDER BY event_date DESC").fetchall()
        return [dict(row) for row in rows]


def get_bet(bet_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()
        return dict(row) if row else None


def insert_bankroll_entry(
    balance: float, deposit: float = 0, withdrawal: float = 0, notes: str = ""
) -> int:
    if balance < 0 or deposit < 0 or withdrawal < 0:
        msg = "Balance, deposit, and withdrawal cannot be negative."
        raise ValueError(msg)
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO bankroll_log (balance, deposit, withdrawal, notes) VALUES (?, ?, ?, ?)",
            (balance, deposit, withdrawal, notes),
        )
        return cursor.lastrowid


def get_bankroll_history() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM bankroll_log ORDER BY logged_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def insert_parlay(data: dict[str, Any]) -> int:
    unknown = set(data) - PARLAY_COLUMNS
    if unknown:
        msg = f"Unknown parlay field(s): {', '.join(sorted(unknown))}"
        raise ValueError(msg)
    columns = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))
    with get_connection() as conn:
        cursor = conn.execute(
            f"INSERT INTO parlays ({columns}) VALUES ({placeholders})",  # noqa: S608
            list(data.values()),
        )
        return cursor.lastrowid


def get_summary_stats(smoke: bool | None = None) -> dict[str, Any]:
    """Aggregate stats for the dashboard overview."""
    where = ""
    params: list[Any] = []
    if smoke is not None:
        where = "WHERE is_smoke = ?"
        params = [int(smoke)]

    with get_connection() as conn:
        row = conn.execute(
            f"""
            SELECT
                COUNT(*) as total_bets,
                SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN result = 'push' THEN 1 ELSE 0 END) as pushes,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                SUM(pnl) as total_pnl,
                SUM(stake) as total_staked,
                AVG(odds_decimal) as avg_odds,
                AVG(edge) as avg_edge,
                AVG(CASE WHEN result = 'win' THEN odds_decimal ELSE NULL END) as avg_winning_odds
            FROM bets {where}
            """,  # noqa: S608
            params,
        ).fetchone()
        return dict(row) if row else {}
