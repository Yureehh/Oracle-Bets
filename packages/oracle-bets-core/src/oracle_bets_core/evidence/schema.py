"""Versioned SQLite schema for the Oracle Bets evidence graph."""

from __future__ import annotations

SCHEMA_VERSION = 3

EVIDENCE_TABLES = (
    "runs",
    "run_events",
    "source_snapshots",
    "identities",
    "provider_links",
    "fixtures",
    "model_versions",
    "predictions",
    "forecasts",
    "market_candidates",
    "market_snapshots",
    "proposals",
    "approvals",
    "paper_positions",
    "settlements",
    "corrections",
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS evidence_schema_version (
    version INTEGER NOT NULL,
    installed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    run_type TEXT NOT NULL,
    started_at TEXT NOT NULL,
    status TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_events (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    event_at TEXT NOT NULL,
    event_type TEXT NOT NULL,
    status TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_snapshots (
    id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES runs(id),
    provider TEXT NOT NULL,
    source_type TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    source_uri TEXT,
    schema_fingerprint TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS identities (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS provider_links (
    id TEXT PRIMARY KEY,
    identity_id TEXT NOT NULL REFERENCES identities(id),
    provider TEXT NOT NULL,
    provider_entity_id TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    content_hash TEXT NOT NULL,
    UNIQUE(provider, provider_entity_id, valid_from)
);

CREATE TABLE IF NOT EXISTS fixtures (
    id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES runs(id),
    sport TEXT NOT NULL,
    competition_id TEXT NOT NULL,
    team_a_identity_id TEXT NOT NULL REFERENCES identities(id),
    team_b_identity_id TEXT NOT NULL REFERENCES identities(id),
    start_time TEXT NOT NULL,
    best_of INTEGER CHECK (best_of IS NULL OR best_of IN (1, 2, 3, 5)),
    status TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    content_hash TEXT NOT NULL,
    CHECK (team_a_identity_id <> team_b_identity_id)
);

CREATE TABLE IF NOT EXISTS model_versions (
    id TEXT PRIMARY KEY,
    sport TEXT NOT NULL,
    target TEXT NOT NULL,
    created_at TEXT NOT NULL,
    artifact_uri TEXT NOT NULL,
    artifact_checksum TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    fixture_id TEXT NOT NULL REFERENCES fixtures(id),
    model_version_id TEXT NOT NULL REFERENCES model_versions(id),
    selection_id TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode = 'prematch'),
    created_at TEXT NOT NULL,
    probability_point TEXT NOT NULL,
    probability_lower TEXT NOT NULL,
    probability_upper TEXT NOT NULL,
    warnings_json TEXT NOT NULL CHECK (json_valid(warnings_json)),
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    content_hash TEXT NOT NULL,
    CHECK (
        CAST(probability_lower AS REAL) BETWEEN 0 AND 1
        AND CAST(probability_point AS REAL) BETWEEN 0 AND 1
        AND CAST(probability_upper AS REAL) BETWEEN 0 AND 1
        AND CAST(probability_lower AS REAL) <= CAST(probability_point AS REAL)
        AND CAST(probability_point AS REAL) <= CAST(probability_upper AS REAL)
    )
);

CREATE TABLE IF NOT EXISTS forecasts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    fixture_id TEXT NOT NULL REFERENCES fixtures(id),
    model_version_id TEXT NOT NULL REFERENCES model_versions(id),
    target TEXT NOT NULL,
    created_at TEXT NOT NULL,
    point_value TEXT NOT NULL,
    uncertainty_json TEXT NOT NULL CHECK (json_valid(uncertainty_json)),
    evidence_status TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_candidates (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    fixture_id TEXT NOT NULL REFERENCES fixtures(id),
    provider TEXT NOT NULL,
    provider_market_id TEXT NOT NULL,
    provider_selection_id TEXT,
    discovered_at TEXT NOT NULL,
    match_status TEXT NOT NULL,
    rejection_reason TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id TEXT PRIMARY KEY,
    market_candidate_id TEXT NOT NULL REFERENCES market_candidates(id),
    observed_at TEXT NOT NULL,
    sequence_number INTEGER NOT NULL CHECK (sequence_number > 0),
    intended_stake_units TEXT NOT NULL,
    expected_decimal_odds TEXT,
    available_stake_units TEXT,
    book_json TEXT NOT NULL CHECK (json_valid(book_json)),
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    content_hash TEXT NOT NULL,
    UNIQUE(market_candidate_id, sequence_number, observed_at)
);

CREATE TABLE IF NOT EXISTS proposals (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    prediction_id TEXT NOT NULL REFERENCES predictions(id),
    market_snapshot_id TEXT REFERENCES market_snapshots(id),
    created_at TEXT NOT NULL,
    state TEXT NOT NULL,
    rejection_reason TEXT,
    ruleset_version TEXT NOT NULL,
    strategy TEXT,
    stake_units TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL REFERENCES proposals(id),
    created_at TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_positions (
    id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL REFERENCES proposals(id),
    opened_at TEXT NOT NULL,
    strategy TEXT NOT NULL,
    stake_units TEXT NOT NULL,
    decimal_odds TEXT NOT NULL,
    state TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settlements (
    id TEXT PRIMARY KEY,
    paper_position_id TEXT NOT NULL REFERENCES paper_positions(id),
    settled_at TEXT NOT NULL,
    result TEXT NOT NULL,
    result_source TEXT NOT NULL,
    pnl_units TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS corrections (
    id TEXT PRIMARY KEY,
    target_table TEXT NOT NULL,
    target_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    reason TEXT NOT NULL,
    replacement_id TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    content_hash TEXT NOT NULL,
    CHECK (target_table IN (
        'runs', 'run_events', 'source_snapshots', 'identities', 'provider_links',
        'fixtures', 'model_versions', 'predictions', 'forecasts', 'market_candidates',
        'market_snapshots', 'proposals', 'approvals', 'paper_positions',
        'settlements', 'corrections'
    ))
);

CREATE INDEX IF NOT EXISTS idx_run_events_run_id ON run_events(run_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_run_id ON source_snapshots(run_id);
CREATE INDEX IF NOT EXISTS idx_provider_links_identity ON provider_links(identity_id);
CREATE INDEX IF NOT EXISTS idx_fixtures_start_time ON fixtures(start_time);
CREATE INDEX IF NOT EXISTS idx_predictions_fixture ON predictions(fixture_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_fixture ON forecasts(fixture_id);
CREATE INDEX IF NOT EXISTS idx_market_candidates_fixture ON market_candidates(fixture_id);
CREATE INDEX IF NOT EXISTS idx_market_snapshots_candidate ON market_snapshots(market_candidate_id);
CREATE INDEX IF NOT EXISTS idx_proposals_prediction ON proposals(prediction_id);
CREATE INDEX IF NOT EXISTS idx_positions_proposal ON paper_positions(proposal_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_settlements_position ON settlements(paper_position_id);
CREATE INDEX IF NOT EXISTS idx_corrections_target ON corrections(target_table, target_id);
"""


def append_only_triggers_sql() -> str:
    """Return immutable-update and immutable-delete triggers for evidence tables."""
    statements: list[str] = []
    for table in EVIDENCE_TABLES:
        statements.extend(
            (
                f"""
                CREATE TRIGGER IF NOT EXISTS prevent_{table}_update
                BEFORE UPDATE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{table} is append-only');
                END;
                """,
                f"""
                CREATE TRIGGER IF NOT EXISTS prevent_{table}_delete
                BEFORE DELETE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{table} is append-only');
                END;
                """,
            )
        )
    return "\n".join(statements)
