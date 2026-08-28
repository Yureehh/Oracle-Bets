from __future__ import annotations

from datetime import UTC, datetime

import pytest
from oracle_bets_core.config import load_product_config
from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.operations.audit import (
    build_monthly_audit,
    record_monthly_audit,
    write_monthly_audit,
)

NOW = datetime(2026, 7, 31, 22, tzinfo=UTC)
AUDIT_SCHEMA_VERSION = 2


def test_monthly_audit_records_owner_review_without_claiming_signoff(tmp_path):
    store = EvidenceStore(tmp_path / "audit.db")
    store.initialize_schema()
    config = load_product_config()
    report = build_monthly_audit(
        store,
        period="2026-07",
        config=config,
        generated_at=NOW,
    )

    run_id = record_monthly_audit(store, report)
    json_path, markdown_path = write_monthly_audit(
        report,
        json_path=tmp_path / "monthly.json",
        markdown_path=tmp_path / "monthly.md",
    )

    assert report.status == "owner_review_required"
    assert report.product_config_changed is None
    assert report.to_dict()["owner_signoff_required"] is True
    assert report.to_dict()["schema_version"] == AUDIT_SCHEMA_VERSION
    assert report.strategy_cohorts["enrolled"] == 0
    assert report.bet_performance["paper"]["settled"] == 0
    assert store.get(EvidenceTable.RUNS, run_id)["run_type"] == "monthly_audit"
    assert store.count(EvidenceTable.SOURCE_SNAPSHOTS) == 1
    assert json_path.is_file()
    assert "Required owner review" in markdown_path.read_text()


def test_next_monthly_audit_compares_versioned_product_rules(tmp_path):
    store = EvidenceStore(tmp_path / "audit.db")
    store.initialize_schema()
    config = load_product_config()
    first = build_monthly_audit(
        store,
        period="2026-06",
        config=config,
        generated_at=datetime(2026, 6, 30, tzinfo=UTC),
    )
    record_monthly_audit(store, first)

    second = build_monthly_audit(
        store,
        period="2026-07",
        config=config,
        generated_at=NOW,
    )

    assert second.product_config_changed is False


def test_monthly_audit_rejects_ambiguous_period(tmp_path):
    store = EvidenceStore(tmp_path / "audit.db")
    store.initialize_schema()

    with pytest.raises(ValueError, match="YYYY-MM"):
        build_monthly_audit(
            store,
            period="July 2026",
            config=load_product_config(),
            generated_at=NOW,
        )
