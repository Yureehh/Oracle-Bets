from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest
from lol_bets.data_generation.ingestion.source import (
    OracleSourceReadinessError,
    inspect_oracle_source,
)

NOW = datetime(2026, 8, 10, 10, tzinfo=UTC)


def _write_year(root, year: int, *, modified_at: datetime = NOW) -> None:
    path = root / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"
    path.write_text(
        f"gameid,date,league\ngame-{year},{year}-07-30T12:00:00Z,LCK\n",
        encoding="utf-8",
    )
    timestamp = modified_at.timestamp()
    os.utime(path, (timestamp, timestamp))


def test_source_readiness_accepts_stable_fresh_three_year_snapshot(tmp_path):
    for year in (2024, 2025, 2026):
        _write_year(tmp_path, year)

    report = inspect_oracle_source(
        tmp_path,
        now=NOW,
        stability_seconds=0,
        minimum_bytes=1,
    )

    assert report.ready
    assert report.required_years == (2024, 2025, 2026)
    assert report.current_year_max_match_at == datetime(2026, 7, 30, 12, tzinfo=UTC)
    assert report.issues == ()


def test_source_readiness_rejects_stale_current_year_file(tmp_path):
    for year in (2024, 2025):
        _write_year(tmp_path, year)
    _write_year(tmp_path, 2026, modified_at=NOW - timedelta(hours=49))

    report = inspect_oracle_source(
        tmp_path,
        now=NOW,
        stability_seconds=0,
        minimum_bytes=1,
    )

    assert not report.ready
    assert "current_year_file_stale" in report.issues
    with pytest.raises(OracleSourceReadinessError, match="current_year_file_stale"):
        report.raise_if_unready()


def test_source_readiness_rejects_missing_and_placeholder_files(tmp_path):
    _write_year(tmp_path, 2024)
    _write_year(tmp_path, 2026)

    report = inspect_oracle_source(
        tmp_path,
        now=NOW,
        stability_seconds=0,
        minimum_bytes=10_000,
    )

    assert not report.ready
    assert "missing_year:2025" in report.issues
    assert "placeholder_or_incomplete:2026" in report.issues


def test_source_readiness_reports_directory_failure(tmp_path):
    report = inspect_oracle_source(
        tmp_path / "missing",
        now=NOW,
        stability_seconds=0,
    )

    assert not report.ready
    assert report.issues == ("source_directory_missing",)


def test_source_readiness_can_require_a_drive_symlink(tmp_path):
    for year in (2024, 2025, 2026):
        _write_year(tmp_path, year)

    report = inspect_oracle_source(
        tmp_path,
        now=NOW,
        stability_seconds=0,
        minimum_bytes=1,
        require_symlink=True,
    )

    assert not report.ready
    assert "source_directory_not_symlink" in report.issues
