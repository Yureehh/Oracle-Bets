from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from lol_bets.data_generation.ingestion.source import (
    PUBLIC_DRIVE_FILE_IDS,
    PUBLIC_DRIVE_FOLDER_URL,
    TAKEOUT_EXPORTS_URL,
    OracleSourceReadinessError,
    OracleSourceRefreshError,
    inspect_oracle_source,
    oracle_source_directory,
    refresh_oracle_source,
)

NOW = datetime(2026, 8, 10, 10, tzinfo=UTC)
POINTER_WRITE_CALL = 2
EXPECTED_GENERATIONS = 2


@dataclass
class _Source:
    content: bytes
    filename: str


class _Response:
    def __init__(self, *, text="", payload=None, content=b""):
        self.text = text
        self.payload = payload
        self.content = content

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        del chunk_size
        yield self.content

    def json(self):
        return self.payload

    def close(self):
        return None


class _Session:
    archive_url = (
        "https://storage.googleapis.com/drive-bulk-export-anonymous/test/archive"
    )

    def __init__(self, sources, *, statuses=("SUCCEEDED",)):
        self.sources = sources
        self.statuses = iter(statuses)

    def get(self, url, **_kwargs):
        if url == PUBLIC_DRIVE_FOLDER_URL:
            return _Response(text='<script data-id="_gd">{"yLTeS":"key"}</script>')
        if url == f"{TAKEOUT_EXPORTS_URL}/job-1":
            status = next(self.statuses)
            archives = (
                [{"storagePath": self.archive_url}] if status == "SUCCEEDED" else []
            )
            return _Response(
                payload={"exportJob": {"status": status, "archives": archives}}
            )
        if url == self.archive_url:
            return _Response(content=_zip_sources(self.sources))
        raise AssertionError(f"unexpected GET: {url}")

    def post(self, url, *, json, **_kwargs):
        assert url == TAKEOUT_EXPORTS_URL
        assert {item["id"] for item in json["items"]} == set(self.sources)
        return _Response(payload={"exportJob": {"id": "job-1"}})


def _zip_sources(sources):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sources.values():
            info = zipfile.ZipInfo(f"OE Public Match Data/{source.filename}")
            info.date_time = (2026, 8, 10, 7, 4, 22)
            archive.writestr(info, source.content)
    return output.getvalue()


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


def test_source_refresh_downloads_and_atomically_promotes_required_files(tmp_path):
    header = b"gameid,date,league\n"
    session = _Session(
        {
            PUBLIC_DRIVE_FILE_IDS[year]: _Source(
                header + f"game-{year},{year}-08-09T12:00:00Z,LCK\n".encode() * 40,
                f"{year}_LoL_esports_match_data_from_OraclesElixir.csv",
            )
            for year in (2024, 2025, 2026)
        },
        statuses=("QUEUED", "SUCCEEDED"),
    )
    sleep_calls = []

    result = refresh_oracle_source(
        tmp_path,
        session=session,
        now=NOW,
        sleeper=sleep_calls.append,
    )
    readiness = inspect_oracle_source(
        tmp_path,
        now=NOW,
        stability_seconds=0,
        minimum_bytes=1,
    )

    assert readiness.ready
    assert {item["year"] for item in result.files} == {2024, 2025, 2026}
    assert {item["status"] for item in result.files} == {"downloaded"}
    assert sleep_calls == [5.0]
    assert (tmp_path / "source_manifest.json").is_file()
    assert not list(tmp_path.glob(".*.part"))


def test_source_refresh_rejects_non_csv_without_replacing_cache(tmp_path):
    destination = tmp_path / "2024_LoL_esports_match_data_from_OraclesElixir.csv"
    destination.write_text("original", encoding="utf-8")
    session = _Session(
        {
            PUBLIC_DRIVE_FILE_IDS[2024]: _Source(
                b"<html>quota page</html>" * 100,
                "2024_LoL_esports_match_data_from_OraclesElixir.csv",
            )
        }
    )

    with pytest.raises(OracleSourceRefreshError, match="not an Oracle's Elixir CSV"):
        refresh_oracle_source(
            tmp_path,
            required_years=(2024,),
            session=session,
            now=NOW,
        )

    assert destination.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".*.part"))


def test_source_refresh_rejects_untrusted_export_archive_url(tmp_path):
    session = _Session(
        {
            PUBLIC_DRIVE_FILE_IDS[2026]: _Source(
                b"gameid,date\ng1,2026-08-09\n" * 100,
                "2026_LoL_esports_match_data_from_OraclesElixir.csv",
            )
        }
    )
    session.archive_url = "https://example.com/untrusted.zip"

    with pytest.raises(OracleSourceRefreshError, match="untrusted archive URL"):
        refresh_oracle_source(
            tmp_path,
            required_years=(2026,),
            session=session,
            now=NOW,
        )


def test_source_refresh_pointer_is_atomic_and_generation_is_immutable(tmp_path):
    header = b"gameid,date,league\n"
    session = _Session(
        {
            PUBLIC_DRIVE_FILE_IDS[year]: _Source(
                header + f"game-{year},{year}-08-09T12:00:00Z,LCK\n".encode() * 40,
                f"{year}_LoL_esports_match_data_from_OraclesElixir.csv",
            )
            for year in (2024, 2025, 2026)
        }
    )
    refresh_oracle_source(tmp_path, session=session, now=NOW, sleeper=lambda _: None)
    pointer = (tmp_path / "current.json").read_text()
    readiness = inspect_oracle_source(
        tmp_path, now=NOW, stability_seconds=0, minimum_bytes=1
    )
    assert readiness.ready
    assert readiness.snapshot_id in pointer
    assert len(list((tmp_path / "generations").iterdir())) == 1


def test_source_generation_rejects_manifest_path_escape(tmp_path):
    header = b"gameid,date,league\n"
    sources = {
        PUBLIC_DRIVE_FILE_IDS[2026]: _Source(
            header + b"game-2026,2026-08-09T12:00:00Z,LCK\n" * 40,
            "2026_LoL_esports_match_data_from_OraclesElixir.csv",
        )
    }
    refresh_oracle_source(
        tmp_path,
        required_years=(2026,),
        session=_Session(sources),
        now=NOW,
        sleeper=lambda _: None,
    )
    generation = next((tmp_path / "generations").iterdir())
    manifest_path = generation / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["filename"] = "../outside.csv"
    manifest_path.write_text(json.dumps(manifest))

    report = inspect_oracle_source(
        tmp_path,
        required_years=(2026,),
        now=NOW,
        stability_seconds=0,
        minimum_bytes=1,
    )

    assert report.issues == ("current_source_pointer_invalid",)


def test_source_refresh_advances_pointer_without_mutating_prior_generation(
    tmp_path, monkeypatch
):
    header = b"gameid,date,league\n"
    first_sources = {
        PUBLIC_DRIVE_FILE_IDS[year]: _Source(
            header + f"game-{year},2026-08-09T12:00:00Z,LCK\n".encode() * 40,
            f"{year}_LoL_esports_match_data_from_OraclesElixir.csv",
        )
        for year in (2024, 2025, 2026)
    }
    second_sources = {
        key: _Source(source.content + b"refresh\n", source.filename)
        for key, source in first_sources.items()
    }
    refresh_oracle_source(
        tmp_path,
        session=_Session(first_sources),
        now=NOW,
        sleeper=lambda _: None,
    )
    first_pointer = (tmp_path / "current.json").read_text()
    first_generation = next((tmp_path / "generations").iterdir())

    refresh_oracle_source(
        tmp_path,
        session=_Session(second_sources),
        now=NOW,
        sleeper=lambda _: None,
    )

    second_pointer = (tmp_path / "current.json").read_text()
    generations = list((tmp_path / "generations").iterdir())
    assert first_pointer != second_pointer
    assert len(generations) == EXPECTED_GENERATIONS
    assert first_generation.is_dir()
    assert (first_generation / "source_manifest.json").is_file()
    monkeypatch.setenv("ORACLES_ELIXIR_LOCAL_DIR", str(tmp_path))
    assert oracle_source_directory() == next(
        generation for generation in generations if generation != first_generation
    )
    second_generation = oracle_source_directory()
    third_sources = {
        key: _Source(source.content + b"third\n", source.filename)
        for key, source in second_sources.items()
    }
    refresh_oracle_source(
        tmp_path,
        session=_Session(third_sources),
        now=NOW,
        sleeper=lambda _: None,
    )
    retained = set((tmp_path / "generations").iterdir())
    assert len(retained) == EXPECTED_GENERATIONS
    assert first_generation not in retained
    assert second_generation in retained


def test_source_refresh_pointer_failure_keeps_previous_generation(
    tmp_path, monkeypatch
):
    header = b"gameid,date,league\n"
    sources = {
        PUBLIC_DRIVE_FILE_IDS[year]: _Source(
            header + f"game-{year},{year}-08-09T12:00:00Z,LCK\n".encode() * 40,
            f"{year}_LoL_esports_match_data_from_OraclesElixir.csv",
        )
        for year in (2024, 2025, 2026)
    }
    refresh_oracle_source(
        tmp_path,
        session=_Session(sources),
        now=NOW,
        sleeper=lambda _: None,
    )
    before = (tmp_path / "current.json").read_text()
    import lol_bets.data_generation.ingestion.source as source_module

    calls = 0
    original = source_module._atomic_json

    def fail_pointer(path, payload):
        nonlocal calls
        calls += 1
        if calls == POINTER_WRITE_CALL:
            raise OSError("injected pointer failure")
        return original(path, payload)

    monkeypatch.setattr(source_module, "_atomic_json", fail_pointer)
    with pytest.raises(OracleSourceRefreshError, match="pointer"):
        refresh_oracle_source(
            tmp_path,
            session=_Session(sources),
            now=NOW,
            sleeper=lambda _: None,
        )
    assert (tmp_path / "current.json").read_text() == before
    assert len(list((tmp_path / "generations").iterdir())) == 1


def test_source_refresh_refuses_to_write_through_symlink(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "source"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(OracleSourceRefreshError, match="Refusing to update"):
        refresh_oracle_source(link, required_years=(2026,), session=_Session({}))
