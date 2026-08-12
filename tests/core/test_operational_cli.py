import argparse
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from lol_bets.operations.models import CandidateManifest, ModelRegistry
from oracle_bets_core.cli import _market_check_fixture_rows, build_parser, main

NOW = datetime(2026, 7, 27, 8, tzinfo=UTC)
BEST_OF_THREE = 3
BLOCKED_EXIT = 2


def test_parser_exposes_required_operational_commands():
    commands = (
        ["lol", "source-check"],
        ["lol", "source-check", "--format", "json"],
        ["lol", "source-refresh"],
        ["lol", "source-refresh", "--format", "json"],
        ["lol", "retune"],
        ["lol", "market-check"],
        ["lol", "market-check", "--match-key", "pandascore:123"],
        ["model", "status"],
        ["model", "promote", "candidate-1", "--reason", "gate passed"],
        ["model", "rollback", "candidate-1", "--reason", "owner review"],
        [
            "model",
            "register-current",
            "candidate-2",
            "--code-version",
            "abc123",
            "--metric",
            "log_loss=0.64",
        ],
        ["evidence", "init"],
        ["evidence", "health"],
        ["evidence", "backup"],
        ["evidence", "export"],
        ["evidence", "restore-verify", "backup.db"],
        ["health", "system"],
        ["audit", "monthly", "--period", "2026-07"],
        [
            "paper",
            "settle",
            "--position-id",
            "position-1",
            "--result",
            "win",
            "--source-reference",
            "provider-1",
        ],
    )

    assert all(build_parser().parse_args(command) for command in commands)


def test_removed_automatic_reconciliation_command_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["paper", "reconcile"])


@pytest.mark.parametrize("action", ["reconcile-history", "train"])
def test_source_failure_blocks_history_and_training_commands(
    action,
    monkeypatch,
    capsys,
):
    from lol_bets.data_generation.ingestion.source import OracleSourceReadinessError

    def blocked():
        raise OracleSourceReadinessError("source stale")

    monkeypatch.setattr(
        "lol_bets.data_generation.ingestion.source.require_oracle_source_ready",
        blocked,
    )
    monkeypatch.setattr(
        "lol_bets.data_generation.ingestion.source.refresh_oracle_source",
        lambda: None,
    )

    assert main(["lol", action]) == BLOCKED_EXIT
    assert "source stale" in capsys.readouterr().err


def test_fixture_market_check_falls_back_to_newest_daily_report(tmp_path):
    reports = tmp_path / "daily"
    reports.mkdir()
    (reports / "20260802.json").write_text(
        json.dumps(
            {
                "schedule": [
                    {
                        "match_key": "pandascore:1",
                        "team_a": "T1",
                        "team_b": "Gen.G",
                        "league": "LCK",
                        "start_utc": "2026-08-02T10:00:00+00:00",
                        "best_of": 3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = _market_check_fixture_rows(
        "pandascore:1",
        schedule_path=tmp_path / "missing.parquet",
        report_dir=reports,
    )

    assert rows.iloc[0]["team_a"] == "T1"
    assert rows.iloc[0]["team_b"] == "Gen.G"
    assert rows.iloc[0]["best_of"] == BEST_OF_THREE


def test_command_runbook_mentions_every_public_leaf_command():
    def leaf_commands(
        parser: argparse.ArgumentParser,
        prefix: tuple[str, ...] = (),
    ) -> list[tuple[str, ...]]:
        subparsers = next(
            (
                action
                for action in parser._actions
                if isinstance(action, argparse._SubParsersAction)
            ),
            None,
        )
        if subparsers is None:
            return [prefix]
        return [
            command
            for name, child in subparsers.choices.items()
            for command in leaf_commands(child, (*prefix, name))
        ]

    docs = (Path(__file__).parents[2] / "docs" / "commands.md").read_text(
        encoding="utf-8"
    )
    missing = [
        " ".join(command)
        for command in leaf_commands(build_parser())
        if not re.search(
            rf"`{re.escape(' '.join(command))}(?:`|\s)",
            docs,
        )
    ]

    assert missing == []


def test_evidence_cli_init_health_backup_verify_and_export(tmp_path, capsys):
    database = tmp_path / "evidence.db"
    backups = tmp_path / "backups"
    exports = tmp_path / "exports"

    assert main(["evidence", "init", "--database", str(database)]) == 0
    assert main(["evidence", "health", "--database", str(database)]) == 0
    assert (
        main(
            [
                "evidence",
                "backup",
                "--database",
                str(database),
                "--output",
                str(backups),
            ]
        )
        == 0
    )
    backup = next(backups.glob("*.db"))
    assert main(["evidence", "restore-verify", str(backup)]) == 0
    assert (
        main(
            [
                "evidence",
                "export",
                "--database",
                str(database),
                "--output",
                str(exports),
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "integrity=ok" in output
    assert "Backup verified" in output
    assert (exports / "manifest.json").is_file()


def test_evidence_health_fails_closed_on_stale_schema(tmp_path):
    database = tmp_path / "evidence.db"
    assert main(["evidence", "init", "--database", str(database)]) == 0
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE evidence_schema_version SET version = 3")

    assert main(["evidence", "health", "--database", str(database)]) == BLOCKED_EXIT


def _manifest(model_id):
    return CandidateManifest(
        model_id=model_id,
        sport="lol",
        target="map_win",
        created_at=NOW,
        code_version="code",
        data_manifest="data",
        feature_fingerprint="features",
        config_hash="config",
        dependency_lock_hash="lock",
        random_seed=7,
        metrics={"log_loss": 0.65},
    )


def test_model_cli_promotes_checks_and_rolls_back_verified_bundles(
    tmp_path,
    capsys,
):
    registry_path = tmp_path / "registry"
    registry = ModelRegistry(registry_path)
    first = tmp_path / "first.pkl"
    second = tmp_path / "second.pkl"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    registry.register_candidate(_manifest("candidate-1"), {"model.pkl": first})
    registry.register_candidate(_manifest("candidate-2"), {"model.pkl": second})

    assert (
        main(
            [
                "model",
                "promote",
                "candidate-1",
                "--reason",
                "initial",
                "--registry",
                str(registry_path),
            ]
        )
        == 0
    )
    assert main(["model", "status", "--registry", str(registry_path)]) == 0
    assert (
        main(
            [
                "model",
                "promote",
                "candidate-2",
                "--reason",
                "passed gate",
                "--registry",
                str(registry_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "model",
                "rollback",
                "candidate-1",
                "--reason",
                "owner review",
                "--registry",
                str(registry_path),
            ]
        )
        == 0
    )

    assert registry.champion_id() == "candidate-1"
    assert "healthy" in capsys.readouterr().out


def test_paper_settlement_dry_run_uses_recorded_position_terms(
    capsys,
    monkeypatch,
):
    monkeypatch.setattr(
        "oracle_bets_core.operations.paper_evidence.paper_show",
        lambda *_args: {
            "stake_units": "1",
            "decimal_odds": "1.90",
        },
    )
    code = main(
        [
            "paper",
            "settle",
            "--position-id",
            "position-1",
            "--result",
            "win",
            "--source-reference",
            "provider-1",
            "--settled-at",
            NOW.isoformat(),
            "--dry-run",
        ]
    )

    assert code == 0
    assert "PnL 0.90 units (dry run)" in capsys.readouterr().out


def test_paper_settlement_rejects_legacy_stake_and_odds_flags():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "paper",
                "settle",
                "--position-id",
                "position-1",
                "--result",
                "win",
                "--source-reference",
                "provider-1",
                "--stake",
                "1",
            ]
        )


def test_system_health_and_monthly_audit_cli_record_reports(tmp_path, capsys):
    database = tmp_path / "evidence.db"
    health_path = tmp_path / "health.json"
    audit_dir = tmp_path / "audits"
    assert main(["evidence", "init", "--database", str(database)]) == 0

    health_code = main(
        [
            "health",
            "system",
            "--database",
            str(database),
            "--output",
            str(health_path),
            "--record",
        ]
    )
    audit_code = main(
        [
            "audit",
            "monthly",
            "--database",
            str(database),
            "--period",
            "2026-07",
            "--output",
            str(audit_dir),
        ]
    )

    assert health_code == 1
    assert audit_code == 0
    assert health_path.is_file()
    assert (audit_dir / "monthly-2026-07.json").is_file()
    assert (audit_dir / "monthly-2026-07.md").is_file()
    assert "Owner review and sign-off are still required" in capsys.readouterr().out
