from datetime import UTC, datetime

from lol_bets.operations.models import CandidateManifest, ModelRegistry
from oracle_bets_core.cli import build_parser, main

NOW = datetime(2026, 7, 27, 8, tzinfo=UTC)


def test_parser_exposes_required_operational_commands():
    commands = (
        ["lol", "retune"],
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
            "--stake",
            "1",
            "--odds",
            "2",
            "--result",
            "win",
            "--source-reference",
            "provider-1",
        ],
    )

    assert all(build_parser().parse_args(command) for command in commands)


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


def test_paper_settlement_dry_run_does_not_require_database(capsys):
    code = main(
        [
            "paper",
            "settle",
            "--position-id",
            "position-1",
            "--stake",
            "1",
            "--odds",
            "1.90",
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
