"""Command-line entrypoint for the Oracle Bets suite."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oracle-bets")
    sub = parser.add_subparsers(dest="domain", required=True)
    _add_lol_commands(sub)
    _add_research_operation_commands(sub)
    _add_delivery_commands(sub)
    _add_monitoring_commands(sub)
    return parser


def _add_lol_commands(sub) -> None:
    lol = sub.add_parser("lol", help="League of Legends workflows")
    lol_sub = lol.add_subparsers(dest="action", required=True)
    lol_sub.add_parser("health", help="Check LoL training and inference artifacts")
    lol_sub.add_parser("ingest", help="Run the LoL ingestion/feature pipeline")
    lol_sub.add_parser(
        "reconcile-history",
        help="Fully refresh and reconcile the retained LoL history",
    )
    lol_sub.add_parser(
        "sync-identities",
        help="Normalize retained LoL players, teams, leagues, series, and maps",
    )
    schedule = lol_sub.add_parser(
        "schedule",
        help="Fetch and print upcoming LoL matches without writing data",
    )
    schedule.add_argument("--days", type=int, default=14)
    schedule.add_argument("--leagues", default=None)
    lol_sub.add_parser(
        "validate-data",
        help="Validate generated LoL training tables without model training",
    )
    train = lol_sub.add_parser("train", help="Train LoL prediction models")
    _add_train_arguments(train)
    retune = lol_sub.add_parser(
        "retune",
        help="Force a fresh hyperparameter search and train research artifacts",
    )
    retune.add_argument(
        "--targets",
        default="all",
    )
    retune.add_argument(
        "--feature-set",
        choices=["full", "compact", "selected"],
        default="compact",
    )
    retune.add_argument("--max-features", type=int, default=120)
    promote_tuning = lol_sub.add_parser(
        "promote-tuning",
        help="Promote one reviewed complete retuning run",
    )
    promote_tuning.add_argument("run_id")


def _add_train_arguments(train) -> None:
    train.add_argument(
        "--feature-selection",
        choices=["none", "importance", "cumulative", "report"],
        default="none",
    )
    train.add_argument("--targets", default="all")
    train.add_argument(
        "--feature-set",
        choices=["full", "compact", "selected"],
        default="full",
    )
    train.add_argument("--max-features", type=int, default=120)
    train.add_argument("--calibration", choices=["auto", "none"], default="auto")
    train.add_argument(
        "--calibration-method",
        choices=["raw", "sigmoid", "isotonic", "auto"],
        default="auto",
    )
    train.add_argument("--calibration-size", type=float, default=0.15)
    train.add_argument("--tune-size", type=float, default=0.10)
    train.add_argument("--test-size", type=float, default=0.15)


def _add_research_operation_commands(sub) -> None:
    model = sub.add_parser("model", help="Immutable candidate registry controls")
    model_sub = model.add_subparsers(dest="action", required=True)
    for action in ("status", "promote", "rollback"):
        command = model_sub.add_parser(action)
        command.add_argument("--registry", default=None)
        if action != "status":
            command.add_argument("model_id")
            command.add_argument("--reason", required=True)
    register = model_sub.add_parser(
        "register-current",
        help="Freeze the current complete inference tree as a candidate",
    )
    register.add_argument("model_id")
    register.add_argument("--registry", default=None)
    register.add_argument("--target", default="map_win")
    register.add_argument("--code-version", required=True)
    register.add_argument(
        "--metric",
        action="append",
        required=True,
        help="Repeat as NAME=VALUE, for example --metric log_loss=0.64",
    )
    register.add_argument("--random-seed", type=int, default=7)

    evidence = sub.add_parser("evidence", help="Evidence database operations")
    evidence_sub = evidence.add_subparsers(dest="action", required=True)
    for action in ("init", "health", "backup", "export"):
        command = evidence_sub.add_parser(action)
        command.add_argument("--database", default=None)
        if action == "backup":
            command.add_argument("--output", default=None)
        if action == "export":
            command.add_argument("--output", default=None)
            command.add_argument(
                "--format",
                choices=["json", "csv"],
                default="json",
            )
    restore = evidence_sub.add_parser(
        "restore-verify",
        help="Verify a backup without replacing the active database",
    )
    restore.add_argument("backup_path")

    paper = sub.add_parser("paper", help="Paper-position operations")
    paper_sub = paper.add_subparsers(dest="action", required=True)
    settle = paper_sub.add_parser("settle", help="Reconcile one paper settlement")
    settle.add_argument("--database", default=None)
    settle.add_argument("--position-id", required=True)
    settle.add_argument("--stake", required=True)
    settle.add_argument("--odds", required=True)
    settle.add_argument(
        "--result",
        required=True,
        choices=["win", "loss", "push", "void"],
    )
    settle.add_argument("--source-reference", required=True)
    settle.add_argument("--internal", action="store_true")
    settle.add_argument("--settled-at", default=None)
    settle.add_argument("--dry-run", action="store_true")


def _add_delivery_commands(sub) -> None:
    from oracle_bets_core.config import load_product_config

    product = load_product_config()
    daily = sub.add_parser("daily", help="Daily report and retraining workflows")
    daily_sub = daily.add_subparsers(dest="action", required=True)
    daily_lol = daily_sub.add_parser("lol", help="Run the daily LoL workflow")
    daily_lol.add_argument("--dry-run", action="store_true")
    daily_lol.add_argument(
        "--horizon-hours",
        type=int,
        default=product.fixture_window_hours,
    )
    daily_lol.add_argument("--leagues", default=None)
    daily_lol.add_argument("--webhook-url", default=None)
    daily_lol.add_argument("--skip-retrain", action="store_true")
    daily_lol.add_argument("--skip-market-search", action="store_true")
    daily_lol.add_argument("--targets", default="all")
    daily_lol.add_argument(
        "--feature-set",
        choices=["full", "compact", "selected"],
        default="compact",
    )
    daily_lol.add_argument("--max-features", type=int, default=120)


def _add_monitoring_commands(sub) -> None:
    health = sub.add_parser("health", help="Unified operational health")
    health_sub = health.add_subparsers(dest="action", required=True)
    system = health_sub.add_parser(
        "system",
        help="Evaluate all evidence-driven operational checks",
    )
    system.add_argument("--database", default=None)
    system.add_argument("--output", default=None)
    system.add_argument(
        "--record",
        action="store_true",
        help="Append the report to canonical run evidence",
    )

    audit = sub.add_parser("audit", help="Owner audit workflows")
    audit_sub = audit.add_subparsers(dest="action", required=True)
    monthly = audit_sub.add_parser(
        "monthly",
        help="Generate and record the light monthly owner audit",
    )
    monthly.add_argument("--database", default=None)
    monthly.add_argument("--period", default=None, help="Audit month in YYYY-MM")
    monthly.add_argument("--output", default=None, help="Output directory")


def _main_lol(args: argparse.Namespace) -> int:  # noqa: PLR0911
    if args.action in {"ingest", "reconcile-history"}:
        from lol_bets.data_generation.ingestion.history import HistoryRefreshMode
        from lol_bets.pipeline import DataGenerator

        mode = (
            HistoryRefreshMode.FULL
            if args.action == "reconcile-history"
            else HistoryRefreshMode.INCREMENTAL
        )
        DataGenerator(history_mode=mode).run()
        return 0
    if args.action == "sync-identities":
        from datetime import UTC, datetime

        from lol_bets.operations.identity import sync_history_identity_graph

        from oracle_bets_core.evidence import EvidenceStore
        from oracle_bets_core.paths import EVIDENCE_DB, RAW_DATA
        from oracle_bets_core.pd import pd

        result = sync_history_identity_graph(
            EvidenceStore(EVIDENCE_DB),
            pd.read_parquet(RAW_DATA),
            observed_at=datetime.now(UTC),
        )
        sys.stdout.write(
            "LoL identity graph synchronized: "
            f"{result.added_identities} identities, "
            f"{result.added_links} provider links added.\n"
        )
        return 0
    if args.action == "schedule":
        _print_lol_schedule(args)
        return 0
    if args.action == "health":
        return _print_lol_health()
    if args.action == "validate-data":
        _validate_lol_data()
        return 0
    if args.action == "promote-tuning":
        from lol_bets.training import promote_tuning_run

        paths = promote_tuning_run(args.run_id)
        sys.stdout.write(f"Promoted {len(paths)} tuned parameter files.\n")
        return 0
    if args.action in {"train", "retune"}:
        _train_lol(args)
        return 0
    return 1


def _print_lol_schedule(args: argparse.Namespace) -> None:
    from lol_bets.data_generation.ingestion.schedule import fetch_and_store_schedule

    schedule = fetch_and_store_schedule(
        window_days=args.days,
        leagues=args.leagues,
        save_path=None,
    )
    if schedule.empty:
        sys.stdout.write("No upcoming matches found.\n")
        return
    display = schedule[
        ["start_utc", "league", "team_a", "team_b", "best_of", "status"]
    ].copy()
    team_a = display.pop("team_a").replace("", "TBD").fillna("TBD")
    team_b = display.pop("team_b").replace("", "TBD").fillna("TBD")
    display["match"] = team_a + " vs " + team_b
    display["best_of"] = "BO" + (
        display["best_of"].astype("Int64").astype(str).replace("<NA>", "?")
    )
    display = display.rename(columns={"start_utc": "start (UTC)", "best_of": "format"})[
        ["start (UTC)", "league", "match", "format", "status"]
    ]
    sys.stdout.write(f"Upcoming LoL schedule: {len(display)} matches\n")
    sys.stdout.write(display.to_string(index=False) + "\n")


def _print_lol_health() -> int:
    from lol_bets.module import LoLBetsModule

    from oracle_bets_core.pd import backend_name

    module = LoLBetsModule()
    health_reports = (module.artifact_health(), module.training_artifact_health())
    sys.stdout.write(f"dataframe backend: {backend_name()}\n")
    for health in health_reports:
        sys.stdout.write(f"{health.module_id}: {'ok' if health.ok else 'unhealthy'}\n")
        for check in health.checks:
            status = "ok" if check.ok else check.reason or "missing/unreadable"
            sys.stdout.write(f"  {check.name}: {status} - {check.path}\n")
    return 0 if all(health.ok for health in health_reports) else 2


def _validate_lol_data() -> None:
    from lol_bets.module import LoLBetsModule
    from lol_bets.training import validate_training_tables

    from oracle_bets_core.io_utils import load_training_data
    from oracle_bets_core.logger import logger
    from oracle_bets_core.paths import TRAINING_PLAYER_DATA, TRAINING_TEAM_DATA

    LoLBetsModule().training_artifact_health().raise_if_unhealthy()
    team_df, player_df = load_training_data(
        TRAINING_TEAM_DATA,
        TRAINING_PLAYER_DATA,
        logger,
    )
    validate_training_tables(team_df, player_df)
    sys.stdout.write(
        "LoL training data ok: "
        f"{team_df['gameid'].nunique()} games, "
        f"{len(team_df)} team rows, {len(player_df)} player rows\n"
    )


def _train_lol(args: argparse.Namespace) -> None:
    from lol_bets.training import train_models

    options = {
        "targets": args.targets,
        "force_retune": args.action == "retune",
        "feature_set": args.feature_set,
        "max_features": args.max_features,
    }
    if args.action == "train":
        options.update(
            {
                "feature_selection": args.feature_selection,
                "calibration": args.calibration,
                "calibration_method": args.calibration_method,
                "calibration_size": args.calibration_size,
                "tune_size": args.tune_size,
                "test_size": args.test_size,
            }
        )
    report_root = train_models(**options)
    sys.stdout.write(f"Training report: {report_root}\n")


def _main_model(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime
    from pathlib import Path

    from lol_bets.operations.models import (
        ModelRegistry,
        ModelRegistryError,
        register_current_candidate,
    )

    from oracle_bets_core.paths import MODEL_REGISTRY_DIR

    registry = ModelRegistry(
        Path(args.registry) if args.registry else MODEL_REGISTRY_DIR
    )
    if args.action == "register-current":
        try:
            metrics = _parse_metrics(args.metric)
            bundle = register_current_candidate(
                registry=registry,
                model_id=args.model_id,
                target=args.target,
                code_version=args.code_version,
                metrics=metrics,
                created_at=datetime.now(UTC),
                random_seed=args.random_seed,
            )
        except (ModelRegistryError, ValueError) as error:
            sys.stderr.write(f"{error}\n")
            return 2
        sys.stdout.write(f"Candidate registered: {bundle}\n")
        return 0
    if args.action == "status":
        champion = registry.champion_id()
        if champion is None:
            sys.stdout.write("No champion selected.\n")
            return 2
        healthy = registry.verify_bundle(champion)
        sys.stdout.write(
            f"Champion: {champion} ({'healthy' if healthy else 'unhealthy'})\n"
        )
        return 0 if healthy else 2
    try:
        if args.action == "promote":
            registry.promote(
                args.model_id,
                promoted_at=datetime.now(UTC),
                reason=args.reason,
            )
        else:
            registry.rollback(
                args.model_id,
                rolled_back_at=datetime.now(UTC),
                reason=args.reason,
            )
    except ModelRegistryError as error:
        sys.stderr.write(f"{error}\n")
        return 2
    sys.stdout.write(f"Champion is now {args.model_id}.\n")
    return 0


def _parse_metrics(values: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or not name.strip():
            raise ValueError("model metrics must use NAME=VALUE")
        try:
            parsed = float(raw)
        except ValueError as error:
            raise ValueError(f"model metric is not numeric: {value}") from error
        metrics[name.strip()] = parsed
    return metrics


def _main_evidence(args: argparse.Namespace) -> int:
    from pathlib import Path

    from oracle_bets_core.evidence import EvidenceStore
    from oracle_bets_core.operations.backup import (
        create_evidence_backup,
        export_all_evidence,
        verify_evidence_backup,
    )
    from oracle_bets_core.paths import BACKUPS_DIR, EVIDENCE_DB, EXPORTS_DIR

    if args.action == "restore-verify":
        result = verify_evidence_backup(Path(args.backup_path))
        sys.stdout.write(
            f"Backup verified: {result.path} "
            f"(schema={result.schema_version}, sha256={result.sha256})\n"
        )
        return 0
    database = Path(args.database) if args.database else EVIDENCE_DB
    store = EvidenceStore(database)
    if args.action == "init":
        store.initialize_schema()
        sys.stdout.write(f"Evidence database initialized: {database}\n")
        return 0
    if args.action == "health":
        if not database.is_file():
            sys.stderr.write(f"Evidence database missing: {database}\n")
            return 2
        sys.stdout.write(
            f"Evidence database: integrity={store.integrity_check()}, "
            f"schema={store.schema_version()}\n"
        )
        return 0
    if args.action == "backup":
        result = create_evidence_backup(
            database,
            Path(args.output) if args.output else BACKUPS_DIR,
        )
        sys.stdout.write(f"Backup created: {result.path} (sha256={result.sha256})\n")
        return 0
    outputs = export_all_evidence(
        store,
        Path(args.output) if args.output else EXPORTS_DIR,
        file_format=args.format,
    )
    sys.stdout.write(f"Exported {len(outputs)} evidence files.\n")
    return 0


def _main_paper(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime
    from pathlib import Path

    from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
    from oracle_bets_core.evidence.settlement import (
        PositionTerms,
        SettlementResult,
        reconcile_settlement,
    )
    from oracle_bets_core.paths import EVIDENCE_DB

    settled_at = (
        datetime.fromisoformat(args.settled_at)
        if args.settled_at
        else datetime.now(UTC)
    )
    result = SettlementResult(args.result)
    record = reconcile_settlement(
        PositionTerms(args.position_id, args.stake, args.odds),
        settled_at=settled_at,
        provider_result=None if args.internal else result,
        provider_reference=None if args.internal else args.source_reference,
        internal_result=result if args.internal else None,
        internal_reference=args.source_reference if args.internal else None,
        internal_verified=args.internal,
    )
    if not args.dry_run:
        store = EvidenceStore(Path(args.database) if args.database else EVIDENCE_DB)
        store.append(
            EvidenceTable.SETTLEMENTS,
            {
                "id": record.settlement_id,
                "paper_position_id": record.position_id,
                "settled_at": record.settled_at,
                "result": record.result,
                "result_source": record.source,
                "pnl_units": record.pnl_units,
                "idempotency_key": record.settlement_id,
                "payload_json": {
                    "source_reference": record.source_reference,
                    "warnings": list(record.warnings),
                },
            },
        )
    sys.stdout.write(
        f"Paper settlement {record.result.value}: PnL {record.pnl_units} units"
        f"{' (dry run)' if args.dry_run else ''}.\n"
    )
    return 0


def _main_daily(args: argparse.Namespace) -> int:
    from collections import Counter

    from lol_bets.daily import DailyWorkflowConfig, run_daily_lol_workflow

    result = run_daily_lol_workflow(
        DailyWorkflowConfig(
            horizon_hours=args.horizon_hours,
            leagues=args.leagues,
            webhook_url=args.webhook_url,
            dry_run=args.dry_run,
            skip_retrain=args.skip_retrain,
            skip_market_search=args.skip_market_search,
            targets=args.targets,
            feature_set=args.feature_set,
            max_features=args.max_features,
        )
    )
    sys.stdout.write(f"Daily LoL workflow: {'OK' if result.ok else 'FAILED'}\n")
    for step in result.steps:
        status = "OK" if step.ok else "FAILED"
        sys.stdout.write(f"- {status}: {step.name} - {step.detail}\n")
    prediction_counts = Counter(
        detail.get("status", "unknown") for detail in result.prediction_details
    )
    if prediction_counts:
        sys.stdout.write(
            "Predictions: "
            f"{prediction_counts['predicted']} generated, "
            f"{prediction_counts['unsupported_team']} unsupported team, "
            f"{prediction_counts['insufficient_history']} insufficient history, "
            f"{prediction_counts['prediction_unavailable']} unavailable.\n"
        )
    if result.report_paths is not None:
        json_path, markdown_path = result.report_paths
        sys.stdout.write(f"Daily JSON report: {json_path}\n")
        sys.stdout.write(f"Daily Markdown report: {markdown_path}\n")
    return 0 if result.ok else 2


def _main_health(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from oracle_bets_core.evidence import EvidenceStore
    from oracle_bets_core.operations.health import (
        evaluate_system_health,
        record_system_health,
        write_system_health_report,
    )
    from oracle_bets_core.paths import EVIDENCE_DB

    database = Path(args.database) if args.database else EVIDENCE_DB
    if not database.is_file():
        sys.stderr.write(f"Evidence database missing: {database}\n")
        return 2
    store = EvidenceStore(database)
    report = evaluate_system_health(store)
    if args.record:
        run_id = record_system_health(store, report)
        sys.stdout.write(f"Health evidence recorded: {run_id}\n")
    if args.output:
        destination = write_system_health_report(report, Path(args.output))
        sys.stdout.write(f"Health report written: {destination}\n")
    sys.stdout.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return report.exit_code


def _main_audit(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime, timedelta
    from pathlib import Path

    from oracle_bets_core.config import load_product_config
    from oracle_bets_core.evidence import EvidenceStore
    from oracle_bets_core.operations.audit import (
        build_monthly_audit,
        record_monthly_audit,
        write_monthly_audit,
    )
    from oracle_bets_core.paths import EVIDENCE_DB, REPORTS_DIR

    database = Path(args.database) if args.database else EVIDENCE_DB
    if not database.is_file():
        sys.stderr.write(f"Evidence database missing: {database}\n")
        return 2
    now = datetime.now(UTC)
    previous_month = now.replace(day=1) - timedelta(days=1)
    period = args.period or previous_month.strftime("%Y-%m")
    store = EvidenceStore(database)
    try:
        report = build_monthly_audit(
            store,
            period=period,
            config=load_product_config(),
            generated_at=now,
        )
    except ValueError as error:
        sys.stderr.write(f"{error}\n")
        return 2
    run_id = record_monthly_audit(store, report)
    output_dir = (
        Path(args.output) if args.output else REPORTS_DIR / "audits" / "monthly"
    )
    json_path, markdown_path = write_monthly_audit(
        report,
        json_path=output_dir / f"monthly-{period}.json",
        markdown_path=output_dir / f"monthly-{period}.md",
    )
    sys.stdout.write(
        f"Monthly audit recorded: {run_id}\n"
        f"JSON: {json_path}\n"
        f"Markdown: {markdown_path}\n"
        "Owner review and sign-off are still required.\n"
    )
    return 2 if report.status == "critical" else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "lol": _main_lol,
        "model": _main_model,
        "evidence": _main_evidence,
        "paper": _main_paper,
        "daily": _main_daily,
        "health": _main_health,
        "audit": _main_audit,
    }
    return handlers[args.domain](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
