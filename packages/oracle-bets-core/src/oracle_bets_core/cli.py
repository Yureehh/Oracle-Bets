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
    source_check = lol_sub.add_parser(
        "source-check",
        help="Validate the managed Oracle's Elixir cache before rebuilding",
    )
    source_check.add_argument("--format", choices=["table", "json"], default="table")
    source_refresh = lol_sub.add_parser(
        "source-refresh",
        help="Atomically refresh public Oracle's Elixir files from Google Drive",
    )
    source_refresh.add_argument("--format", choices=["table", "json"], default="table")
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
    market_check = lol_sub.add_parser(
        "market-check",
        help="Verify public read-only Polymarket discovery and CLOB data",
    )
    market_check.add_argument("--match-key", default=None)
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


def _add_research_operation_commands(sub) -> None:  # noqa: PLR0915
    model = sub.add_parser("model", help="Immutable candidate registry controls")
    model_sub = model.add_subparsers(dest="action", required=True)
    for action in ("status", "promote", "rollback"):
        command = model_sub.add_parser(action)
        command.add_argument("--registry", default=None)
        if action != "status":
            command.add_argument("model_id")
            command.add_argument("--reason", required=True)
    listing = model_sub.add_parser("list", help="List immutable model candidates")
    listing.add_argument("--registry", default=None)
    listing.add_argument("--format", choices=["table", "json"], default="table")
    review = model_sub.add_parser("review", help="Review one candidate manifest")
    review.add_argument("model_id")
    review.add_argument("--registry", default=None)
    review.add_argument("--format", choices=["table", "json"], default="table")
    register_run = model_sub.add_parser(
        "register-run", help="Confirm an automatically registered training run"
    )
    register_run.add_argument("run_id", help="Training run ID or 'latest'")
    register_run.add_argument("--registry", default=None)
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
    listing = paper_sub.add_parser("list", help="List paper proposals and positions")
    listing.add_argument("--database", default=None)
    listing.add_argument("--state", choices=["pending", "open", "settled"])
    listing.add_argument("--target")
    listing.add_argument("--league")
    listing.add_argument("--format", choices=["table", "json"], default="table")
    show = paper_sub.add_parser("show", help="Show one proposal or position")
    show.add_argument("record_id")
    show.add_argument("--database", default=None)
    show.add_argument("--format", choices=["table", "json"], default="table")
    quote = paper_sub.add_parser("quote-prop", help="Price a manual prop line")
    quote.add_argument("--database", default=None)
    quote.add_argument("--forecast-id", required=True)
    quote.add_argument("--line", required=True, type=float)
    quote.add_argument("--over-odds", required=True, type=float)
    quote.add_argument("--under-odds", required=True, type=float)
    quote.add_argument("--source", required=True)
    decide = paper_sub.add_parser("decide", help="Accept or reject a paper proposal")
    decide.add_argument("--database", default=None)
    decide.add_argument("--proposal-id", required=True)
    decide.add_argument("--decision", required=True, choices=["accept", "reject"])
    decide.add_argument("--reason")
    decide.add_argument("--actor-id", default="owner-cli")
    settle = paper_sub.add_parser("settle", help="Record one owner-verified settlement")
    settle.add_argument("--database", default=None)
    settle.add_argument("--position-id", required=True)
    settle.add_argument(
        "--result",
        required=True,
        choices=["win", "loss", "push", "void"],
    )
    settle.add_argument("--source-reference", required=True)
    settle.add_argument("--note")
    settle.add_argument("--actor-id", default="owner-cli")
    settle.add_argument("--settled-at", default=None)
    settle.add_argument("--dry-run", action="store_true")
    closing = paper_sub.add_parser(
        "capture-closing",
        help="Capture read-only pre-start Polymarket closing books",
    )
    closing.add_argument("--database", default=None)
    closing.add_argument("--window-minutes", type=int, default=15)
    closing.add_argument("--dry-run", action="store_true")
    closing.add_argument("--format", choices=["table", "json"], default="table")
    performance = paper_sub.add_parser("performance", help="Report paper performance")
    performance.add_argument("--database", default=None)
    performance.add_argument("--since")
    performance.add_argument("--target")
    performance.add_argument("--league")
    performance.add_argument("--format", choices=["table", "json"], default="table")


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
    daily_lol.add_argument(
        "--discord-delivery-mode",
        choices=["gateway", "webhook", "off"],
        default=None,
    )
    daily_lol.add_argument("--skip-retrain", action="store_true")
    daily_lol.add_argument("--skip-market-search", action="store_true")
    daily_lol.add_argument("--targets", default="all")
    daily_lol.add_argument(
        "--feature-set",
        choices=["full", "compact", "selected"],
        default="compact",
    )
    daily_lol.add_argument("--max-features", type=int, default=120)
    daily_lol.add_argument(
        "--ai-review",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use one optional bounded advisory review when OPENAI_API_KEY exists",
    )
    daily_lol.add_argument("--openai-model", default="gpt-5.6-luna")
    discord = sub.add_parser("discord", help="Discord delivery operations")
    discord_sub = discord.add_subparsers(dest="action", required=True)
    doctor = discord_sub.add_parser("doctor", help="Validate Discord configuration")
    doctor.add_argument("--live", action="store_true")
    discord_sub.add_parser("run", help="Run the owner-only Gateway bot")
    publish = discord_sub.add_parser("publish", help="Publish a saved daily report")
    publish.add_argument("--run-id", default="latest")


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
    if args.action == "source-check":
        return _print_lol_source_check(args.format)
    if args.action == "source-refresh":
        return _refresh_lol_source(args.format)
    if args.action in {"ingest", "reconcile-history"}:
        from lol_bets.data_generation.ingestion.history import HistoryRefreshMode
        from lol_bets.data_generation.ingestion.source import (
            OracleSourceReadinessError,
            OracleSourceRefreshError,
            refresh_oracle_source,
            require_oracle_source_ready,
        )
        from lol_bets.pipeline import DataGenerator

        try:
            refresh_oracle_source()
            require_oracle_source_ready()
        except (OracleSourceReadinessError, OracleSourceRefreshError) as error:
            sys.stderr.write(f"{error}\n")
            return 2
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
    if args.action == "market-check":
        return _check_lol_markets(args.match_key)
    if args.action == "promote-tuning":
        from lol_bets.training import promote_tuning_run

        paths = promote_tuning_run(args.run_id)
        sys.stdout.write(f"Promoted {len(paths)} tuned parameter files.\n")
        return 0
    if args.action in {"train", "retune"}:
        from lol_bets.data_generation.ingestion.source import (
            OracleSourceReadinessError,
            require_oracle_source_ready,
        )

        try:
            require_oracle_source_ready()
        except OracleSourceReadinessError as error:
            sys.stderr.write(f"{error}\n")
            return 2
        _train_lol(args)
        return 0
    return 1


def _print_lol_source_check(output_format: str) -> int:
    import json

    from lol_bets.data_generation.ingestion.source import inspect_oracle_source

    report = inspect_oracle_source()
    payload = report.to_dict()
    if output_format == "json":
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(
            f"Oracle's Elixir source: {'READY' if report.ready else 'BLOCKED'}\n"
        )
        for item in report.files:
            sys.stdout.write(
                f"- {item.year}: {item.size_bytes:,} bytes, "
                f"modified {item.modified_at.isoformat()}\n"
            )
        sys.stdout.write(
            "- current-year max match: "
            f"{report.current_year_max_match_at or 'unavailable'}\n"
        )
        for issue in report.issues:
            sys.stdout.write(f"- BLOCKED: {issue}\n")
    return 0 if report.ready else 2


def _refresh_lol_source(output_format: str) -> int:
    import json

    from lol_bets.data_generation.ingestion.source import (
        OracleSourceRefreshError,
        refresh_oracle_source,
    )

    try:
        result = refresh_oracle_source()
    except OracleSourceRefreshError as error:
        sys.stderr.write(f"{error}\n")
        return 2
    payload = result.to_dict()
    if output_format == "json":
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write("Oracle's Elixir source refreshed:\n")
        for item in result.files:
            sys.stdout.write(
                f"- {item['year']}: {item['size_bytes']:,} bytes, "
                f"modified {item['remote_modified_at']}\n"
            )
    return 0


def _check_lol_markets(match_key: str | None) -> int:  # noqa: PLR0915
    """Exercise only public Polymarket reads and print a sanitized JSON result."""
    import json

    import requests

    from oracle_bets_core.markets import (
        MarketFixture,
        PolymarketClobClient,
        PolymarketGammaAdapter,
        SupportedMarketType,
        capture_minimum_order_book_batch,
        confirmed_executable_fill,
        select_best_market,
    )
    from oracle_bets_core.paths import REPORTS_DIR, SCHEDULE
    from oracle_bets_core.pd import pd

    result: dict[str, object] = {
        "read_only": True,
        "trading_surface": False,
        "match_key": match_key,
    }
    try:
        geoblock = requests.get(
            "https://polymarket.com/api/geoblock",
            timeout=10,
        )
        geoblock.raise_for_status()
        geoblock_payload = geoblock.json()
        result["geoblock"] = {
            "blocked": bool(geoblock_payload.get("blocked")),
            "country": geoblock_payload.get("country"),
        }

        if match_key:
            rows = _market_check_fixture_rows(
                match_key,
                schedule_path=SCHEDULE,
                report_dir=REPORTS_DIR / "daily",
            )
            row = _single_market_check_row(rows, match_key)
            team_a = str(row["team_a"]).strip()
            team_b = str(row["team_b"]).strip()
            league = str(row["league"]).strip()
            start = pd.Timestamp(row["start_utc"])
            if start.tzinfo is None:
                start = start.tz_localize("UTC")
            start_time = _required_datetime(start.tz_convert("UTC").to_pydatetime())
            query = f"{team_a} {team_b} League of Legends"
            fixture = MarketFixture(
                fixture_id=match_key,
                competition_names=(league,),
                team_a_id=team_a.casefold(),
                team_b_id=team_b.casefold(),
                team_a_names=(team_a,),
                team_b_names=(team_b,),
                start_time=start_time,
                best_of=int(row.get("best_of") or 1),
            )
        else:
            query = "League of Legends"
            fixture = None

        gamma = PolymarketGammaAdapter()
        markets = gamma.search_markets(query, limit=50)
        supported = [
            market
            for market in markets
            if market.sports_market_type in {"child_moneyline", "moneyline", "totals"}
            and market.active
            and not market.closed
            and market.accepting_orders
        ]
        result["gamma"] = {
            "query": query,
            "market_count": len(markets),
            "supported_open_count": len(supported),
        }
        if fixture is not None:
            selection = select_best_market(fixture, supported)
            result["typed_match"] = {
                "requested_type": SupportedMarketType.SERIES_WINNER.value,
                "selected_market_id": selection.selected_market_id,
                "assessments": [
                    {
                        "market_id": item.market_id,
                        "matched": item.matched,
                        "reasons": list(item.reasons),
                    }
                    for item in selection.assessments
                ],
            }
            selected_market = _required_selected_market(
                supported,
                selection.selected_market_id,
            )
        else:
            selected_market = _first_supported_market(supported)
        batch = capture_minimum_order_book_batch(
            PolymarketClobClient(),
            token_ids=tuple(outcome.token_id for outcome in selected_market.outcomes),
        )
        quotes = []
        for outcome in selected_market.outcomes:
            pair = batch.observations.get(outcome.token_id)
            failure = batch.failures.get(outcome.token_id)
            if failure is not None or pair is None:
                quotes.append(
                    {
                        "outcome": outcome.name,
                        "token_id": outcome.token_id,
                        "ok": False,
                        "reason": failure.reason if failure else "book_unavailable",
                        "detail": failure.detail
                        if failure
                        else "book capture unavailable",
                    }
                )
                continue
            fill = confirmed_executable_fill(pair)
            quotes.append(
                {
                    "outcome": outcome.name,
                    "token_id": outcome.token_id,
                    "ok": True,
                    "requested_shares": str(fill.requested_shares),
                    "hypothetical_cost": str(fill.total_cost),
                    "executable_decimal_odds": fill.decimal_odds,
                    "minimum_order_sizes": [
                        str(observation.book.minimum_order_size) for observation in pair
                    ],
                    "book_timestamps": [
                        observation.book.timestamp for observation in pair
                    ],
                    "book_hashes": [observation.book.book_hash for observation in pair],
                }
            )
        result["clob"] = {
            "market_id": selected_market.market_id,
            "token_orientation_present": True,
            "observation_interval_seconds": 45,
            "quote_basis": "minimum_order_shares",
            "outcomes": quotes,
        }
    except Exception as error:
        result["ok"] = False
        result["error"] = f"{type(error).__name__}: {error}"
        sys.stdout.write(json.dumps(result, indent=2, default=str) + "\n")
        return 2
    result["ok"] = True
    sys.stdout.write(json.dumps(result, indent=2, default=str) + "\n")
    return 0


def _single_market_check_row(rows, match_key: str):
    if len(rows) != 1:
        raise ValueError(f"schedule match-key must resolve exactly once: {match_key}")
    return rows.iloc[0]


def _market_check_fixture_rows(match_key: str, *, schedule_path, report_dir):
    """Read one fixture from the saved schedule or newest daily report."""
    import json
    from pathlib import Path

    from oracle_bets_core.pd import pd

    schedule_file = Path(schedule_path)
    if schedule_file.is_file():
        schedule = pd.read_parquet(schedule_file)
        rows = schedule.loc[schedule["match_key"].astype(str).eq(match_key)]
        if not rows.empty:
            return rows.head(1)
    for report_path in sorted(Path(report_dir).glob("*.json"), reverse=True):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            schedule = pd.DataFrame(payload.get("schedule") or [])
        except (OSError, ValueError, TypeError):
            continue
        if "match_key" not in schedule:
            continue
        rows = schedule.loc[schedule["match_key"].astype(str).eq(match_key)]
        if not rows.empty:
            return rows.head(1)
    return pd.DataFrame()


def _required_datetime(value):
    from datetime import datetime

    if not isinstance(value, datetime):
        raise TypeError("matched fixture has no valid start time")
    return value


def _first_supported_market(markets):
    if not markets:
        raise ValueError("Gamma returned no supported open LoL market")
    return markets[0]


def _required_selected_market(markets, selected_market_id):
    selected = next(
        (market for market in markets if market.market_id == selected_market_id),
        None,
    )
    if selected is None:
        raise ValueError("Gamma returned no exact supported market for fixture")
    return selected


def _print_lol_schedule(args: argparse.Namespace) -> None:
    from lol_bets.data_generation.ingestion.schedule import fetch_and_store_schedule

    from oracle_bets_core.league_selection import actionable_leagues

    schedule = fetch_and_store_schedule(
        window_days=args.days,
        leagues=args.leagues or ",".join(actionable_leagues()),
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


def _main_model(args: argparse.Namespace) -> int:  # noqa: PLR0911, PLR0912
    import json
    from datetime import UTC, datetime
    from pathlib import Path

    from lol_bets.operations.models import (
        ModelRegistry,
        ModelRegistryError,
        load_candidate_review,
        register_current_candidate,
    )

    from oracle_bets_core.paths import MODEL_REGISTRY_DIR

    registry = ModelRegistry(
        Path(args.registry) if args.registry else MODEL_REGISTRY_DIR
    )
    if args.action in {"list", "review", "register-run"}:
        candidates = sorted(
            path.parent for path in registry.candidates.glob("*/manifest.json")
        )
        if args.action == "register-run":
            selected = (
                candidates[-1] if args.run_id == "latest" and candidates else None
            )
            if selected is None and args.run_id != "latest":
                selected = registry.candidates / args.run_id
            if selected is None or not registry.verify_bundle(selected.name):
                sys.stderr.write(
                    f"Registered training candidate not found: {args.run_id}\n"
                )
                return 2
            sys.stdout.write(f"Candidate already registered: {selected.name}\n")
            return 0
        manifests = [
            json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            for path in candidates
        ]
        if args.action == "review":
            manifests = [row for row in manifests if row["model_id"] == args.model_id]
            if not manifests:
                sys.stderr.write(f"Unknown model candidate: {args.model_id}\n")
                return 2
            manifests[0]["promotion_review"] = load_candidate_review(
                registry, args.model_id
            )
        if args.format == "json":
            sys.stdout.write(json.dumps(manifests, indent=2, sort_keys=True) + "\n")
        else:
            champion = registry.champion_id()
            for row in manifests:
                marker = "champion" if row["model_id"] == champion else "candidate"
                review = row.get("promotion_review") or {}
                review_text = (
                    f" review={review.get('status')}" if args.action == "review" else ""
                )
                sys.stdout.write(
                    f"{row['model_id']} {marker} target={row['target']} "
                    f"created={row['created_at']}{review_text}\n"
                )
        return 0
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


def _main_paper(args: argparse.Namespace) -> int:  # noqa: PLR0912
    import json
    from datetime import datetime
    from pathlib import Path

    from oracle_bets_core.evidence import EvidenceStore
    from oracle_bets_core.evidence.settlement import SettlementResult
    from oracle_bets_core.operations.paper_evidence import (
        PaperEvidenceError,
        capture_closing_snapshots,
        decide_paper,
        paper_rows,
        paper_show,
        performance_summary,
        quote_prop,
        settle_paper,
    )
    from oracle_bets_core.paths import EVIDENCE_DB

    store = EvidenceStore(Path(args.database) if args.database else EVIDENCE_DB)
    try:
        if args.action == "list":
            result = paper_rows(
                store, state=args.state, target=args.target, league=args.league
            )
        elif args.action == "show":
            result = paper_show(store, args.record_id)
        elif args.action == "quote-prop":
            proposal = quote_prop(
                store,
                forecast_id=args.forecast_id,
                line=args.line,
                over_odds=args.over_odds,
                under_odds=args.under_odds,
                source=args.source,
            )
            sys.stdout.write(f"Created research proposal {proposal}.\n")
            return 0
        elif args.action == "decide":
            position = decide_paper(
                store,
                proposal_id=args.proposal_id,
                decision=args.decision,
                reason=args.reason,
                actor_id=args.actor_id,
            )
            sys.stdout.write(
                f"Decision recorded{f'; opened {position}' if position else ''}.\n"
            )
            return 0
        elif args.action == "settle":
            if args.dry_run:
                from oracle_bets_core.evidence.settlement import (
                    PositionTerms,
                    settlement_pnl,
                )

                position = paper_show(store, args.position_id)
                pnl = settlement_pnl(
                    PositionTerms(
                        args.position_id,
                        position["stake_units"],
                        position["decimal_odds"],
                    ),
                    SettlementResult(args.result),
                )
                sys.stdout.write(f"PnL {pnl} units (dry run); no evidence written.\n")
                return 0
            settled_at = (
                datetime.fromisoformat(args.settled_at) if args.settled_at else None
            )
            settlement = settle_paper(
                store,
                position_id=args.position_id,
                result=SettlementResult(args.result),
                source_reference=args.source_reference,
                settled_at=settled_at,
                actor_id=args.actor_id,
                note=args.note,
            )
            sys.stdout.write(f"Recorded settlement {settlement}.\n")
            return 0
        elif args.action == "capture-closing":
            from oracle_bets_core.markets import PolymarketClobClient

            result = capture_closing_snapshots(
                store,
                client=PolymarketClobClient(),
                window_minutes=args.window_minutes,
                dry_run=args.dry_run,
            )
        else:
            since = datetime.fromisoformat(args.since) if args.since else None
            result = performance_summary(
                store,
                since=since,
                target=args.target,
                league=args.league,
            )
    except (PaperEvidenceError, FileNotFoundError) as error:
        sys.stderr.write(f"{error}\n")
        return 2
    if getattr(args, "format", "table") == "json":
        sys.stdout.write(json.dumps(result, indent=2, default=str) + "\n")
    elif isinstance(result, list):
        for row in result:
            sys.stdout.write(
                f"{row['state']:<8} {row['proposal_id']} "
                f"{row['league']} {row['target']}\n"
            )
    else:
        for key, value in result.items():
            sys.stdout.write(f"{key}: {value}\n")
    return 0


def _main_daily(args: argparse.Namespace) -> int:
    from collections import Counter

    from lol_bets.daily import DailyWorkflowConfig, run_daily_lol_workflow

    result = run_daily_lol_workflow(
        DailyWorkflowConfig(
            horizon_hours=args.horizon_hours,
            leagues=args.leagues,
            webhook_url=args.webhook_url,
            delivery_mode=args.discord_delivery_mode,
            dry_run=args.dry_run,
            skip_retrain=args.skip_retrain,
            skip_market_search=args.skip_market_search,
            targets=args.targets,
            feature_set=args.feature_set,
            max_features=args.max_features,
            ai_review=args.ai_review,
            openai_model=args.openai_model,
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


def _main_discord(args: argparse.Namespace) -> int:  # noqa: PLR0911
    import os

    from oracle_bets_discord.delivery import (
        DiscordDeliveryMode,
        resolve_delivery_mode,
    )

    mode = resolve_delivery_mode()
    required = ("DISCORD_TOKEN", "DISCORD_CHANNEL_ID", "DISCORD_OWNER_USER_ID")
    missing = [name for name in required if not os.getenv(name)]
    if args.action == "doctor":
        try:
            import discord  # noqa: F401

            sdk_available = True
        except ImportError:
            sdk_available = False
        sys.stdout.write(
            "Discord SDK: "
            f"{'available' if sdk_available else 'missing; install --extra discord-bot'}\n"
        )
        sys.stdout.write(
            f"Configuration: {'missing ' + ', '.join(missing) if missing else 'ok'}\n"
        )
        sys.stdout.write(f"Delivery mode: {mode.value}\n")
        if args.live and sdk_available and not missing:
            from oracle_bets_discord.delivery import check_gateway_access

            try:
                access = check_gateway_access(
                    os.environ["DISCORD_TOKEN"],
                    os.environ["DISCORD_CHANNEL_ID"],
                )
            except Exception as error:
                sys.stderr.write(f"Discord live check failed: {error}\n")
                return 2
            sys.stdout.write(
                f"Live access: bot={access['bot']} channel={access['channel']}\n"
            )
        return 0 if not args.live or (sdk_available and not missing) else 2
    if args.action == "run":
        if mode is not DiscordDeliveryMode.GATEWAY:
            sys.stderr.write("discord run requires DISCORD_DELIVERY_MODE=gateway\n")
            return 2
        if missing:
            sys.stderr.write(f"Missing Discord configuration: {', '.join(missing)}\n")
            return 2
        from oracle_bets_discord.bot import run_bot

        run_bot()
        return 0
    if mode is not DiscordDeliveryMode.WEBHOOK:
        sys.stderr.write("discord publish requires DISCORD_DELIVERY_MODE=webhook\n")
        return 2
    from oracle_bets_discord.delivery import publish_saved_report

    publish_saved_report(args.run_id)
    return 0


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
        "discord": _main_discord,
        "health": _main_health,
        "audit": _main_audit,
    }
    return handlers[args.domain](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
