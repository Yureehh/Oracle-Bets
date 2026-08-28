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
        help="Fetch, save, and print upcoming LoL matches",
    )
    schedule.add_argument("--days", type=int, default=14)
    schedule.add_argument("--leagues", default=None)
    market_check = lol_sub.add_parser(
        "market-check",
        help="Verify public read-only Polymarket discovery and CLOB data",
    )
    market_check.add_argument("--match-key", default=None)
    market_review = lol_sub.add_parser(
        "market-review",
        help="Review one owner-selected Polymarket/Thunderpick LoL fixture",
    )
    market_review.add_argument("urls", nargs="+")
    market_review.add_argument(
        "--manual-lines",
        help="JSON file containing owner-entered Thunderpick lines",
    )
    market_review.add_argument("--format", choices=["table", "json"], default="table")
    lol_sub.add_parser(
        "validate-data",
        help="Validate generated LoL training tables without model training",
    )
    lol_sub.add_parser(
        "build-series",
        help="Reconstruct complete historical series and frozen prematch datasets",
    )
    winner_validation = lol_sub.add_parser(
        "validate-winner-model",
        help="Validate the independent direct-series serving contract",
    )
    winner_validation.add_argument(
        "--format", choices=["table", "json"], default="table"
    )
    market_validation = lol_sub.add_parser(
        "validate-market-strategies",
        help="Validate exact-link series, map, totals, and handicap contracts",
    )
    market_validation.add_argument(
        "--format", choices=["table", "json"], default="table"
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
        choices=["auto", "full", "compact", "selected"],
        default="auto",
    )
    retune.add_argument("--max-features", type=int, default=120)
    promote_tuning = lol_sub.add_parser(
        "promote-tuning",
        help="Promote one reviewed complete retuning run",
    )
    promote_tuning.add_argument("run_id")
    review_tuning = lol_sub.add_parser(
        "review-tuning",
        help="Review Winner V2 tuning on sealed evidence before promotion",
    )
    review_tuning.add_argument("run_id")
    review_tuning.add_argument("--format", choices=["table", "json"], default="table")
    research = lol_sub.add_parser(
        "research",
        help="Run isolated target studies without promoting parameters or models",
    )
    research.add_argument("--targets", default="all")
    research.add_argument("--include-next-map", action="store_true")


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
        default="compact",
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
    for action in ("status", "promote", "rollback", "quarantine"):
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
    review.add_argument(
        "--refresh",
        action="store_true",
        help="Replay the immutable candidate under its existing review policy",
    )
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

    bet = sub.add_parser("bet", help="Owner-recorded paper and real bet evidence")
    bet_sub = bet.add_subparsers(dest="action", required=True)
    listing = bet_sub.add_parser("list", help="List unified bet records")
    listing.add_argument("--database", default=None)
    listing.add_argument("--state", choices=["open", "settled"])
    listing.add_argument("--mode", choices=["paper", "real"])
    listing.add_argument("--provider")
    listing.add_argument("--target")
    listing.add_argument("--format", choices=["table", "json"], default="table")
    show = bet_sub.add_parser("show", help="Show one bet and its lifecycle")
    show.add_argument("bet_id")
    show.add_argument("--database", default=None)
    show.add_argument("--format", choices=["table", "json"], default="table")
    record = bet_sub.add_parser("record", help="Record a manually placed bet")
    record.add_argument("--database", default=None)
    record.add_argument("--review-id", required=True)
    record.add_argument("--market-id", required=True)
    record.add_argument("--mode", required=True, choices=["paper", "real"])
    record.add_argument("--currency", required=True)
    record.add_argument("--bankroll-before", required=True)
    record.add_argument("--stake-percent", required=True)
    record.add_argument("--stake-amount")
    record.add_argument("--accepted-odds", required=True)
    record.add_argument("--reason", required=True)
    record.add_argument("--actor-id", default="owner-cli")
    record.add_argument("--opened-at")
    record.add_argument(
        "--idempotency-key",
        help="Reuse only when retrying the same owner record action",
    )
    settle = bet_sub.add_parser("settle", help="Record one owner-verified settlement")
    settle.add_argument("--database", default=None)
    settle.add_argument("--bet-id", required=True)
    settle.add_argument(
        "--result",
        required=True,
        choices=["win", "loss", "push", "void"],
    )
    settle.add_argument("--source-reference", required=True)
    settle.add_argument("--note")
    settle.add_argument("--actor-id", default="owner-cli")
    settle.add_argument("--settled-at", default=None)
    settle.add_argument("--closing-odds")
    result = bet_sub.add_parser("result", help="Record one owner-sourced result fact")
    result.add_argument("--database", default=None)
    result.add_argument("--bet-id", required=True)
    result.add_argument("--source-reference", required=True)
    result.add_argument("--winning-selection")
    result.add_argument("--observed-value")
    result.add_argument("--actor-id", default="owner-cli")
    result.add_argument("--recorded-at")
    correct = bet_sub.add_parser(
        "correct-settlement", help="Supersede a mistaken settlement append-only"
    )
    correct.add_argument("--database", default=None)
    correct.add_argument("--bet-id", required=True)
    correct.add_argument(
        "--result", required=True, choices=["win", "loss", "push", "void"]
    )
    correct.add_argument("--source-reference", required=True)
    correct.add_argument("--correction-reason", required=True)
    correct.add_argument("--note")
    correct.add_argument("--closing-odds")
    correct.add_argument("--actor-id", default="owner-cli")
    correct.add_argument("--corrected-at")
    performance = bet_sub.add_parser("performance", help="Report performance by mode")
    performance.add_argument("--database", default=None)
    performance.add_argument("--mode", required=True, choices=["paper", "real"])
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
    daily_lol.add_argument(
        "--discord-delivery-mode",
        choices=["gateway", "off"],
        default=None,
    )
    daily_lol.add_argument("--skip-retrain", action="store_true")
    daily_lol.add_argument("--targets", default="all")
    daily_lol.add_argument(
        "--feature-set",
        choices=["full", "compact", "selected"],
        default="compact",
    )
    daily_lol.add_argument("--max-features", type=int, default=120)
    discord = sub.add_parser("discord", help="Discord delivery operations")
    discord_sub = discord.add_subparsers(dest="action", required=True)
    doctor = discord_sub.add_parser("doctor", help="Validate Discord configuration")
    doctor.add_argument("--live", action="store_true")
    discord_sub.add_parser("run", help="Run the owner-only Gateway bot")


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


def _main_lol(args: argparse.Namespace) -> int:  # noqa: PLR0911, PLR0912, PLR0915
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

        from lol_bets.data_generation.ingestion.history import current_history_data_path
        from lol_bets.operations.identity import sync_history_identity_graph

        from oracle_bets_core.evidence import EvidenceStore
        from oracle_bets_core.paths import EVIDENCE_DB, RAW_CURRENT_POINTER, RAW_DATA
        from oracle_bets_core.pd import pd

        result = sync_history_identity_graph(
            EvidenceStore(EVIDENCE_DB),
            pd.read_parquet(
                current_history_data_path(
                    RAW_DATA,
                    pointer_path=RAW_CURRENT_POINTER,
                )
            ),
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
    if args.action == "build-series":
        import json

        from lol_bets.data_generation.series import build_series_artifacts

        sys.stdout.write(
            json.dumps(build_series_artifacts(), indent=2, sort_keys=True) + "\n"
        )
        return 0
    if args.action == "validate-winner-model":
        import json

        from lol_bets.operations.winner_validation import validate_winner_model

        report = validate_winner_model()
        if args.format == "json":
            sys.stdout.write(
                json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
            )
        else:
            sys.stdout.write(
                f"Winner V2: {'READY' if report.ok else 'BLOCKED'} "
                f"(champion={report.champion_id or 'none'}, "
                f"features={report.feature_count})\n"
            )
            for failure in report.failures:
                sys.stdout.write(f"- {failure}\n")
        return 0 if report.ok else 2
    if args.action == "market-check":
        return _check_lol_markets(args.match_key)
    if args.action == "validate-market-strategies":
        import json

        from lol_bets.operations.market_validation import validate_market_strategies

        report = validate_market_strategies()
        if args.format == "json":
            sys.stdout.write(
                json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
            )
        else:
            status = "READY" if report.ok else "BLOCKED"
            sys.stdout.write(f"LoL market strategies: {status}\n")
            for name, passed in report.checks.items():
                sys.stdout.write(f"- {'OK' if passed else 'FAILED'}: {name}\n")
            sys.stdout.write(
                "- Experimental: " + ", ".join(report.experimental_strategies) + "\n"
            )
            for strategy, metrics in report.derived_backtest_metrics.items():
                sys.stdout.write(
                    f"- Held-out {strategy}: log_loss={metrics['log_loss']:.4f}, "
                    f"brier={metrics['brier']:.4f}, accuracy={metrics['accuracy']:.1%}\n"
                )
        return 0 if report.ok else 2
    if args.action == "market-review":
        import json

        from lol_bets.operations.manual_market import review_polymarket_events

        from oracle_bets_core.evidence import EvidenceStore

        manual_lines = []
        if args.manual_lines:
            from pathlib import Path

            try:
                manual_lines = json.loads(
                    Path(args.manual_lines).read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                sys.stderr.write(f"Invalid --manual-lines file: {error}\n")
                return 2
            if not isinstance(manual_lines, list):
                sys.stderr.write("--manual-lines must contain a JSON list.\n")
                return 2
        result = review_polymarket_events(
            args.urls,
            manual_lines=manual_lines,
            store=EvidenceStore(),
        )
        if args.format == "json":
            sys.stdout.write(
                json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
            )
        else:
            sys.stdout.write(
                f"Reviewed {result.fixtures} fixture(s); "
                f"{result.predictions} prediction(s); "
                f"{result.comparisons} supported comparison(s).\n"
                f"JSON: {result.report_paths[0]}\n"
                f"Markdown: {result.report_paths[1]}\n"
            )
            for failure in result.failures:
                sys.stdout.write(
                    f"- {failure['url']}: {failure['reason']} ({failure['detail']})\n"
                )
            for ignored in result.ignored_links:
                sys.stdout.write(f"- Ignored: {ignored['detail']}\n")
        return 0 if result.ok else 2
    if args.action == "promote-tuning":
        from lol_bets.training import promote_tuning_run

        paths = promote_tuning_run(args.run_id)
        sys.stdout.write(f"Promoted {len(paths)} tuned parameter files.\n")
        return 0
    if args.action == "review-tuning":
        import json

        from lol_bets.training import review_tuning_run

        report = review_tuning_run(args.run_id)
        if args.format == "json":
            sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        else:
            reasons = report.get("reasons")
            reason_text = (
                ", ".join(str(reason) for reason in reasons)
                if isinstance(reasons, list)
                else ""
            )
            sys.stdout.write(
                f"Tuning review: {report['status']}\nReasons: {reason_text or 'none'}\n"
            )
        return 0 if report["status"] == "approved" else 2
    if args.action in {"train", "retune", "research"}:
        from lol_bets.data_generation.ingestion.source import (
            OracleSourceReadinessError,
            require_oracle_source_ready,
        )

        try:
            require_oracle_source_ready()
        except OracleSourceReadinessError as error:
            sys.stderr.write(f"{error}\n")
            return 2
        if args.action == "research":
            _research_lol(args)
        else:
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
    from lol_bets.operations.market_actions import LOL_RESOLUTION_RULE_TERMS

    from oracle_bets_core.markets import (
        MarketFixture,
        PolymarketClobClient,
        PolymarketGammaAdapter,
        SupportedMarketType,
        capture_current_order_books,
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
                resolution_rule_terms=LOL_RESOLUTION_RULE_TERMS,
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
        batch = capture_current_order_books(
            PolymarketClobClient(),
            token_ids=tuple(outcome.token_id for outcome in selected_market.outcomes),
        )
        quotes = []
        for outcome in selected_market.outcomes:
            observation = batch.observations.get(outcome.token_id)
            failure = batch.failures.get(outcome.token_id)
            if failure is not None or observation is None:
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
            fill = observation.fill
            quotes.append(
                {
                    "outcome": outcome.name,
                    "token_id": outcome.token_id,
                    "ok": True,
                    "requested_shares": str(fill.requested_shares),
                    "hypothetical_cost": str(fill.total_cost),
                    "executable_decimal_odds": fill.decimal_odds,
                    "minimum_order_size": str(observation.book.minimum_order_size),
                    "book_timestamp": observation.book.timestamp,
                    "book_hash": observation.book.book_hash,
                    "warnings": list(batch.warnings.get(outcome.token_id, ())),
                }
            )
        result["clob"] = {
            "market_id": selected_market.market_id,
            "token_orientation_present": True,
            "observation_count": 1,
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
    from lol_bets.data_generation.ingestion.schedule import (
        fetch_and_store_schedule,
        resolved_team_name,
    )

    from oracle_bets_core.league_selection import actionable_leagues
    from oracle_bets_core.paths import SCHEDULE

    schedule = fetch_and_store_schedule(
        window_days=args.days,
        leagues=args.leagues or ",".join(actionable_leagues()),
        save_path=SCHEDULE,
    )
    if schedule.empty:
        sys.stdout.write("No upcoming matches found.\n")
        return
    display = schedule[
        ["start_utc", "league", "team_a", "team_b", "best_of", "status"]
    ].copy()
    display["team_a"] = display["team_a"].map(resolved_team_name)
    display["team_b"] = display["team_b"].map(resolved_team_name)
    display = display[display["team_a"].ne("") & display["team_b"].ne("")].copy()
    if display.empty:
        sys.stdout.write("No upcoming matches with confirmed teams found.\n")
        return
    team_a = display.pop("team_a")
    team_b = display.pop("team_b")
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


def _research_lol(args: argparse.Namespace) -> None:
    from lol_bets.training import run_research_studies

    reports = run_research_studies(
        targets=args.targets,
        include_next_map=args.include_next_map,
    )
    for target, report in reports.items():
        sys.stdout.write(f"{target}: {report}\n")


def _main_model(args: argparse.Namespace) -> int:  # noqa: PLR0911, PLR0912, PLR0915
    import json
    from datetime import UTC, datetime
    from pathlib import Path

    from lol_bets.operations.models import (
        ModelRegistry,
        ModelRegistryError,
        PromotionPolicy,
        load_candidate_review,
        register_current_candidate,
        review_candidate_on_sealed_rows,
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
            review_payload = load_candidate_review(registry, args.model_id)
            if args.refresh:
                policy = PromotionPolicy(
                    (review_payload or {}).get("policy", PromotionPolicy.ROUTINE.value)
                )
                review_candidate_on_sealed_rows(
                    registry,
                    args.model_id,
                    policy=policy,
                    automatic=False,
                    reviewed_at=datetime.now(UTC),
                )
                review_payload = load_candidate_review(registry, args.model_id)
            manifests[0]["promotion_review"] = review_payload
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
        actionability = registry.actionability(champion)
        sys.stdout.write(
            f"Champion: {champion} ({'healthy' if healthy else 'unhealthy'}, "
            f"{actionability['status']})\n"
        )
        return 0 if healthy and registry.is_actionable(champion) else 2
    try:
        if args.action == "quarantine":
            registry.quarantine(
                args.model_id,
                quarantined_at=datetime.now(UTC),
                reason=args.reason,
            )
            sys.stdout.write(f"Model {args.model_id} is research-only.\n")
            return 0
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
    from oracle_bets_core.evidence.schema import SCHEMA_VERSION
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
        integrity = store.integrity_check()
        version = store.schema_version()
        sys.stdout.write(
            f"Evidence database: integrity={integrity}, schema={version} "
            f"(expected={SCHEMA_VERSION})\n"
        )
        return 0 if integrity == "ok" and version == SCHEMA_VERSION else 2
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


def _main_bet(args: argparse.Namespace) -> int:  # noqa: PLR0912
    import json
    from datetime import datetime
    from pathlib import Path

    from oracle_bets_core.evidence import EvidenceStore
    from oracle_bets_core.operations.bets import (
        BetEvidenceError,
        list_bets,
        performance_summary,
        record_bet,
        record_bet_result,
        replace_bet_settlement,
        settle_bet,
        show_bet,
    )
    from oracle_bets_core.paths import EVIDENCE_DB

    store = EvidenceStore(Path(args.database) if args.database else EVIDENCE_DB)
    try:
        if args.action == "list":
            result = list_bets(
                store,
                state=args.state,
                mode=args.mode,
                provider=args.provider,
                target=args.target,
            )
        elif args.action == "show":
            result = show_bet(store, args.bet_id)
        elif args.action == "record":
            bet_id = record_bet(
                store,
                review_id=args.review_id,
                market_id=args.market_id,
                mode=args.mode,
                currency=args.currency,
                bankroll_before=args.bankroll_before,
                stake_percent=args.stake_percent,
                stake_amount=args.stake_amount,
                accepted_odds=args.accepted_odds,
                reason=args.reason,
                actor_id=args.actor_id,
                opened_at=(
                    datetime.fromisoformat(args.opened_at) if args.opened_at else None
                ),
                idempotency_key=args.idempotency_key,
            )
            sys.stdout.write(
                f"Recorded {args.mode} bet {bet_id}; no wager was placed.\n"
            )
            return 0
        elif args.action == "settle":
            event_id = settle_bet(
                store,
                bet_id=args.bet_id,
                result=args.result,
                source_reference=args.source_reference,
                actor_id=args.actor_id,
                note=args.note,
                settled_at=(
                    datetime.fromisoformat(args.settled_at) if args.settled_at else None
                ),
                closing_odds=args.closing_odds,
            )
            sys.stdout.write(f"Recorded settlement {event_id}.\n")
            return 0
        elif args.action == "result":
            event_id = record_bet_result(
                store,
                bet_id=args.bet_id,
                source_reference=args.source_reference,
                winning_selection=args.winning_selection,
                observed_value=args.observed_value,
                actor_id=args.actor_id,
                recorded_at=(
                    datetime.fromisoformat(args.recorded_at)
                    if args.recorded_at
                    else None
                ),
            )
            sys.stdout.write(f"Recorded result fact {event_id}.\n")
            return 0
        elif args.action == "correct-settlement":
            event_id = replace_bet_settlement(
                store,
                bet_id=args.bet_id,
                result=args.result,
                source_reference=args.source_reference,
                correction_reason=args.correction_reason,
                actor_id=args.actor_id,
                note=args.note,
                corrected_at=(
                    datetime.fromisoformat(args.corrected_at)
                    if args.corrected_at
                    else None
                ),
                closing_odds=args.closing_odds,
            )
            sys.stdout.write(f"Recorded corrected settlement {event_id}.\n")
            return 0
        else:
            result = performance_summary(store, mode=args.mode)
    except (BetEvidenceError, FileNotFoundError) as error:
        sys.stderr.write(f"{error}\n")
        return 2
    if getattr(args, "format", "table") == "json":
        sys.stdout.write(json.dumps(result, indent=2, default=str) + "\n")
    elif isinstance(result, list):
        for row in result:
            sys.stdout.write(
                f"{row['state']:<8} {row['mode']:<5} {row['id']} "
                f"{row['provider']} {row['target']} {row['selection']}\n"
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
            delivery_mode=args.discord_delivery_mode,
            dry_run=args.dry_run,
            skip_retrain=args.skip_retrain,
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


def _main_discord(args: argparse.Namespace) -> int:
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
    return 1


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
        "bet": _main_bet,
        "daily": _main_daily,
        "discord": _main_discord,
        "health": _main_health,
        "audit": _main_audit,
    }
    return handlers[args.domain](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
