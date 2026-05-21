"""Command-line entrypoint for the Oracle Bets suite."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oracle-bets")
    sub = parser.add_subparsers(dest="domain", required=True)

    lol = sub.add_parser("lol", help="League of Legends workflows")
    lol_sub = lol.add_subparsers(dest="action", required=True)
    lol_sub.add_parser("health", help="Check LoL training and inference artifacts")
    lol_sub.add_parser("ingest", help="Run the LoL ingestion/feature pipeline")
    train = lol_sub.add_parser("train", help="Train LoL prediction models")
    train.add_argument(
        "--model-type", choices=["lightgbm", "tabnet"], default="lightgbm"
    )
    train.add_argument(
        "--feature-selection",
        choices=["none", "importance", "cumulative", "rfecv", "boruta"],
        default="none",
    )

    discord = sub.add_parser("discord", help="Discord bot workflows")
    discord_sub = discord.add_subparsers(dest="action", required=True)
    discord_sub.add_parser("run", help="Run the Discord bot")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.domain == "lol" and args.action == "ingest":
        from lol_bets.pipeline import DataGenerator

        DataGenerator().run()
        return 0

    if args.domain == "lol" and args.action == "health":
        from lol_bets.module import LoLBetsModule

        module = LoLBetsModule()
        for health in (module.artifact_health(), module.training_artifact_health()):
            sys.stdout.write(
                f"{health.module_id}: {'ok' if health.ok else 'unhealthy'}\n"
            )
            for check in health.checks:
                status = "ok" if check.ok else f"missing ({check.reason})"
                sys.stdout.write(f"  {check.name}: {status} - {check.path}\n")
        return (
            0
            if module.artifact_health().ok and module.training_artifact_health().ok
            else 2
        )

    if args.domain == "lol" and args.action == "train":
        from lol_bets.training import train_models

        train_models(
            model_type=args.model_type,
            feature_selection=args.feature_selection,
        )
        return 0

    if args.domain == "discord" and args.action == "run":
        from oracle_bets_discord.bot import run_bot

        run_bot()
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
