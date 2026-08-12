# Oracle Bets

Oracle Bets is a local-first League of Legends betting-research system focused
on calibrated probability accuracy, leakage prevention, conservative market
matching, and real paper-profit evidence.

It downloads the public 2024–2026 Oracle's Elixir files into a validated local
cache, generates chronological team/player features and ratings, reconstructs
complete historical series, and trains six research targets. The actionable
Winner V2 model is a direct, symmetric series-winner model with a rating
baseline, ten week-block LightGBM members, held-out calibration, and a
conservative probability bound. Upcoming PandaScore fixtures are compared
read-only with Polymarket, recorded in JSON/Markdown evidence, and delivered by
one owner-only Discord Gateway bot.

There is no automated betting, wallet, signing, private-key, order-submission,
or fund-movement code.

## Install

```bash
cd /Users/yureeh/dev/oracle_bets
uv sync --extra dev --extra discord-bot --extra ai-review
source .venv/bin/activate
```

Refresh the public source with `uv run oracle-bets lol source-refresh`. Set
provider secrets outside Git:

```bash
export PANDASCORE_API_KEY="..."
export DISCORD_TOKEN="..."
export DISCORD_CHANNEL_ID="..."
export DISCORD_OWNER_USER_ID="..."
export DISCORD_DELIVERY_MODE="gateway"
# Optional advisory service:
export OPENAI_API_KEY="..."
```

## Normal workflow

```bash
uv run oracle-bets lol source-refresh
uv run oracle-bets lol source-check
uv run oracle-bets lol ingest
uv run oracle-bets lol validate-data
uv run oracle-bets lol build-series
uv run oracle-bets lol train --targets all --feature-set compact
uv run oracle-bets model list
uv run oracle-bets model review <candidate-id> --format json
uv run oracle-bets model promote <candidate-id> \
  --reason "reviewed first Winner V2 champion"
uv run oracle-bets lol validate-winner-model
uv run oracle-bets lol health
uv run oracle-bets daily lol --dry-run --no-ai-review
```

Routine training refits weights and calibrators with reviewed production
hyperparameters and never runs Optuna.

## Explicit research retuning

```bash
uv run oracle-bets lol retune \
  --targets series_winner,next_map_winner --feature-set compact
uv run oracle-bets lol promote-tuning <run-id>
uv run oracle-bets lol train --targets all --feature-set compact
```

Retuning writes isolated research artifacts under a timestamped report.
Promotion is explicit and publishes only the complete reviewed target set from
that run. Any model bundle derived from Optuna remains manual-promotion only.

## Repository layout

- `packages/oracle-bets-core`: CLI, evidence, health, market, settlement, and lifecycle support.
- `packages/lol-bets`: LoL ingestion, features, ratings, training, calibration, inference, and daily workflow.
- `packages/oracle-bets-discord`: safe report delivery and owner-only paper controls.
- `config`: source, league, identity, feature, and reviewed parameter contracts.
- `data/lol`: generated datasets.
- `data/state`: canonical evidence and immutable model registry.
- `models/lol`: bootstrap serving artifacts and temporary training staging.
- `reports/lol`: ingestion, training, and daily review artifacts.
- `notebooks/lol`: thin read-only analysis notebooks.
- `ops/launchd`: daily, monthly, hourly market-watch, and Gateway-bot examples.

Generated data, databases, models, logs, and reports are ignored. Reviewed
configuration, docs, cleared notebooks, and migration manifests are tracked.

## Verification

```bash
uv run pytest tests/lol
uv run pytest tests/core
uv run ruff check packages tests
uv run ty check packages
uv run mkdocs build --strict
uv run oracle-bets lol health
uv run oracle-bets lol validate-data
uv run oracle-bets lol validate-winner-model
uv run oracle-bets lol market-check
uv run oracle-bets evidence health
uv run oracle-bets daily lol --dry-run --no-ai-review
uv run oracle-bets discord doctor
```

Read the [documentation](docs/index.md) and [command reference](docs/commands.md).

Historical data is supplied by [Oracle's Elixir](https://www.oracleselixir.com).
This is research software, not financial advice; betting can lose money.
