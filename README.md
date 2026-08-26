# Oracle Bets

Oracle Bets is a local-first League of Legends betting-research system focused
on calibrated probability accuracy, leakage prevention, conservative market
matching, and real paper-profit evidence.

It downloads the public 2024–2026 Oracle's Elixir files into a validated local
cache, generates chronological team/player features and ratings, reconstructs
complete historical series, and normally trains five targets. The actionable
Winner V2 model is a direct, symmetric series-winner model with a rating
baseline, ten week-block LightGBM members, held-out calibration, and a
conservative probability bound. The owner submits exact Polymarket event links;
the system inventories their contracts, compares independent probabilities with
one current public quote per supported outcome, records compact JSON/Markdown
evidence, and serves a unified paper/real tracking ledger through one owner-only
Discord Gateway bot. Thunderpick lines are entered manually and never scraped.

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
```

## First data setup

```bash
uv run oracle-bets lol ingest
uv run oracle-bets lol validate-data
uv run oracle-bets lol build-series
```

These data commands are not needed before every paper review. For the normal
owner flow, keep the Gateway bot running and use either Discord or the CLI:

```bash
uv run oracle-bets discord run
uv run oracle-bets lol schedule --days 14
uv run oracle-bets lol market-review "<URL1> <URL2>"
uv run oracle-bets bet list --state open --mode paper
```

The source refresh is built into ingestion. `market-review` uses only one or two
exact owner-selected Polymarket/Thunderpick links. Polymarket is read-only;
Thunderpick is link-plus-manual-lines. The review compares independent series,
experimental map/derived, and calibrated prop probabilities where compatible.
Unsupported specials are counted in the compact report and preserved in JSON.

Run `uv run oracle-bets discord run`, then use the owner-only `/oracle` hub. The
first row contains Review Markets, Record Bet, Open Bets, and Closed Bets; the
second contains Schedule, Performance, and Health. Every paper or real entry is
owner-confirmed and settlement is manual.

## Explicit research retuning

```bash
uv run oracle-bets lol retune \
  --targets series_winner --feature-set full
uv run oracle-bets lol review-tuning <series-run-id> --format json
uv run oracle-bets lol promote-tuning <series-run-id>
uv run oracle-bets lol train --targets all --feature-set full
```

Winner tuning remains an explicit research action. Continue only when the
series review is `approved` or `approved_with_warnings` and every hard safety
gate passes. Warnings remain recorded for later evidence review. Routine
training refits with reviewed parameters and never runs Optuna. The experimental
next-map model is trained only when named explicitly and is not required by the
normal bundle.

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
- `ops/launchd`: optional always-on Gateway-bot example.

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
uv run oracle-bets lol validate-market-strategies
uv run oracle-bets lol market-check
uv run oracle-bets evidence health
uv run oracle-bets daily lol --dry-run
uv run oracle-bets discord doctor
```

Start with the [ordered CLI and Discord usage guide](docs/commands.md), then read
the [system documentation](docs/index.md).

Historical data is supplied by [Oracle's Elixir](https://www.oracleselixir.com).
This is research software, not financial advice; betting can lose money.
