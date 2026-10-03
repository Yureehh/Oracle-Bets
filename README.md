# Oracle Bets

Oracle Bets is a local-first League of Legends betting-research system focused
on calibrated probability accuracy, leakage prevention, conservative market
matching, and real paper-profit evidence.

It validates public 2024–2026 Oracle's Elixir history, generates strictly
chronological team/player features and ratings, reconstructs complete series,
and trains direct-series, map, next-map, duration, kills, and towers research
targets. The direct Winner V2 model is symmetric, calibrated, and compared with
a rating-only baseline. The owner submits exact Polymarket/Thunderpick links;
Polymarket books are read publicly while Thunderpick lines are entered manually.
One deterministic policy labels comparable outcomes `recommended`,
`exploration`, or `not_comparable` and records every reason. Positive-edge paper
proposals use provisional quarter-Kelly sizing with ticket, fixture, and total
exposure caps; other fractions are research comparisons, not active policies.
An optional LLM may
later explain stored decisions but can never create or alter them.

There is no automated betting, wallet, signing, private-key, order-submission,
or fund-movement code.

## Install

```bash
git clone https://github.com/Yureehh/Oracle-Bets.git
cd Oracle-Bets
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
uv run oracle-bets lol source-check
uv run oracle-bets lol reconcile-history
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

Source refresh is built into history ingestion. `market-review` uses only one or two
exact owner-selected Polymarket/Thunderpick links. Polymarket is read-only;
Thunderpick is link-plus-manual-lines. The review compares independent series,
experimental map/derived, and calibrated prop probabilities where compatible.
Unsupported specials are counted in the compact report and preserved in JSON.

Direct series is the only initial recommendation-capable target. Map, next-map,
derived totals/handicaps, and props begin as exploration. Forced exploration
coverage is kept separate from recommendation evidence and is never proof that
a wager is profitable.

Run `uv run oracle-bets discord run`, then use the owner-only `/oracle` hub. The
first row contains Review Markets, Record Bet, Open Bets, and Closed Bets; the
second contains Schedule, Performance, and Health. Every paper or real entry is
owner-confirmed and settlement is manual.

The [pipeline status](docs/pipeline-status.md) distinguishes implemented code,
local generated artifacts, and live paper evidence. A healthy model or a passed
backtest does not establish a profitable betting strategy.

## Explicit research retuning

```bash
uv run oracle-bets lol research --targets all
uv run oracle-bets lol refit-research \
  <MAP_RUN_ID> <SERIES_RUN_ID> <LENGTH_RUN_ID> <KILLS_RUN_ID> <TOWERS_RUN_ID>
```

These commands require a clean, reviewed Git worktree. `research` runs one
independent Optuna study per supported target; `refit-research` validates the
five study inputs and trains a research-only bundle. Neither promotes a model
or parameters. Serving changes require separate replay, review, and promotion
gates. The experimental next-map model is not part of the normal bundle.

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
