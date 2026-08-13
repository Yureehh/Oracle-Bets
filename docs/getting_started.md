# Getting started

## Install

```bash
cd /Users/yureeh/dev/oracle_bets
uv sync --extra dev --extra discord-bot --extra ai-review
source .venv/bin/activate
```

Create an ignored `.env`, then run `chmod 600 .env`:

```dotenv
PANDASCORE_API_KEY=
DISCORD_TOKEN=
DISCORD_CHANNEL_ID=
DISCORD_OWNER_USER_ID=
DISCORD_DELIVERY_MODE=gateway
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5.6-luna
```

PandaScore plus the three Discord bot values are needed for the complete
interactive workflow; OpenAI is optional and failure-safe. Polymarket public reads
need no key or VPN.

Oracle's Elixir is fetched from the reviewed public Google Drive file IDs into
the generated local cache `data/lol/raw/oracles_elixir_cache`. Google Drive
Desktop is not required. `ORACLES_ELIXIR_LOCAL_DIR` may point to another
owner-managed local cache, but `source-refresh` refuses cloud-backed symlinks
so it cannot accidentally modify the public source.

The download uses Drive's anonymous bulk-export path—the same temporary ZIP
workflow as **Download all**. It needs no Google credential but is an
undocumented upstream interface, so every archive and CSV is validated before
the existing cache is replaced.

The pipeline does not use AWS and never modifies Google Drive files.

## First clean research rebuild

For a completely new assessment workspace, preserve only tracked documentation:

```bash
find data/lol/raw -mindepth 1 -maxdepth 1 \
  ! -name README.md -delete
find data/lol/interim -mindepth 1 -depth -delete
find data/lol/processed -mindepth 1 -depth -delete
find data/state -mindepth 1 -depth ! -name README.md -delete
find models/lol -mindepth 1 -depth ! -name README.md -delete
find reports/lol -mindepth 1 -depth ! -name README.md -delete
find logs -type f ! -name README.md ! -name .gitkeep -delete
```

Then:

```bash
uv run oracle-bets lol source-refresh
uv run oracle-bets lol source-check
uv run oracle-bets lol reconcile-history
uv run oracle-bets lol validate-data
uv run oracle-bets lol build-series
uv run oracle-bets lol retune \
  --targets series_winner --feature-set full
uv run oracle-bets lol review-tuning <run-id> --format json
```

Review the printed run directory, especially temporal metrics, calibration,
probability-sum checks, cohorts, and feature attribution. Continue only when
`review-tuning` returns `status: approved`; a blocked study must remain isolated.
The experimental next-map target still needs its own sealed tuning-review gate
and is not part of the bootstrap procedure yet. After both Winner V2 targets
have reviewed fixed parameters:

```bash
uv run oracle-bets lol promote-tuning <run-id>
uv run oracle-bets lol train --targets all --feature-set compact
uv run oracle-bets model list
uv run oracle-bets model review <candidate-id> --format json
uv run oracle-bets model promote <candidate-id> \
  --reason "reviewed bootstrap champion"
uv run oracle-bets evidence init
uv run pytest tests/lol tests/core
uv run ruff check packages tests
uv run ty check packages
uv run mkdocs build --strict
uv run oracle-bets lol health
uv run oracle-bets lol validate-winner-model
uv run oracle-bets evidence health
uv run oracle-bets lol market-check
uv run oracle-bets lol schedule --days 2
uv run oracle-bets daily lol --dry-run --skip-market-search
uv run oracle-bets daily lol --dry-run
uv run oracle-bets discord doctor --live
```

This explicit first pass retunes because the canonical symmetric feature schema
changed. Future scheduled runs retrain without retuning. As of August 14, 2026,
study `20260813T233204_354446Z` is blocked and the subsequent promotion/training
commands must not be run for that study.

If existing evidence and the model registry are valuable, back them up first
and omit the `data/state` cleanup line. Removing it intentionally resets paper
history and champion/candidate state; it is not part of a normal retrain.

## Start paper testing

Run the normal workflow once, inspect the one JSON/Markdown pair it prints,
then review only deterministic proposals:

```bash
uv run oracle-bets daily lol
uv run oracle-bets paper list --state pending
uv run oracle-bets paper show <proposal-id> --format json
uv run oracle-bets paper requote <proposal-id> --format json
uv run oracle-bets paper decide \
  --proposal-id <proposal-id> --decision accept \
  --requote-token <120-second-token> --reason "owner confirmed fresh quote"
uv run oracle-bets paper list --state open
uv run oracle-bets paper capture-closing --dry-run --format json
uv run oracle-bets paper settle \
  --position-id <position-id> --result win \
  --source-reference <result-url-or-id>
uv run oracle-bets paper performance --format json
```

For a manual scalar prop line:

```bash
uv run oracle-bets paper quote-prop \
  --forecast-id <forecast-id> --line 26.5 \
  --over-odds 1.91 --under-odds 1.91 --source bookmaker
```

Use `uv run oracle-bets discord run` for persistent owner-only buttons,
two-phase confirmation, settlement modals, and closing observations. The
launchd examples are documented under `ops/README.md`.
