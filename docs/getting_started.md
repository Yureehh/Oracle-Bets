# Getting started

## Install

```bash
cd /Users/yureeh/dev/oracle_bets
uv sync --extra dev
source .venv/bin/activate
```

Google Drive Desktop must expose the Oracle's Elixir folder locally and keep
the 2024, 2025, and 2026 CSV files available offline. The default symlink is
`data/lol/raw/oracles_elixir`; alternatively set
`ORACLES_ELIXIR_LOCAL_DIR=/absolute/path/to/OE Public Match Data`.

The pipeline does not use AWS and does not scrape Google Drive HTTP pages.

## First clean research rebuild

For a completely new assessment workspace, preserve only tracked documentation
and the `raw/oracles_elixir` Google Drive symlink:

```bash
find data/lol/raw -mindepth 1 -maxdepth 1 \
  ! -name oracles_elixir -delete
find data/lol/interim -mindepth 1 -depth -delete
find data/lol/processed -mindepth 1 -depth -delete
find data/state -mindepth 1 -depth ! -name README.md -delete
find models/lol -mindepth 1 -depth ! -name README.md -delete
find reports/lol -mindepth 1 -depth ! -name README.md -delete
find logs -type f ! -name README.md ! -name .gitkeep -delete
```

Then:

```bash
uv run oracle-bets lol reconcile-history
uv run oracle-bets lol validate-data
uv run oracle-bets lol retune --targets all --feature-set compact
```

Review the printed run directory, especially temporal metrics, calibration,
probability-sum checks, cohorts, and feature attribution. Then:

```bash
uv run oracle-bets lol promote-tuning <run-id>
uv run oracle-bets lol train --targets all --feature-set compact
uv run pytest tests/lol tests/core
uv run ruff check packages tests
uv run mkdocs build --strict
uv run oracle-bets lol health
uv run oracle-bets evidence health
uv run oracle-bets lol schedule --days 2
uv run oracle-bets daily lol --dry-run --skip-market-search
```

This explicit first pass retunes because the canonical symmetric feature schema
changed. Future scheduled runs retrain without retuning.

If existing evidence and the model registry are valuable, back them up first
and omit the `data/state` cleanup line. Removing it intentionally resets paper
history and champion/candidate state; it is not part of a normal retrain.
