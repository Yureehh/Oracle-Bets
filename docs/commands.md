# Commands

Run commands from the repository root with `uv run`.

## Data and schedules

```bash
# Incremental 2024–2026 refresh from local Google Drive Desktop files
uv run oracle-bets lol ingest

# Full source-history reconciliation
uv run oracle-bets lol reconcile-history

# Validate generated supervised tables
uv run oracle-bets lol validate-data

# Inspect upcoming fixtures only; no ingestion or training
uv run oracle-bets lol schedule --days 2

# Optional explicit league subset
uv run oracle-bets lol schedule --days 7 --leagues LCK,LEC,LPL
```

`lol ingest` and `lol reconcile-history` never tune hyperparameters.

## Training

```bash
# Routine full retraining with reviewed production hyperparameters; no Optuna
uv run oracle-bets lol train \
  --targets all \
  --feature-set compact

# Explicit research retuning; writes candidates under reports/, not production
uv run oracle-bets lol retune \
  --targets all \
  --feature-set compact

# After reviewing all four target reports, promote the complete parameter bundle
uv run oracle-bets lol promote-tuning <run-id>

# Retrain once with the newly reviewed production parameters
uv run oracle-bets lol train \
  --targets all \
  --feature-set compact
```

Target selectors are `all`, `outcome`, `props`, or comma-separated target names.
Routine retraining refits model weights and calibrators on newer data. Retuning
uses Optuna to search hyperparameters and should be rare and explicit.

## Daily workflow

```bash
# Safe end-to-end smoke test; writes JSON and Markdown but makes no mutations
uv run oracle-bets daily lol --dry-run --skip-market-search

# Normal midnight workflow
uv run oracle-bets daily lol

# Generate without the retraining trigger
uv run oracle-bets daily lol --skip-retrain
```

The default 36-hour window and `tier1_plus_erls` profile come from product
configuration. Set `PANDASCORE_API_KEY` and `DISCORD_WEBHOOK_URL` in the
ignored, mode-`0600` `.env`; never commit them. Polymarket access is read-only.

## Clean end-to-end assessment

The exact cleanup and rebuild sequence is in
[Getting started](getting_started.md#first-clean-research-rebuild). It removes
generated data, models, reports, logs, evidence, and registry state while
preserving the local Google Drive symlink. This is destructive research setup,
not routine retraining; back up `data/state/` first if its paper evidence
matters.

## Evidence, models, and audit

```bash
uv run oracle-bets evidence init
uv run oracle-bets evidence health
uv run oracle-bets evidence backup
uv run oracle-bets evidence export --format json

uv run oracle-bets model status
uv run oracle-bets model register-current <id> \
  --code-version <git-sha> \
  --metric log_loss=0.64 \
  --metric brier=0.22
uv run oracle-bets model promote <id> --reason "reviewed temporal evidence"
uv run oracle-bets model rollback <id> --reason "health regression"

uv run oracle-bets paper settle \
  --position-id <id> \
  --stake 0.50 \
  --odds 1.95 \
  --result win \
  --source-reference <result-url>

uv run oracle-bets health system
uv run oracle-bets audit monthly
```

`models/lol/` is the mutable training workspace. `model register-current`
freezes it into the immutable checksum-verified registry.
