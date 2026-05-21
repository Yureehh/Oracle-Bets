# Quickstart

This assumes you have a `.env` file with the required keys and have installed dependencies.

## 1) Build data artifacts

```bash
uv run oracle-bets lol ingest
```

This writes processed data to `data/lol/processed/` and rating artifacts to `models/lol/`.

## 2) Train models

```bash
uv run oracle-bets lol train --model-type lightgbm
```

Models are stored in `models/lol/<ModelName>/`.

## 3) Check artifact health

```bash
uv run oracle-bets lol health
```

Health returns non-zero while required artifacts are missing.

## 4) Run the Discord bot (optional)

```bash
uv run oracle-bets discord run
```

Requires `DISCORD_TOKEN` and trained model artifacts.

## Compact training config

To use the compact training feature set, set:

```bash
export TRAINING_CONFIG_VARIANT=compact
```
