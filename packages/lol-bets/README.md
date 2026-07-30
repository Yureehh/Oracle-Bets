# lol-bets

League of Legends prediction module for Oracle Bets.

Main package: `lol_bets`.

It owns Oracle's Elixir ingestion, LoL feature engineering, rating systems,
model training, artifact health, and match inference.

The four root modules are intentional public entrypoints:

- `pipeline.py` — ingestion and feature generation;
- `training.py` — target orchestration;
- `daily.py` — scheduled workflow;
- `module.py` — sport-module health/inference adapter.

Implementation details live under `data_generation/`, `prediction_models/`,
`inference/`, and `operations/`.

Examples:

```bash
uv run oracle-bets lol ingest
uv run oracle-bets lol train --targets all
uv run oracle-bets lol train --targets props
```
