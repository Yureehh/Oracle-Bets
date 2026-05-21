# lol-bets

League of Legends prediction module for Oracle Bets.

Main package: `lol_bets`.

It owns Oracle's Elixir ingestion, LoL feature engineering, rating systems,
model training, artifact health, and match inference.

Examples:

```bash
uv run oracle-bets lol ingest
uv run oracle-bets lol train --model-type lightgbm
```
