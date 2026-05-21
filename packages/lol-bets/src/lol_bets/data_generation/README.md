# data_generation

LoL data pipeline internals.

This folder ingests Oracle's Elixir data, cleans team/player rows, creates
features, computes ratings, and writes parquet artifacts consumed by training and
Discord inference.

Example:

```bash
uv run oracle-bets lol ingest
```
