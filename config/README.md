# config

Runtime and model configuration is scoped by module.

- `lol/`: League of Legends configuration for `lol-bets`.

Example:

```bash
uv run oracle-bets lol ingest
uv run oracle-bets lol train --model-type lightgbm --feature-set compact
```
