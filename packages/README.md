# Packages

Installable Oracle Bets packages live here.

- `oracle-bets-core`: shared config, CLI, evidence, operations, betting math,
  and read-only markets.
- `lol-bets`: League of Legends data, features, ratings, training, and inference.
- `oracle-bets-discord`: one-way Discord report formatting.

Package roots contain only stable public entrypoints and small shared
primitives. Domain implementation belongs in named subpackages such as
`evidence`, `operations`, `data_generation`, `prediction_models`, `inference`,
and `predictions`; do not add another loose root module without a public
entrypoint reason.

Example:

```bash
uv run oracle-bets lol health
```
