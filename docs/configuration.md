# Configuration

Configuration is intentionally small and each retained value has a runtime
consumer.

## Product contract

`config/product/product.json` controls:

- timezone: `Europe/Rome`;
- default fixture window: 36 hours;
- market comparison: read-only;
- league profile: `tier1_plus_erls`;
- candidate-training triggers: 50 new valid maps or 20 major-league maps;
- promotion: manual only.

There is no execution, API, dashboard, or automated-staking configuration
because those products do not exist.

## League scope

`config/lol/data_ingestion/considered_leagues.json` defines named research
profiles. `tier1_plus_erls` is the one operational default across ingestion,
training, schedule filtering, prediction, and reporting. An explicit CLI league
override is for research and should be recorded with the resulting report.

## Team identity

`config/lol/data_ingestion/team_aliases.json` has two separate contracts:

- `external_aliases` maps current provider/event branding to a canonical team;
- `historical_identity_merges` joins verified historical IDs after review.

Aliases never justify fabricating history. Unknown or ambiguous teams are
skipped and reported.

## Source and feature columns

`import_columns.json` is the source allowlist. Training JSON files classify
team/player inputs used by full, compact, and flattened artifacts. Every
ingestion produces a column-reconciliation report classifying source columns as
present, missing, or new and as required, metadata, feature, or candidate.

## Hyperparameters

`default_models_parameters.json` contains conventional initialization and
search priors:

- half-life 9 maps;
- Elo 1500 / K 32 / divisor 400;
- Glicko-2 1500 / deviation 350 / volatility 0.06 / tau 0.5;
- Plackett-Luce and TrueSkill conventional 25 / 8.333 priors;
- inactivity grace 45 days and half-life 180 days;
- Optuna 100 trials with seed 42.

These are not silent fallbacks.

Reviewed production rating values live in `hyperparameters/tuned/ratings/`.
Reviewed LightGBM values live in `hyperparameters/tuned/lightgbm/`. Missing
reviewed files are hard errors. Routine workflows never run Optuna.

## Secrets and local paths

Use environment variables or a user-owned `0600` file outside the repository:

- `PANDASCORE_API_KEY`
- `DISCORD_WEBHOOK_URL`
- `ORACLES_ELIXIR_LOCAL_DIR` when the generated local source cache location is unsuitable

Never commit secrets, webhook URLs, private keys, or wallet material.
