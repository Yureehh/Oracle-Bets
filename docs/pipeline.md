# Pipeline

The data pipeline turns raw match logs into model-ready features.

## Ingestion

- Source: Oracle's Elixir CSV dumps in S3.
- Entrypoint: `packages/lol-bets/src/lol_bets/data_generation/ingestion/oracles_elixir.py`.
- Filtering: active profile in `config/lol/data_ingestion/considered_leagues.json`.

## Feature engineering

- `packages/lol-bets/src/lol_bets/data_generation/feature_engineering/features_generator.py` adds league taxonomy, season, and team context.
- `packages/lol-bets/src/lol_bets/data_generation/feature_engineering/performance_features/` produces leak-free EMA stats.

## Ratings

- Team and player ratings live in `packages/lol-bets/src/lol_bets/data_generation/feature_engineering/ratings_features/`.
- League Elo is computed for cross-league comparisons and stored in `models/lol/league_elo.parquet`.
- League strength priors are derived from league Elo and stored in `models/lol/league_strength_priors.json`.

## Outputs

- Raw and interim data in `data/lol/raw/` and `data/lol/interim/`.
- Processed, training, and flattened datasets in `data/lol/processed/`.
- Model artifacts in `models/lol/`.
