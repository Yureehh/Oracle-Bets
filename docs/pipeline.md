# Pipeline

The data pipeline turns raw match logs into model-ready features.

## Ingestion

- Source: Oracle's Elixir CSV dumps in S3.
- Entrypoint: `src/ingestion/oracles_elixir.py`.
- Filtering: `config/data_ingestion/leagues_handling/considered_leagues.json`.

## Feature engineering

- `src/feature_engineering/features_generator.py` adds league taxonomy, season, and team context.
- `src/feature_engineering/performance_features/` produces leak-free EMA stats.

## Ratings

- Team and player ratings live in `src/feature_engineering/ratings_features/`.
- League Elo is computed for cross-league comparisons and stored in `models/artifacts/league_elo.parquet`.
- League strength priors are derived from league Elo and stored in `models/artifacts/league_strength_priors.json`.

## Outputs

- Raw and interim data in `data/raw/` and `data/interim/`.
- Processed, training, and flattened datasets in `data/processed/`.
- Model artifacts in `models/artifacts/`.
