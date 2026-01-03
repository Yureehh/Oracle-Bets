# Configuration

Key configuration files live under `config/`.

## Data ingestion

- `config/data_ingestion/import_columns.json` - columns to load from Oracle's Elixir.
- `config/data_ingestion/team_name_replacements_and_invalid_games.json` - data cleanup rules.
- `config/data_ingestion/considered_leagues.json` - leagues, taxonomy, and tiers.
- `config/data_ingestion/league_prior_settings.json` - league prior calibration.
- `config/data_ingestion/extras/teams_by_league.json` - generated team lists by league.
- `config/data_ingestion/extras/filtered_teams_by_league.json` - generated team lists filtered to considered leagues.

## Training

- `config/training/training_team_config.json` and `config/training/training_team_config_compact.json`
- `config/training/training_player_config.json` and `config/training/training_player_config_compact.json`
- `config/target_features.json`

## Discord

- `config/discord_config.json` contains safe defaults for bot behavior (no secrets).

## Environment variables

See `docs/getting_started/installation.md` for required environment keys.
