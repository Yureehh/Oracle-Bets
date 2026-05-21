# Configuration

League of Legends configuration files live under `config/lol/`.

## Data ingestion

- `config/lol/data_ingestion/import_columns.json` - columns to load from Oracle's Elixir.
- `config/lol/data_ingestion/team_name_replacements_and_invalid_games.json` - data cleanup rules.
- `config/lol/data_ingestion/considered_leagues.json` - leagues, taxonomy, and tiers.
- `config/lol/data_ingestion/league_taxonomy.json` - league taxonomy + prior calibration settings.
- `config/lol/data_ingestion/extras/teams_by_league.json` - generated team lists by league.
- `config/lol/data_ingestion/extras/filtered_teams_by_league.json` - generated team lists filtered to considered leagues.

## Training

- `config/lol/training/training_team_config.json` and `config/lol/training/training_compact_team_config.json`
- `config/lol/training/training_player_config.json` and `config/lol/training/training_compact_player_config.json`
- `config/lol/target_features.json`

## Discord

- `config/lol/discord_config.json` contains safe defaults for bot behavior (no secrets).

## Environment variables

See `docs/getting_started/installation.md` for required environment keys.
