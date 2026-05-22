# Configuration

League of Legends configuration files live under `config/lol/`.

## Data ingestion

- `config/lol/data_ingestion/import_columns.json` - columns to load from Oracle's Elixir.
- `config/lol/data_ingestion/team_name_replacements_and_invalid_games.json` - data cleanup rules.
- `config/lol/data_ingestion/considered_leagues.json` - named league-selection profiles.
- `config/lol/data_ingestion/league_taxonomy.json` - league region/tier metadata.
- `reports/lol/ingestion/teams_by_league.json` - generated team lists by league.
- `reports/lol/ingestion/filtered_teams_by_league.json` - generated team lists filtered to the active league profile.

The default active profile is `tier1_plus_erls`, which keeps current Tier 1,
2025 Americas bridge leagues, international events, major ERLs, and APAC feeder
leagues. Switch to `tier1_current` for a stricter market-facing dataset or
`research_all_supported` for broad experiments.

## Training

- `config/lol/training/training_team_config.json` and `config/lol/training/training_compact_team_config.json`
- `config/lol/training/training_player_config.json` and `config/lol/training/training_compact_player_config.json`
- `config/lol/target_features.json`

## Discord

- `config/lol/discord_config.json` contains safe defaults for bot behavior (no secrets).

## Environment variables

See `docs/getting_started/installation.md` for required environment keys.
