# data_ingestion

Configuration for Oracle's Elixir ingestion.

- `import_columns.json`: source columns kept for team/player rows.
- `considered_leagues.json`: named league-selection profiles for ingestion.
- `team_name_replacements_and_invalid_games.json`: manual data fixes.
- `league_taxonomy.json`: region/tier metadata.

Generated league/team summaries live in `reports/lol/ingestion/`.

Example:

```bash
uv run oracle-bets lol ingest
```
