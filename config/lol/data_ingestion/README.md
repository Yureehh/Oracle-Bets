# data_ingestion

Configuration for Oracle's Elixir ingestion.

- `import_columns.json`: source columns kept for team/player rows.
- `considered_leagues.json`: leagues included in modeling.
- `team_name_replacements_and_invalid_games.json`: manual data fixes.
- `league_taxonomy.json`: region/tier metadata.

Example:

```bash
uv run oracle-bets lol ingest
```
