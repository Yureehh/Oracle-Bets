# logs/lol

Generated League of Legends runtime logs. Ingestion, schedule, training, and
daily workflow messages all use the bounded `oracle-bets.log`; topic-specific
legacy log files are no longer created.

Example:

```bash
uv run oracle-bets lol ingest
```
