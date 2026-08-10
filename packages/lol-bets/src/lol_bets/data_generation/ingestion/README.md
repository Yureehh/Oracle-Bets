# ingestion

Oracle's Elixir ingestion and schedule fetching.

- `source.py`: atomic public Google Drive downloads into
  `data/lol/raw/oracles_elixir_cache/`, plus fail-closed freshness and schema
  validation. It never modifies Drive and rejects cloud-backed cache symlinks.
- `oracles_elixir.py`: local Oracle's Elixir CSV ingestion, cleaning, league
  filtering, and opponent pairing. Every requested year must exist in the
  validated cache. There is no AWS dependency. Source spellings such as `earned gpm` are
  reconciled to stable internal snake_case names. `first_pick` is retained as
  source metadata but prohibited from the pre-match winner model.
- `schedule.py`: upcoming LoL schedule helpers. Returned schedules use stable
  snake_case columns such as `team_a`, `team_b`, `start_utc`, `best_of`,
  `market_query`, and `match_key` so Discord and market matching can parse them
  without display-column cleanup.

Example:

```python
from lol_bets.data_generation.ingestion.oracles_elixir import OraclesElixir
```
