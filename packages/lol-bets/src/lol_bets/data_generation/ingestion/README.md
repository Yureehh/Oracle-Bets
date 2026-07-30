# ingestion

Oracle's Elixir ingestion and schedule fetching.

- `oracles_elixir.py`: local-first Oracle's Elixir CSV ingestion, cleaning,
  league filtering, and opponent pairing. It reads canonical yearly files from
  `data/lol/raw/oracles_elixir/` (which may be a Google Drive for Desktop
  symlink). Every requested year must exist and be available offline; there is
  no AWS or HTTP download fallback. Source spellings such as `earned gpm` are
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
