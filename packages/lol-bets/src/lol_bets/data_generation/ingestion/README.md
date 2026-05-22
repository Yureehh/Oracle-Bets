# ingestion

Oracle's Elixir ingestion and schedule fetching.

- `oracles_elixir.py`: S3 CSV ingestion, cleaning, league filtering, and opponent pairing.
  Team rows keep `first_pick` so 2026 First Selection can be modeled separately
  from map side.
- `schedule.py`: upcoming LoL schedule helpers. Returned schedules use stable
  snake_case columns such as `team_a`, `team_b`, `start_utc`, `best_of`,
  `market_query`, and `match_key` so Discord and market matching can parse them
  without display-column cleanup.

Example:

```python
from lol_bets.data_generation.ingestion.oracles_elixir import OraclesElixir
```
