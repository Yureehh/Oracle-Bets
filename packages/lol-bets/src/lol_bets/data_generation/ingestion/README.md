# ingestion

Oracle's Elixir ingestion and schedule fetching.

- `oracles_elixir.py`: S3 CSV ingestion, cleaning, league filtering, and opponent pairing.
- `schedule.py`: upcoming LoL schedule helpers for Discord commands.

Example:

```python
from lol_bets.data_generation.ingestion.oracles_elixir import OraclesElixir
```
