# oracle_bets_core

Runtime code shared by every prediction module.

- `cli.py`: `oracle-bets` command router.
- `paths.py`: artifact and config locations.
- `interfaces.py`: prediction-module contracts.
- `betting.py` and `markets.py`: read-only market decision support.
- `pd.py`: FireDucks-first pandas shim.

Example:

```bash
uv run oracle-bets lol health
```
