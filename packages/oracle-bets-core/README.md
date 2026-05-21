# oracle-bets-core

Shared infrastructure for Oracle Bets.

Main package: `oracle_bets_core`.

Use it for path resolution, logging, pandas/FireDucks selection, module contracts,
market quotes, and betting math.

Example:

```python
from oracle_bets_core.betting import build_edge_signal

signal = build_edge_signal(model_probability=0.57, market_odds=2.05)
```
