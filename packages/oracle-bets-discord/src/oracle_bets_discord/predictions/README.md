# predictions

Discord presentation helpers for prediction modules.

- `lol.py`: LoL-specific schedules, profiles, roster parsing, and prediction responses.
- `best_ofs.py`: reusable best-of series probability formatting.

Shared Discord formatting lives in `oracle_bets_discord.formatting`. Betting
command helpers live in `oracle_bets_discord.betting`.

Example:

```python
from oracle_bets_discord.predictions.lol import validate_and_predict
```
