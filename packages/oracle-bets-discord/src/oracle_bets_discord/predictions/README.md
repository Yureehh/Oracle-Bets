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

Neutral commands do not assume map side or first pick. First Selection commands
accept explicit Blue/Red map side plus the team with draft priority so LoL can
model the 2026 side/draft decoupling after retraining.
