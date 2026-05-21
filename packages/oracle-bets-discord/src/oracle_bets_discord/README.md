# oracle_bets_discord

Discord bot shell and command layer.

- `bot.py`: bot lifecycle and commands.
- `formatting.py`: shared Discord message/error formatting.
- `betting.py`: Discord-facing odds/probability parsing helpers.
- `registry.py`: prediction-module registry.
- `predictions/`: module-specific prediction presentation helpers.

Example:

```python
from oracle_bets_discord.registry import default_registry
```
