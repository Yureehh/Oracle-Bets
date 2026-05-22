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

Runtime examples:

```text
!lol predict "Team WE" "LNG Esports" --side Blue --first-pick "Team WE"
!lol props "Team WE" "LNG Esports" --towers-line 12.5 --towers-over-odds 1.90
```
