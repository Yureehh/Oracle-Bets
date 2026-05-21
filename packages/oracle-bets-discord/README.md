# oracle-bets-discord

Discord delivery package for Oracle Bets.

Main package: `oracle_bets_discord`.

It owns bot lifecycle, command formatting, market commands, and module registry
wiring. Prediction logic remains in domain modules such as `lol_bets`.

Example:

```bash
uv run oracle-bets discord run
```
