# oracle-bets-discord

Discord delivery package for Oracle Bets.

Main package: `oracle_bets_discord`.

It owns bot lifecycle, command formatting, market commands, and module registry
wiring. Prediction logic remains in domain modules such as `lol_bets`.

Example:

```bash
uv run oracle-bets discord run
```

Copy-paste examples:

```text
!lol predict "Team WE" "LNG Esports" --bo5
!lol props "Team WE" "LNG Esports" --kills-line 26.5 --kills-over-odds 1.85
!lol edge "Team WE" "LNG Esports"
```
