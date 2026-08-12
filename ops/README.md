# Operations

This folder contains deployment examples, not application code or planning.

- `com.oracle-bets.daily-lol.plist.example`: daily 00:15 Rome workflow;
- `com.oracle-bets.market-watch.plist.example`: hourly read-only market timing observations;
- `com.oracle-bets.monthly-audit.plist.example`: first-day evidence review;
- `com.oracle-bets.discord-bot.plist.example`: the sole persistent Discord
  delivery path, owner decisions, confirmation, settlement, and closing reads.

There is no scheduled weekly retune and no webhook/second closing daemon.
Training reports are created by training itself. The work board is
`docs/roadmap.md`.

Copy the examples, replace absolute `uv` and repository paths, then run:

```bash
mkdir -p ~/Library/LaunchAgents
for job in daily-lol market-watch monthly-audit discord-bot; do
  cp "ops/launchd/com.oracle-bets.${job}.plist.example" \
    "$HOME/Library/LaunchAgents/com.oracle-bets.${job}.plist"
  plutil -lint "$HOME/Library/LaunchAgents/com.oracle-bets.${job}.plist"
  launchctl bootstrap "gui/$(id -u)" \
    "$HOME/Library/LaunchAgents/com.oracle-bets.${job}.plist"
done
```

Inspect or trigger a job with `launchctl print` and `launchctl kickstart -k`.
Unload with `launchctl bootout` before replacing a plist. Keep credentials only
in the ignored `chmod 600 .env`, never in a plist. Application logs rotate at
10 MiB with five backups; launchd stdout/stderr therefore point to `/dev/null`.
