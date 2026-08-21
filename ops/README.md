# Operations

This folder contains the optional always-on Discord Gateway deployment example,
not application code or planning. Daily data refresh, model research, exact
market review, and monthly evidence review remain owner-triggered commands.
The work board is `docs/roadmap.md`.

Copy the examples, replace absolute `uv` and repository paths, then run:

```bash
mkdir -p ~/Library/LaunchAgents
cp ops/launchd/com.oracle-bets.discord-bot.plist.example \
  "$HOME/Library/LaunchAgents/com.oracle-bets.discord-bot.plist"
plutil -lint "$HOME/Library/LaunchAgents/com.oracle-bets.discord-bot.plist"
launchctl bootstrap "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.oracle-bets.discord-bot.plist"
```

Inspect or trigger a job with `launchctl print` and `launchctl kickstart -k`.
Unload with `launchctl bootout` before replacing a plist. Keep credentials only
in the ignored `chmod 600 .env`, never in a plist. Application logs rotate at
10 MiB with five backups; launchd stdout/stderr therefore point to `/dev/null`.
