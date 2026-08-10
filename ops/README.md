# Operations

This folder is deployment configuration, not a planner or application package.
It contains the daily/monthly jobs plus two alternatives for continuous paper
operations:

- `launchd/com.oracle-bets.daily-lol.plist.example` — deterministic 00:15 LoL
  workflow and optional Discord report;
- `launchd/com.oracle-bets.monthly-audit.plist.example` — evidence review on the
  first day of each month.
- `launchd/com.oracle-bets.discord-bot.plist.example` — persistent owner-only
  paper decision buttons, manual settlement confirmation, prop-line modal, and
  pre-start closing-line capture;
  install only for interactive use.
- `launchd/com.oracle-bets.closing-lines.plist.example` — five-minute read-only
  closing-line capture for webhook-only deployments. Do not install it when the
  Gateway bot is running because the bot already performs this check.

There is deliberately no weekly pipeline. Training-event reports are produced
by training itself, and a weekly model rebuild without enough new data would
only add churn. The current work board lives in `docs/roadmap.md`.

## Install on macOS

Replace the `uv` and repository placeholders in copies of the examples, then:

```bash
mkdir -p ~/Library/LaunchAgents
cp ops/launchd/com.oracle-bets.daily-lol.plist.example \
  ~/Library/LaunchAgents/com.oracle-bets.daily-lol.plist
cp ops/launchd/com.oracle-bets.monthly-audit.plist.example \
  ~/Library/LaunchAgents/com.oracle-bets.monthly-audit.plist
# Webhook-only paper tracking (omit when installing the Gateway bot):
cp ops/launchd/com.oracle-bets.closing-lines.plist.example \
  ~/Library/LaunchAgents/com.oracle-bets.closing-lines.plist
# Optional, after `uv sync --extra discord-bot`:
cp ops/launchd/com.oracle-bets.discord-bot.plist.example \
  ~/Library/LaunchAgents/com.oracle-bets.discord-bot.plist

plutil -lint ~/Library/LaunchAgents/com.oracle-bets.*.plist
launchctl bootstrap gui/$(id -u) \
  ~/Library/LaunchAgents/com.oracle-bets.daily-lol.plist
launchctl bootstrap gui/$(id -u) \
  ~/Library/LaunchAgents/com.oracle-bets.monthly-audit.plist
launchctl bootstrap gui/$(id -u) \
  ~/Library/LaunchAgents/com.oracle-bets.closing-lines.plist
# Optional:
launchctl bootstrap gui/$(id -u) \
  ~/Library/LaunchAgents/com.oracle-bets.discord-bot.plist
```

Inspect or trigger an installed job with:

```bash
launchctl print gui/$(id -u)/com.oracle-bets.daily-lol
launchctl kickstart -k gui/$(id -u)/com.oracle-bets.daily-lol
```

Unload before replacing an installed file:

```bash
launchctl bootout gui/$(id -u)/com.oracle-bets.daily-lol
```

Keep `PANDASCORE_API_KEY`, Discord values, and the optional OpenAI key in the
ignored repository `.env`, never in a plist or commit, and restrict it with
`chmod 600 .env`.
Application logs rotate at 10 MiB with five backups by default (about 60 MiB
maximum). The plist sends launchd stdout/stderr to `/dev/null` because the
application already owns its bounded log.
