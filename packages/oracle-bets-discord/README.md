# oracle-bets-discord

Discord presentation and owner-only paper-review package. Production uses one
Gateway bot so report delivery and interactive controls cannot duplicate each
other. It needs an always-on Mac, VPS, or VM (Oracle Cloud free-tier capacity
and reclamation are not guaranteed).

Install with `uv sync --extra discord-bot`, set `DISCORD_TOKEN`,
`DISCORD_CHANNEL_ID`, `DISCORD_OWNER_USER_ID`, and
`DISCORD_DELIVERY_MODE=gateway`, then run
`uv run oracle-bets discord doctor --live` and `uv run oracle-bets discord run`.
The bot also records size-aware public closing books for accepted positions
shortly before start so later settlement can calculate CLV. It exposes
persistent owner-only Win/Loss/Push/Void controls; each result requires a source
reference in a confirmation modal. Nothing settles automatically and the LLM
is never involved in settlement. A durable intent plus deterministic message
marker lets the bot recover an already-sent card after a crash without
duplicating it, and a process lock rejects a second running bot. All decisions
are append-only paper records.
The package exposes no wallet,
signing, order-placement, fund movement, shell, or arbitrary-prompt command.
