# Security and hosting

Oracle Bets reads public esports data, PandaScore schedules, Discord webhooks,
and public Polymarket market data. It never automates betting, wallet access,
signing, private keys, or fund movement.

Secrets belong outside the repository in environment variables or a `0600`
user-owned file. Reports exclude webhook URLs. Discord payloads disable all
mentions. The removed dashboard/API/bot surfaces are not network attack
surfaces anymore.

## Hosting recommendation

For one-way midnight Discord webhook reports, prefer a free or near-free
scheduled runner:

- the current Mac with `launchd`;
- GitHub Actions/another cron runner if private secrets and generated-state
  persistence are handled carefully;
- a low-cost scheduled VM job.

This does not require an always-on bot.

An interactive always-on Discord bot would require a continuously running PC,
VPS, or free VM such as Oracle Cloud Always Free. Free VM capacity, account
verification, idle-resource reclamation, regional availability, maintenance,
and outbound-network limits are real caveats. The repository intentionally
ships only the simpler one-way webhook path.
