# Security and hosting

Oracle Bets reads public esports data, PandaScore schedules, Discord webhooks,
and public Polymarket market data. It never automates betting, wallet access,
signing, private keys, or fund movement.

Secrets belong outside the repository in environment variables or a `0600`
user-owned file. Reports exclude webhook URLs. Discord payloads disable all
mentions. The Gateway bot exposes only owner-checked paper accept/reject
controls and a prop Line/Odds modal; duplicate decisions are idempotent. It has
no shell, arbitrary prompt, trading, signing, or wallet capability.

## Hosting recommendation

For one-way midnight Discord webhook reports, prefer a free or near-free
scheduled runner:

- the current Mac with `launchd`;
- GitHub Actions/another cron runner if private secrets and generated-state
  persistence are handled carefully;
- a low-cost scheduled VM job.

This does not require an always-on bot. Public Gamma discovery and public CLOB
books require no credentials or VPN from the current Italian connection.

The optional interactive Discord bot requires a continuously running PC,
VPS, or free VM such as Oracle Cloud Always Free. Free VM capacity, account
verification, idle-resource reclamation, regional availability, maintenance,
and outbound-network limits are real caveats. Italy's Polymarket order geoblock
does not affect read-only research and reinforces the permanent no-order
boundary.
