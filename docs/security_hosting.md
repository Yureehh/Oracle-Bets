# Security and hosting

Oracle Bets reads public esports data, PandaScore schedules, Discord Gateway
events, and public Polymarket data. It never automates betting, wallet access,
signing, private keys, order placement, or fund movement.

Secrets belong in the ignored `0600` `.env`. PandaScore uses a bearer header;
provider errors are sanitized. Discord cards disable mentions and accept owner
interactions only. Accept requires a fresh read-only quote and separate
120-second confirmation. Durable send intent and bounded history recovery avoid
duplicate cards. Webhook delivery is disabled.

The Gateway bot must stay online. The simplest first host is this Mac with
`launchd`. A VPS or Oracle Cloud Always Free VM is possible, with capacity,
verification, reclamation, regional, maintenance, and outbound-network caveats.
Public Gamma/CLOB reads require no VPN or trading credentials from the current
Italian connection; the order geoblock reinforces the read-only boundary.

Rotate any PandaScore, Discord, AWS, or OpenAI credential exposed in a chat,
screenshot, paste, or commit. Remove obsolete AWS variables because ingestion
uses the public Drive cache.
