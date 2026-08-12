# Daily operations

## Cadence

| Cadence | Owner | Command | Purpose |
| --- | --- | --- | --- |
| Hourly | `launchd` | `oracle-bets lol market-watch` | Read-only timing and CLV observations; never proposals |
| Daily, 00:15 Rome | `launchd` | `oracle-bets daily lol` | History, conditional fixed-parameter retrain, fixtures, Winner V2, quotes, evidence, report |
| Continuous | Gateway bot | `oracle-bets discord run` | Publish each live card once, re-quote/confirm, manual settlement controls, closing observations |
| Every training event | training code | `reports/lol/training/runs/<run-id>/` | Sealed metrics, calibration, attribution, lineage, model cards, provenance |
| Monthly, day 1 | owner/`launchd` | `oracle-bets audit monthly` | Paper calibration, CLV, ROI, drawdown, and cohort review |
| On demand | owner | retune, promotion, rollback, settlement, notebooks | Judgment-heavy research or recovery |

## Daily workflow

The normal run:

1. reports open paper positions without settling them;
2. refreshes and validates 2024–2026 `research_all_supported` history;
3. rebuilds deterministic complete-series datasets when history changes;
4. conditionally refits fixed reviewed parameters—never Optuna;
5. registers a clean, checksummed candidate and compares V2 on sealed rows;
6. fetches `tier1_plus_erls` fixtures while omitting LCP/CBLOL and invalid rows;
7. validates the actionable direct-series champion and expected roster evidence;
8. produces direct series probabilities and retains prop/next-map artifacts for
   separate shadow research;
9. strictly matches series-moneyline contracts and captures two executable books;
10. applies favorite, 52.5%, 5% bound-edge, 48–24h, anomaly, roster, and market gates;
11. appends evidence and writes exactly one JSON/Markdown report pair;
12. leaves delivery to the single persistent Gateway bot.

Before the first manually promoted Winner V2 champion exists, step 4 is
deliberately skipped. Midnight automation cannot bootstrap or retune the new
model family; the one-time Optuna study, parameter review, fixed-parameter
rebuild, and first promotion remain explicit owner actions.

The model is trained on every supported imported league. Promotion evidence is
reported separately for the actionable `tier1_plus_erls - {LCP, CBLOL}` cohort,
so minor-league gains cannot hide a betting-universe regression.

PandaScore match-detail enrichment never requests a missing ID. Authentication
uses a bearer header so credentials are absent from URLs and errors. A 401/403
stops further detail calls for that run and retains embedded schedule lineups.
Only a completely absent provider lineup may fall back to the exact role-mapped
five repeated across the latest three completed consecutive historical series.
Partial or conflicting provider lineups remain blocked.

Monday, Thursday, and first-of-month reminders use `Europe/Rome` and are
advisory only. The daily command never automatically settles a position. An
optional bounded LLM summary cannot alter probabilities, gates, stakes,
promotion, acceptance, or settlement.

`--dry-run` skips ingestion, training, evidence writes, promotion, and Discord,
while still writing one report pair. A quarantined or structurally invalid
champion may produce diagnostics but never a proposal.
