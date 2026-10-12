# Daily operation

This is the owner’s daily guide. The active model is `lol-20261010-paper-v1`;
weights, calibration, and uncertainty are fixed for the October 11–November 30,
2026 window, Europe/Rome. Current forecasts are exploratory. No strategy is
recommendation-active and no real-stake activation is enabled.

## First setup

Run from the repository root. Install the locked dependencies with
`uv sync --extra dev --extra discord-bot`. Supply PandaScore and Discord
credentials outside Git as described in the README. Use one managed Gateway
worker; do not start another if launchd already owns it. A GitHub clone contains
source and reviewed configuration, not your local datasets, models, or ledger.

```bash
uv run oracle-bets discord doctor
uv run oracle-bets model status
uv run oracle-bets evidence health
uv run oracle-bets evidence paper-capital
```

Expect the paper champion and schema/integrity checks to pass. Technical health
means files and services work; it does not mean a market strategy is validated.
Discord Health displays paper scope and readiness warnings separately.

## The declared population

Majors: LEC, LCK, LPL, LCS, LCP, CBLOL.
European/Turkish ERLs: LFL, PRM, LES, LIT, LPLOL, NLC, RL, ROL, EBL, HM, HLL, TCL.
Cross-region events: Worlds (`WLDs`), MSI, First Stand (`FST`), EWC, EMEA Masters (`EM`).
Other cups: Demacia Cup (`DCup`) and KeSPA.

These 25 competitions form `paper_selected`. Historical Spanish `LVP SL` aliases
remain for identity/training, while Arabian League, feeders, other Asian domestic
leagues, and legacy Americas codes stay outside this paper population. Training
retains broader supported history. Unknown identities and unavailable rosters
remain explicit gaps; they never justify substituting another fixture.

## One manual run each day

```bash
uv run oracle-bets evidence backup --output ~/OracleBetsArchives/backups
uv run oracle-bets daily lol
```

The daily run refreshes public history, validates source/data/model health,
fetches the next **336 hours**, enrolls eligible fixtures before inference,
runs supported forecasts, and checks public Polymarket listing candidates.
The generated JSON/Markdown report includes missing forecasts, unknown opponents,
provider failures, listing status, and open positions. Thunderpick stays manual.
It does not train, retune, accept a quote, create a ticket, settle, or place orders.

Unchanged features are reused only after source identity, code/configuration,
manifest and complete output hashes match. Changed or damaged inputs rebuild.
Leave your machine awake during a full rebuild. A failed source or model check
suppresses forecasts; resolve the named failure before acting on a report.

Use `/oracle` → **Schedule** and read the daily report under `reports/lol/daily/`.
Forecast means are not prop win probabilities: an exact line, period and calibrated
distribution are needed later. Read all roster/calibration/coverage warnings.
Supported families are series winner, prematch map winner, series totals and
handicaps, duration, kills, and towers. Unsupported specials stay unsupported.

## Why scan 14 days?

Discovery is separate from entry. An early listing may have no usable price,
insufficient size, uncertain rosters, or incompatible terms. First-seen records
show when your daily check observed a candidate; they are not the exact provider
opening time. Missing today followed by present tomorrow gives an observation
interval. Neither provider has a fixed listing lead time established by the
reviewed official material. Polymarket’s public discovery is documented at
[Discover Markets](https://docs.polymarket.com/market-data/discover-markets).

At the later entry stage, use the first qualified opportunity you can actually
verify, not an automatic “bet at opening” rule. Recheck forecasts/evidence and
fresh executable odds before owner confirmation. Compare early/later lead-time
observations after collection rather than presuming early entries or props win.

## Capital and provisional Kelly

Total virtual capital is **€500**, allocated **€250 per venue**. Thunderpick
is declared in EUR for this paper policy. Polymarket uses pUSD; no EUR/pUSD parity or fee-free conversion is
assumed. See the [official collateral explanation](https://help.polymarket.com/en/articles/13364250-does-polymarket-have-a-token). `config/research/lol-paper-capital.json` holds the declaration.

For Polymarket, fill `native_units_per_eur`, `conversion_cost_native`,
`source_reference`, `verified_at` with an aware ISO timestamp, and
`costs_verified` using an owner-verified conversion quote. The initial native
bankroll is allocation × rate minus conversion cost. Do this before the first
paper ticket freezes the complete declaration. Missing conversion blocks entry;
it does not block schedule/forecast research. Never invent conversion evidence.

The primary provisional policy is quarter Kelly with caps of **2% per ticket,
5% per fixture, and 20% total open exposure**, applied to each venue allocation.
At the initial Thunderpick allocation these are €5, €12.50, and €50. They are
maximums, not intended stake sizes. Unsettled positions consume capacity.
Bankroll input must match declared capital plus corrected realized PnL. Negative
edge means observation only, with no stake.

The balanced research drawdown target is **20%**, not a guaranteed ceiling.
Eighth/quarter/half Kelly comparisons use temporal development and later
held-out evidence. Zero settled tickets means no empirical optimum can be
selected. Do not change the primary policy from interim outcomes.

## Stop here for the current phase

Points 1–3 prepare and refresh the system. The next owner phase starts at
**point 4: exact quotes**. Follow the [remaining checklist](paper-owner-checklist.md)
when you are ready. No paper tickets or settlements were created during preparation.

## Weekly and longer-term use

On Monday, review population coverage, missing stages and open positions.
On Thursday, review health and provider failures. Back up/export monthly and
run `audit monthly --period YYYY-MM`; activity counts are monthly, while
explicitly labeled lifetime summaries remain lifetime. Do not confuse them.
The [system guide](system.md) explains the code, model lifecycle and evaluation.

The JSON daily report contains the numeric forecasts and discovery records. The
Markdown report summarizes operational status; use the JSON when reviewing
individual probabilities.
