# Daily operations

## Ownership and cadence

| Cadence | Owner | Command/artifact | Why |
| --- | --- | --- | --- |
| Daily, 00:15 | `launchd` | `oracle-bets daily lol` | Fixtures, incremental history, conditional retraining, health gates, predictions, read-only market comparison, reports, Discord |
| Every 5 minutes before fixtures | Gateway bot or `launchd` | `paper capture-closing` | Record a fresh, fully fillable public close for later CLV; no request is made when no open position is near start |
| Every training event | training code | `reports/lol/training/runs/<run-id>/` | Metrics, calibration, attribution, manifest, model cards; no separate report job |
| Weekly | none required | Review the latest compact daily/training reports only | Do not retrain or retune just because a week elapsed |
| Monthly, day 1 | `launchd` | `oracle-bets audit monthly` | Settled evidence, calibration/profit sample health, drawdown and owner review |
| On demand | owner | retune, promotion, rollback, settlement, notebooks, scratch rebuild | These actions need evidence or an explicit research decision |

The production operator is normal Python plus `launchd`. An optional OpenAI
Responses API review can summarize only already-actionable proposals in one
bounded call. It is advisory, recorded in the report, and cannot alter gates,
stakes, promotion, evidence, or settlement. Failure or a missing key never
blocks deterministic output. It must never place a bet, sign anything, access
a wallet, or move funds. The owner remains the only decision-maker.

## Daily workflow

The midnight job performs:

1. report open paper positions for owner settlement without closing anything;
2. refresh all `research_all_supported` 2024–2026 history;
3. validate data and evaluate the fixed-parameter retraining trigger;
4. register and compare any new candidate against the champion on the
   candidate's exact sealed rows;
5. fetch the next 36 hours of `tier1_plus_erls` PandaScore fixtures;
6. refresh expected lineups and exclude blank/TBD and Equal eSports Cup;
7. validate serving health and produce symmetric map, derived series/totals,
   and scalar prop forecasts;
8. strictly match public Polymarket contracts and observe every CLOB book
   twice with one shared 45-second wait;
9. apply league, model, roster, uncertainty, edge, correlation, and exposure
   gates;
10. append evidence and optionally make one bounded AI advisory call;
11. write exactly one atomic JSON/Markdown pair;
12. send deterministic Discord output with mentions disabled, then atomically
    update that same pair with the delivery result.

The daily report also emits advisory Monday, Thursday, and first-of-month
review reminders using `Europe/Rome`. Overlapping reminders are combined. They
do not perform the review, change workflow health, or settle a position.

The model is trained and rated on `research_all_supported`. Daily schedules,
predictions, reports, and Discord output use `tier1_plus_erls` after removing
the configured actionability exclusions. LCP and CBLOL therefore contribute to
training but are not fetched, predicted, or displayed by the daily workflow.

Every run, including `--dry-run`, writes
`reports/lol/daily/<UTC timestamp>.json` and `.md`. Unsupported teams and
excluded fixtures appear compactly in the report rather than as noisy repeated
terminal errors. Webhook URLs are never serialized, and Discord mentions are
disabled. Schema version 4 also records source freshness, exact historical
roster evidence when used, and the latest warning-only model drift review.

If schedule refresh fails, a stored schedule may be used with an explicit
warning. A failed required mutation suppresses predictions. Report persistence
failure prevents webhook delivery.

### PandaScore lineup enrichment

Schedule discovery and match-detail enrichment are separate. The workflow uses
PandaScore's generic `/matches/{id}` detail endpoint and never requests a
missing provider ID. A 401/403 disables further detail requests for that run,
keeps any lineups embedded in the schedule response, and records a single
compact enrichment warning. It does not turn a valid schedule into a pipeline
failure. Predictions without a complete expected lineup remain shadow-only.
When PandaScore supplies no lineup at all, the workflow may use the exact
role-mapped five only if that same roster appears in each map of the latest
three completed consecutive series before fixture start. Partial, conflicting,
or malformed provider lineups never fall back to history.

Use the examples under `ops/launchd/` for the supported recurring jobs.
Application logging is bounded; launchd stdout/stderr point to `/dev/null`.
The optional `com.oracle-bets.discord-bot` launchd example keeps owner-only
buttons and the prop Line/Odds modal available across restarts, and captures
closing lines itself. Webhook-only deployments use the separate closing-line
job; do not run both closing capture mechanisms together.
