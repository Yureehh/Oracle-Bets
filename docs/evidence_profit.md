# Evidence and profit

`data/state/oracle_bets.db` is the single append-only evidence graph for runs,
fixtures, identities, model versions, predictions, market observations,
proposals, paper positions, settlement, corrections, and audit events.

The old `data/ledger.db` was checkpointed under
`data/state/archives/legacy-ledger/` with hashes and a row count. Its
byte-for-byte original and sidecars are retained under that archive; no legacy
ledger remains loose in `data/`.

## What to evaluate

Probability quality comes first:

- temporal log loss and Brier score;
- ECE, reliability, slope/intercept;
- league, patch, roster, edge, and confidence cohorts.

Profit evidence then adds:

- actual quoted/opening odds;
- closing-line value;
- ROI with uncertainty;
- maximum drawdown;
- realistic capped staking;
- performance by league, market, and edge bucket.

Small samples and theoretical model edges do not activate a strategy.
No-bet decisions are useful evidence and should remain visible.

## Models versus registry

Full training stages artifacts and freezes a complete checksum-verified
candidate before any serving change. The registry stores candidate metadata,
history, and a champion pointer. The first champion and every Optuna-derived
candidate require explicit owner promotion. A routine candidate may promote
automatically only after same-row paired non-inferiority, calibration, cohort,
regression-target, and artifact-health gates pass; otherwise the current
champion remains active.

Scalar game-length, kills, and towers means are stored as forecasts. A manual
line is priced later with that forecast's exact registered residual calibrator,
then recorded as a line-specific probability and fixed 0.25-unit research-only
proposal. Settlement loads the immutable odds and stake from the position.

Paper positions remain open until the owner records `win`, `loss`, `push`, or
`void` with `paper settle` or the owner-only Discord confirmation modal. A
non-empty result URL or source ID is required, every record is marked
`owner_verified`, and conflicting resubmissions fail visibly. No daily or
provider process automatically closes a position. `paper performance` reports
turnover, ROI and its bootstrap interval, drawdown, hit rate, calibration, and
target/league cohorts, with optional `--since`, `--target`, and `--league`
filters. `paper capture-closing` reads a fresh public CLOB book only for open
Polymarket positions starting within the configured window, walks the recorded
paper stake, and stores a close only when the depth is complete. Run it every
five minutes, or let the Gateway bot do so every minute. Settlement selects the
latest fillable pre-start close and records probability and odds-ratio CLV. CLV
remains explicitly unavailable when no valid close was captured; display prices
are never substituted. Provider-backed automatic settlement is future work and
requires verified outcome identity, captured resolution rules,
correction/conflict support, and explicit owner opt-in.
