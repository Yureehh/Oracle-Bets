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

`models/lol/` is replaced by successful full training. The model registry
freezes complete artifacts with checksums, metadata, candidate history, and a
champion pointer. Promotion and rollback are explicit owner actions.
