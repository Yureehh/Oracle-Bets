# Predictions and markets

Winner output is a calibrated map probability converted into coherent
best-of-series outcomes. Props report expected game length, total kills, and
total towers with residual/calibrated uncertainty when available.

Confidence is evidence-based: probability distance from 50% alone is not high
confidence. Calibration health, uncertainty width, roster stability, team
identity, data freshness, and cohort support all matter.

Polymarket comparison is read-only. Matching requires compatible competition,
both teams, start time, best-of format, selection orientation, and market
meaning. Ambiguous or incomplete candidates are rejected. This deliberately
prefers missed markets over false precision.

An edge is:

```text
conservative model probability - executable implied probability
```

It is not profit evidence until the actual observed price, stake constraints,
fees/slippage assumptions, decision, closing price, and settlement are recorded.
No code places bets or interacts with wallets.
