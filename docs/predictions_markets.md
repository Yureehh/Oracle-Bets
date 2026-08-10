# Predictions and markets

Winner output is a calibrated map probability converted into coherent
best-of-series outcomes. Props report expected game length, total kills, and
total towers with residual/calibrated uncertainty when available.

Prematch Game 1, Game 2, and Game 3 use the same pre-series map probability.
Match winner, exact scores, and over/under total maps are enumerated from that
probability under an explicit exchangeable/independent-map approximation. This
is not an in-play model: once a map starts or finishes, score, side, draft, and
roster changes require a future live workflow.

Confidence is evidence-based: probability distance from 50% alone is not high
confidence. Calibration health, uncertainty width, roster stability, team
identity, data freshness, and cohort support all matter.

Polymarket comparison is read-only. Matching requires compatible competition,
both teams, start time, best-of format, `sportsMarketType`, game number or
totals line, resolution terms, outcome-to-token orientation, and market meaning.
Only Game N `child_moneyline`, series `moneyline`, and series `totals` are
supported. Handicaps and kill/objective contracts are rejected. Ambiguous or
incomplete candidates are rejected; missed markets are preferable to false
precision.

An edge is:

```text
conservative model probability - executable implied probability
```

It is not profit evidence until the actual observed price, stake constraints,
fees/slippage assumptions, decision, closing price, and settlement are recorded.
No code places bets or interacts with wallets.

Every supported token book is read twice 30–60 seconds apart. The system walks
asks by intended risk, requires complete depth both times, and uses the worse
executable average price. A 5% conservative edge, quarter-Kelly sizing, 1-unit
position cap, 3-unit daily cap, and one correlated position per fixture apply.
Game length, kills, and towers remain fixed-0.25-unit research-only props.
