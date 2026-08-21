# Oracle Bets

Oracle Bets is a local-first League of Legends betting-research system. It
turns historical Oracle's Elixir data and upcoming PandaScore fixtures into
calibrated probabilities, prop estimates, reviewable reports, one-way Discord
messages, and read-only Polymarket comparisons.

The objective is long-run decision quality and profit evidence, in this order:

1. prevent temporal and identity leakage;
2. minimize out-of-sample log loss and Brier score;
3. maintain calibration across time and important cohorts;
4. match markets conservatively and record the price actually available;
5. assess paper ROI, CLV, drawdown, and calibration together.

The repository contains no automated betting, wallet, signing, private-key,
order-submission, or fund-movement path. Counter-Strike and real sports are
future architecture concerns, not active implementations.

Read the [system design and operating decisions](system.md), use the
[command runbook](commands.md), and check the [roadmap](roadmap.md) before any
model promotion or paper proposal.
