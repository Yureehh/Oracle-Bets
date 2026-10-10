# Remaining owner steps for paper testing

The engineering preparation is complete: `lol-20261010-paper-v1` is the active
paper champion, the causal baseline is fixed, and the model protocol is locked.
The August artifact is quarantined. All current market strategies remain
exploratory or display-only; recommendation and real-stake activation are disabled.
No complete prospective quote-to-settlement cycle has been recorded yet.

The window covers preregistered fixtures scheduled from **October 11 through
November 30, 2026, Europe/Rome**. Forecasts must be recorded after model selection
and before fixture start. Daily features can advance with completed games;
weights, parameters, calibration, and uncertainty remain fixed. See the
[protocol](lol-paper-protocol.md) and [command runbook](commands.md).

## 1. Declare the population before inspecting forecasts or odds

Choose the leagues and tournaments you actually want to study, eligible dates,
fixture rules, and markets. Record those rules before your first forecast or
quote. Include exact map, series, handicap, duration, kills, and towers contracts
where supported; unsupported contracts remain observations. Keep every enrolled
fixture in the denominator, including missing predictions, missing prices, and
no-entry decisions. Do not drop fixtures after seeing their result.

Keep a dated population declaration with your paper records. Use `/oracle` →
**Schedule** to inspect future fixtures, then run the daily workflow below
before **Review Markets**. The daily workflow records prospective enrollment
for its future schedule population. Check that the market report says
`enrolled_before_review`; an unregistered fixture does not belong to the
systematic paper sample. Your predeclared league rules select the study subset
from that complete recorded population. Historical reconstructed forecasts
cannot replace these prospective records.

## 2. Declare the paper bankroll, currency, and sizing policy

Use a separate virtual bankroll and one declared currency. Record the initial
bankroll and the policy before the first entry. The implemented provisional
policy is quarter Kelly with caps of 2% per ticket, 5% per sporting fixture,
and 20% total open exposure. These are research defaults, not an optimized
strategy or a promised drawdown ceiling. Existing unsettled positions consume
exposure capacity. Non-positive-edge cases remain observations with no stake.

Use the existing capped policy consistently for this window; do not select a
new fraction from the same outcomes used to report performance. Alternative
fractions can be compared as explicitly identified research tracks. A change
to the actual entry policy requires a separately registered experiment.

## 3. Check health and refresh the daily data

Back up the evidence database, then check the active model and runtime:

```bash
uv run oracle-bets evidence backup --output ~/OracleBetsArchives/backups
uv run oracle-bets daily lol
uv run oracle-bets model status
uv run oracle-bets lol health
uv run oracle-bets evidence health
uv run oracle-bets discord doctor
```

Expect champion `lol-20261010-paper-v1`. Run the daily data workflow before
new decisions; retain its failures rather than forecasting from unchecked
inputs. Do not run training, Optuna, or model promotion during the window.
The managed Discord gateway is already running; do not start a second worker.

## 4. Capture an exact contract and executable price

In `/oracle` → **Review Markets**, supply one or two links for the same fixture.
Thunderpick requires manual transcription of visible lines:

```text
target | selection | decimal odds | line | game number
```

Check teams, map/series period, selection, line, scheduled start, best-of format,
and push/void rules against the provider. Retain the URL and observation time.
For Polymarket, inspect the executable book evidence and settlement terms.
An approximate or mismatched contract cannot be treated as a comparable price.

## 5. Inspect the comparison and record only eligible paper entries

Read the full report, including calibration, conservative-bound, roster,
freshness, and missing-market warnings. Current valid comparisons are
**exploration**, not recommendations. Unsupported/display-only contracts cannot
create tickets. Missing quotes and non-positive-edge opportunities stay in the
coverage record without a paper stake.

For an eligible positive-edge exploratory sample, use **Record Bet**, select
**paper**, and confirm the executable price, current virtual bankroll, currency,
and capped stake. Check the model ID and fixture before confirmation. The
acceptance step recomputes the exposure limit; reprice if odds or available
size have changed. A positive estimated edge remains a hypothesis.

## 6. Capture the closing quote

Before fixture start, record a separate closing observation for the same exact
contract and selection. Retain its source and timestamp. A changed line or
settlement rule is a different contract. If the close is unavailable, mark it
missing; do not reconstruct it later or substitute the entry price.

## 7. Record the official result and settle

Use **Open Bets** to record a sourced **Win/Loss/Push/Void** result. Verify the
specific map or series outcome and provider settlement rule. Keep unresolved
components open when a fixture is only partially settled. Use **Closed Bets**
and **Performance** to inspect the ledger; correct evidence through recorded
corrections rather than deleting history.

```bash
uv run oracle-bets bet list --state open --mode paper
uv run oracle-bets bet performance --mode paper
```

## 8. Repeat across the complete population

Retain all eligible fixtures and missing stages, not just entries or winners.
Review probability quality (log loss, Brier, calibration), cohort coverage,
closing-line value, net paper return, exposure, and drawdown by league and
target. Report derived handicap/map-path markets separately from direct series
predictions and props. Keep the model and entry policy fixed throughout the
window. A critical defect stops the affected experiment and starts a separately
registered epoch after repair.

## 9. Return the completed evidence for review

After November 30, review the locked window from December 1. Send the population
rules, bankroll/policy declaration, evidence export or database backup, and
performance report for analysis. Insufficient settled or representative support
means **inconclusive**, followed by a separately registered later window.
Real stakes remain disabled. Tennis activation and further model work follow
review of the operational cycle and target-specific evidence.

Second-machine installation checks and profiling remain engineering follow-ups;
they do not replace the prospective evidence you collect here.
