# Locked LoL paper protocol

The protocol is recorded in `config/research/lol-paper-v1.json`. Its evaluation
window is October 11 through November 30, 2026, Europe/Rome. Model weights,
hyperparameters, calibration, and uncertainty artifacts remain fixed throughout
that window. Daily feature state may advance using completed games available
before each decision. Forecasts retain the model, feature generation, and actual
decision timestamp; later reconstructions are not prospective forecasts.

## Why the August score does not establish a clean champion

The August bundle records source commit
`44f76c7cfcd12bb4b1c69a0b5afc94abad36bb46`. At that commit, the default Elo,
Glicko, Plackett–Luce, and TrueSkill paths loaded a final league-rating table
into earlier entity initialization or transfers. The bundle's source window
ends August 30, 2026, while its series training partition ends May 10, 2025.
This code permitted later results to influence historical training features.
The causal-prior corrections removed that dependency and regression tests
verify the corrected defaults are independent of the final league table.

This is evidence of a look-ahead defect in the recorded training recipe,
not proof of conventional overfitting or a measurement of how much it explains
the August model's better replay score. August remains quarantined. Historical
comparisons with it are retained as diagnostics, not treated as a clean causal
standard that justifies keeping it in service.

## Replacement and explicit paper override

The five fresh studies and full refit have already completed. Their weight
origin is `lol-20261009T231405_688954Z`; the paper package is
`lol-20261010-paper-v1`. It retains the exact trained weights and parameters,
with corrected evaluation reports and verified binding to the rebuilt input
generation. Repackaging is not another training run. The series base learner
was fitted through February 1, 2026, and the source extends through October 8.

The owner requested replacing the defective legacy benchmark with the corrected
model for paper research. This manual selection does not turn failed automatic
quality gates into passes. Series log loss is 0.562790; Brier is 0.190816;
calibration error is 0.035164. Its conservative probability bounds failed several
league coverage checks. Map and prop targets also remain exploratory.

An index-alignment bug caused the original training coverage report to omit
league/actionable membership on sliced dataframes. The later review correctly
detected failures. The corrected report includes those cohorts; old reports
remain archived. Fixing the report does not improve model probabilities or
their statistical coverage.

The paper champion permits research forecasts and owner-confirmed exploratory
paper samples. All recommendation states remain disabled. Model checksum and
structural correctness checks still apply. The manual transition retains the
failed review, source/weight provenance, replay receipt, protocol hash, and
owner-override reason. It makes no claim of proven edge or optimized sizing.

## Evaluation rules

1. The owner declares the leagues, fixture rules, and included markets before
   forecasts or prices are inspected. Every enrolled fixture stays in coverage
   reports, including unavailable and no-bet cases.
2. Pin the paper candidate and its causal rating baseline. Do not select model
   parameters or calibration methods using evaluation-window outcomes.
3. Record exact market contracts, observed executable prices, source references,
   closing prices, and sourced settlements. Missing evidence stays missing.
4. Declare the paper bankroll and capped fractional-Kelly policy before the
   first entry. Assess that locked policy on subsequent observations. A
   provisional policy is not an empirically optimal policy.
5. Keep historical diagnostics separate from prospective probability quality,
   coverage, closing-line value, return, exposure, and drawdown reports.
6. At the window's end, apply the existing quantitative gates and cohort support
   requirements. Insufficient evidence means inconclusive; register a separate
   later window rather than rewriting this one.
7. A critical bug stops the affected experiment. Retain the records and
   preregister a replacement epoch; do not quietly change the model mid-window.

This protocol locks the model and dates. The owner still needs to supply the
fixture population, bankroll, exact markets, and paper confirmations. There
are currently no settled prospective paper bets or demonstrated earnings.
