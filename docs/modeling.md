# Modeling

## Targets

Six research targets are retained:

- legacy map winner classification (diagnostic only);
- direct prematch series winner classification (the only actionable target);
- experimental next-map winner classification (shadow only);
- game length regression;
- total kills regression;
- total towers regression.

Props and map forecasts are not substitutes for the direct-series model. Each
target has independent calibration/evaluation evidence, but only the prematch
series winner can create a paper proposal.

## Exact winner symmetry

Complete historical series collapse to one row ordered by stable resolved team
ID. The target is whether canonical Team A won the series. Numeric
team/player/roster/inactivity
features are Team A minus Team B; invariant context remains unchanged.
Inference canonicalizes once, predicts once, and complements for reversed caller
order. Therefore `P(A beats B) + P(B beats A) == 1` by construction.

Mirrored-call averaging is not used. Side, first pick, and other unavailable
fixture facts cannot enter the winner model.

## Temporal evaluation and calibration

Winner V2 uses timestamp- and series-atomic 55% training, 5% tuning, 10%
calibration-fit, 10% calibration/blend-selection, 5% uncertainty, and 15%
untouched test partitions. Optuna rolling-origin folds are confined to the
first 60%. Winner acceptance centers
on log loss, Brier score, ECE, calibration slope/intercept, reliability tables,
probability-sum invariants, and league/patch/roster cohorts. Accuracy is
secondary.

Calibration candidates are evaluated on held-out game rows and selected
conservatively. Props store residual distributions and calibrated line
probabilities, not only point estimates.

## Retraining versus retuning

Routine retraining:

- loads reviewed JSON hyperparameters;
- refits weights on refreshed data;
- refits calibrators;
- generates a report and immutable complete candidate;
- replays champion and candidate on the candidate's exact sealed test rows;
- auto-promotes only a routine candidate that passes log-loss non-inferiority,
  Brier, ECE, actionable/per-cohort, prop-MAE, status, symmetry, leakage, and
  artifact gates;
- never invokes Optuna.

Explicit retuning:

- runs seeded Optuna research;
- writes candidate JSON inside the run report;
- does not mutate production parameters or models;
- promotes only the complete requested target set after owner review (normally
  `series_winner,next_map_winner` for the one-time V2 study);
- marks reviewed parameter files with their Optuna run provenance;
- is followed by one full training whose model candidate still requires manual
  champion promotion.

## Attribution

Every training run stores final feature lineage (source, prematch availability,
family, swap behavior, and eligibility), LightGBM gain and split counts, held-out permutation
importance, mean absolute SHAP, local SHAP reasons, model cards, calibration,
cohorts, split metadata, feature availability/missingness, league coverage,
prediction-distribution shift, and top feature/family stability. Drift findings
are warning-only review evidence; they do not create or bypass promotion gates.
Importance is diagnostic, not proof of causality.

The production default is the tracked `compact` feature contract. The
`selected` mode is research-only and requires a prior temporal recommendation
report; it fails clearly when that report is absent instead of silently
training every available feature or ignoring `--max-features`.
