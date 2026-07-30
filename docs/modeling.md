# Modeling

## Targets

Four LightGBM models are retained:

- canonical game winner classification;
- game length regression;
- total kills regression;
- total towers regression.

Props are not substitutes for the winner model. The daily report needs both,
and each target has independent calibration/evaluation evidence.

## Exact winner symmetry

Historical games collapse to one row ordered by stable resolved team ID. The
target is whether canonical Team A won. Numeric team/player/roster/inactivity
features are Team A minus Team B; invariant context remains unchanged.
Inference canonicalizes once, predicts once, and complements for reversed caller
order. Therefore `P(A beats B) + P(B beats A) == 1` by construction.

Mirrored-call averaging is not used. Side, first pick, and other unavailable
fixture facts cannot enter the winner model.

## Temporal evaluation and calibration

Splits are chronological and keep entire games together. The tuning,
calibration, and untouched test windows are separate. Winner acceptance centers
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
- generates a new report;
- never invokes Optuna.

Explicit retuning:

- runs seeded Optuna research;
- writes candidate JSON inside the run report;
- does not mutate production parameters or models;
- requires review and all-four-target promotion;
- is followed by one routine full retraining.

## Attribution

Every training run stores LightGBM gain and split counts, held-out permutation
importance, mean absolute SHAP, local SHAP reasons, model cards, calibration,
cohorts, and split metadata. Importance is diagnostic, not proof of causality.

The production default is the tracked `compact` feature contract. The
`selected` mode is research-only and requires a prior temporal recommendation
report; it fails clearly when that report is absent instead of silently
training every available feature or ignoring `--max-features`.
