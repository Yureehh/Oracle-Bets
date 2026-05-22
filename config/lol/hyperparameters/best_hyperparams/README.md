# best_hyperparams

Generated best-known rating hyperparameters.

This folder intentionally ships without tuned JSON files. The next full
ingestion run should create fresh Optuna outputs here, then those outputs should
be reviewed against walk-forward validation before being committed.
