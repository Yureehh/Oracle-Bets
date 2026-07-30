# Contributing

Keep changes surgical and preserve unrelated worktree edits.

For model or feature changes:

1. state the temporal availability of every input;
2. show that swapped teams produce exactly complementary winner probabilities;
3. compare untouched temporal log loss, Brier, ECE, and cohort calibration;
4. record feature-schema and dataset fingerprints;
5. do not promote automatically.

For ingestion changes, update source-column reconciliation and identity tests.
For market changes, prefer false negatives over false matches. No contribution
may add automated betting, signing, wallet, private-key, or fund movement.

Before handoff run the commands in [Testing](testing.md). Generated data,
models, reports, logs, and secrets are not committed; reviewed configuration,
notebooks with cleared outputs, migration manifests, and documentation are.
