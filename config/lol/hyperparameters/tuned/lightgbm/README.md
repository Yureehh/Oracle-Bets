# tuned LightGBM

Reviewed production hyperparameters for the winner, game-length, total-kills,
and total-towers models live here as JSON.

`oracle-bets lol retune` writes candidates under a timestamped training report.
After reviewing temporal calibration, loss, and cohort evidence, promote the
complete four-target candidate bundle with:

```bash
uv run oracle-bets lol promote-tuning <run-id>
```

Routine `lol train` loads these files and never runs Optuna.
