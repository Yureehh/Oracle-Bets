# LoL reports

- `ingestion/`: source history, data-quality, and column-reconciliation reports.
- `training/runs/<run-id>/`: model cards, metrics, calibration, attribution,
  figures, candidates from explicit retuning, and a combined JSON/Markdown
  summary. `training/latest.json` points to the latest completed run.
- `daily/`: timestamped JSON and Markdown reports, including dry runs.

Training keeps the latest 12 completed runs. Notebooks read these artifacts;
the production pipeline does not execute notebooks.
