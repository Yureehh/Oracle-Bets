# Reports and notebooks

Production Python writes deterministic artifacts:

- ingestion quality and history reconciliation;
- one timestamped training directory with per-model metrics, calibration,
  attribution, model cards, figures, manifest, JSON summary, and Markdown
  summary;
- one JSON/Markdown pair per daily run;
- monthly evidence audits.

The latest 12 completed training reports are retained. Partial/failed runs remain
available for diagnosis and are not published as latest.

Notebooks are thin read-only review surfaces:

1. ingestion quality;
2. training and calibration;
3. feature attribution;
4. paper-profit evidence.

The pipeline never executes notebooks. This avoids hidden state and keeps every
required calculation testable in package code. Committed notebook outputs are
cleared.

Run the ingestion notebook after a source/schema investigation, the
training/calibration and attribution notebooks after a research retune or
challenger comparison, and the profit notebook during the monthly evidence
review. None is required for the daily workflow. Notebooks consume report
artifacts; they do not create production evidence or change models.

An optional low-token agent should summarize existing JSON, not execute a
notebook or scan the repository. The deterministic Markdown report remains the
fallback when no agent is available.
