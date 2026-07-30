# lol_bets

LoL domain package.

- `pipeline.py`: ingest and materialize training/inference parquet artifacts.
- `training.py`: train configured prediction models.
- `daily.py`: orchestrate the scheduled report workflow.
- `module.py`: Oracle Bets module contract and artifact health checks.
- `data_generation/`: ingestion, features, and ratings.
- `prediction_models/`: preprocessing, model training, and evaluation.
- `inference/`: team/roster resolution, symmetric matchup, and series prediction.
- `operations/`: evidence, identity, and model-lifecycle orchestration.

Example:

```python
from lol_bets.module import LoLBetsModule

LoLBetsModule().artifact_health().raise_if_unhealthy()
```
