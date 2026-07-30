# notebooks/lol

League of Legends review notebooks:

1. `01_ingestion_quality.ipynb`
2. `02_training_and_calibration.ipynb`
3. `03_feature_attribution.ipynb`
4. `04_profit_evidence.ipynb`

They read deterministic artifacts from `reports/` and `data/state/`. Production
training never executes notebooks. Keep committed outputs cleared and promote
any reusable calculation into a package before relying on it operationally.
