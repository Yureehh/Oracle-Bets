# hyperparameters

Default search priors for ratings and model experiments.

`tuned/ratings/` contains reviewed rating inputs and `tuned/lightgbm/`
contains reviewed supervised-model inputs. Routine ingestion and training load
these files and never start Optuna. Defaults define initial/search priors; they
are not an automatic fallback for missing tuned artifacts.
