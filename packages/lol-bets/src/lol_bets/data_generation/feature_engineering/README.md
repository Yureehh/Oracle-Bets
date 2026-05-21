# feature_engineering

LoL feature creation.

- `features_generator.py`: team/player derived features, series context, breaks, and head-to-head.
- `performance_features/`: EMA win rates and in-game stat trends.
- `ratings_features/`: Elo, Glicko2, Plackett-Luce, TrueSkill, league Elo, and WHR experiments.

Example:

```python
from lol_bets.data_generation.feature_engineering.features_generator import FeatureGenerator
```
