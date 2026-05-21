# ratings_features

Rating-system features for LoL.

This folder contains Elo, Glicko2, Plackett-Luce, TrueSkill, league Elo, and WHR
implementations. Ratings should be tuned and validated with walk-forward splits.

Example:

```python
from lol_bets.data_generation.feature_engineering.ratings_features.elo import calculate_elo
```
