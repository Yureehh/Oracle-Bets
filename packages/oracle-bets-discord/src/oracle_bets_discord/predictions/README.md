# predictions

- `lol.py`: LoL schedule, winner, prop, and read-only market presentation.
- `best_ofs.py`: compatibility dictionaries and handicap formatting built from
  the canonical series engine in `lol_bets.inference.series`.

The daily winner model ignores map side and first pick because they are unknown
before the fixture.
