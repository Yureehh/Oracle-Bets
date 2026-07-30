# Testing

Run the safe repository checks:

```bash
uv run pytest tests/lol
uv run pytest tests/core
uv run ruff check packages tests
uv run mkdocs build --strict
uv run oracle-bets lol health
uv run oracle-bets lol validate-data
uv run oracle-bets daily lol --dry-run --skip-market-search
```

Before a release or after changing ingestion/model schemas also run:

```bash
uv run oracle-bets lol reconcile-history
uv run oracle-bets lol retune --targets all --feature-set compact
uv run oracle-bets lol promote-tuning <run-id>
uv run oracle-bets lol train --targets all --feature-set compact
```

Required invariants include chronological availability, complete two-team/ten-
player games, exact complementary winner inference under team swapping, no
side/first-pick winner features, complete tuning/model-bundle publication,
atomic reports before webhook delivery, conservative market matching, bounded
logs, and valid launchd property lists.
