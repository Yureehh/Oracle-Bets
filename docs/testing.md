# Testing

Run the safe repository checks:

```bash
uv run pytest tests/lol
uv run pytest tests/core
uv run ruff check packages tests
uv run ty check packages
uv run mkdocs build --strict
uv run oracle-bets lol health
uv run oracle-bets lol validate-data
uv run oracle-bets lol validate-winner-model
uv run oracle-bets lol market-check
uv run oracle-bets evidence health
uv run oracle-bets model list
uv run oracle-bets daily lol --dry-run --no-ai-review
uv run oracle-bets discord doctor
```

Before a release or after changing ingestion/model schemas also run:

```bash
uv run oracle-bets lol reconcile-history
uv run oracle-bets lol build-series
uv run oracle-bets lol retune \
  --targets series_winner --feature-set full
uv run oracle-bets lol review-tuning <series-run-id> --format json
uv run oracle-bets lol promote-tuning <series-run-id>
uv run oracle-bets lol retune \
  --targets next_map_winner --feature-set full
uv run oracle-bets lol review-tuning <next-map-run-id> --format json
uv run oracle-bets lol promote-tuning <next-map-run-id>
uv run oracle-bets lol train --targets all --feature-set compact
```

Do not execute the research commands merely as a smoke test: they are expensive
and completion exposes their final holdouts. Use fixtures for CI. A real review records its
label hash and requires every row to be later than the target's previously
exposed test window.

Required invariants include chronological availability, complete two-team/ten-
player maps, deterministic complete-series reconstruction, exact complementary
winner inference under team swapping, no
side/first-pick/market winner features, complete feature lineage and
tuning/model-bundle publication,
one report pair per run, conservative market matching and CLOB depth, correlated
exposure caps, conflict-safe manual settlement, idempotent paper/Discord decisions,
LLM failure fallback, bounded messages/logs, no write-side market surface, and
valid launchd property lists.
