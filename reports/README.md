# reports

Generated, immutable review artifacts are scoped by module. `lol/` contains
ingestion, training, and daily reports; `audits/` contains periodic owner
reviews. Reports are inputs to notebooks, not interchangeable with them:
production code writes reports and notebooks only read them.
