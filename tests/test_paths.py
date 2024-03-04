import os
from pathlib import Path

from utils.paths import (
    BASE_DIR,
    CONFIG_DIR,
    DATA_DIR,
    IMPORT_COLUMNS,
    INTERIM_DIR,
    INVALID_GAMES,
    PROCESSED_DIR,
    RAW_DIR,
)


class TestPaths:
    def test_paths(self):
        assert BASE_DIR == Path(os.getcwd())
        assert DATA_DIR == BASE_DIR / "data"
        assert RAW_DIR == DATA_DIR / "raw"
        assert INTERIM_DIR == DATA_DIR / "interim"
        assert PROCESSED_DIR == DATA_DIR / "processed"
        assert CONFIG_DIR == BASE_DIR / "config"
        assert INVALID_GAMES == CONFIG_DIR / "invalid_games.json"
        assert IMPORT_COLUMNS == CONFIG_DIR / "import_columns.json"

        # Check that the directories exist
        assert DATA_DIR.exists()
        assert RAW_DIR.exists()
        assert INTERIM_DIR.exists()
        assert PROCESSED_DIR.exists()
        assert CONFIG_DIR.exists()
        assert INVALID_GAMES.exists()
        assert IMPORT_COLUMNS.exists()
