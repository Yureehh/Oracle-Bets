import ast
import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).parents[2] / "notebooks" / "lol"
NOTEBOOK_FORMAT = 4
EXPECTED_NOTEBOOKS = [
    "01_ingestion_quality.ipynb",
    "02_training_and_calibration.ipynb",
    "03_feature_attribution.ipynb",
    "04_profit_evidence.ipynb",
]


def test_analysis_notebooks_are_valid_cleared_python_notebooks():
    assert [path.name for path in sorted(NOTEBOOK_DIR.glob("*.ipynb"))] == (
        EXPECTED_NOTEBOOKS
    )

    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == NOTEBOOK_FORMAT
        assert notebook["metadata"]["kernelspec"]["name"] == "python3"
        assert notebook["cells"][0]["cell_type"] == "markdown"
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]), filename=str(path))
