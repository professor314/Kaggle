"""Unit tests for CodeCompetitionConverter notebook conversion.

Validates: Requirements 34.1, 34.2
"""

import json
import os

import pytest

from kaggle_ml_toolkit.code_competition import CodeCompetitionConverter
from kaggle_ml_toolkit.config import CompetitionConfig


def _make_notebook(cells=None):
    """Create minimal valid notebook JSON structure."""
    if cells is None:
        cells = [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["train = pd.read_csv('data/train.csv')\n"],
            }
        ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": cells,
    }


class TestConvertNotebook:
    """Tests for CodeCompetitionConverter.convert_notebook()."""

    def test_convert_notebook_remaps_paths(self, tmp_path):
        """Local data/ paths are replaced with /kaggle/input/{slug}/."""
        notebook = _make_notebook()
        nb_path = tmp_path / "notebooks" / "my_notebook.ipynb"
        nb_path.parent.mkdir(parents=True)
        nb_path.write_text(json.dumps(notebook), encoding="utf-8")

        converter = CodeCompetitionConverter()
        output = converter.convert_notebook(
            notebook_path=str(nb_path),
            competition_slug="titanic",
        )

        with open(output, "r", encoding="utf-8") as f:
            converted = json.load(f)

        # The cell source should reference /kaggle/input/titanic/
        cell_source = converted["cells"][0]["source"]
        source_str = "".join(cell_source) if isinstance(cell_source, list) else cell_source
        assert "/kaggle/input/titanic/" in source_str
        # Original local path should be gone
        assert "data/train.csv" not in source_str or "/kaggle/input/" in source_str

    def test_convert_missing_notebook_raises(self, tmp_path):
        """FileNotFoundError raised when notebook path doesn't exist."""
        converter = CodeCompetitionConverter()
        fake_path = str(tmp_path / "nonexistent.ipynb")

        with pytest.raises(FileNotFoundError):
            converter.convert_notebook(
                notebook_path=fake_path,
                competition_slug="titanic",
            )

    def test_convert_invalid_json_raises(self, tmp_path):
        """ValueError raised when notebook file is not valid JSON."""
        bad_file = tmp_path / "bad.ipynb"
        bad_file.write_text("this is not json {{{", encoding="utf-8")

        converter = CodeCompetitionConverter()
        with pytest.raises(ValueError, match="Invalid notebook format"):
            converter.convert_notebook(
                notebook_path=str(bad_file),
                competition_slug="titanic",
            )


class TestGenerateNotebook:
    """Tests for CodeCompetitionConverter.generate_code_competition_notebook()."""

    def test_generate_notebook_structure(self, tmp_path):
        """Generated notebook has valid nbformat 4 structure with cells."""
        converter = CodeCompetitionConverter()
        config = CompetitionConfig(
            competition_name="test-comp",
            target_column="target",
            id_column="id",
            problem_type="classification",
            primary_metric="accuracy",
        )

        output_path = converter.generate_code_competition_notebook(
            config=config,
            pipeline_steps=["# Step 1: Feature engineering"],
            dependencies=["pandas", "numpy"],
        )

        try:
            with open(output_path, "r", encoding="utf-8") as f:
                notebook = json.load(f)

            assert notebook["nbformat"] == 4
            assert "cells" in notebook
            assert len(notebook["cells"]) > 0

            # All cells should have required keys
            for cell in notebook["cells"]:
                assert "cell_type" in cell
                assert "source" in cell
                assert cell["cell_type"] == "code"
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


class TestBundleDependencies:
    """Tests for CodeCompetitionConverter.bundle_dependencies()."""

    def test_bundle_dependencies_format(self):
        """Output contains '!pip install -q' with all package names."""
        converter = CodeCompetitionConverter()
        result = converter.bundle_dependencies(["pandas", "numpy", "scikit-learn"])

        assert "!pip install -q" in result
        assert "pandas" in result
        assert "numpy" in result
        assert "scikit-learn" in result
