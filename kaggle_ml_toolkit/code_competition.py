"""Code Competition Converter for the Kaggle ML Toolkit.

Converts local pipeline notebooks into Kaggle Code Competition-compatible
format by remapping file paths and bundling dependencies. Also generates
fresh notebooks from scratch with proper Kaggle path conventions.

Validates: Requirements 23.1, 23.2, 23.3
"""

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from kaggle_ml_toolkit.config import CompetitionConfig


class CodeCompetitionConverter:
    """Converts local notebooks to Kaggle Code Competition format.

    Handles path remapping from local directories to Kaggle's standard
    /kaggle/input/ and /kaggle/working/ paths, generates dependency
    installation cells, and produces complete notebook JSON structures.

    Attributes:
        KAGGLE_INPUT_PREFIX: Standard Kaggle input data path prefix.
        KAGGLE_WORKING_PREFIX: Standard Kaggle working/output path prefix.
    """

    KAGGLE_INPUT_PREFIX = "/kaggle/input/"
    KAGGLE_WORKING_PREFIX = "/kaggle/working/"

    def convert_notebook(
        self,
        notebook_path: str,
        competition_slug: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Convert a local pipeline notebook to Code Competition format.

        Loads an existing .ipynb file, remaps all local data paths to
        /kaggle/input/{competition_slug}/ and output paths to
        /kaggle/working/, then saves the modified notebook.

        Args:
            notebook_path: Path to the source .ipynb notebook file.
            competition_slug: The Kaggle competition slug used in paths.
            output_path: Optional path for the converted notebook.
                Defaults to {notebook_name}_kaggle.ipynb in the same directory.

        Returns:
            Path to the converted notebook file.

        Raises:
            FileNotFoundError: If notebook_path doesn't exist.
            ValueError: If the file is not valid JSON or not a notebook
                (missing "cells" key).
        """
        nb_path = Path(notebook_path)

        if not nb_path.exists():
            raise FileNotFoundError(
                f"Notebook not found: {notebook_path}"
            )

        # Load and parse the notebook
        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                notebook = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid notebook format") from exc

        if "cells" not in notebook:
            raise ValueError("Invalid notebook format")

        # Deep copy to avoid mutating the original data structure
        notebook = copy.deepcopy(notebook)

        # Determine the local data directory from the notebook's location
        local_data_dir = str(nb_path.parent.parent / "data")

        # Process each code cell
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                source = cell.get("source", "")
                # Handle source as list of lines or single string
                if isinstance(source, list):
                    source_str = "".join(source)
                else:
                    source_str = source

                remapped = self.remap_paths(
                    source_str, local_data_dir, competition_slug
                )

                # Preserve format (list vs string)
                if isinstance(cell.get("source"), list):
                    cell["source"] = remapped.splitlines(keepends=True)
                    # Handle case where source doesn't end with newline
                    if remapped and not cell["source"]:
                        cell["source"] = [remapped]
                else:
                    cell["source"] = remapped

        # Determine output path
        if output_path is None:
            stem = nb_path.stem
            output_path = str(nb_path.parent / f"{stem}_kaggle.ipynb")

        # Save the converted notebook
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)

        return str(out_path)

    def generate_code_competition_notebook(
        self,
        config: CompetitionConfig,
        pipeline_steps: List[str],
        dependencies: Optional[List[str]] = None,
    ) -> str:
        """Generate a new Code Competition notebook from scratch.

        Creates a complete .ipynb notebook with proper Kaggle path
        conventions. The notebook includes dependency installation,
        imports, pipeline steps, and submission output.

        Args:
            config: CompetitionConfig with competition settings.
            pipeline_steps: List of pipeline step descriptions/code blocks
                to include as individual cells.
            dependencies: Optional list of pip package names to install.

        Returns:
            Path to the generated notebook file.
        """
        cells: List[Dict[str, Any]] = []

        # Cell 1: Dependency installation (if dependencies provided)
        if dependencies:
            dep_source = self.bundle_dependencies(dependencies)
            cells.append(self._make_code_cell(dep_source))

        # Cell 2: Imports and config loading
        input_path = f"{self.KAGGLE_INPUT_PREFIX}{config.competition_name}/"
        imports_source = (
            "import pandas as pd\n"
            "import numpy as np\n"
            "from pathlib import Path\n"
            "\n"
            f"# Competition configuration\n"
            f"COMPETITION = \"{config.competition_name}\"\n"
            f"TARGET_COLUMN = \"{config.target_column}\"\n"
            f"ID_COLUMN = \"{config.id_column}\"\n"
            f"INPUT_DIR = \"{input_path}\"\n"
            f"OUTPUT_DIR = \"{self.KAGGLE_WORKING_PREFIX}\"\n"
            "\n"
            f"# Load data from Kaggle input\n"
            f"train = pd.read_csv(INPUT_DIR + \"train.csv\")\n"
            f"test = pd.read_csv(INPUT_DIR + \"test.csv\")\n"
            f"print(f\"Train shape: {{train.shape}}, Test shape: {{test.shape}}\")\n"
        )
        cells.append(self._make_code_cell(imports_source))

        # Cell 3+: One cell per pipeline step
        for step in pipeline_steps:
            # Replace any generic data path references with Kaggle paths
            step_source = step.replace(
                "./data/", input_path
            ).replace(
                "data/", input_path
            )
            cells.append(self._make_code_cell(step_source))

        # Final cell: Save submission
        submission_source = (
            "# Generate submission file\n"
            f"submission = pd.DataFrame({{\n"
            f"    \"{config.id_column}\": test[\"{config.id_column}\"],\n"
            f"    \"{config.target_column}\": predictions\n"
            f"}})\n"
            f"submission.to_csv(\"{self.KAGGLE_WORKING_PREFIX}submission.csv\", index=False)\n"
            f"print(f\"Submission saved: {{submission.shape[0]}} rows\")\n"
            f"submission.head()\n"
        )
        cells.append(self._make_code_cell(submission_source))

        # Assemble full notebook structure
        notebook: Dict[str, Any] = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0",
                },
            },
            "cells": cells,
        }

        # Save the notebook
        output_filename = f"{config.competition_name}_submission.ipynb"
        output_path = Path(output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)

        return str(output_path)

    def remap_paths(
        self,
        cell_source: str,
        local_data_dir: str,
        competition_slug: str,
    ) -> str:
        """Remap file paths in a single cell's source code.

        Replaces local data directory references with Kaggle input paths
        and local output/submission paths with Kaggle working paths.

        Args:
            cell_source: The source code string from a notebook cell.
            local_data_dir: The local data directory path to replace.
            competition_slug: The competition slug for Kaggle input paths.

        Returns:
            Transformed cell source with remapped paths.
        """
        result = cell_source
        kaggle_input = f"{self.KAGGLE_INPUT_PREFIX}{competition_slug}/"

        # Normalize the local_data_dir for consistent matching
        # Handle both forward and backslash variants
        local_dir_normalized = local_data_dir.replace("\\", "/")

        # Replace the local data directory (absolute paths) with Kaggle input
        for local_variant in [
            local_data_dir,
            local_dir_normalized,
            local_data_dir.rstrip("/\\"),
            local_dir_normalized.rstrip("/"),
        ]:
            if local_variant:
                # Escape for regex (handle backslashes and special chars)
                escaped = re.escape(local_variant)
                # Match with optional trailing slash
                pattern = escaped + r"[/\\]?"
                result = re.sub(pattern, kaggle_input, result)

        # Replace common relative data path patterns with Kaggle input
        data_patterns = [
            r"\./data/",
            r"data/",
            r"\.\\/data\\/",
            r"data\\/",
        ]
        for pattern in data_patterns:
            result = re.sub(pattern, kaggle_input, result)

        # Replace common local output patterns with /kaggle/working/
        output_patterns = [
            r"\./submissions/",
            r"submissions/",
            r"\./output/",
            r"output/",
            r"\./results/",
            r"results/",
        ]
        for pattern in output_patterns:
            result = re.sub(pattern, self.KAGGLE_WORKING_PREFIX, result)

        return result

    def bundle_dependencies(self, dependencies: List[str]) -> str:
        """Generate a pip install cell source for offline use.

        Creates a cell that installs all specified dependencies using pip
        with the quiet flag for cleaner notebook output.

        Args:
            dependencies: List of pip package names to install.

        Returns:
            Cell source string for dependency installation.
        """
        deps_str = " ".join(dependencies)
        return f"!pip install -q {deps_str}\n"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_code_cell(self, source: str) -> Dict[str, Any]:
        """Create a notebook code cell dictionary.

        Args:
            source: The source code for the cell.

        Returns:
            A dictionary representing a valid notebook code cell.
        """
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }
