"""Property-based tests for Code Competition path correctness.

Validates: Requirements 34.1
"""

import json
import tempfile
import os

from hypothesis import given, settings, strategies as st

from kaggle_ml_toolkit.code_competition import CodeCompetitionConverter
from kaggle_ml_toolkit.config import CompetitionConfig


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Competition slugs: lowercase alphanumeric with hyphens
competition_slugs = st.from_regex(r"[a-z][a-z0-9\-]{2,20}", fullmatch=True)

# Pipeline step code snippets that reference data paths (read operations)
pipeline_code_steps = st.lists(
    st.sampled_from([
        "train = pd.read_csv('data/train.csv')",
        "test = pd.read_csv('./data/test.csv')",
        "feature_data = pd.read_csv('data/features.csv')",
        "# Feature engineering step",
        "X = train.drop(columns=['target'])",
    ]),
    min_size=1,
    max_size=3,
)

# Dependencies list
dependencies_list = st.lists(
    st.sampled_from(["pandas", "numpy", "scikit-learn", "lightgbm", "xgboost"]),
    min_size=1,
    max_size=4,
    unique=True,
)


# ---------------------------------------------------------------------------
# Property 36: Code Competition Path Correctness
# ---------------------------------------------------------------------------


@given(slug=competition_slugs, steps=pipeline_code_steps, deps=dependencies_list)
@settings(max_examples=20)
def test_code_competition_path_correctness(slug, steps, deps):
    """All data paths use /kaggle/input/, all output paths use /kaggle/working/.

    For any generated notebook, the notebook's own generated cells must satisfy:
    1. Data read paths reference /kaggle/input/
    2. Output/submission paths reference /kaggle/working/
    3. The notebook uses the correct competition slug in input paths

    **Validates: Requirements 34.1**
    """
    converter = CodeCompetitionConverter()
    config = CompetitionConfig(
        competition_name=slug,
        target_column="target",
        id_column="id",
        problem_type="classification",
        primary_metric="accuracy",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = converter.generate_code_competition_notebook(
            config=config,
            pipeline_steps=steps,
            dependencies=deps,
        )

        try:
            with open(output_path, "r", encoding="utf-8") as f:
                notebook = json.load(f)

            # Collect all source code from the notebook
            all_source = ""
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    source = cell.get("source", "")
                    if isinstance(source, list):
                        all_source += "".join(source)
                    else:
                        all_source += source

            # The generated notebook MUST reference /kaggle/input/{slug}/
            expected_input = f"/kaggle/input/{slug}/"
            assert expected_input in all_source, (
                f"Expected input path '{expected_input}' not found in notebook"
            )

            # The generated notebook MUST reference /kaggle/working/ for output
            assert "/kaggle/working/" in all_source, (
                "Expected output path '/kaggle/working/' not found in notebook"
            )

            # The submission cell specifically writes to /kaggle/working/
            # Find the last code cell (submission cell)
            last_cell = notebook["cells"][-1]
            last_source = last_cell.get("source", "")
            if isinstance(last_source, list):
                last_source_str = "".join(last_source)
            else:
                last_source_str = last_source
            assert "/kaggle/working/" in last_source_str, (
                "Submission cell must write to /kaggle/working/"
            )
        finally:
            # Clean up generated notebook
            if os.path.exists(output_path):
                os.remove(output_path)
