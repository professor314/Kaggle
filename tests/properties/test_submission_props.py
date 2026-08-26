"""Property-based tests for SubmissionGenerator CSV format and naming.

Validates: Requirements 15.3, 15.5
"""

import re
import tempfile

import pandas as pd
from hypothesis import given, settings

from kaggle_ml_toolkit.config import CompetitionConfig
from kaggle_ml_toolkit.submission_generator import SubmissionGenerator
from tests.conftest import matched_predictions


# ---------------------------------------------------------------------------
# Property 27: Submission CSV Format
# Feature: kaggle-ml-toolkit, Property 27: Submission CSV Format
# ---------------------------------------------------------------------------


@given(data=matched_predictions())
@settings(max_examples=50)
def test_submission_csv_format(data):
    """For any predictions array and config, the generated CSV has exactly
    2 columns (id_column and target_column), a header row, no index column,
    and row count equal to len(predictions).

    **Validates: Requirements 15.3**
    """
    predictions, test_ids = data

    config = CompetitionConfig(
        competition_name="testcomp",
        target_column="target",
        id_column="id",
        problem_type="classification",
        primary_metric="accuracy",
    )

    generator = SubmissionGenerator()

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = generator.generate(
            predictions=predictions,
            test_ids=test_ids,
            config=config,
            model_name="prop_model",
            output_dir=tmp_dir,
        )

        # Read the CSV back
        df = pd.read_csv(file_path)

        # Exactly 2 columns
        assert df.shape[1] == 2

        # Column names match config
        assert list(df.columns) == [config.id_column, config.target_column]

        # Row count matches predictions length
        assert len(df) == len(predictions)

        # No index column (read_csv without index_col means the file had no index)
        # Verify by reading raw lines: header + data rows = len(predictions) + 1
        with open(file_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == len(predictions) + 1  # header + data rows

        # Header row matches expected column names
        header = lines[0].strip()
        assert header == f"{config.id_column},{config.target_column}"


# ---------------------------------------------------------------------------
# Property 28: Submission File Naming Pattern
# Feature: kaggle-ml-toolkit, Property 28: Submission File Naming Pattern
# ---------------------------------------------------------------------------

FILENAME_PATTERN = re.compile(
    r"^(?P<model_name>.+)_\d{8}_\d{6}\.csv$"
)


@given(data=matched_predictions())
@settings(max_examples=50)
def test_submission_file_naming_pattern(data):
    """For any predictions, the generated filename matches the pattern
    `{model_name}_YYYYMMDD_HHmmss.csv`.

    **Validates: Requirements 15.5**
    """
    predictions, test_ids = data

    config = CompetitionConfig(
        competition_name="testcomp",
        target_column="survived",
        id_column="passengerid",
        problem_type="classification",
        primary_metric="accuracy",
    )

    generator = SubmissionGenerator()
    model_name = "xgboost_v2"

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = generator.generate(
            predictions=predictions,
            test_ids=test_ids,
            config=config,
            model_name=model_name,
            output_dir=tmp_dir,
        )

        # Extract just the filename from the path
        from pathlib import Path

        filename = Path(file_path).name

        # Must match the naming pattern
        match = FILENAME_PATTERN.match(filename)
        assert match is not None, f"Filename '{filename}' does not match pattern"

        # The model_name prefix must match what we provided
        assert match.group("model_name") == model_name
