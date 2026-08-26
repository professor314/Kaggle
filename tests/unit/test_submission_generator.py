"""Unit tests for SubmissionGenerator.

Validates: Requirements 15.3, 15.4, 15.5, 15.6, 15.7, 15.8
"""

import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from kaggle_ml_toolkit.config import CompetitionConfig
from kaggle_ml_toolkit.submission_generator import SubmissionGenerator


@pytest.fixture
def config():
    """A standard CompetitionConfig for submission tests."""
    return CompetitionConfig(
        competition_name="titanic",
        target_column="Survived",
        id_column="PassengerId",
        problem_type="classification",
        primary_metric="accuracy",
    )


@pytest.fixture
def generator():
    """A SubmissionGenerator instance."""
    return SubmissionGenerator()


@pytest.fixture
def sample_predictions():
    """Sample predictions and test IDs."""
    predictions = np.array([0, 1, 1, 0, 1])
    test_ids = pd.Series([892, 893, 894, 895, 896], name="PassengerId")
    return predictions, test_ids


class TestGenerateCreatesCsv:
    """test_generate_creates_csv — verify file created with correct content."""

    def test_generate_creates_csv(self, generator, config, sample_predictions):
        """Verify that generate() creates a CSV file with correct content."""
        predictions, test_ids = sample_predictions

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = generator.generate(
                predictions=predictions,
                test_ids=test_ids,
                config=config,
                model_name="test_model",
                output_dir=tmp_dir,
            )

            # File exists
            assert Path(file_path).exists()

            # Read it back and verify content
            df = pd.read_csv(file_path)
            assert list(df[config.id_column]) == list(test_ids)
            np.testing.assert_array_equal(df[config.target_column].values, predictions)


class TestGenerateCorrectColumns:
    """test_generate_correct_columns — verify exactly 2 columns matching config."""

    def test_generate_correct_columns(self, generator, config, sample_predictions):
        """Verify generated CSV has exactly 2 columns with names from config."""
        predictions, test_ids = sample_predictions

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = generator.generate(
                predictions=predictions,
                test_ids=test_ids,
                config=config,
                model_name="col_test",
                output_dir=tmp_dir,
            )

            df = pd.read_csv(file_path)
            assert df.shape[1] == 2
            assert list(df.columns) == [config.id_column, config.target_column]


class TestGenerateCorrectRowCount:
    """test_generate_correct_row_count — verify row count matches predictions length."""

    def test_generate_correct_row_count(self, generator, config, sample_predictions):
        """Verify generated CSV has the same number of rows as predictions."""
        predictions, test_ids = sample_predictions

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = generator.generate(
                predictions=predictions,
                test_ids=test_ids,
                config=config,
                model_name="row_test",
                output_dir=tmp_dir,
            )

            df = pd.read_csv(file_path)
            assert len(df) == len(predictions)

    def test_generate_large_predictions(self, generator, config):
        """Verify row count with a larger set of predictions."""
        n = 500
        predictions = np.random.rand(n)
        test_ids = pd.Series(range(1, n + 1))

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = generator.generate(
                predictions=predictions,
                test_ids=test_ids,
                config=config,
                model_name="large_test",
                output_dir=tmp_dir,
            )

            df = pd.read_csv(file_path)
            assert len(df) == n


class TestGenerateFileNamingPattern:
    """test_generate_file_naming_pattern — verify timestamp pattern in filename."""

    def test_generate_file_naming_pattern(self, generator, config, sample_predictions):
        """Verify filename follows {model_name}_YYYYMMDD_HHmmss.csv pattern."""
        predictions, test_ids = sample_predictions
        model_name = "random_forest"

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = generator.generate(
                predictions=predictions,
                test_ids=test_ids,
                config=config,
                model_name=model_name,
                output_dir=tmp_dir,
            )

            filename = Path(file_path).name
            pattern = re.compile(
                rf"^{re.escape(model_name)}_\d{{8}}_\d{{6}}\.csv$"
            )
            assert pattern.match(filename), (
                f"Filename '{filename}' does not match expected pattern"
            )


class TestGenerateCreatesDirectory:
    """test_generate_creates_directory — verify output_dir created if missing."""

    def test_generate_creates_directory(self, generator, config, sample_predictions):
        """Verify that generate() creates output_dir if it doesn't exist."""
        predictions, test_ids = sample_predictions

        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_dir = str(Path(tmp_dir) / "nested" / "submissions")
            assert not Path(nested_dir).exists()

            file_path = generator.generate(
                predictions=predictions,
                test_ids=test_ids,
                config=config,
                model_name="dir_test",
                output_dir=nested_dir,
            )

            assert Path(nested_dir).exists()
            assert Path(file_path).exists()


class TestGenerateMismatchedLengthsRaises:
    """test_generate_mismatched_lengths_raises — ValueError with expected/actual counts."""

    def test_generate_mismatched_lengths_raises(self, generator, config):
        """Verify ValueError raised when predictions and test_ids have different lengths."""
        predictions = np.array([0, 1, 1])
        test_ids = pd.Series([892, 893, 894, 895, 896])  # 5 IDs vs 3 predictions

        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValueError, match=r"3.*5|5.*3"):
                generator.generate(
                    predictions=predictions,
                    test_ids=test_ids,
                    config=config,
                    model_name="mismatch",
                    output_dir=tmp_dir,
                )


class TestSubmitNoKaggleCliRaises:
    """test_submit_no_kaggle_cli_raises — RuntimeError with remediation message."""

    def test_submit_no_kaggle_cli_raises(self, generator):
        """Verify RuntimeError when kaggle CLI is not installed, with install instructions."""
        with patch("kaggle_ml_toolkit.submission_generator.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="pip install kaggle"):
                generator.submit(
                    file_path="dummy.csv",
                    competition_name="titanic",
                    message="test submission",
                )
