"""Unit tests for CompetitionConfig.

Tests cover validation, default values, YAML serialization/deserialization,
and error handling for invalid inputs.

Validates: Requirements 18.1, 18.2, 18.3, 18.4, 18.5, 18.6
"""

import os
from pathlib import Path

import pytest
import yaml

from kaggle_ml_toolkit.config import CompetitionConfig


# ---------------------------------------------------------------------------
# Valid construction tests
# ---------------------------------------------------------------------------


class TestValidConfigs:
    """Tests for successfully creating CompetitionConfig instances."""

    def test_valid_classification_config(self):
        """Create with valid classification params, verify all fields."""
        config = CompetitionConfig(
            competition_name="titanic",
            target_column="Survived",
            id_column="PassengerId",
            problem_type="classification",
            primary_metric="accuracy",
            random_seed=123,
            domain_params={"max_age": 80},
        )
        assert config.competition_name == "titanic"
        assert config.target_column == "Survived"
        assert config.id_column == "PassengerId"
        assert config.problem_type == "classification"
        assert config.primary_metric == "accuracy"
        assert config.random_seed == 123
        assert config.domain_params == {"max_age": 80}

    def test_valid_regression_config(self):
        """Create with valid regression params."""
        config = CompetitionConfig(
            competition_name="house-prices",
            target_column="SalePrice",
            id_column="Id",
            problem_type="regression",
            primary_metric="rmse",
        )
        assert config.problem_type == "regression"
        assert config.primary_metric == "rmse"

    def test_valid_clustering_config(self):
        """Create with valid clustering params."""
        config = CompetitionConfig(
            competition_name="customer-segmentation",
            target_column="cluster",
            id_column="customer_id",
            problem_type="clustering",
            primary_metric="silhouette_score",
        )
        assert config.problem_type == "clustering"
        assert config.primary_metric == "silhouette_score"


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestValidationErrors:
    """Tests for validation failures raising ValueError."""

    def test_invalid_problem_type_raises(self):
        """Unsupported problem_type raises ValueError listing supported types."""
        with pytest.raises(ValueError, match="Supported types are"):
            CompetitionConfig(
                competition_name="test",
                target_column="target",
                id_column="id",
                problem_type="unsupervised",
                primary_metric="accuracy",
            )

    def test_invalid_metric_for_problem_type_raises(self):
        """Incompatible metric raises ValueError listing valid metrics."""
        with pytest.raises(ValueError, match="Valid metrics are"):
            CompetitionConfig(
                competition_name="test",
                target_column="target",
                id_column="id",
                problem_type="classification",
                primary_metric="rmse",
            )

    def test_competition_name_too_long_raises(self):
        """Name > 128 chars raises ValueError."""
        long_name = "a" * 129
        with pytest.raises(ValueError, match="at most 128 characters"):
            CompetitionConfig(
                competition_name=long_name,
                target_column="target",
                id_column="id",
                problem_type="classification",
                primary_metric="accuracy",
            )


# ---------------------------------------------------------------------------
# Default values tests
# ---------------------------------------------------------------------------


class TestDefaults:
    """Tests for default field values."""

    def test_default_random_seed(self):
        """Defaults to 42 when not specified."""
        config = CompetitionConfig(
            competition_name="test",
            target_column="target",
            id_column="id",
            problem_type="classification",
            primary_metric="accuracy",
        )
        assert config.random_seed == 42

    def test_domain_params_optional(self):
        """None by default, can be set to a dict."""
        config = CompetitionConfig(
            competition_name="test",
            target_column="target",
            id_column="id",
            problem_type="regression",
            primary_metric="mae",
        )
        assert config.domain_params is None

        config_with_params = CompetitionConfig(
            competition_name="test",
            target_column="target",
            id_column="id",
            problem_type="regression",
            primary_metric="mae",
            domain_params={"noise_level": 0.1, "valid_range": [0, 100]},
        )
        assert config_with_params.domain_params == {
            "noise_level": 0.1,
            "valid_range": [0, 100],
        }


# ---------------------------------------------------------------------------
# YAML serialization tests
# ---------------------------------------------------------------------------


class TestYamlSerialization:
    """Tests for to_yaml and from_yaml methods."""

    def test_to_yaml_creates_file(self, tmp_dir):
        """to_yaml creates a readable YAML file."""
        config = CompetitionConfig(
            competition_name="titanic",
            target_column="Survived",
            id_column="PassengerId",
            problem_type="classification",
            primary_metric="f1",
            random_seed=7,
            domain_params={"class_weights": [1, 2]},
        )
        yaml_path = os.path.join(tmp_dir, "config.yaml")
        config.to_yaml(yaml_path)

        assert Path(yaml_path).exists()

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["competition_name"] == "titanic"
        assert data["target_column"] == "Survived"
        assert data["random_seed"] == 7
        assert data["domain_params"] == {"class_weights": [1, 2]}

    def test_from_yaml_loads_correctly(self, tmp_dir):
        """from_yaml loads all fields including domain_params."""
        yaml_path = os.path.join(tmp_dir, "config.yaml")
        data = {
            "competition_name": "house-prices",
            "target_column": "SalePrice",
            "id_column": "Id",
            "problem_type": "regression",
            "primary_metric": "r_squared",
            "random_seed": 99,
            "domain_params": {"outlier_threshold": 3.5},
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f)

        config = CompetitionConfig.from_yaml(yaml_path)

        assert config.competition_name == "house-prices"
        assert config.target_column == "SalePrice"
        assert config.id_column == "Id"
        assert config.problem_type == "regression"
        assert config.primary_metric == "r_squared"
        assert config.random_seed == 99
        assert config.domain_params == {"outlier_threshold": 3.5}

    def test_from_yaml_missing_file_raises(self, tmp_dir):
        """FileNotFoundError for non-existent path."""
        fake_path = os.path.join(tmp_dir, "nonexistent.yaml")
        with pytest.raises(FileNotFoundError, match="not found"):
            CompetitionConfig.from_yaml(fake_path)

    def test_from_yaml_missing_fields_raises(self, tmp_dir):
        """ValueError listing missing fields."""
        yaml_path = os.path.join(tmp_dir, "incomplete.yaml")
        data = {
            "competition_name": "test",
            # missing target_column, id_column, problem_type, primary_metric
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f)

        with pytest.raises(ValueError, match="Missing required fields"):
            CompetitionConfig.from_yaml(yaml_path)

    def test_from_yaml_invalid_yaml_raises(self, tmp_dir):
        """ValueError for unparseable YAML content."""
        yaml_path = os.path.join(tmp_dir, "bad.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(":\n  :\n    - [invalid\n  {{broken")

        with pytest.raises(ValueError, match="Failed to parse YAML"):
            CompetitionConfig.from_yaml(yaml_path)
