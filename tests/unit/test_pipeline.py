"""Unit tests for Pipeline orchestration.

Validates: Requirements 24.1, 24.2
"""

from kaggle_ml_toolkit.config import CompetitionConfig
from kaggle_ml_toolkit.pipeline import Pipeline


def _make_config(seed: int = 42) -> CompetitionConfig:
    """Create a simple config for testing."""
    return CompetitionConfig(
        competition_name="unit_test_comp",
        target_column="target",
        id_column="id",
        problem_type="classification",
        primary_metric="accuracy",
        random_seed=seed,
    )


class TestPipelineRun:
    """Tests for Pipeline.run() method."""

    def test_run_returns_experiment_record(self):
        """Pipeline.run() returns a dict with expected keys."""
        config = _make_config()
        pipeline = Pipeline(config)
        record = pipeline.run(["load", "clean"])

        expected_keys = {"library_versions", "pipeline_config", "random_seed", "steps_executed"}
        assert expected_keys == set(record.keys())

    def test_library_versions_present(self):
        """Experiment record includes pandas, scikit-learn, numpy, kaggle_ml_toolkit."""
        config = _make_config()
        pipeline = Pipeline(config)
        record = pipeline.run(["load"])

        lib_versions = record["library_versions"]
        assert "pandas" in lib_versions
        assert "scikit-learn" in lib_versions
        assert "numpy" in lib_versions
        assert "kaggle_ml_toolkit" in lib_versions

        # Each should have a version string (not empty)
        for key in ["pandas", "numpy", "kaggle_ml_toolkit"]:
            assert isinstance(lib_versions[key], str)
            assert len(lib_versions[key]) > 0

    def test_random_seed_propagated(self):
        """Experiment record contains the config's random_seed value."""
        seed = 123
        config = _make_config(seed=seed)
        pipeline = Pipeline(config)
        record = pipeline.run(["load", "clean", "model_select"])

        assert record["random_seed"] == seed
