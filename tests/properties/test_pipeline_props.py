"""Property-based tests for Pipeline reproducibility and experiment records.

Validates: Requirements 24.1, 24.2
"""

from hypothesis import given, settings, strategies as st

from kaggle_ml_toolkit.config import CompetitionConfig
from kaggle_ml_toolkit.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def pipeline_configs(draw):
    """Generate valid CompetitionConfig instances with varying seeds."""
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    return CompetitionConfig(
        competition_name="test_comp",
        target_column="target",
        id_column="id",
        problem_type="classification",
        primary_metric="accuracy",
        random_seed=seed,
    )


pipeline_steps = st.lists(
    st.sampled_from(["load", "clean", "feature_engineer", "model_select", "evaluate"]),
    min_size=1,
    max_size=5,
)


# ---------------------------------------------------------------------------
# Property 32: Reproducibility via Seed
# ---------------------------------------------------------------------------


@given(config=pipeline_configs(), steps=pipeline_steps)
@settings(max_examples=20)
def test_reproducibility_via_seed(config, steps):
    """Same seed + config → same experiment record structure.

    Running the pipeline twice with identical config and steps produces
    experiment records with matching keys, library_versions keys,
    pipeline_config, and random_seed.

    **Validates: Requirements 24.1**
    """
    pipeline_a = Pipeline(config)
    record_a = pipeline_a.run(steps)

    pipeline_b = Pipeline(config)
    record_b = pipeline_b.run(steps)

    # Same top-level keys
    assert set(record_a.keys()) == set(record_b.keys())

    # Same library version keys reported
    assert set(record_a["library_versions"].keys()) == set(record_b["library_versions"].keys())

    # Same pipeline config
    assert record_a["pipeline_config"] == record_b["pipeline_config"]

    # Same random seed
    assert record_a["random_seed"] == record_b["random_seed"]

    # Same steps executed
    assert record_a["steps_executed"] == record_b["steps_executed"]


# ---------------------------------------------------------------------------
# Property 33: Experiment Record Completeness
# ---------------------------------------------------------------------------


@given(config=pipeline_configs(), steps=pipeline_steps)
@settings(max_examples=20)
def test_experiment_record_completeness(config, steps):
    """Experiment record contains library_versions, pipeline_config, random_seed.

    For any valid config and set of steps, the resulting experiment record
    must contain all required keys for reproducibility.

    **Validates: Requirements 24.2**
    """
    pipeline = Pipeline(config)
    record = pipeline.run(steps)

    # Required top-level keys
    assert "library_versions" in record
    assert "pipeline_config" in record
    assert "random_seed" in record
    assert "steps_executed" in record

    # Library versions must include key packages
    lib_versions = record["library_versions"]
    assert "pandas" in lib_versions
    assert "scikit-learn" in lib_versions
    assert "numpy" in lib_versions
    assert "kaggle_ml_toolkit" in lib_versions

    # Pipeline config must have competition settings
    pc = record["pipeline_config"]
    assert "competition_name" in pc
    assert "target_column" in pc
    assert "problem_type" in pc

    # Random seed must match what was configured
    assert record["random_seed"] == config.random_seed
