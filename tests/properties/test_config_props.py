"""Property-based tests for CompetitionConfig serialization and validation.

Validates: Requirements 18.1, 18.2, 18.5
"""

import tempfile
import os

import pytest
from hypothesis import given, settings, strategies as st

from kaggle_ml_toolkit.config import (
    CompetitionConfig,
    SUPPORTED_PROBLEM_TYPES,
    VALID_METRICS,
)

# Import the shared strategy from conftest
from tests.conftest import valid_competition_configs


# ---------------------------------------------------------------------------
# Property 30: Config Serialization Round-Trip
# Feature: kaggle-ml-toolkit, Property 30: Competition Config Serialization Round-Trip
# ---------------------------------------------------------------------------


@given(config_data=valid_competition_configs())
@settings(max_examples=100)
def test_config_serialization_round_trip(config_data):
    """For any valid CompetitionConfig, serializing to YAML and then
    deserializing produces an equivalent config with all fields preserved.

    **Validates: Requirements 18.5**
    """
    # Create a config from generated data
    original = CompetitionConfig(**config_data)

    # Serialize to a temp file and deserialize back
    with tempfile.TemporaryDirectory() as tmp_dir:
        yaml_path = os.path.join(tmp_dir, "config.yaml")
        original.to_yaml(yaml_path)
        loaded = CompetitionConfig.from_yaml(yaml_path)

    # All fields must match
    assert loaded.competition_name == original.competition_name
    assert loaded.target_column == original.target_column
    assert loaded.id_column == original.id_column
    assert loaded.problem_type == original.problem_type
    assert loaded.primary_metric == original.primary_metric
    assert loaded.random_seed == original.random_seed
    assert loaded.domain_params == original.domain_params


# ---------------------------------------------------------------------------
# Property 31: Config Validation
# Feature: kaggle-ml-toolkit, Property 31: Config Validation
# ---------------------------------------------------------------------------


# Strategy for valid (problem_type, primary_metric) combinations
@st.composite
def valid_problem_metric_pairs(draw):
    """Generate valid (problem_type, metric) pairs from supported set."""
    problem_type = draw(st.sampled_from(SUPPORTED_PROBLEM_TYPES))
    metric = draw(st.sampled_from(VALID_METRICS[problem_type]))
    return problem_type, metric


@given(pair=valid_problem_metric_pairs())
@settings(max_examples=100)
def test_valid_problem_metric_instantiation_succeeds(pair):
    """For any valid (problem_type, primary_metric) combination from the
    supported set, instantiation should succeed without error.

    **Validates: Requirements 18.1, 18.2**
    """
    problem_type, metric = pair

    # Should not raise any exception
    config = CompetitionConfig(
        competition_name="test_competition",
        target_column="target",
        id_column="id",
        problem_type=problem_type,
        primary_metric=metric,
    )

    assert config.problem_type == problem_type
    assert config.primary_metric == metric


# Strategy for unsupported problem types
unsupported_problem_types = st.text(
    min_size=1,
    max_size=30,
    alphabet=st.characters(whitelist_categories=("Ll", "Nd")),
).filter(lambda x: x not in SUPPORTED_PROBLEM_TYPES)


@given(bad_type=unsupported_problem_types)
@settings(max_examples=100)
def test_unsupported_problem_type_raises_value_error(bad_type):
    """For any config with an unsupported problem_type, instantiation
    should raise ValueError.

    **Validates: Requirements 18.2**
    """
    with pytest.raises(ValueError, match="Unsupported problem_type"):
        CompetitionConfig(
            competition_name="test_competition",
            target_column="target",
            id_column="id",
            problem_type=bad_type,
            primary_metric="accuracy",
        )


# Strategy for invalid metrics given a valid problem type
@st.composite
def invalid_metric_for_problem_type(draw):
    """Generate a valid problem_type paired with a metric NOT in its valid set."""
    problem_type = draw(st.sampled_from(SUPPORTED_PROBLEM_TYPES))
    valid_for_type = VALID_METRICS[problem_type]
    # Collect all metrics from OTHER problem types that aren't valid for this one
    all_other_metrics = []
    for other_type, metrics in VALID_METRICS.items():
        if other_type != problem_type:
            for m in metrics:
                if m not in valid_for_type:
                    all_other_metrics.append(m)
    # Also add some completely invalid metric names
    bad_metric = draw(
        st.one_of(
            st.sampled_from(all_other_metrics) if all_other_metrics else st.just("invalid_metric"),
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(whitelist_categories=("Ll",)),
            ).filter(lambda x: x not in valid_for_type),
        )
    )
    return problem_type, bad_metric


@given(pair=invalid_metric_for_problem_type())
@settings(max_examples=100)
def test_invalid_metric_raises_value_error(pair):
    """For any config with a metric not in VALID_METRICS[problem_type],
    instantiation should raise ValueError.

    **Validates: Requirements 18.2**
    """
    problem_type, bad_metric = pair

    with pytest.raises(ValueError, match="Invalid primary_metric"):
        CompetitionConfig(
            competition_name="test_competition",
            target_column="target",
            id_column="id",
            problem_type=problem_type,
            primary_metric=bad_metric,
        )
