"""Property-based tests for competition directory structure creation.

Validates: Requirements 28.1
"""

import os
import tempfile

from hypothesis import given, settings, strategies as st

from kaggle_ml_toolkit.utils import create_competition_directory


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Competition names: alphanumeric with hyphens and underscores
competition_names = st.text(
    min_size=1,
    max_size=40,
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="-_"),
).filter(lambda x: len(x.strip()) > 0)


# ---------------------------------------------------------------------------
# Property 29: Competition Directory Structure
# ---------------------------------------------------------------------------


@given(name=competition_names)
@settings(max_examples=20)
def test_competition_directory_structure(name):
    """Creating a competition directory produces the expected structure.

    The created directory must contain:
    - notebooks/ subdirectory
    - data/ subdirectory
    - submissions/ subdirectory
    - research/ subdirectory
    - content/ subdirectory
    - competition_config.yaml file

    **Validates: Requirements 28.1**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        result_path = create_competition_directory(name, base_dir=tmp_dir)

        # Result path exists
        assert os.path.isdir(result_path)

        # All required subdirectories exist
        expected_subdirs = ["notebooks", "data", "submissions", "research", "content"]
        for subdir in expected_subdirs:
            subdir_path = os.path.join(result_path, subdir)
            assert os.path.isdir(subdir_path), f"Missing subdirectory: {subdir}"

        # Config file exists
        config_path = os.path.join(result_path, "competition_config.yaml")
        assert os.path.isfile(config_path), "Missing competition_config.yaml"
