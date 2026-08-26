"""Integration tests for package-level imports.

Validates: Requirements 25.1
"""

import importlib
import re


class TestPackageImports:
    """Tests for kaggle_ml_toolkit package importability."""

    def test_all_public_classes_importable(self):
        """Every class listed in __all__ is importable from kaggle_ml_toolkit."""
        import kaggle_ml_toolkit

        all_exports = kaggle_ml_toolkit.__all__

        for name in all_exports:
            if name == "__version__":
                # __version__ is a string, not a class
                assert hasattr(kaggle_ml_toolkit, name)
                continue

            obj = getattr(kaggle_ml_toolkit, name, None)
            assert obj is not None, f"{name} listed in __all__ but not accessible"

    def test_version_defined(self):
        """__version__ is a string in semver-like format (e.g. '0.1.0')."""
        import kaggle_ml_toolkit

        version = kaggle_ml_toolkit.__version__
        assert isinstance(version, str)
        # Should match a pattern like X.Y.Z
        assert re.match(r"^\d+\.\d+\.\d+", version), (
            f"__version__ '{version}' does not match expected format X.Y.Z"
        )
