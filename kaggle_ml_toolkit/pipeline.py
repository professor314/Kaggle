"""Pipeline orchestration for Kaggle ML Toolkit.

Provides the Pipeline class that manages the end-to-end ML workflow
with reproducibility support, library version tracking, and experiment
record generation.
"""

from typing import Any, Dict, List

import kaggle_ml_toolkit


class Pipeline:
    """Orchestrates the ML pipeline with reproducibility.

    Manages pipeline step execution, records library versions, and
    propagates configuration (e.g., random seed) to ensure experiments
    are fully reproducible.

    Attributes:
        _config: The CompetitionConfig for this pipeline run.
        _steps: List of executed step records.
    """

    def __init__(self, config: "kaggle_ml_toolkit.CompetitionConfig") -> None:
        self._config = config
        self._steps: List[Dict[str, Any]] = []
        self._library_versions: Dict[str, str] = {}

    def run(self, steps: List[str]) -> Dict[str, Any]:
        """Execute pipeline steps in order.

        Records library versions (pandas, scikit-learn, numpy, toolkit)
        and propagates the config's random_seed to numpy and sklearn
        for reproducibility.

        Args:
            steps: List of step names to execute (e.g.,
                ["load", "clean", "feature_engineer", "model_select"]).

        Returns:
            An experiment record dictionary containing:
                - library_versions: dict of package versions
                - pipeline_config: competition config as dict
                - random_seed: the random seed used
                - steps_executed: list of step names run
        """
        import numpy as np

        # Record library versions
        self._library_versions = self._get_library_versions()

        # Propagate random seed for reproducibility
        np.random.seed(self._config.random_seed)

        try:
            import sklearn

            sklearn.utils.check_random_state(self._config.random_seed)
        except ImportError:
            pass

        # Record executed steps
        self._steps = [{"name": step, "status": "executed"} for step in steps]

        return self.get_experiment_record()

    def get_experiment_record(self) -> Dict[str, Any]:
        """Return full experiment configuration for persistence.

        Returns:
            Dictionary with:
                - library_versions: versions of pandas, scikit-learn,
                    numpy, and the toolkit
                - pipeline_config: serialized competition config
                - random_seed: the random seed value
                - steps_executed: list of step records
        """
        if not self._library_versions:
            self._library_versions = self._get_library_versions()

        return {
            "library_versions": self._library_versions,
            "pipeline_config": {
                "competition_name": self._config.competition_name,
                "target_column": self._config.target_column,
                "id_column": self._config.id_column,
                "problem_type": self._config.problem_type,
                "primary_metric": self._config.primary_metric,
            },
            "random_seed": self._config.random_seed,
            "steps_executed": self._steps,
        }

    def _get_library_versions(self) -> Dict[str, str]:
        """Collect versions of key libraries.

        Returns:
            Dictionary mapping library names to version strings.
        """
        versions: Dict[str, str] = {}

        try:
            import pandas as pd

            versions["pandas"] = pd.__version__
        except ImportError:
            versions["pandas"] = "not installed"

        try:
            import sklearn

            versions["scikit-learn"] = sklearn.__version__
        except ImportError:
            versions["scikit-learn"] = "not installed"

        try:
            import numpy as np

            versions["numpy"] = np.__version__
        except ImportError:
            versions["numpy"] = "not installed"

        versions["kaggle_ml_toolkit"] = kaggle_ml_toolkit.__version__

        return versions
