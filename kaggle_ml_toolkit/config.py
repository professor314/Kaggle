"""Competition configuration for the Kaggle ML Toolkit.

Provides the CompetitionConfig dataclass that defines competition-specific
settings such as target column, problem type, and evaluation metric.

Validates: Requirements 18.1, 18.2, 18.3, 18.4
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_PROBLEM_TYPES: List[str] = ["classification", "regression", "clustering"]

VALID_METRICS: Dict[str, List[str]] = {
    "classification": ["accuracy", "precision", "recall", "f1", "auc_roc"],
    "regression": ["rmse", "mae", "r_squared"],
    "clustering": ["silhouette_score"],
}


# ---------------------------------------------------------------------------
# CompetitionConfig
# ---------------------------------------------------------------------------


@dataclass
class CompetitionConfig:
    """Competition-specific settings.

    Encapsulates all configuration needed to adapt the toolkit to a
    particular Kaggle competition, including target column, ID column,
    problem type, and evaluation metric.

    Attributes:
        competition_name: Name of the Kaggle competition (max 128 characters).
        target_column: Name of the target/label column in the dataset.
        id_column: Name of the ID column used for submission files.
        problem_type: One of 'classification', 'regression', or 'clustering'.
        primary_metric: Evaluation metric valid for the specified problem_type.
        random_seed: Random seed for reproducibility. Defaults to 42.
        domain_params: Optional dictionary of domain-specific parameters
            informed by research (e.g., valid value ranges, noise levels).
    """

    competition_name: str
    target_column: str
    id_column: str
    problem_type: str
    primary_metric: str
    random_seed: int = 42
    domain_params: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self) -> None:
        """Validate configuration immediately after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate all configuration fields.

        Checks that:
            - competition_name is at most 128 characters.
            - problem_type is one of the supported types.
            - primary_metric is valid for the given problem_type.

        Raises:
            ValueError: If competition_name exceeds 128 characters.
            ValueError: If problem_type is not supported, listing valid types.
            ValueError: If primary_metric is invalid for problem_type,
                listing valid metrics.
        """
        if len(self.competition_name) > 128:
            raise ValueError(
                f"competition_name must be at most 128 characters, "
                f"got {len(self.competition_name)}."
            )

        if self.problem_type not in SUPPORTED_PROBLEM_TYPES:
            raise ValueError(
                f"Unsupported problem_type '{self.problem_type}'. "
                f"Supported types are: {SUPPORTED_PROBLEM_TYPES}."
            )

        valid = VALID_METRICS[self.problem_type]
        if self.primary_metric not in valid:
            raise ValueError(
                f"Invalid primary_metric '{self.primary_metric}' "
                f"for problem_type '{self.problem_type}'. "
                f"Valid metrics are: {valid}."
            )

    def to_yaml(self, path: str) -> None:
        """Serialize the configuration to a YAML file.

        Creates parent directories if they don't exist.

        Args:
            path: File path where the YAML will be written.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data: Dict[str, Any] = {
            "competition_name": self.competition_name,
            "target_column": self.target_column,
            "id_column": self.id_column,
            "problem_type": self.problem_type,
            "primary_metric": self.primary_metric,
            "random_seed": self.random_seed,
            "domain_params": self.domain_params,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> "CompetitionConfig":
        """Load a CompetitionConfig from a YAML file.

        Args:
            path: File path to the YAML configuration file.

        Returns:
            A new CompetitionConfig instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the YAML is unparseable or required fields
                are missing.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse YAML file: {exc}") from exc

        if data is None:
            data = {}

        required_fields = [
            "competition_name",
            "target_column",
            "id_column",
            "problem_type",
            "primary_metric",
        ]

        missing = [field for field in required_fields if field not in data]
        if missing:
            raise ValueError(
                f"Missing required fields in configuration: {missing}"
            )

        return cls(
            competition_name=data["competition_name"],
            target_column=data["target_column"],
            id_column=data["id_column"],
            problem_type=data["problem_type"],
            primary_metric=data["primary_metric"],
            random_seed=data.get("random_seed", 42),
            domain_params=data.get("domain_params", None),
        )
