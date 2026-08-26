"""Submission generator for the Kaggle ML Toolkit.

Produces Kaggle-format submission CSV files and interfaces with the
Kaggle CLI to submit entries programmatically.

Validates: Requirements 15.3, 15.4, 15.5, 15.6, 15.7, 15.8
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from kaggle_ml_toolkit.config import CompetitionConfig


class SubmissionGenerator:
    """Kaggle submission file generation and CLI integration.

    Generates properly formatted submission CSV files with exactly two
    columns (ID and target), and provides an interface to submit files
    to Kaggle competitions via the CLI.
    """

    def generate(
        self,
        predictions: np.ndarray,
        test_ids: pd.Series,
        config: "CompetitionConfig",
        model_name: str = "model",
        output_dir: str = "./submissions",
    ) -> str:
        """Generate a submission CSV file.

        Creates a CSV with two columns: the ID column (from test_ids) and
        the target column (from predictions), using column names defined
        in the competition config.

        Args:
            predictions: Array of predicted values for the test set.
            test_ids: Series of test sample identifiers.
            config: Competition configuration with column name definitions.
            model_name: Name prefix for the submission file.
            output_dir: Directory to save the submission file.

        Returns:
            Full file path to the generated CSV as a string.

        Raises:
            ValueError: If len(predictions) != len(test_ids), with a
                message including both the expected and actual counts.
        """
        if len(predictions) != len(test_ids):
            raise ValueError(
                f"Length mismatch: predictions has {len(predictions)} entries "
                f"but test_ids has {len(test_ids)} entries. "
                f"Expected {len(test_ids)} predictions."
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.csv"
        file_path = output_path / filename

        submission_df = pd.DataFrame(
            {
                config.id_column: test_ids.values,
                config.target_column: predictions,
            }
        )

        submission_df.to_csv(file_path, header=True, index=False)

        return str(file_path)

    def submit(
        self,
        file_path: str,
        competition_name: str,
        message: str = "Automated submission",
    ) -> Dict[str, Any]:
        """Submit a file to a Kaggle competition via the CLI.

        Executes the ``kaggle competitions submit`` command as a subprocess.

        Args:
            file_path: Path to the submission CSV file.
            competition_name: Kaggle competition slug/name.
            message: Submission message describing the entry.

        Returns:
            Dictionary with keys: status, competition, file, message.

        Raises:
            RuntimeError: If the Kaggle CLI is not installed, with
                remediation instructions for installation and configuration.
            RuntimeError: If the subprocess command fails, with the
                error output from the CLI.
        """
        if shutil.which("kaggle") is None:
            raise RuntimeError(
                "Kaggle CLI not found. "
                "Install with: pip install kaggle. "
                "Configure API key at ~/.kaggle/kaggle.json"
            )

        cmd = [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            competition_name,
            "-f",
            file_path,
            "-m",
            message,
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            error_output = exc.stderr or exc.stdout or str(exc)
            raise RuntimeError(
                f"Kaggle submission failed: {error_output}"
            ) from exc

        return {
            "status": "submitted",
            "competition": competition_name,
            "file": file_path,
            "message": message,
        }
