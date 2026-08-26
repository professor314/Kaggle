"""Utility functions for Kaggle ML Toolkit.

Provides helper functions for competition directory setup and other
common operations used across the toolkit.
"""

import os

import yaml


def create_competition_directory(
    competition_name: str, base_dir: str = "competitions"
) -> str:
    """Create a new competition directory with standard structure.

    Creates the following subdirectories:
        - notebooks/
        - data/
        - submissions/
        - research/
        - content/

    Also generates a default competition_config.yaml template in the
    competition root directory.

    Args:
        competition_name: Name of the competition. Used as the directory
            name (spaces and special characters preserved).
        base_dir: Parent directory for all competitions. Defaults to
            "competitions".

    Returns:
        Path to the created competition directory.
    """
    competition_dir = os.path.join(base_dir, competition_name)
    os.makedirs(competition_dir, exist_ok=True)

    # Create standard subdirectories
    subdirs = ["notebooks", "data", "submissions", "research", "content"]
    for subdir in subdirs:
        os.makedirs(os.path.join(competition_dir, subdir), exist_ok=True)

    # Write default competition_config.yaml template
    config_path = os.path.join(competition_dir, "competition_config.yaml")
    if not os.path.exists(config_path):
        default_config = {
            "competition_name": competition_name,
            "target_column": "target",
            "id_column": "id",
            "problem_type": "classification",
            "primary_metric": "accuracy",
            "random_seed": 42,
            "domain_params": None,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(default_config, f, default_flow_style=False)

    return competition_dir
