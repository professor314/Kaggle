"""Kaggle ML Toolkit - AI-assisted research and learning platform.

An importable Python package for structured machine learning competition
workflows, combining domain research, EDA, modeling, and educational
content generation.
"""

__version__ = "0.1.0"

from kaggle_ml_toolkit.config import (
    CompetitionConfig,
    SUPPORTED_PROBLEM_TYPES,
    VALID_METRICS,
)
from kaggle_ml_toolkit.loader import DataBundle, load_csv, load_competition_data
from kaggle_ml_toolkit.cleaner import DataCleaner
from kaggle_ml_toolkit.feature_engineer import FeatureEngineer
from kaggle_ml_toolkit.feature_selector import FeatureSelector
from kaggle_ml_toolkit.eda_engine import EDAEngine
from kaggle_ml_toolkit.model_selector import ModelSelector
from kaggle_ml_toolkit.model_arena import ModelArena, ArenaResult
from kaggle_ml_toolkit.arena_generator import ArenaGenerator
from kaggle_ml_toolkit.model_optimizer import ModelOptimizer
from kaggle_ml_toolkit.cross_validator import CrossValidator
from kaggle_ml_toolkit.ensemble_builder import EnsembleBuilder
from kaggle_ml_toolkit.evaluator import Evaluator
from kaggle_ml_toolkit.interpreter import Interpreter
from kaggle_ml_toolkit.submission_generator import SubmissionGenerator
from kaggle_ml_toolkit.content_generator import ContentGenerator
from kaggle_ml_toolkit.augmenter import Augmenter
from kaggle_ml_toolkit.research import ResearchDocumentGenerator
from kaggle_ml_toolkit.pipeline import Pipeline


def __getattr__(name):
    """Lazily expose deep-learning classes.

    The deep learning module depends on torch/transformers, which only exist in
    the dedicated `.venv-dl` (Python 3.12) environment. Importing it lazily keeps
    `import kaggle_ml_toolkit` working on the main Python 3.14 tabular env where
    torch is intentionally absent.
    """
    if name in ("TransformerClassifier", "DeepLearningTrainer", "CNNClassifier",
                "get_device", "gpu_info"):
        from kaggle_ml_toolkit import deep_learning

        return getattr(deep_learning, name)
    if name in ("seed_everything", "autocast_context"):
        from kaggle_ml_toolkit import gpu_utils

        return getattr(gpu_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    # Config
    "CompetitionConfig",
    "VALID_METRICS",
    "SUPPORTED_PROBLEM_TYPES",
    # Data loading
    "DataBundle",
    "load_csv",
    "load_competition_data",
    # Data cleaning
    "DataCleaner",
    # Feature engineering
    "FeatureEngineer",
    # Feature selection
    "FeatureSelector",
    # EDA
    "EDAEngine",
    # Modeling
    "ModelSelector",
    "ModelArena",
    "ArenaResult",
    "ArenaGenerator",
    "ModelOptimizer",
    "CrossValidator",
    "EnsembleBuilder",
    # Evaluation
    "Evaluator",
    # Interpretability
    "Interpreter",
    # Submission
    "SubmissionGenerator",
    # Augmentation
    "Augmenter",
    # Content
    "ContentGenerator",
    # Research
    "ResearchDocumentGenerator",
    # Pipeline
    "Pipeline",
    # Deep learning (lazy — requires the .venv-dl environment)
    "TransformerClassifier",
    "DeepLearningTrainer",
    "CNNClassifier",
    "get_device",
    "gpu_info",
    "seed_everything",
    "autocast_context",
]
