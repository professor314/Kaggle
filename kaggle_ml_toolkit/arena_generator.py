"""Arena Generator — Automatically produces pipeline configurations for the ModelArena.

Generates systematic combinations of models, hyperparameters, and feature subsets,
calibrated to fit within a time budget. Estimates per-pipeline cost using a quick
timing probe, then determines how many pipelines can run within the budget.

Usage:
    from kaggle_ml_toolkit import ModelArena
    from kaggle_ml_toolkit.arena_generator import ArenaGenerator
    
    gen = ArenaGenerator(
        X=train_df, y=y,
        problem_type="classification",
        feature_sets={"core": core_cols, "extended": ext_cols},
        time_budget_minutes=10,
    )
    
    arena = gen.build_arena()  # Returns a populated ModelArena ready to run
    results = arena.run()
"""

from typing import Dict, List, Any, Optional
import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from itertools import product

from kaggle_ml_toolkit.model_arena import ModelArena


# Default parameter grids (conservative → moderate → aggressive)
CLASSIFICATION_CONFIGS = {
    "LogisticRegression": [
        {"C": 0.01, "max_iter": 2000, "solver": "lbfgs"},
        {"C": 0.1, "max_iter": 2000, "solver": "lbfgs"},
        {"C": 1.0, "max_iter": 2000, "solver": "lbfgs"},
        {"C": 10.0, "max_iter": 2000, "solver": "lbfgs"},
    ],
    "RandomForest": [
        {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 15},
        {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 10},
        {"n_estimators": 150, "max_depth": 5, "min_samples_leaf": 5},
        {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 5},
        {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 3},
    ],
    "GradientBoosting": [
        {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.15, "min_samples_leaf": 20, "subsample": 0.8},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "min_samples_leaf": 10, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "min_samples_leaf": 8, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "min_samples_leaf": 5, "subsample": 0.8},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "min_samples_leaf": 5, "subsample": 0.8},
    ],
    "DecisionTree": [
        {"max_depth": 2, "min_samples_leaf": 20},
        {"max_depth": 3, "min_samples_leaf": 10},
        {"max_depth": 5, "min_samples_leaf": 5},
    ],
    "KNN": [
        {"n_neighbors": 3},
        {"n_neighbors": 5},
        {"n_neighbors": 9},
        {"n_neighbors": 15},
    ],
}

REGRESSION_CONFIGS = {
    "Ridge": [
        {"alpha": 0.1},
        {"alpha": 1.0},
        {"alpha": 10.0},
        {"alpha": 50.0},
    ],
    "Lasso": [
        {"alpha": 0.0001, "max_iter": 5000},
        {"alpha": 0.001, "max_iter": 5000},
        {"alpha": 0.01, "max_iter": 5000},
    ],
    "ElasticNet": [
        {"alpha": 0.001, "l1_ratio": 0.3, "max_iter": 5000},
        {"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 5000},
        {"alpha": 0.001, "l1_ratio": 0.7, "max_iter": 5000},
    ],
    "RandomForest": [
        {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 15},
        {"n_estimators": 100, "max_depth": 6, "min_samples_leaf": 10},
        {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 5},
        {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 3},
    ],
    "GradientBoosting": [
        {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.15, "min_samples_leaf": 20, "subsample": 0.8},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "min_samples_leaf": 10, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "min_samples_leaf": 8, "subsample": 0.8},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "min_samples_leaf": 5, "subsample": 0.8},
    ],
}


def _make_model(model_type: str, params: Dict[str, Any], problem_type: str, random_state: int = 42) -> BaseEstimator:
    """Instantiate a model from type name and parameters."""
    params = {**params, "random_state": random_state}
    
    if problem_type == "classification":
        constructors = {
            "LogisticRegression": LogisticRegression,
            "RandomForest": RandomForestClassifier,
            "GradientBoosting": GradientBoostingClassifier,
            "DecisionTree": DecisionTreeClassifier,
            "KNN": KNeighborsClassifier,
        }
    else:
        constructors = {
            "Ridge": Ridge,
            "Lasso": Lasso,
            "ElasticNet": ElasticNet,
            "RandomForest": RandomForestRegressor,
            "GradientBoosting": GradientBoostingRegressor,
        }
    
    cls = constructors[model_type]
    # Filter params to only those the constructor accepts
    import inspect
    valid_params = inspect.signature(cls).parameters
    filtered = {k: v for k, v in params.items() if k in valid_params}
    return cls(**filtered)


class ArenaGenerator:
    """Automatically generates pipeline configurations for the ModelArena.
    
    Produces combinations of model types × parameter configs × feature sets,
    estimates timing per pipeline, and caps the total to fit within a time budget.
    
    Args:
        X: Training feature DataFrame.
        y: Target Series.
        problem_type: "classification" or "regression".
        feature_sets: Dict mapping set names to lists of column names.
            Example: {"core": [...], "extended": [...]}
        time_budget_minutes: Maximum arena run time in minutes. Default 10.
        metric: Scoring metric for the arena.
        random_state: Seed for reproducibility.
        prior_cv_lb_gap: Optional CV-LB gap estimate to pass to the arena.
        model_types: Optional list of model type names to include.
            If None, uses all defaults for the problem_type.
    """
    
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str = "classification",
        feature_sets: Optional[Dict[str, List[str]]] = None,
        time_budget_minutes: float = 10.0,
        metric: Optional[str] = None,
        random_state: int = 42,
        prior_cv_lb_gap: Optional[float] = None,
        model_types: Optional[List[str]] = None,
    ):
        self.X = X
        self.y = y
        self.problem_type = problem_type
        self.feature_sets = feature_sets or {"all": list(X.columns)}
        self.time_budget_minutes = time_budget_minutes
        self.metric = metric or ("accuracy" if problem_type == "classification" else "rmse")
        self.random_state = random_state
        self.prior_cv_lb_gap = prior_cv_lb_gap
        
        # Determine which configs to use
        if problem_type == "classification":
            all_configs = CLASSIFICATION_CONFIGS
        else:
            all_configs = REGRESSION_CONFIGS
        
        if model_types:
            self._configs = {k: v for k, v in all_configs.items() if k in model_types}
        else:
            self._configs = all_configs
    
    def estimate_pipeline_time(self) -> float:
        """Run a quick timing probe to estimate seconds per pipeline.
        
        Trains the fastest model (LogisticRegression or Ridge) on a subset
        and extrapolates. Returns estimated seconds per pipeline for 5-fold CV.
        
        Returns:
            Estimated seconds per pipeline evaluation.
        """
        # Use a small sample for timing
        sample_size = min(500, len(self.X))
        X_sample = self.X.iloc[:sample_size]
        y_sample = self.y.iloc[:sample_size]
        
        # Pick first feature set
        first_features = list(self.feature_sets.values())[0]
        X_feat = X_sample[first_features]
        
        # Time a quick model
        if self.problem_type == "classification":
            probe_model = LogisticRegression(max_iter=200, random_state=42)
        else:
            probe_model = Ridge(alpha=1.0)
        
        start = time.time()
        try:
            cross_val_score(probe_model, X_feat, y_sample, cv=3, scoring="accuracy" if self.problem_type == "classification" else "r2")
        except Exception:
            pass
        probe_time = time.time() - start
        
        # Scale up: full dataset is larger, 5 folds instead of 3, heavier models ~3-5x
        scale_factor = (len(self.X) / sample_size) * (5 / 3) * 3  # 3x for heavier models (conservative)
        estimated = probe_time * scale_factor
        
        return max(estimated, 1.0)  # minimum 1.0s per pipeline
    
    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate all pipeline configurations (model × params × features).
        
        Returns:
            List of dicts with keys: name, model_type, params, feature_set_name, feature_cols.
        """
        configs = []
        
        for model_type, param_list in self._configs.items():
            for i, params in enumerate(param_list):
                for feat_name, feat_cols in self.feature_sets.items():
                    name = f"{model_type}_{feat_name}_v{i+1}"
                    configs.append({
                        "name": name,
                        "model_type": model_type,
                        "params": params,
                        "feature_set_name": feat_name,
                        "feature_cols": feat_cols,
                    })
        
        return configs
    
    def build_arena(self, verbose: bool = True) -> ModelArena:
        """Generate configs, cap to time budget, and populate a ModelArena.
        
        Estimates timing, determines how many pipelines fit within the budget,
        prioritizes diverse model types, and returns a ready-to-run arena.
        
        Args:
            verbose: Print generation statistics.
            
        Returns:
            A populated ModelArena ready to call .run() on.
        """
        all_configs = self.generate_configs()
        total_possible = len(all_configs)
        
        # Estimate time per pipeline
        est_time = self.estimate_pipeline_time()
        budget_seconds = self.time_budget_minutes * 60
        max_pipelines = max(5, int(budget_seconds / est_time))
        
        if verbose:
            print(f"ArenaGenerator: {total_possible} possible configs")
            print(f"  Estimated {est_time:.1f}s per pipeline")
            print(f"  Budget: {self.time_budget_minutes} min = {budget_seconds:.0f}s")
            print(f"  Max pipelines that fit: {max_pipelines}")
        
        # If we can fit all, use all
        if max_pipelines >= total_possible:
            selected = all_configs
        else:
            # Prioritize: ensure diversity across model types and feature sets
            selected = self._select_diverse(all_configs, max_pipelines)
        
        if verbose:
            print(f"  Selected: {len(selected)} pipelines")
            # Count by model type
            type_counts = {}
            for c in selected:
                type_counts[c["model_type"]] = type_counts.get(c["model_type"], 0) + 1
            print(f"  Model types: {type_counts}")
            print()
        
        # Build the arena
        arena = ModelArena(
            X=self.X,
            y=self.y,
            cv_folds=5,
            metric=self.metric,
            random_state=self.random_state,
            prior_cv_lb_gap=self.prior_cv_lb_gap,
        )
        
        for config in selected:
            model = _make_model(
                config["model_type"], config["params"],
                self.problem_type, self.random_state
            )
            arena.add(
                name=config["name"],
                model=model,
                feature_cols=config["feature_cols"],
                description=f"{config['model_type']} with {config['feature_set_name']} features",
            )
        
        return arena
    
    def _select_diverse(self, configs: List[Dict], max_n: int) -> List[Dict]:
        """Select a diverse subset of configs that fits the budget.
        
        Ensures representation from each model type and feature set,
        then fills remaining slots with variety.
        
        Args:
            configs: All possible configurations.
            max_n: Maximum number to select.
            
        Returns:
            Selected subset of configurations.
        """
        # Group by model_type
        by_type: Dict[str, List[Dict]] = {}
        for c in configs:
            by_type.setdefault(c["model_type"], []).append(c)
        
        n_types = len(by_type)
        per_type = max(2, max_n // n_types)  # at least 2 per type
        
        selected = []
        
        for model_type, type_configs in by_type.items():
            # Take evenly spaced configs (conservative to aggressive spread)
            n_to_take = min(per_type, len(type_configs))
            if len(type_configs) <= n_to_take:
                selected.extend(type_configs)
            else:
                indices = np.linspace(0, len(type_configs) - 1, n_to_take, dtype=int)
                for idx in indices:
                    selected.append(type_configs[idx])
        
        # If still under budget, add more from underrepresented types
        remaining_budget = max_n - len(selected)
        if remaining_budget > 0:
            unused = [c for c in configs if c not in selected]
            np.random.seed(self.random_state)
            np.random.shuffle(unused)
            selected.extend(unused[:remaining_budget])
        
        return selected[:max_n]
