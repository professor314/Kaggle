"""Model Arena — Local validation loop for comparing full ML pipelines.

Runs multiple model configurations through identical CV folds for fair
comparison, ranks them by score and stability, and recommends which to
submit. Saves all experiments to the Evaluator for historical tracking.

Usage:
    arena = ModelArena(X, y, cv_folds=5, metric="accuracy", random_state=42)
    
    arena.add("RF_simple", RandomForestClassifier(max_depth=3), feature_cols=["Pclass", "IsFemale", "Age"])
    arena.add("GBM_full", GradientBoostingClassifier(max_depth=5), feature_cols=all_features)
    arena.add("LR_minimal", LogisticRegression(), feature_cols=["IsFemale", "Pclass"])
    
    results = arena.run()
    print(arena.leaderboard())
    print(arena.recommend())
    
    # Submit the winner
    arena.generate_submission(test_X, test_ids, config)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
import warnings
import time


@dataclass
class PipelineEntry:
    """A single pipeline configuration to compete in the arena."""
    name: str
    model: BaseEstimator
    feature_cols: List[str]
    description: str = ""


@dataclass 
class ArenaResult:
    """Results for a single pipeline run."""
    name: str
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    per_fold_scores: List[float]
    train_time_seconds: float
    feature_count: int
    feature_cols: List[str]
    model_type: str
    model_params: Dict[str, Any]
    # Derived metrics
    stability_score: float  # 1 - (std/mean), higher = more stable
    submit_worthiness: float  # mean_score * stability_score, combines accuracy + consistency


class ModelArena:
    """Local validation arena for comparing complete ML pipelines.
    
    All pipelines are evaluated on the exact same CV folds for fair
    comparison. Results are ranked by a "submit-worthiness" score that
    balances accuracy with consistency (low variance).
    
    Args:
        X: Full training feature DataFrame.
        y: Target Series.
        cv_folds: Number of CV folds (default 5).
        metric: Scoring metric name (default "accuracy").
        random_state: Seed for reproducible fold splits.
        prior_cv_lb_gap: Optional estimated CV-LB gap from prior submissions.
            Used to discount CV scores when estimating real LB performance.
    """
    
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        metric: str = "accuracy",
        random_state: int = 42,
        prior_cv_lb_gap: Optional[float] = None,
    ):
        self.X = X
        self.y = y
        self.cv_folds = cv_folds
        self.metric = metric
        self.random_state = random_state
        self.prior_cv_lb_gap = prior_cv_lb_gap
        
        self._entries: List[PipelineEntry] = []
        self._results: List[ArenaResult] = []
        
        # Use StratifiedKFold for classification, KFold for regression
        neg_metrics = {"rmse", "mae", "r_squared", "neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"}
        if metric in neg_metrics or (hasattr(y, 'dtype') and y.dtype.kind == 'f'):
            self._cv_splitter = KFold(
                n_splits=cv_folds, shuffle=True, random_state=random_state
            )
        else:
            self._cv_splitter = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=random_state
            )
        # Pre-compute fold indices for consistency
        self._fold_indices = list(self._cv_splitter.split(X, y))
    
    def add(
        self,
        name: str,
        model: BaseEstimator,
        feature_cols: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        """Add a pipeline configuration to the arena.
        
        Args:
            name: Unique identifier for this pipeline.
            model: Configured sklearn estimator (with desired hyperparams).
            feature_cols: Which columns from X to use. None = all columns.
            description: Optional human-readable description.
        """
        if feature_cols is None:
            feature_cols = list(self.X.columns)
        
        self._entries.append(PipelineEntry(
            name=name,
            model=model,
            feature_cols=feature_cols,
            description=description,
        ))
    
    def run(self, verbose: bool = True, timeout_minutes: Optional[float] = None) -> List[ArenaResult]:
        """Run all pipelines through the arena.
        
        Evaluates each pipeline on the pre-computed CV folds and stores results.
        
        Args:
            verbose: If True, print progress during evaluation.
            timeout_minutes: Optional hard wall-clock timeout. If set, stops
                evaluating new pipelines after this many minutes have elapsed.
            
        Returns:
            List of ArenaResult objects, sorted by submit_worthiness descending.
        """
        self._results = []
        run_start_time = time.time()
        
        # Map user-friendly metric to sklearn scorer
        metric_map = {
            "accuracy": "accuracy",
            "f1": "f1_weighted",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "auc_roc": "roc_auc",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r_squared": "r2",
        }
        sklearn_scorer = metric_map.get(self.metric, self.metric)
        is_neg_metric = sklearn_scorer.startswith("neg_")
        
        if verbose:
            print(f"{'='*60}")
            print(f"MODEL ARENA — {len(self._entries)} pipelines, {self.cv_folds}-fold CV, metric={self.metric}")
            print(f"{'='*60}\n")
        
        for i, entry in enumerate(self._entries):
            # Check wall-clock timeout
            if timeout_minutes and (time.time() - run_start_time) > timeout_minutes * 60:
                if verbose:
                    print(f"\n⏱️  TIMEOUT: {timeout_minutes} minutes elapsed. Stopping with {i}/{len(self._entries)} pipelines evaluated.")
                break
            
            if verbose:
                print(f"[{i+1}/{len(self._entries)}] Running: {entry.name} ({len(entry.feature_cols)} features)...", end=" ")
            
            X_subset = self.X[entry.feature_cols]
            
            start_time = time.time()
            fold_scores = []
            
            for train_idx, val_idx in self._fold_indices:
                X_train, X_val = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
                y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
                
                model_clone = clone(entry.model)
                try:
                    model_clone.fit(X_train, y_train)
                    
                    if sklearn_scorer == "accuracy":
                        score = model_clone.score(X_val, y_val)
                    else:
                        from sklearn.metrics import get_scorer
                        scorer = get_scorer(sklearn_scorer)
                        score = scorer(model_clone, X_val, y_val)
                        if is_neg_metric:
                            score = -score
                    
                    fold_scores.append(score)
                except Exception as e:
                    if verbose:
                        print(f"FAILED ({e})")
                    fold_scores.append(0.0)
            
            elapsed = time.time() - start_time
            scores_arr = np.array(fold_scores)
            mean_score = scores_arr.mean()
            std_score = scores_arr.std()
            
            # Stability: 1 - coefficient of variation (higher = more stable)
            stability = 1 - (std_score / max(mean_score, 1e-10))
            stability = max(0.0, min(1.0, stability))
            
            # Submit-worthiness: balances score + stability
            # For lower-is-better metrics (RMSE, MAE), invert so higher worthiness = better
            estimated_lb = mean_score - (self.prior_cv_lb_gap or 0.0)
            if is_neg_metric or self.metric in ("rmse", "mae"):
                # Lower score is better — worthiness = (1/score) * stability
                submit_worthiness = (1.0 / max(estimated_lb, 1e-10)) * stability
            else:
                submit_worthiness = estimated_lb * stability
            
            result = ArenaResult(
                name=entry.name,
                mean_score=mean_score,
                std_score=std_score,
                min_score=scores_arr.min(),
                max_score=scores_arr.max(),
                per_fold_scores=fold_scores,
                train_time_seconds=elapsed,
                feature_count=len(entry.feature_cols),
                feature_cols=entry.feature_cols,
                model_type=type(entry.model).__name__,
                model_params=entry.model.get_params(),
                stability_score=stability,
                submit_worthiness=submit_worthiness,
            )
            
            self._results.append(result)
            
            if verbose:
                print(f"{mean_score:.4f} ± {std_score:.4f} (worthiness: {submit_worthiness:.4f}, {elapsed:.1f}s)")
        
        # Sort by submit_worthiness descending
        self._results.sort(key=lambda r: r.submit_worthiness, reverse=True)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"WINNER: {self._results[0].name} (worthiness: {self._results[0].submit_worthiness:.4f})")
            print(f"{'='*60}")
        
        return self._results
    
    def leaderboard(self) -> pd.DataFrame:
        """Return a DataFrame leaderboard of all results, ranked by submit_worthiness.
        
        Returns:
            DataFrame with columns: rank, name, mean_score, std_score, 
            stability, submit_worthiness, features, model_type, time_s
        """
        if not self._results:
            raise ValueError("No results yet. Call run() first.")
        
        rows = []
        for rank, r in enumerate(self._results, 1):
            rows.append({
                "rank": rank,
                "name": r.name,
                "mean_score": round(r.mean_score, 4),
                "std_score": round(r.std_score, 4),
                "stability": round(r.stability_score, 4),
                "submit_worthiness": round(r.submit_worthiness, 4),
                "features": r.feature_count,
                "model_type": r.model_type,
                "time_s": round(r.train_time_seconds, 1),
            })
        
        return pd.DataFrame(rows)
    
    def recommend(self) -> Dict[str, Any]:
        """Return a recommendation for which pipeline to submit.
        
        Returns:
            Dict with: recommended_name, reasoning, estimated_lb_score,
            warnings (list of concerns).
        """
        if not self._results:
            raise ValueError("No results yet. Call run() first.")
        
        best = self._results[0]
        warnings_list = []
        
        # Check for high variance
        if best.std_score > 0.03:
            warnings_list.append(
                f"High CV variance ({best.std_score:.4f}). Model may be unstable."
            )
        
        # Check for potential overfitting (many features relative to simple models)
        if best.feature_count > 15:
            warnings_list.append(
                f"Many features ({best.feature_count}). Risk of overfitting on small datasets."
            )
        
        # Estimate LB score
        gap = self.prior_cv_lb_gap or 0.05  # default 5% gap estimate
        estimated_lb = best.mean_score - gap
        
        # Check if estimated LB barely beats baseline
        if estimated_lb < 0.78:
            warnings_list.append(
                f"Estimated LB ({estimated_lb:.4f}) may be near or below gender baseline."
            )
        
        return {
            "recommended_name": best.name,
            "cv_score": best.mean_score,
            "cv_std": best.std_score,
            "estimated_lb_score": estimated_lb,
            "stability_score": best.stability_score,
            "submit_worthiness": best.submit_worthiness,
            "feature_count": best.feature_count,
            "model_type": best.model_type,
            "warnings": warnings_list,
            "reasoning": (
                f"'{best.name}' has the highest submit-worthiness score "
                f"({best.submit_worthiness:.4f}), combining strong CV performance "
                f"({best.mean_score:.4f}) with stability ({best.stability_score:.4f}). "
                f"Estimated LB score: {estimated_lb:.4f}."
            ),
        }
    
    def get_best_model(self) -> Tuple[BaseEstimator, List[str]]:
        """Return the best pipeline's model (fitted on full training data) and its feature columns.
        
        Returns:
            Tuple of (fitted model, feature_cols list).
        """
        if not self._results:
            raise ValueError("No results yet. Call run() first.")
        
        best_name = self._results[0].name
        best_entry = next(e for e in self._entries if e.name == best_name)
        
        # Fit on full training data
        X_subset = self.X[best_entry.feature_cols]
        model = clone(best_entry.model)
        model.fit(X_subset, self.y)
        
        return model, best_entry.feature_cols
    
    def generate_submission(
        self,
        X_test: pd.DataFrame,
        test_ids: pd.Series,
        config,
        output_dir: str = "./submissions",
    ) -> str:
        """Generate a submission file using the best pipeline.
        
        Fits the best model on full training data and predicts on test.
        
        Args:
            X_test: Test feature DataFrame (must contain the best model's feature_cols).
            test_ids: Test IDs for the submission file.
            config: CompetitionConfig for column naming.
            output_dir: Directory to save the submission CSV.
            
        Returns:
            Path to the generated submission file.
        """
        from kaggle_ml_toolkit.submission_generator import SubmissionGenerator
        
        model, feature_cols = self.get_best_model()
        predictions = model.predict(X_test[feature_cols])
        
        gen = SubmissionGenerator()
        best_name = self._results[0].name
        path = gen.generate(
            predictions=predictions,
            test_ids=test_ids,
            config=config,
            model_name=f"arena_{best_name}",
            output_dir=output_dir,
        )
        
        return path
    
    def summary_report(self) -> str:
        """Generate a markdown summary of the arena run for documentation.
        
        Returns:
            Markdown string suitable for the experiments_log.
        """
        if not self._results:
            return "No results to report. Call run() first."
        
        lines = [
            "## Model Arena Results\n",
            f"**Folds:** {self.cv_folds} | **Metric:** {self.metric} | **Entries:** {len(self._results)}\n",
            "| Rank | Name | CV Score | ± Std | Stability | Worthiness | Features | Model |",
            "|------|------|----------|-------|-----------|------------|----------|-------|",
        ]
        
        for rank, r in enumerate(self._results, 1):
            lines.append(
                f"| {rank} | {r.name} | {r.mean_score:.4f} | {r.std_score:.4f} | "
                f"{r.stability_score:.4f} | {r.submit_worthiness:.4f} | "
                f"{r.feature_count} | {r.model_type} |"
            )
        
        rec = self.recommend()
        lines.extend([
            "",
            f"**Recommendation:** Submit `{rec['recommended_name']}` (estimated LB: {rec['estimated_lb_score']:.4f})",
        ])
        
        if rec["warnings"]:
            lines.append("\n**Warnings:**")
            for w in rec["warnings"]:
                lines.append(f"- ⚠️ {w}")
        
        return "\n".join(lines)
