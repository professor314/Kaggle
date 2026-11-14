---
phase: 4
title: Model Selection
triggers: kaggle_ml_toolkit.model_selector, kaggle_ml_toolkit.cross_validator
---

# Model Selection

## Context
Model selection compares multiple candidates under controlled conditions to find the best starting point for optimization. We use cross-validation to get reliable performance estimates and handle failures gracefully. The goal isn't to find the final model — it's to identify which 2-3 candidates deserve hyperparameter tuning.

## Decision Points

### Model Candidates
**Question:** Which models should we compare?
**Options:**
- Standard classification: Logistic Regression, Random Forest, Gradient Boosting, KNN, SVM. Covers linear and non-linear approaches.
- Boosting-focused: XGBoost, LightGBM, CatBoost, Random Forest. Best for tabular competitions where boosting dominates.
- Broad sweep: All available models including slower ones. Takes longer but may surface surprises.
- Custom set: User specifies exact model list based on research phase findings.
**Recommendation:** Standard classification for Phase 1; switch to boosting-focused once XGBoost/LightGBM are integrated in Phase 2.

### Evaluation Metric
**Question:** Which metric should drive model comparison?
**Options:**
- Competition metric: Use the exact metric from the competition evaluation page (e.g., accuracy, log loss, RMSE).
- Multiple metrics: Track competition metric as primary, but also report precision/recall/F1 for fuller picture.
- Custom metric: Define a domain-specific metric (e.g., weighted F1 for imbalanced classes).
**Recommendation:** Competition metric as primary — this is what the leaderboard uses. Report secondary metrics for insight.

### Cross-Validation Strategy
**Question:** Which CV strategy should we use?
**Options:**
- Stratified K-Fold (k=5): Standard for classification. Preserves class ratios in each fold.
- Stratified K-Fold (k=10): More folds = lower variance estimate, but slower.
- Repeated Stratified (5×2): Multiple repeats reduce variance further. Good for small datasets.
- Group K-Fold: When observations are grouped (e.g., multiple rows per patient).
**Recommendation:** Stratified K-Fold (k=5) as default — good balance of reliability and speed.

### Handling Failed Models
**Question:** What should happen when a model fails during comparison?
**Options:**
- Skip and continue: Record the error, move on. Fastest path forward.
- Retry with simpler config: Reduce complexity (fewer features, simpler params) and retry once.
- Investigate: Pause and diagnose why the model failed before continuing.
**Recommendation:** Skip and continue — failed models are recorded with error messages for later investigation if needed.

### Baseline Establishment
**Question:** Should we establish a baseline model first?
**Options:**
- Yes, dummy baseline: Train a dummy classifier/regressor first to set the floor.
- Yes, simple baseline: Use logistic regression or linear model as baseline.
- No, jump to comparison: Start the full comparison directly.
**Recommendation:** Yes, simple baseline — gives a meaningful reference point for improvement metrics.

## Actions
Based on decisions above, Kiro will:
1. Establish baseline if selected
2. Execute `ModelSelector.compare()` with chosen models, metric, and CV strategy
3. Execute `CrossValidator.validate()` for detailed fold-level analysis
4. Present ranked results table with scores, standard deviations, and status
5. Recommend top 2-3 models for optimization phase
6. Ask user: "Which models should we tune? Do you want to investigate any failures?"
7. Record model selection decisions and results in the design doc
