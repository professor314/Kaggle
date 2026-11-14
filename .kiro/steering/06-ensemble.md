---
phase: 6
title: Ensemble Construction
triggers: kaggle_ml_toolkit.ensemble_builder
---

# Ensemble Construction

## Context
Ensembles combine multiple models to produce predictions more robust than any individual model. This is often the final performance push in competitions — diverse models that make different errors complement each other. We build on the optimized models from the previous phase.

## Decision Points

### Models to Combine
**Question:** Which models should be included in the ensemble?
**Options:**
- Top 2: Combine the two best optimized models. Simple and often effective.
- Top 3: Add a third model for more diversity. Standard competitive approach.
- Diverse set: Include models that are accurate but make different errors, even if individually weaker.
- All optimized: Include every model that exceeded baseline. Maximum diversity.
**Recommendation:** Diverse set — diversity matters more than individual accuracy for ensembles. A slightly weaker model that disagrees with others adds value.

### Ensemble Strategy
**Question:** Which combination strategy should we use?
**Options:**
- Hard voting: Majority vote across models. Simple, interpretable, works for classification only.
- Soft voting: Average predicted probabilities. Leverages confidence levels, usually better than hard voting.
- Stacking: Train a meta-learner on base model predictions. Most powerful but risk of overfitting.
- Blending: Like stacking but uses a held-out set instead of CV. Simpler, less risk of leakage.
- Weighted average: Weight models by their CV scores. Good middle ground.
**Recommendation:** Soft voting as baseline, then try stacking if improvement is needed. Stacking gives 1-3% lift on most competitions.

### Meta-Learner Choice (for Stacking)
**Question:** If using stacking, which meta-learner should combine the base predictions?
**Options:**
- Logistic Regression: Simple, regularized, less likely to overfit. Best default meta-learner.
- Ridge Regression: Linear with L2 penalty. Good when base models are well-calibrated.
- Gradient Boosting: Non-linear meta-learner. Powerful but higher overfitting risk on small stacking datasets.
- Simple average: No learner — just average predictions. Zero overfitting risk, surprisingly competitive.
**Recommendation:** Logistic Regression — regularization prevents overfitting the meta-learner, which is the main stacking risk.

### Validation Strategy
**Question:** How should we validate the ensemble?
**Options:**
- Same CV as selection: Use the same folds for consistency in comparison.
- Fresh holdout: Set aside data not used in any training. Cleanest estimate but reduces training data.
- Nested CV: Full nested cross-validation for unbiased estimate. Most rigorous but slowest.
**Recommendation:** Same CV as selection — enables direct comparison against individual model scores from earlier phases.

### Contribution Analysis
**Question:** Should we analyze each model's contribution to the ensemble?
**Options:**
- Skip: Just report the ensemble score and move on.
- Ablation study: Remove each model one at a time and measure impact. Shows who contributes most.
- Full analysis: Ablation + correlation of errors + diversity metrics. Comprehensive understanding.
**Recommendation:** Ablation study — quick to run and directly answers "is this model helping?"

## Actions
Based on decisions above, Kiro will:
1. Prepare base model predictions using cross-validation (prevent leakage)
2. Execute `EnsembleBuilder.build()` with chosen strategy and models
3. Execute `EnsembleBuilder.evaluate_contributions()` if analysis selected
4. Compare ensemble score against best individual model
5. Present improvement metrics and contribution report
6. Ask user: "Is this ensemble satisfactory, or should we try a different strategy?"
7. Record ensemble architecture and performance in the design doc
