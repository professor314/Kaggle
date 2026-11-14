---
phase: 2
title: Exploratory Data Analysis
triggers: kaggle_ml_toolkit.eda_engine
---

# Exploratory Data Analysis

## Context
EDA transforms raw data into actionable understanding. We systematically examine distributions, relationships, missing patterns, and anomalies — generating both statistical summaries and visualizations. Findings here directly inform feature engineering decisions and validate or refute research hypotheses.

## Decision Points

### Statistical Analysis Depth
**Question:** Which statistical analyses should we run?
**Options:**
- Essential: Summary stats, missing values, correlation matrix, class balance. Covers basics quickly.
- Comprehensive: Add distribution tests (normality, skewness), multicollinearity (VIF), near-zero variance detection.
- Full suite: Add statistical hypothesis tests, effect sizes, interaction detection. For complex datasets.
**Recommendation:** Comprehensive — VIF and distribution analysis catch issues that derail models later.

### Visualization Strategy
**Question:** Which visualizations should we generate?
**Options:**
- Core: Histograms, correlation heatmap, target distribution. Minimal but informative.
- Standard: Add box plots, scatter matrices, missing value patterns, feature-target relationships.
- Publication-ready: Add styled plots with annotations, multi-panel figures, interactive plots. For blog content.
**Recommendation:** Standard for analysis; regenerate key plots as publication-ready during content creation phase.

### PCA Exploration
**Question:** How deep should we go on dimensionality analysis?
**Options:**
- Skip: Not needed for low-dimensional datasets (< 15 features).
- Variance check: Run PCA, report explained variance ratios, identify dominant components.
- Full analysis: Variance + component loadings + biplot + suggest dimensionality reduction threshold.
**Recommendation:** Variance check for most tabular competitions. Full analysis when features exceed 30 or multicollinearity is high.

### Target Analysis
**Question:** How should we analyze the target variable?
**Options:**
- Distribution only: Class balance for classification, distribution shape for regression.
- Distribution + relationships: Add target vs. each feature analysis (correlations, grouped means).
- Segmented analysis: Add subgroup analysis, interaction effects with target, leakage detection.
**Recommendation:** Distribution + relationships — this directly feeds feature importance intuition.

### Investigation Triggers
**Question:** What should trigger deeper investigation?
**Options:**
- High missing rate (>30% in any column): Investigate mechanism (MCAR/MAR/MNAR)
- Extreme class imbalance (>10:1 ratio): Flag for augmentation or sampling strategy
- Suspicious correlations (>0.95): Check for data leakage or redundancy
- All of the above: Investigate any flag that appears
**Recommendation:** All of the above — each flag can significantly impact model performance.

## Actions
Based on decisions above, Kiro will:
1. Execute `EDAEngine.run()` with configured analysis depth
2. Generate visualizations at chosen quality level
3. Run PCA analysis if selected
4. Produce narrative summary via `EDAEngine.generate_narrative()`
5. Flag anomalies and investigation triggers
6. Present findings with specific questions: "Should we investigate X further?" or "Does this confirm hypothesis Y?"
7. Record confirmed insights in the design doc
