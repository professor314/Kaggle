---
phase: 3
title: Feature Engineering
triggers: kaggle_ml_toolkit.feature_engineer, kaggle_ml_toolkit.feature_selector
---

# Feature Engineering

## Context
Feature engineering is where domain knowledge meets data science. We create new features informed by research hypotheses and EDA findings, then select the most predictive subset. Good feature engineering often matters more than model choice — this phase directly builds on insights from research and EDA.

## Decision Points

### Interaction Features
**Question:** Which feature interactions should we create?
**Options:**
- Research-driven only: Create interactions suggested by domain hypotheses (e.g., Age × Pclass for Titanic).
- EDA-driven: Create interactions for highly correlated feature pairs identified in EDA.
- Combinatorial: Generate all pairwise interactions for top-N features, let selection filter them.
- Targeted: Combine research-driven and EDA-driven without exhaustive combinations.
**Recommendation:** Targeted — research gives strong priors, EDA validates which combinations actually correlate with the target.

### Binning Strategy
**Question:** How should we discretize continuous features?
**Options:**
- No binning: Keep continuous features as-is. Simplest, works well with tree-based models.
- Domain-informed bins: Use meaningful thresholds (e.g., age groups: child/adult/senior).
- Statistical bins: Equal-width or quantile binning with configurable number of bins.
- Hybrid: Domain bins for key features, statistical bins for others.
**Recommendation:** Hybrid — domain bins capture known non-linearities; statistical bins help linear models.

### Encoding Choices
**Question:** How should we encode categorical features?
**Options:**
- One-hot encoding: Creates binary columns for each category. Simple, works everywhere, but high-cardinality explodes dimensions.
- Label encoding: Integer mapping. Compact, works with tree models, but implies false ordering.
- Target encoding: Replace category with mean of target. Powerful but requires careful regularization to avoid leakage.
- Mixed: One-hot for low-cardinality (< 10 levels), label for ordinal, target for high-cardinality.
**Recommendation:** Mixed strategy — matches encoding to the nature of each feature.

### Domain-Informed Features
**Question:** Which domain-specific features should we engineer?
**Options:**
- Minimal: Only features directly suggested by research (e.g., FamilySize = SibSp + Parch + 1).
- Moderate: Add ratio features, group aggregates, and flag variables from domain knowledge.
- Aggressive: Create all plausible domain features, including speculative ones, and let selection prune.
**Recommendation:** Moderate — enough to capture domain signal without overfitting to noise.

### Feature Selection Method
**Question:** Which feature selection approach should we use?
**Options:**
- Single method: Pick one method (e.g., mutual information) and trust its ranking.
- Ensemble selection: Run multiple methods (mutual_info, RFE, L1, correlation) and use consensus ranking.
- Iterative: Start with ensemble selection, then refine based on model performance feedback.
**Recommendation:** Ensemble selection — consensus across methods is more robust than any single ranking.

## Actions
Based on decisions above, Kiro will:
1. Execute `FeatureEngineer.create_interaction()` for chosen feature pairs
2. Execute `FeatureEngineer.bin_continuous()` with chosen strategy
3. Apply encoding via `DataCleaner.encode()` with chosen method per feature
4. Create domain-specific features using `FeatureEngineer`
5. Run `FeatureSelector.select()` with chosen methods and domain priorities
6. Present ranked features with explanation report
7. Ask user to confirm final feature set before model selection
8. Record feature decisions in the design doc
