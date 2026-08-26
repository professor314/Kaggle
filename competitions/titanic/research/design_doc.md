# Design Document: Titanic Survival Prediction

## Competition Overview

- **Competition**: Titanic - Machine Learning from Disaster (Kaggle Getting Started)
- **Objective**: Predict which passengers survived the Titanic shipwreck
- **Metric**: Accuracy (percentage of correct predictions)
- **Submission**: CSV with PassengerId and Survived columns

## Strategic Decisions

### Primary Hypothesis

**Gender + Class + Age are the dominant survival factors.**

The "women and children first" evacuation protocol, combined with class-based access to lifeboats, explains the majority of survival variance. Gender alone achieves 77% accuracy; adding class and age should push toward 80%+. Engineered features (Title, FamilySize, Deck) capture finer-grained signals within these broad categories.

### Modeling Strategy

| Phase | Model | Purpose |
|-------|-------|---------|
| Baseline | Gender-only rule | Establish floor (77%) |
| Phase 1 | Logistic Regression | Interpretable reference, validate feature importance |
| Phase 2 | Random Forest | Strong tree-based model, handles mixed features well |
| Phase 2 | Gradient Boosting (XGBoost) | Typically top performer for structured tabular data |
| Phase 3 | Ensemble (Stack/Blend) | Combine top 2 models for final submission |

**Optimization plan:**
1. Train all three models with default parameters on engineered feature set
2. Identify top 2 performers based on CV accuracy
3. Run hyperparameter optimization (randomized search) on top 2
4. If top 2 are sufficiently different (RF + GBM), build a stacking ensemble
5. If marginal gains, submit best single model

### Evaluation Priorities

1. **Primary**: Accuracy (competition leaderboard metric)
2. **Diagnostic**: Precision and Recall by class (Survived=0 vs Survived=1) to understand where the model fails
3. **Validation**: Stratified 5-fold CV to ensure stable estimates given ~38% positive rate
4. **Comparison**: Track improvement over gender baseline at every stage

### Content Goals

Two pieces of educational content from this competition:

1. **Blog Post** (intermediate audience):
   - "Beating the Titanic Baseline: A Research-First Approach to Kaggle Competitions"
   - Covers: Why research matters, key domain insights, feature engineering rationale, model comparison
   - Target length: 2000-3000 words with visualizations and code snippets
   - Demonstrates the toolkit workflow and decision-driven methodology

2. **Kaggle Writeup** (competition community):
   - Concise solution writeup for Kaggle Discussion forum
   - Sections: Summary, Approach, What Worked, What Didn't, Final Model, Score
   - Includes AI disclosure (developed with Kiro assistance)
   - Focus on reproducibility and actionable insights for other competitors

## Feature Engineering Strategy

Based on research findings, the following features will be engineered:

### Priority 1 (High confidence from research)
- **Title**: Extract from Name using regex, group rare titles
- **FamilySize**: SibSp + Parch + 1
- **IsAlone**: Binary flag for solo travelers

### Priority 2 (Moderate confidence)
- **Deck**: Extract from Cabin letter (where available)
- **HasCabin**: Binary flag for cabin information presence
- **AgeBin**: Domain-informed age groups (Child, Teen, Young Adult, Adult, Senior)
- **FareBin**: Quartile-based fare groups

### Priority 3 (Speculative, validate with feature selection)
- **FarePerPerson**: Fare / FamilySize
- **Age × Pclass interaction**: Captures class-dependent age effects

### Encoding Plan
- **Sex**: Label encode (binary)
- **Embarked**: One-hot encode (3 levels)
- **Title groups**: One-hot encode (4-5 levels)
- **Pclass**: Keep numeric (ordinal)
- **Binned features**: Label encode (ordinal bins)

## Missing Data Strategy

| Feature | Strategy | Rationale |
|---------|----------|-----------|
| Age | Title-group median imputation | Titles are strong age proxies (Master=child, Miss=young woman) |
| Cabin | Extract deck where available, create HasCabin flag | 77% missing makes imputation unreliable; missingness is informative |
| Embarked | Mode imputation ('S') | Only 2 values missing, mode is safe |
| Fare | Class-specific median | Fare varies dramatically by class |

## Decisions Log

| # | Decision | Rationale | Date |
|---|----------|-----------|------|
| 1 | Research scope: Standard | Getting Started competition with abundant prior work | TBD |
| 2 | Primary hypothesis: Gender + Class + Age | Historical records and prior Kaggle solutions confirm these as dominant factors | TBD |
| 3 | Models: LR, RF, GBM then ensemble | Covers interpretable baseline + competitive options without overcomplicating | TBD |
| 4 | Feature engineering: Moderate | Domain knowledge is strong; create research-backed features without speculation | TBD |
| 5 | CV strategy: Stratified 5-fold | Standard for binary classification with mild imbalance (38% positive) | TBD |

## Success Criteria

- **Minimum**: Beat gender baseline (>77% accuracy)
- **Target**: Achieve 80-82% accuracy with well-documented approach
- **Stretch**: Reach 83%+ with optimized ensemble
- **Educational**: Produce publishable blog post regardless of final score

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Overfitting to CV | High leaderboard drop | Use conservative model complexity, compare CV vs LB gap |
| Age imputation noise | Reduced accuracy | Use title-based imputation rather than global median |
| Feature leakage | Inflated CV scores | Ensure all transformations fit only on training folds |
| Over-engineering | Diminishing returns | Stop adding features when CV improvement < 0.5% |
