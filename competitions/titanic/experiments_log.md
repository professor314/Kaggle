# Titanic Competition — Experiment Log

**Competition:** Titanic - Machine Learning from Disaster
**Metric:** Accuracy (public leaderboard)
**Baseline:** Gender-only prediction = 0.7868 (local CV) / ~0.766 (LB)

---

## Experiment 1: RF Optimized v1

**Date:** 2026-08-20
**LB Score:** 0.76076
**CV Score:** 0.8339
**CV-LB Gap:** 7.3% (overfitting)

### Strategy
- 18 features (one-hot encoded Sex, Embarked, Title + numeric features)
- Title extraction from Name (Mr, Mrs, Miss, Master, Rare)
- FamilySize, IsAlone, HasCabin
- Age imputation by title-group median
- Randomized search on RandomForest (n_estimators, max_depth, min_samples_split, min_samples_leaf)
- Best params: n_estimators=100, min_samples_split=2, min_samples_leaf=4, max_depth=10

### What Went Wrong
- Too many features (18) for 891 training rows → overfitting
- One-hot encoding expanded the feature space unnecessarily
- Model complexity (max_depth=10) too high for this dataset size
- CV score was misleadingly high — the model memorized training patterns that don't generalize

### Code
```python
# Feature set (18 features)
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone',
            'HasCabin', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q',
            'Embarked_S', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare']

model = RandomForestClassifier(
    n_estimators=100, min_samples_split=2, min_samples_leaf=4, max_depth=10, random_state=42
)
```

---

## Experiment 2: GBM Conservative v2

**Date:** 2026-08-20
**LB Score:** 0.77272
**CV Score:** 0.8317
**CV-LB Gap:** 5.9% (better, still overfitting)

### Strategy
- 8 features only (no one-hot encoding, binary IsFemale instead)
- Same feature engineering (Title for imputation, FamilySize, IsAlone)
- GradientBoosting with heavy regularization: n_estimators=50, max_depth=3, min_samples_leaf=10, subsample=0.8
- Fewer trees + shallow depth + large leaves = less overfitting

### What Improved
- Simpler feature set reduces overfitting (8 vs 18 features)
- Lower max_depth (3 vs 10) prevents memorizing noise
- min_samples_leaf=10 forces broader generalization
- subsample=0.8 adds randomization (bagging effect)
- CV variance much lower (±0.01 vs ±0.03)

### What's Still Wrong
- Still 6% CV-LB gap — some overfitting remains
- LB score barely above gender baseline (0.766)
- The model adds marginal value beyond the gender signal

### Code
```python
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'IsFemale']

model = GradientBoostingClassifier(
    n_estimators=50, max_depth=3, learning_rate=0.1,
    min_samples_leaf=10, subsample=0.8, random_state=42
)
```

---

## Key Learnings (So Far)

### What Works on Titanic
1. **Gender is king** — Sex/IsFemale is by far the strongest single feature
2. **Title extraction** — useful primarily for age imputation (Master=child), adds modest signal
3. **Heavy regularization** — small dataset (891 rows) + noise = overfitting risk. Simpler is better.
4. **Low CV variance** — a model with ±0.01 std is more trustworthy than one with ±0.03

### What Doesn't Work on Titanic
1. **Many features** — 18 features for 891 rows causes overfitting
2. **One-hot encoding** — expands feature space, adds columns that are mostly redundant with Sex
3. **Deep trees** — max_depth > 5 memorizes training noise
4. **High CV score** — misleading on small datasets. A 0.83 CV that gives 0.76 LB is worse than a 0.81 CV that gives 0.79 LB.

### Strategies to Try Next
1. **Ultra-simple model** — just Sex + Pclass + child flag. Might beat both attempts.
2. **Feature interactions** — Sex × Pclass (women in 1st class survived at 97%)
3. **Cabin deck** — risky (77% missing) but top solutions use it carefully
4. **Ensemble of simple models** — blend gender-rule with a shallow tree

---

## Submission History

| # | Model | Features | CV | LB | Gap | Notes |
|---|-------|----------|----|----|-----|-------|
| 1 | RF Optimized | 18 | 0.8339 | 0.76076 | 7.3% | Overfit, too complex |
| 2 | GBM Conservative | 8 | 0.8317 | 0.77272 | 5.9% | **Best** — sweet spot |
| 3 | GBM Strong (Arena winner) | 8 | 0.8575 | 0.75837 | 10.0% | Most overfit — high CV is a trap |
| 4 | GBM Sweetspot | 8 | 0.8249 | 0.77272 | 5.3% | Tied best — confirms the ceiling |
| 5 | Research V2a (LEAKAGE) | 8 | 0.9888 | 0.72727 | 26.2% | ❌ Target leakage from survival features |
| 6 | Research V2b (no leakage) | 8 | 0.8406 | 0.76794 | 7.3% | Structural group features only |

## Key Discovery: The Overfitting Curve

```
CV Score vs LB Score (Titanic, 891 rows):

  CV 0.86 ─── GBM_strong ──────────────── LB 0.758 (worst!)
  CV 0.83 ─── RF_optimized / GBM_cons ─── LB 0.761-0.773 (sweet spot)
  CV 0.82 ─── GBM_sweetspot ───────────── LB 0.773 (tied best)
  CV 0.79 ─── Gender baseline ──────────── LB ~0.766

On this dataset, the relationship between CV and LB is INVERTED past ~0.83 CV.
Higher CV = more overfitting = worse LB score.
The optimal zone is CV 0.82-0.83 with std < 0.012.
```

## Arena Results (21 pipelines compared)

The Model Arena compared 21 configurations. Top 5 by submit-worthiness:

| Rank | Name | CV | Std | Worthiness |
|------|------|-----|-----|------------|
| 1 | GBM_strong | 0.858 | 0.010 | 0.783 |
| 2 | GBM_moderate | 0.856 | 0.018 | 0.775 |
| 3 | GBM_interaction | 0.846 | 0.015 | 0.767 |
| 4 | RF_conservative | 0.833 | 0.008 | 0.760 |
| 5 | RF_moderate | 0.832 | 0.009 | 0.758 |

**Lesson:** The arena's worthiness metric correctly rewards stability, but on a small dataset where ALL models overfit, the worthiness ranking still doesn't predict LB well. The true signal is: simpler models (lower CV, lower variance) generalize better.

## Experiment 5: Research V2a — Target Leakage (FAILED)

**Date:** 2026-08-21
**LB Score:** 0.72727 ❌
**CV Score:** 0.9888 (!!!)
**CV-LB Gap:** 26.2% (catastrophic — target leakage)

### What Happened
- Computed TicketGroupSurvival and FamilySurvival FROM training labels
- These features perfectly encode the target during CV (leak within folds)
- CV was 98.88% — should have been an immediate red flag
- LB was 0.727 — worse than gender-only baseline

### Lesson Learned
**NEVER compute features from the target variable unless you do out-of-fold encoding.**
A CV score of 98.8% on Titanic is physically impossible without leakage.
Any feature derived from `Survived` must be computed leave-one-out or out-of-fold.

---

## Experiment 6: Research V2b — No-Leakage Features

**Date:** 2026-08-21
**LB Score:** 0.76794
**CV Score:** 0.8406
**CV-LB Gap:** 7.3%

### Strategy
- Removed all target-derived features (TicketGroupSurvival, FamilySurvival)
- Kept structural group features (TicketGroupSize, SurnameGroupSize)
- Added SexPclass interaction (6-level ordinal)
- Added WomenChildren1st (women/children priority × class)
- Added FarePerPerson (fare / ticket group size)
- Added FamilyBucket (alone=0, small=1, large=2)
- Quantile-transformed Age and Fare
- GBM conservative (50 trees, depth=2, min_leaf=15) on 8 minimal features

### What Worked
- CV-LB gap back to expected range (7.3%)
- SexPclass interaction captures the "women in first class" effect
- FarePerPerson is a better fare signal than raw Fare

### What Didn't Help
- More features (18) didn't beat 8 features for Titanic
- The research_v2 set (18 features) scored lower than minimal (8 features) in the arena
- Titanic's ceiling appears to be ~0.773 with standard approaches

### Code
```python
minimal_features = ["IsFemale", "Pclass", "Age_qt", "Fare_qt", "SexPclass",
                    "WomenChildren1st", "FarePerPerson_qt", "IsChild", "FamilyBucket"]
model = GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.1,
                                    min_samples_leaf=15, subsample=0.8, random_state=42)
```

---

## What We'd Do Differently Next Time

1. **Start with the simplest possible model** and only add complexity when LB improves
2. **Submit early** to establish the CV-LB gap before running expensive arena loops  
3. **Use the gap to calibrate the arena's prior_cv_lb_gap** — we estimated 6.5% but reality was 5-10% depending on complexity
4. **Don't chase CV score** — on small datasets, CV > 0.83 is a red flag, not a goal
5. **The arena is most valuable for ELIMINATING bad models** (KNN, deep trees) rather than finding the single best

---

## ArenaGenerator V3 Sweep — 2026-08-21 21:45

**Arena Winner:** GradientBoosting_minimal_v3
**CV Score:** 0.84737
**Optimized Params:** {'subsample': 0.7, 'n_estimators': 75, 'min_samples_leaf': 25, 'max_depth': 3, 'learning_rate': 0.2}
**Previous Best LB:** 0.77272
**Submission Status:** success
**File:** arena_v3_optimized_20260821_213231.csv
**Method:** ArenaGenerator (10-min budget, 2 feature sets) → ModelOptimizer (randomized, 50 iter)
