# Strategy Library

Patterns and lessons learned across Kaggle competitions. Each entry captures what works, what doesn't, and the evidence (leaderboard scores) behind it.

---

## General Principles

### 1. Simpler Models Generalize Better on Small Datasets
**Evidence:** Titanic (891 rows) — 18-feature RF scored 0.76 LB vs 8-feature GBM scoring 0.77 LB. Models with CV > 0.83 scored WORSE on LB than models with CV 0.82-0.83.
**Rule of thumb:** If rows < 1000, keep features < 10 and max_depth ≤ 3. Target CV in the 0.80-0.83 range, NOT higher.

### 2. CV-LB Gap Is Your Overfitting Diagnostic
**Evidence:** Titanic — 5.3% gap (best), 5.9% gap (good), 7.3% gap (bad), 10% gap (worst). The gap INCREASES with model complexity.
**Rule of thumb:** If CV-LB gap > 5%, reduce model complexity. If gap > 8%, the model is memorizing noise.

### 3. High CV Score Can Be a Red Flag on Small Datasets
**Evidence:** Titanic — the model with the HIGHEST CV (0.858) had the WORST LB (0.758). The relationship between CV and LB inverts past a complexity threshold.
**Rule of thumb:** On datasets < 2000 rows, treat CV > 0.85 as suspicious, not aspirational. The optimal CV zone is typically 2-5 points below what's achievable.

### 4. One-Hot Encoding Can Hurt on Small Datasets
**Evidence:** Titanic — one-hot encoding Sex + Embarked + Title → 10 extra columns, contributed to overfitting.
**Alternative:** Use binary flags (IsFemale) and label encoding for ordinal features. Keep feature count low.

### 5. Domain Knowledge > Feature Count
**Evidence:** Title extraction for age imputation (Master=child) added more value than 10 statistical features combined.
**Rule of thumb:** One domain-informed feature > five blind statistical features.

---

## Competition-Specific Strategies

### Titanic (Binary Classification, Small Dataset)

| Strategy | CV | LB | Verdict |
|----------|----|----|---------|
| GBM Conservative (depth=3, 50 trees, leaf=10) | 0.832 | **0.773** | ✅ Best |
| GBM Sweetspot (depth=2, 40 trees, leaf=15) | 0.825 | **0.773** | ✅ Tied best |
| RF Optimized (depth=10, 18 features) | 0.834 | 0.761 | ❌ Overfit |
| GBM Strong (depth=3, 200 trees, lr=0.05) | 0.858 | 0.758 | ❌ Most overfit |
| Gender-only baseline | 0.787 | ~0.766 | Reference |

**Ceiling:** ~0.773 with standard approaches. Breaking 0.78+ requires ticket grouping, surname-based family survival patterns, or cabin deck engineering.

**What works:** 8 features (IsFemale, Pclass, Age, Fare, FamilySize, IsAlone, IsChild, TitleCode), GBM with depth=2-3, min_leaf=10-15, 40-50 trees.
**What doesn't:** More features, deeper trees, more trees, one-hot encoding on this dataset size.
**Critical insight:** On this dataset, CV > 0.83 means you're overfitting. The sweet spot is CV 0.82-0.83 with std < 0.012.

### Model Arena Findings (Titanic)

The ModelArena compared 21 configurations in one run. Key findings:
- KNN scored worst (0.734 CV) — not suitable for this mixed-type small dataset
- Decision Trees at depth 2-3 scored well (0.777-0.828) but with higher variance
- The arena's "submit-worthiness" metric correctly identified stability-accuracy tradeoffs
- **Arena lesson:** Use the arena to ELIMINATE bad options quickly, not to find the absolute best. Submit the most conservative option from the top cluster.

---

## Model Selection Heuristics

| Dataset Size | Recommended Models | Max Depth | Feature Count |
|-------------|-------------------|-----------|---------------|
| < 500 rows | Logistic Regression, shallow trees | 3 | ≤ 5 |
| 500-2000 rows | GBM (conservative), RF | 3-5 | 5-10 |
| 2000-10000 rows | GBM, RF, XGBoost | 5-8 | 10-30 |
| > 10000 rows | XGBoost, LightGBM, deep learning | 6-12 | 30+ |

---

## Regularization Patterns

### For Small Datasets (< 2000 rows)
```python
# GBM — conservative
GradientBoostingClassifier(
    n_estimators=50,       # few trees
    max_depth=3,           # shallow
    learning_rate=0.1,     # moderate
    min_samples_leaf=10,   # large leaves
    subsample=0.8,         # row sampling
)

# RF — constrained
RandomForestClassifier(
    n_estimators=100,
    max_depth=5,           # capped
    min_samples_leaf=5,    # prevent overfitting
    max_features='sqrt',   # feature sampling
)
```

### For Large Datasets (> 10000 rows)
```python
# XGBoost — standard competitive settings
XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
)
```

---

## Feature Engineering Patterns

### Title Extraction (Name → Social Status)
```python
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
# Group rare titles: Dr, Rev, Col → 'Rare'
# Use title medians for age imputation (Master=~5, Miss=~22, Mr=~30)
```
**When:** Dataset has a Name field with titles. Common in passenger/customer data.

### Family Size (Non-Linear Effect)
```python
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
```
**When:** Solo vs group behavior differs (evacuation, purchasing, churn).

### Binary Instead of One-Hot (Small Datasets)
```python
# Instead of pd.get_dummies(df['Sex']) → Sex_female, Sex_male
df['IsFemale'] = (df['Sex'] == 'female').astype(int)
```
**When:** Binary categorical feature on datasets with < 2000 rows. Halves the column count.

### House Prices (Regression, Medium Dataset)

| Strategy | CV RMSE | LB RMSLE | Gap | Verdict |
|----------|---------|----------|-----|---------|
| GBM Very Conservative (depth=2, 50 trees, 10 features) | 0.1405 | 0.14545 | 3.5% | ✅ Solid first try |
| Stack Ridge+Lasso+GBM (95 features, full encoding) | 0.1154 | **0.12728** | 10.2% | ✅ **Best** — stacking wins |

**What works:** Log-transform target, OverallQual as #1 feature, TotalSF (engineered), full categorical encoding, stacking diverse models.
**Key insight:** Stacking linear + tree models gives ~3-5% free improvement. Full categorical encoding adds ~6% over numeric-only.
**Leaderboard context:** Top scores ~0.10, median ~0.15, poor ~0.30+. Our 0.127 is competitive (top ~35%).

### Spaceship Titanic (Binary Classification, Larger Dataset)

| Strategy | CV | LB | Gap | Verdict |
|----------|----|----|-----|---------|
| GBM Moderate (depth=4, 200 trees, 22 features) | 0.8073 | 0.7994 | 0.8% | ✅ Excellent |
| HGB Strong (depth=5, 300 iter, 50 features) | 0.8119 | **0.8024** | 0.9% | ✅ **Best** — HGB + ratios |

**What works:** CryoSleep (dominant), spending features (Total, Log, NoSpending, ratios), Cabin parsing (Deck, Side, Region), GroupSize from PassengerId, HomePlanet×CryoSleep interactions, HistGradientBoosting.
**Key insight:** On 8700 rows, 50 features + depth=5 is fine (0.9% gap). HistGBM outperforms standard GBM. Spending ratios add marginal but consistent value.
**Leaderboard context:** Top scores ~0.82, median ~0.78, poor ~0.70. Our 0.802 is competitive.

---

## ModelArena Usage Pattern

The ModelArena is the toolkit's local iteration loop. Use it to burn through ideas without spending Kaggle submissions:

```python
from kaggle_ml_toolkit import ModelArena

arena = ModelArena(
    X=train_df, y=y,
    cv_folds=5,
    metric='accuracy',
    prior_cv_lb_gap=0.06,  # calibrate from first submission
)

# Add many configurations spanning simple → complex
arena.add('simple', LogisticRegression(), feature_cols=['IsFemale', 'Pclass'])
arena.add('moderate', GBM(depth=2, trees=40), feature_cols=eight_features)
arena.add('complex', GBM(depth=5, trees=200), feature_cols=twelve_features)

results = arena.run()
print(arena.leaderboard())
# Submit the MOST CONSERVATIVE model from the top cluster, not #1
```

**Key lesson from Titanic:** The arena winner (highest worthiness) isn't always the best LB scorer. On small datasets, submit the most conservative model from the top-scoring cluster. The arena's value is in eliminating bad options, not optimizing the last 0.1%.

---

*Last updated: 2026-08-21*
*Competitions analyzed: Titanic (6 submissions), House Prices (2 submissions), Spaceship Titanic (3 submissions)*
*Total arena configs tested: 103*

---

## NEW LEARNINGS (Research V2 Cycle)

### 6. Target Leakage Is the #1 Risk in Feature Engineering
**Evidence:** Both Titanic and Spaceship Titanic had catastrophic leakage from survival/transport rate features computed from training labels. CV was 95-99% but LB was 49-73%.
**Rule of thumb:** If CV score is implausibly high (>95% on a noisy problem), suspect leakage immediately. Never compute features from the target variable without out-of-fold encoding.

### 7. Stacking Beats Individual Models (Regression)
**Evidence:** House Prices — simple average of Ridge + Lasso + GBM (CV 0.1154) beat the best individual model GBM (CV 0.1194) by 3.4%. LB improved from 0.14545 to 0.12728.
**Rule of thumb:** For regression problems, always try a simple average stack of linear + tree models. The diversity gain is nearly free.

### 8. Full Categorical Encoding Matters (Medium Datasets)
**Evidence:** House Prices — 95 features (with categoricals) beat 30 features (numeric only): 0.117 vs 0.125 CV RMSE. A 6.4% improvement.
**Rule of thumb:** On datasets with 1000+ rows, encoding categoricals (ordinal for quality, label for nominal) adds significant signal.

### 9. HistGradientBoosting Is the New Default
**Evidence:** Spaceship Titanic — HGB beat standard GBM (0.8119 vs 0.8052 CV), was 7x faster, and handles NaN natively.
**Rule of thumb:** Use HistGradientBoostingClassifier/Regressor as the first choice for datasets > 1000 rows. Falls back gracefully on missing values.

### 10. Larger Datasets Can Handle More Features
**Evidence:** 
- Titanic (891 rows): 18 features ≤ 8 features (more = worse)
- House Prices (1460 rows): 95 features > 30 features (more = better)
- Spaceship (8700 rows): 50 features > 22 features (more = better)
**Rule of thumb:** features-to-rows ratio matters. Keep ratio < 1:50 for safety. At 1:10 expect overfitting.
