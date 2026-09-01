# House Prices — Experiment Log

**Competition:** House Prices - Advanced Regression Techniques
**Metric:** RMSLE (Root Mean Squared Log Error)
**Dataset:** 1460 train rows, 80 features, 1459 test rows

---

## Experiment 1: Arena Run (12 pipelines)

**Date:** 2026-08-20
**LB Score:** 0.14545 (RMSLE)
**CV Score:** 0.1405 (RMSE on log-transformed target)
**CV-LB Gap:** 3.5% (excellent!)

### Winner: GBM_very_conservative
```python
model = GradientBoostingRegressor(
    n_estimators=50, max_depth=2, learning_rate=0.15,
    min_samples_leaf=20, subsample=0.7, random_state=42
)
features = ['OverallQual', 'TotalSF', 'GrLivArea', 'TotalBath', 'GarageCars',
            'HouseAge', 'YearsSinceRemodel', 'TotalBsmtSF', 'OverallCond', 'LotArea']
```

### Key Decisions
1. **Log-transformed target** — SalePrice is right-skewed. log1p transform makes RMSE equivalent to RMSLE.
2. **10 core numeric features** — domain-driven selection: quality, size, age, garage, lot.
3. **Ultra-conservative GBM** — 50 trees, depth=2, min_leaf=20. Learned from Titanic overfitting.
4. **No categorical encoding** — used only numeric features to avoid expanding feature space.

### Arena Rankings (top 5)
| Rank | Model | RMSE | Std | Worthiness |
|------|-------|------|-----|------------|
| 1 | GBM_very_conservative | 0.1405 | 0.016 | 6.78 |
| 2 | GBM_conservative | 0.1410 | 0.018 | 6.66 |
| 3 | GBM_moderate | 0.1392 | 0.020 | 6.65 |
| 4 | GBM_strong | 0.1402 | 0.022 | 6.49 |
| 5 | RF_moderate | 0.1455 | 0.019 | 6.42 |

### What Worked
- **OverallQual is king** — single strongest predictor of house prices
- **TotalSF (engineered)** — combining basement + 1st floor + 2nd floor captures total living space
- **Log transform** — makes the target approximately normal, reduces impact of expensive outliers
- **Conservative model** — tiny CV-LB gap proves we're not overfitting
- **Strategies from Titanic applied** — same conservative philosophy, different domain, still works

### What to Try Next
1. Add encoded categorical features (Neighborhood, ExterQual, KitchenQual)
2. Try feature interactions (OverallQual × TotalSF)
3. Stacking: blend GBM with Ridge (they make different errors)
4. Handle the ~5 influential outliers (houses > $500K)

---

## Submission History

| # | Model | Features | CV RMSE | LB RMSLE | Gap | Notes |
|---|-------|----------|---------|----------|-----|-------|
| 1 | GBM Very Conservative | 10 | 0.1405 | 0.14545 | 3.5% | ✅ Solid first attempt |
| 2 | Stack (Ridge+Lasso+GBM) | 95 | 0.1154 | 0.12728 | 10.2% | ✅ **New best** — stacking + full features |

---

## Experiment 2: Research V2 — Stacking with Full Features

**Date:** 2026-08-21
**LB Score:** 0.12728 (RMSLE)
**CV Score:** 0.1154 (RMSE on log target)
**CV-LB Gap:** 10.2%

### Strategy
1. **Outlier removal** — dropped 2 famous GrLivArea > 4000 outliers
2. **Log-transform skewed features** — 54 features with skewness > 0.75
3. **Full categorical encoding** — ordinal for quality features (Ex=5..Po=1), label encoding for rest
4. **Polynomial features** — OverallQual², TotalSF², GrLivArea²
5. **Interaction features** — OverallQual × GrLivArea, OverallQual × TotalBsmtSF
6. **Aggregate features** — TotalPorchSF, TotalBath, HasBasement/Garage/Pool flags
7. **Stacking** — simple average of Ridge + Lasso + GBM predictions

### Arena Winner: GBM_full_v4
- GradientBoostingRegressor (300 trees, depth=4, lr=0.05, min_leaf=5, subsample=0.8)
- 95 full features
- CV RMSE: 0.1194 (arena), worthiness: 8.39

### Stacking Beat Arena
- Ridge: CV 0.1188
- Lasso: CV 0.1177
- GBM: CV 0.1230
- **Stack average: CV 0.1154** (beat arena's 0.1194)

### Key Findings
- Full feature set (95) dramatically outperformed numeric-only (30): 0.117 vs 0.125
- Encoded categoricals (Neighborhood, MSSubClass, etc.) add major value
- Stacking complementary models (linear + tree) beats any single model
- Lasso with full features was competitive with GBM
- CV-LB gap increased from 3.5% to 10.2% — more complex models overfit more

### What to Try Next
- Tune stacking weights (Lasso seems strongest — weight it higher)
- Add XGBoost/LightGBM to the stack
- Try target encoding for high-cardinality categoricals (Neighborhood)
- Feature selection to reduce the 95 features to ~40 most important

---

## ArenaGenerator V3 Sweep — 2026-08-21 21:45

**Arena Winner:** GradientBoosting_full_v4
**CV Score:** 0.11586
**Optimized Params:** {'subsample': 0.8, 'n_estimators': 300, 'min_samples_leaf': 5, 'max_depth': 3, 'learning_rate': 0.1}
**Previous Best LB:** 0.12728
**Submission Status:** success
**File:** arena_v3_stack_20260821_213943.csv
**Method:** ArenaGenerator (10-min budget, 2 feature sets) → ModelOptimizer (randomized, 50 iter)


---

## Experiment: Boosting blend + outlier removal (Phase 3, 2026-08-27)

**OOF RMSLE:** 0.11281 (simple-average blend) | **LB RMSLE:** 0.12345 (new best)

Reused the 92-feature engineering, dropped the two known GrLivArea outliers
(>4000 sqft, low price), and blended 5 diverse base models:

| Base model | OOF RMSLE |
|---|---|
| CatBoost | 0.11461 |
| XGBoost | 0.11485 |
| LightGBM | 0.12059 |
| Lasso | 0.12059 |
| Ridge | 0.12101 |
| **Simple-average blend** | **0.11281** |

The simple average narrowly beat a Ridge-meta stack (0.11281 vs 0.11301), so the
blend was chosen. As on Home-Data-ML, combining trees + linear models beat any
single model — the two families make different errors. Progression on this comp:
0.13560 → 0.12644 → **0.12345**.

### Next levers
- Target encoding for Neighborhood; per-model tuning; try a Huber/quantile blend weight.

---

## Experiment: Importance-based feature selection (2026-08-31)

**OOF RMSLE:** full 0.11281 vs top-45 selected 0.11337 | **LB RMSLE:** 0.12345 (no change)

Tested the research doc's hypothesis that cutting the 92 features to the top ~45
(by aggregated LightGBM gain) would reduce the ~9% CV-LB overfitting gap.

**Result: it did not help.** The full-feature blend (OOF 0.11281) beat the
selected set (0.11337), so the full blend was submitted and scored **0.12345** —
identical to the prior best. **This competition has plateaued at ~0.1234** for
this feature set + tree/linear blend.

### Conclusion / what would actually move it
Not another blend/selection variant. To break below 0.12 needs genuinely new
signal or method:
- Target/mean encoding of Neighborhood (high-cardinality, strong price signal)
  with proper OOF to avoid leakage.
- More aggressive/robust outlier handling than the 2 known GrLivArea points.
- A properly weight-tuned stack (Lasso is strongest; weight it higher) rather
  than an equal-weight average.
- These are the documented next steps; equal-weight blends have topped out.
