# Home Data for ML Course (Ames Housing) — Experiment Log

**Competition:** Housing Prices Competition for Kaggle Learn Users
**Metric:** MAE (mean absolute error on SalePrice)
**Dataset:** 1,460 train / 1,459 test, 80 features (Ames housing)
**Submission:** direct CSV (no kernel needed)

---

## Submission History

| # | Model | OOF MAE | LB MAE | Notes |
|---|-------|:-------:|:------:|-------|
| 1 | Ridge+Lasso+GBM stack (92 feat) | ~0.129 (log RMSE) | 14184 | Ported from House-Prices |
| 2 | LightGBM + Lasso 60/40 blend | 15042 | 13499 | |
| 3 | Boosting stack (5 models, Ridge meta) | 14616 | **13202** | **New best** |

## Experiment 3: Boosting ensemble + OOF stack (Phase 3)

**Date:** 2026-08-27
**OOF MAE:** 14,616 | **LB MAE:** 13,202 (new best)

### Approach
Reused the proven 92-feature engineering (log-skew transform, ordinal quality
encoding, polynomial interactions), then built an OOF stack of 5 base models
with a Ridge meta-learner:

| Base model | OOF MAE |
|---|---|
| XGBoost | 14,742 |
| CatBoost | 14,779 |
| LightGBM | 15,336 |
| Ridge | 17,234 |
| Lasso | 17,187 |
| **Stacked (Ridge meta)** | **14,616** |

### Key finding — stacking worked here (unlike Spaceship-Titanic)
On Spaceship, stacking four boosters *underperformed* the best single model because
they were highly correlated. Here it **beat every base model** (14,616 vs XGB's
14,742). The difference: the linear models (Ridge/Lasso) make genuinely different
errors than the trees, so the meta-learner has real diversity to exploit. Lesson:
stacking pays off when bases are diverse (tree + linear), not when they're similar
(four boosters).

### Next levers
- Target encoding for high-cardinality categoricals (Neighborhood)
- Drop the two known GrLivArea outliers (>4000 sqft, low price)
- Tune per-model hyperparameters (currently reasonable defaults)
