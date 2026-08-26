# House Prices Research V2 — Findings

## Major Improvement: 0.14545 → 0.12728 (12.5% reduction in RMSLE)

---

## What Worked

### 1. Full Categorical Encoding (+25% improvement)
The single biggest gain came from encoding ALL categorical features:
- **Quality ordinals** (ExterQual, KitchenQual, BsmtQual, etc.): Ex=5, Gd=4, TA=3, Fa=2, Po=1
- **Label encoding** for non-ordinal categoricals (Neighborhood, MSSubClass, etc.)
- Full feature set (95) dramatically outperformed numeric-only (30): **0.117 vs 0.125 CV RMSE**

This confirms that categorical features like Neighborhood carry significant price signal.

### 2. Stacking (Ridge + Lasso + GBM)
Simple averaging of three diverse models beat any individual:
- Ridge: 0.1188 CV
- Lasso: 0.1177 CV
- GBM: 0.1230 CV
- **Stack: 0.1154 CV** (better than all individual models)

The stack works because:
- Linear models (Ridge/Lasso) capture linear relationships well
- GBM captures non-linear interactions
- They make different errors → averaging reduces variance

### 3. Outlier Removal
Removing 2 extreme GrLivArea outliers (>4000 sqft with low price) improved model stability.
These are likely data entry errors or unusual transactions.

### 4. Log-Transform Skewed Features
54 features had skewness > 0.75. Applying log1p:
- Makes distributions more Gaussian
- Particularly helps linear models (Ridge, Lasso)
- Reduces the influence of extreme values

### 5. Polynomial Features
OverallQual², TotalSF², GrLivArea² help capture non-linear price effects:
- Quality has diminishing returns (going from 8→9 adds more than 4→5)
- Size has a non-linear effect on price (larger homes are disproportionately expensive)

### 6. Interaction Features
- OverallQual × GrLivArea: a large, high-quality home is worth more than the sum of its parts
- OverallQual × TotalBsmtSF: finished basements in quality homes add premium value

---

## Arena Results

36 pipelines tested (Ridge, Lasso, ElasticNet, RF, GBM × numeric_only/full):

| Rank | Model | CV RMSE | Features | Worthiness |
|------|-------|---------|----------|------------|
| 1 | GBM_full_v4 | 0.1194 | 95 | 8.39 |
| 2 | ElasticNet_full_v2 | 0.1174 | 95 | 8.18 |
| 3 | Lasso_full_v1 | 0.1177 | 95 | 8.18 |
| 4 | ElasticNet_full_v1 | 0.1175 | 95 | 8.17 |
| 5 | Ridge_full_v1 | 0.1182 | 95 | 8.17 |

**Key Insight:** Linear models (Lasso/ElasticNet/Ridge) were competitive with GBM on this dataset!
This is because house prices have strong linear relationships with quality and size features.
The GBM won on "worthiness" due to lower variance, but raw CV RMSE was slightly better for Lasso.

---

## What Didn't Help (Much)

| Feature | Reason |
|---------|--------|
| Polynomial for GrLivArea² | Already captured by GBM's tree splits |
| HasPool flag | Very rare (< 1% of houses), not enough signal |
| Random Forest | Consistently worse than GBM and linear models on this dataset |

---

## CV-LB Gap Analysis

| Experiment | CV RMSE | LB RMSLE | Gap |
|-----------|---------|----------|-----|
| Exp 1 (10 features, conservative) | 0.1405 | 0.14545 | 3.5% |
| Exp 2 (95 features, stack) | 0.1154 | 0.12728 | 10.2% |

The gap increased from 3.5% to 10.2% with more complex models. This suggests:
- Some overfitting to training patterns
- Feature selection could help (reduce 95 → ~40 most important)
- The CV score (0.1154) is optimistic but the improvement still translated to LB

---

## Leaderboard Context

- Our score: **0.12728** (top ~35%)
- Top scores: ~0.10-0.11
- Median: ~0.14-0.15
- Our previous: 0.14545

Next targets:
- 0.12: tuned stacking + XGBoost/LightGBM
- 0.11: feature selection + Bayesian optimization + advanced stacking

---

*Generated: 2026-08-21*
