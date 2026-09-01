# House Prices — Design Doc

**Competition:** House Prices – Advanced Regression Techniques
**Metric:** RMSLE | **Best LB:** 0.1235 (top ~35%)
**Full research:** see `research_v2_findings.md` and `../experiments_log.md`

## Domain research
Ames, Iowa housing. Price is driven mainly by overall quality and living area,
with strong neighborhood effects. Classic regression benchmark; linear models
are surprisingly competitive because quality/size relate near-linearly to price.

## Prior art
Top Kaggle solutions: aggressive categorical encoding, log-transform of the
target and skewed features, outlier removal (the two >4000 sqft low-price
GrLivArea points), and stacked linear + boosting ensembles.

## Feature engineering plan (what shipped)
- Ordinal encoding of quality features (Ex=5 … Po=1); label-encode nominal
  categoricals. Full 95-feature set beat numeric-only (0.117 vs 0.125 CV).
- log1p on 54 skewed features; log target.
- Polynomial (OverallQual², TotalSF²) and interactions
  (OverallQual×GrLivArea, OverallQual×TotalBsmtSF).
- Remove 2 GrLivArea outliers.

## Design decisions
| Decision | Choice | Rationale |
|---|---|---|
| Target transform | log(SalePrice) | Metric is RMSLE; log makes it RMSE |
| Encoding | Ordinal + label, full feature set | +25% gain over numeric-only |
| Models | Ridge + Lasso + GBM, later LGB/XGB/Cat blend | Linear + tree diversity |
| Ensemble | Averaged/blended stack | Stack CV 0.1154 beat all singles |
| CV | KFold RMSE on log target | Matches metric |
| Excluded | Random Forest, HasPool flag | RF worse; pool too rare (<1%) |

## CV-LB gap
Gap widened from 3.5% (simple) to 10.2% (95-feature stack) — some overfitting.
Next lever: feature selection (95 → ~40) + tuned boosting to push toward 0.11.
