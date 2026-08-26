# House Prices — Arena Results

## Model Arena Results

**Folds:** 5 | **Metric:** rmse | **Entries:** 12

| Rank | Name | CV Score | ± Std | Stability | Worthiness | Features | Model |
|------|------|----------|-------|-----------|------------|----------|-------|
| 1 | GBM_very_conservative | 0.1405 | 0.0162 | 0.8844 | 6.7770 | 10 | GradientBoostingRegressor |
| 2 | GBM_conservative | 0.1410 | 0.0181 | 0.8719 | 6.6552 | 10 | GradientBoostingRegressor |
| 3 | GBM_moderate | 0.1392 | 0.0197 | 0.8586 | 6.6474 | 15 | GradientBoostingRegressor |
| 4 | GBM_strong | 0.1402 | 0.0218 | 0.8444 | 6.4870 | 15 | GradientBoostingRegressor |
| 5 | RF_moderate | 0.1455 | 0.0190 | 0.8691 | 6.4158 | 15 | RandomForestRegressor |
| 6 | RF_conservative | 0.1609 | 0.0168 | 0.8953 | 5.9321 | 10 | RandomForestRegressor |
| 7 | RF_shallow | 0.1679 | 0.0156 | 0.9073 | 5.7446 | 15 | RandomForestRegressor |
| 8 | Ridge_extended | 0.1584 | 0.0388 | 0.7551 | 5.0898 | 15 | Ridge |
| 9 | ElasticNet | 0.1585 | 0.0388 | 0.7550 | 5.0854 | 15 | ElasticNet |
| 10 | Ridge_strong_reg | 0.1584 | 0.0396 | 0.7502 | 5.0558 | 15 | Ridge |
| 11 | Ridge_core | 0.1615 | 0.0399 | 0.7526 | 4.9692 | 10 | Ridge |
| 12 | Lasso_core | 0.1615 | 0.0402 | 0.7510 | 4.9567 | 10 | Lasso |

**Recommendation:** Submit `GBM_very_conservative` (estimated LB: 0.1305)

**Warnings:**
- ⚠️ Estimated LB (0.1305) may be near or below gender baseline.