# Spaceship Titanic — Arena Results

## Model Arena Results

**Folds:** 5 | **Metric:** accuracy | **Entries:** 9

| Rank | Name | CV Score | ± Std | Stability | Worthiness | Features | Model |
|------|------|----------|-------|-----------|------------|----------|-------|
| 1 | GBM_moderate | 0.8073 | 0.0065 | 0.9920 | 0.7810 | 22 | GradientBoostingClassifier |
| 2 | GBM_strong | 0.8065 | 0.0068 | 0.9916 | 0.7799 | 22 | GradientBoostingClassifier |
| 3 | RF_extended | 0.7990 | 0.0058 | 0.9927 | 0.7734 | 22 | RandomForestClassifier |
| 4 | LR_extended | 0.7925 | 0.0100 | 0.9874 | 0.7627 | 22 | LogisticRegression |
| 5 | GBM_core_strong | 0.7511 | 0.0060 | 0.9920 | 0.7252 | 11 | GradientBoostingClassifier |
| 6 | GBM_conservative | 0.7472 | 0.0083 | 0.9889 | 0.7191 | 11 | GradientBoostingClassifier |
| 7 | RF_moderate | 0.7424 | 0.0052 | 0.9930 | 0.7174 | 11 | RandomForestClassifier |
| 8 | RF_conservative | 0.7389 | 0.0056 | 0.9925 | 0.7134 | 11 | RandomForestClassifier |
| 9 | LR_core | 0.7309 | 0.0045 | 0.9939 | 0.7066 | 11 | LogisticRegression |

**Recommendation:** Submit `GBM_moderate` (estimated LB: 0.7873)

**Warnings:**
- ⚠️ Many features (22). Risk of overfitting on small datasets.