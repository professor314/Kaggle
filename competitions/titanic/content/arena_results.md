## Model Arena Results

**Folds:** 5 | **Metric:** accuracy | **Entries:** 21

| Rank | Name | CV Score | ± Std | Stability | Worthiness | Features | Model |
|------|------|----------|-------|-----------|------------|----------|-------|
| 1 | GBM_strong | 0.8575 | 0.0099 | 0.9884 | 0.7833 | 8 | GradientBoostingClassifier |
| 2 | GBM_moderate | 0.8563 | 0.0175 | 0.9796 | 0.7751 | 8 | GradientBoostingClassifier |
| 3 | GBM_interaction | 0.8462 | 0.0154 | 0.9818 | 0.7670 | 12 | GradientBoostingClassifier |
| 4 | RF_conservative | 0.8328 | 0.0082 | 0.9901 | 0.7602 | 8 | RandomForestClassifier |
| 5 | RF_moderate | 0.8316 | 0.0094 | 0.9887 | 0.7580 | 8 | RandomForestClassifier |
| 6 | GBM_simple | 0.8316 | 0.0108 | 0.9871 | 0.7567 | 4 | GradientBoostingClassifier |
| 7 | GBM_very_conservative | 0.8328 | 0.0126 | 0.9849 | 0.7562 | 4 | GradientBoostingClassifier |
| 8 | GBM_conservative | 0.8305 | 0.0108 | 0.9870 | 0.7556 | 8 | GradientBoostingClassifier |
| 9 | RF_shallow | 0.8283 | 0.0104 | 0.9875 | 0.7537 | 8 | RandomForestClassifier |
| 10 | RF_interaction | 0.8260 | 0.0082 | 0.9901 | 0.7535 | 12 | RandomForestClassifier |
| 11 | LR_interaction | 0.8272 | 0.0128 | 0.9845 | 0.7504 | 12 | LogisticRegression |
| 12 | DT_depth4 | 0.8283 | 0.0158 | 0.9809 | 0.7487 | 8 | DecisionTreeClassifier |
| 13 | DT_depth3 | 0.8249 | 0.0149 | 0.9819 | 0.7462 | 8 | DecisionTreeClassifier |
| 14 | LR_regularized | 0.8148 | 0.0136 | 0.9833 | 0.7373 | 8 | LogisticRegression |
| 15 | LR_moderate | 0.8115 | 0.0103 | 0.9874 | 0.7370 | 8 | LogisticRegression |
| 16 | LR_simple | 0.8070 | 0.0114 | 0.9859 | 0.7315 | 4 | LogisticRegression |
| 17 | RF_simple | 0.8092 | 0.0187 | 0.9769 | 0.7270 | 4 | RandomForestClassifier |
| 18 | LR_minimal | 0.7868 | 0.0188 | 0.9761 | 0.7045 | 2 | LogisticRegression |
| 19 | DT_depth2 | 0.7766 | 0.0117 | 0.9850 | 0.7009 | 4 | DecisionTreeClassifier |
| 20 | KNN_5_moderate | 0.7340 | 0.0125 | 0.9829 | 0.6576 | 8 | KNeighborsClassifier |
| 21 | KNN_9_moderate | 0.7340 | 0.0172 | 0.9765 | 0.6533 | 8 | KNeighborsClassifier |

**Recommendation:** Submit `GBM_strong` (estimated LB: 0.7925)