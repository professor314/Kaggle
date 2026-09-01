# Playground Series S6E8 — Design Doc

**Competition:** Kaggle Playground Series, Season 6 Episode 8
**Metric:** AUC | **Best LB:** 0.96536 (rank ~1610)
**Target:** `addicted_label` (binary) | **Deadline:** 2026-08-31

## Domain research
Synthetic tabular binary classification (predicting an "addicted" label from
demographic/behavioral columns like gender, stress_level, academic_work_impact,
and usage features). Standard Playground synthetic set: gradient boosting with
light feature work is the winning shape.

## Prior art
Playground episodes reward clean 5-fold CV, tuned LightGBM, and modest blending
of boosters. Deep feature engineering rarely beats a well-tuned booster here.

## Feature engineering plan (what shipped)
- Category-encode gender, stress_level, academic_work_impact (shared train+test
  categorical codes to keep encodings consistent).
- 21 engineered features feeding a tuned LightGBM.

## Design decisions
| Decision | Choice | Rationale |
|---|---|---|
| Primary model | Tuned LightGBM (lr=0.03, 21 features) | Best single-model OOF AUC 0.965 |
| Diversity | + XGBoost + CatBoost blend | Probability blend, weights tuned on OOF AUC |
| CV | StratifiedKFold, 5 splits | Binary target, matches AUC evaluation |
| Ensemble | OOF-weighted probability blend | Standard robust Playground move |

## Status
Iterated to 0.96536 AUC. Closes 2026-08-31. Further gains would need new feature
signal or a wider ensemble; low ROI at deadline. Successor: S6E9 (2026-09-30).
