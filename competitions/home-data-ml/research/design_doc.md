# Home Data for ML Course (Ames) — Design Doc

**Competition:** Housing Prices Competition for Kaggle Learn Users
**Metric:** MAE | **Best LB:** 13,202 (rank ~111, top ~3%)
**Full log:** see `../experiments_log.md`

## Domain research
Same Ames housing data as House-Prices but scored on MAE (not RMSLE) and using
the Kaggle Learn split. Reused the House-Prices research and feature pipeline.

## Prior art
Boosting ensembles dominate; the differentiator here was that stacking tree +
linear bases beats any single model because their errors are genuinely diverse.

## Feature engineering plan (what shipped)
Ported the 92-feature House-Prices pipeline: log-skew transforms, ordinal
quality encoding, polynomial + interaction features.

## Design decisions
| Decision | Choice | Rationale |
|---|---|---|
| Models | XGB + CatBoost + LGB + Ridge + Lasso | Tree + linear diversity |
| Meta-learner | Ridge on OOF predictions | Simple, robust stacker |
| Ensemble | 5-model OOF stack | Beat every base (14,616 vs XGB 14,742 OOF) |
| CV | KFold, OOF stacking | Honest meta-training, no leakage |

## Key finding
Stacking paid off here (unlike Spaceship-Titanic) precisely because the bases
were diverse (trees + linear). Correlated bases (four boosters) do not stack
well. Next levers: target encoding for Neighborhood, drop GrLivArea outliers,
per-model tuning.
