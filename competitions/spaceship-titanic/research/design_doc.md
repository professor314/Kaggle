# Spaceship Titanic — Design Doc

**Competition:** Spaceship Titanic
**Metric:** Accuracy | **Best LB:** 0.8024 (rank ~639; plateaued)
**Full research:** see `research_v2_findings.md` and `../experiments_log.md`

## Domain research
Predict whether a passenger was "Transported." Strongest signal is CryoSleep
(~72% accuracy alone), then spending behavior (cryo passengers don't spend),
cabin deck, and age. 8,700 rows tolerates ~50 features without overfitting.

## Prior art
HistGradientBoosting-style models with light feature engineering; big warning
about target-leakage features (group-survival-rate style encodings).

## Feature engineering plan (what shipped)
- Structural group features: GroupSize, IsAlone (NOT group survival rate).
- Spending ratios + per-amenity spent flags + NumServicesUsed.
- Cabin region binning, granular 7-bin age groups.
- HomePlanet × CryoSleep interactions.

## Design decisions
| Decision | Choice | Rationale |
|---|---|---|
| Model | HistGradientBoostingClassifier | Native NaN handling, regularized, fast; best CV (0.8119) |
| Leakage guard | Drop GroupSurvivalRate | CV 95% / LB 0.49 = textbook leakage |
| CV | 5-fold, track CV-LB gap | Gap stayed ~1%, features validated |
| Ensemble | Stacking boosters | Did NOT help (bases too correlated) |

## Key finding
PLATEAUED at ~0.80 with tree models. Stacking correlated boosters gave nothing.
To break through needs NEW signal (surname/family groups, CryoSleep imputation,
leakage-free OOF group encoding), not more models.
