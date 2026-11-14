# Playground Series S6E11 (credit card fraud, tabular) - Research & Design Doc

**Kaggle ref:** playground-series-s6e11
**Deadline:** 2026-11-30 (VERIFIED 2026-11-12 via kaggle competitions list) | **Reward:** Swag | **Teams:** 1637
**Type:** Tabular binary classification (fraud)
**Status:** First pass, 2026-11-12. Rules accepted 2026-11-12; data downloaded. Baseline submitted, see experiments_log.md.

## 1. What we know (file list only, no data downloaded)
Files: train.csv 98.5MB, test.csv 41MB, sample_submission.csv 8.4MB

## 2. Prior art and facts
Top LB 0.95465 (VERIFIED: kaggle competitions leaderboard). Public notebooks titled 'Credit Card Fraud' report LB 0.9524-0.9530 using shallow-tree stacks / XGB blends (VERIFIED: kaggle kernels list). Metric is ASSUMED AUC-like because scores sit near 0.95; confirm from the Overview tab.

## 3. Reuse from existing code
Port competitions/playground-s6e9/notebooks/lgbm_pipeline.py (template written at notebooks/lgbm_pipeline.py, auto-detects target from sample_submission). Then boosting_blend.py and te_linear_blend.py from S6E9. Toolkit: model_arena.py, ensemble_builder.py.

## 4. Design decisions
| Decision | Choice | Status |
|---|---|---|
| Metric | Read from Overview/Evaluation tab | ASSUMED until read |
| Baseline | Reproduce a top public notebook before any custom work | planned |
| CV | Match metric, keep OOF to compare against LB | planned |

## 5. Next steps
1) Accept rules on kaggle.com. 2) kaggle competitions download. 3) Run notebooks/lgbm_pipeline.py for a baseline OOF. 4) Read the top public stacks (Code tab) before blending. S6E9 lesson: correlated boosters plateau; try OOF target encoding early.

