# Playground Series S6E11: credit card fraud

## 1. Problem and metric
- Predict `is_fraud` (positive rate 0.0167) from 24 transaction features. 699,960 train rows, 299,984 test rows.
- Submission is a probability per `id`. Public LB tracks out-of-fold (OOF) AUC within about 0.001, so the metric is AUC-like (VERIFIED by behavior; not yet read on the Overview tab).
- Deadline 2026-11-30.

## 2. Results

| # | Approach | OOF AUC | Public LB | Status |
|---|---|---|---|---|
| 1 | LightGBM, raw features | 0.94594 | 0.94673 | VERIFIED |
| 2 | +12 engineered features, LGB + XGB rank blend | 0.94661 | 0.94705 | VERIFIED |
| 3 | +CatBoost (native categoricals), 3-way rank blend w=0.3 | 0.94676 | 0.94719 | VERIFIED |
| 4 | 4-leaf LGBM + in-fold TE + original rows, blend w=0.7 with #3 | 0.94795 | 0.94843 | VERIFIED |

## 3. Design decisions
- **CV: 5-fold stratified, seed 42.** Why: rows are independent transactions, and the 1.7% positive rate needs stratification. Evidence: LB tracks OOF within about 0.001.
- **Features: ratios, interactions, hour sin/cos, risk-flag count.** Why: fraud is unusual behavior relative to the account (amount vs balance, distance x velocity). Evidence: only +0.0003 to +0.0007 OOF, so most signal is already in the raw columns.
- **Categorical handling: shared integer codes for LGB/XGB, native handling for CatBoost.** Why: CatBoost's ordered target statistics are leakage-safe target encoding, the lever that broke the House Prices plateau.
- **Ensemble: rank average, not probability average.** Why: AUC depends only on ranking, and boosters calibrate differently.
- **CatBoost blend weight 0.3.** Alone it is weaker (OOF 0.94565) and only adds +0.00015 in the blend. Weight chosen by OOF over 0.3/0.5/0.7.
- **Rejected: more LGB-style models.** S6E9 showed correlated boosters plateau. Diversity matters more.
- **Gap to top (about 0.008 AUC)** is not explained by features. Next suspects are extra training data (original dataset) and deeper or longer-trained models.
- **Shallow trees (4 leaves).** Fold-0: 4 leaves 0.94565 vs 31 leaves 0.94425. The target is close to a sum of per-column effects, so deep trees fit noise.
- **In-fold target encoding on exact values.** sklearn TargetEncoder fitted only on each training fold (cross-fitted), so no label leakage. Gives a 4-leaf tree a per-value fraud curve.
- **Original dataset appended to training folds only.** Competition data is synthetic from uditjain13/credit-card-fraud-detection-2026. Gain +0.0001 to +0.0002. Local copy: C:\Users\profe\kaggle-data\s6e11-original.
- **Not copying public submission blends.** Top public notebooks (~0.9575) mostly rank-mix other authors' prediction files. We build our own models.

## 4. Toolkit reuse
Ported from `playground-s6e9/notebooks/lgbm_pipeline.py`. Nothing new added to `kaggle_ml_toolkit` yet. Candidate: a shared `rank_blend` helper.

## 5. Reproduce
Data lives in `data/` (git-ignored). Use `C:\Users\profe\anaconda3\python.exe`.
1. `notebooks/lgbm_pipeline.py` (about 5 min)
2. `notebooks/fe_blend.py` (about 6 min)
3. `notebooks/cat_te_blend.py` (about 25 min, CPU)
4. `notebooks/shallow_te_full.py` (about 5 min; needs the original CSV)

## 6. What's next
1. Pair-crossed target encodings among the 10 strongest columns.
2. Shallow XGB and CatBoost members; a spline GLM.
3. RealMLP/TabM neural nets (needs torch; run as a Kaggle GPU kernel).
4. Seed-average the best blend. Competition closes 2026-11-30.
