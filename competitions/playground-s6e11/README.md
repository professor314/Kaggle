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

## 3. Design decisions
- **CV: 5-fold stratified, seed 42.** Why: rows are independent transactions, and the 1.7% positive rate needs stratification. Evidence: LB tracks OOF within about 0.001.
- **Features: ratios, interactions, hour sin/cos, risk-flag count.** Why: fraud is unusual behavior relative to the account (amount vs balance, distance x velocity). Evidence: only +0.0003 to +0.0007 OOF, so most signal is already in the raw columns.
- **Categorical handling: shared integer codes for LGB/XGB, native handling for CatBoost.** Why: CatBoost's ordered target statistics are leakage-safe target encoding, the lever that broke the House Prices plateau.
- **Ensemble: rank average, not probability average.** Why: AUC depends only on ranking, and boosters calibrate differently.
- **CatBoost blend weight 0.3.** Alone it is weaker (OOF 0.94565) and only adds +0.00015 in the blend. Weight chosen by OOF over 0.3/0.5/0.7.
- **Rejected: more LGB-style models.** S6E9 showed correlated boosters plateau. Diversity matters more.
- **Gap to top (about 0.008 AUC)** is not explained by features. Next suspects are extra training data (original dataset) and deeper or longer-trained models.

## 4. Toolkit reuse
Ported from `playground-s6e9/notebooks/lgbm_pipeline.py`. Nothing new added to `kaggle_ml_toolkit` yet. Candidate: a shared `rank_blend` helper.

## 5. Reproduce
Data lives in `data/` (git-ignored). Use `C:\Users\profe\anaconda3\python.exe`.
1. `notebooks/lgbm_pipeline.py` (about 5 min)
2. `notebooks/fe_blend.py` (about 6 min)
3. `notebooks/cat_te_blend.py` (about 25 min, CPU)

## 6. What's next
1. Look for the original source dataset and add it as training data.
2. Tune depth and leaves for the deeper-model idea.
3. Seed-average the best blend.
