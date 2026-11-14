# Playground Series S6E11 (credit card fraud, tabular) - Experiment Log

**Kaggle ref:** playground-series-s6e11 | **Deadline:** 2026-11-30

## 2026-11-12 - First pass (research only)
- Verified deadline, reward, team count, file list via kaggle CLI.
- Listed top public notebooks via kaggle kernels list (see design_doc.md section 2).
- Tried data download: not possible, rules not accepted (403 on DownloadDataFiles) - for S6E11 confirmed directly; the others assumed the same until the account joins.
- No submissions made. No experiments run.
- Priority: 1 (best fit, closes soonest)

## 2026-11-12 (night) - Experiment 1: LightGBM 5-fold baseline
**Data (VERIFIED):** 699,960 train / 299,984 test, 24 features (5 categoricals, 6 bools), target `is_fraud`, pos rate 0.0167. Sample submission is probabilities.
**Approach:** competitions/playground-s6e11/notebooks/lgbm_pipeline.py (port of S6E9 pipeline), shared category codes, 5-fold StratifiedKFold, 2000 trees lr 0.03, 63 leaves, early stop 100. No feature engineering.
**Result:** OOF AUC 0.94594 (folds 0.941-0.951), OOF AUPRC 0.50506. **Public LB 0.94673** (submission 58298326). VERIFIED: LB tracks OOF AUC, so metric is AUC-like (still confirm on Overview tab).
**Gap to top:** leader 0.95465, public stacks ~0.953 (VERIFIED earlier). Gap ~0.008.
**Next levers:** read top public stack notebooks; engineered features (amount vs balance, velocity x distance, hour cyclic); XGB/CatBoost; OOF target encoding (helped House-Prices); rank-blend.
