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

## 2026-11-26 - Experiment 2: feature engineering + LGB/XGB rank blend
**Approach:** notebooks/fe_blend.py. 12 new features (amount/balance ratio, amount x dist/velocity, hour sin/cos, risk-flag sum), LGBM (31 leaves) + XGB (depth 5), rank average.
**Result (VERIFIED, OOF):** LGB 0.94624, XGB 0.94648, blend **0.94661** (baseline 0.94594). Small gain; ratios add little, so signal is mostly in raw columns.
**Next levers:** gap to top (~0.953) is not from FE. Try OOF target encoding on categoricals, CatBoost, deeper/longer trees, original-dataset augmentation (check Code tab for the source dataset).

## 2026-11-26 (night) - Experiment 3: CatBoost + rank blend
**Approach:** notebooks/cat_te_blend.py. CatBoost depth 6, native categoricals, 5-fold. About 5 min per fold on CPU.
**Result (VERIFIED):** Cat OOF 0.94565 (weaker alone than LGB/XGB). Blend with FE blend: w_cat 0.3 -> OOF 0.94676, 0.5 -> 0.94666, 0.7 -> 0.94638. Submitted w=0.3: **LB 0.94719** (was 0.94705).
**Takeaway:** CatBoost adds only +0.00015 OOF. Boosters on these features are saturated near 0.9468. Remaining gap (~0.008) needs new data or signal, not more models.
**Next levers:** find original source dataset and add as training data; seed averaging; check public top notebooks for the trick.
