# Playground S6E9 (Will_Buy_EV) — Experiment Log

**Competition:** Kaggle Playground Series S6E9
**Task:** Binary classification, predict `Will_Buy_EV` (EV purchase intent)
**Metric:** AUC (probability target) | **Deadline:** 2026-09-30
**Data:** 668,665 train / 286,571 test, 14 features, pos rate 0.1746

---

## Experiment 1: LightGBM 5-fold + engineered features

**Date:** 2026-09-01
**LB (public) AUC:** 0.94161 | **first submission**

### Approach
- Target Yes/No -> 1/0.
- Shared train+test categorical encoding (codes computed on the concatenation).
  NOTE: pandas 3 reports text columns as `str` dtype, so detect categoricals via
  `not is_numeric_dtype(col)`, not `== "object"`.
- Engineered features: Charging_Access (home+work stations), Income_per_Car,
  Commute_x_Cars.
- 5-fold StratifiedKFold LightGBM (2000 trees, lr 0.03, 63 leaves, early stop 100),
  averaged test probabilities.

### Next levers
- Add XGBoost + CatBoost and blend (S6E8 playbook).
- More interactions (Range_Anxiety x Charging_Access, Subsidy x Income).
- Tune LightGBM; check public/private split behavior (trust OOF over LB).
- Note: OOF AUC print was lost when the run was detached to the background; the
  0.94161 is the public LB. Re-run capturing OOF for CV-LB gap analysis.

## Experiment 2: LGB + XGB + CatBoost blend

**Date:** 2026-09-01
**Per-model OOF AUC:** lgb 0.94169 / xgb 0.94182 / cat 0.94171
**Blends:** mean 0.94205, rank 0.94206 (chosen) | **LB AUC:** 0.94169

The three boosters were nearly identical (~0.9417), so blending only nudged OOF
(+0.0004) and LB barely moved (0.94161 -> 0.94169). Correlated bases don't stack
well (same lesson as Spaceship-Titanic and House-Prices).

### What would actually move it (not more boosters)
- New signal: stronger interactions (Range_Anxiety x Charging_Access, Subsidy x
  Income x City_Type), target/count encoding of the categoricals with OOF.
- A genuinely different base (linear model, or a small MLP/TabNet) for real
  diversity in the blend.
- Hyperparameter tuning per model (Optuna) rather than reasonable defaults.

## Submission History

| # | Model | OOF AUC | LB AUC | Notes |
|---|-------|:-------:|:------:|-------|
| 1 | LGBM 5-fold + engineered | (not captured) | 0.94161 | First entry |
| 2 | LGB+XGB+Cat rank blend | 0.94206 | **0.94169** | Correlated bases; marginal gain |
| 3 | LGBM big-iter (lr0.015, 8000, rich feats) | 0.94171 | 0.94138 | More compute didn't help |
| 4 | OOF target-encode + Logistic base, rank blend | 0.94170 | 0.94160 | The "diverse base" lever — still no gain |

## Experiment 3: LGBM big-iteration (2026-09-02)

Used the new all-cores compute defaults to run lr 0.015 / up to 8000 rounds with
extra interaction features. OOF AUC **0.94171** = no gain over the baseline.
Early stopping fired at ~1000-1300 trees, so more rounds were never the binding
constraint. **S6E9 is not compute-bound** either; gains need OOF target encoding
or a non-booster base, not more iterations.

## Experiment 4: OOF target encoding + linear base (2026-09-27)

Ran the exact "what would actually move it" lever from Experiments 1-2:
leakage-safe **OOF mean-target encoding** of all 6 categoricals (smoothing=20),
fed into (a) LightGBM on label+TE features and (b) a **Logistic Regression** on
numeric+TE features as a decorrelated base, then a weight-searched rank blend.
Script: `notebooks/te_linear_blend.py` (runs on the anaconda3 python; `.venv-dl`
is gone).

Results:
- LGBM (label+TE): OOF AUC **0.94168**
- Logistic (num+TE): OOF AUC **0.93803**
- base rank-corr (LGB vs Logistic): **0.9864** — the "different" base is still
  highly correlated with the trees, so it adds almost no diversity.
- Best rank blend (w_lgb=0.95): OOF AUC **0.94170**; **LB 0.94160**.

**Conclusion — S6E9 is a genuine plateau at ~0.9417.** All three levers have now
been tried and exhausted: more compute (Exp 3), more/correlated boosters (Exp 2),
and target encoding + a linear base (Exp 4). None break 0.9417 on OOF or LB. The
synthetic feature set is signal-saturated for this pipeline. Remaining unexplored
(low expected value): appending the "original" source dataset if Kaggle published
one, or a truly different architecture (TabNet/MLP). Deadline 2026-09-30. Leaving
best submission as-is; not worth more runway.
