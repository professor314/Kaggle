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
