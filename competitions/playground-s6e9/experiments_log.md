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

## Submission History

| # | Model | OOF AUC | LB AUC | Notes |
|---|-------|:-------:|:------:|-------|
| 1 | LGBM 5-fold + engineered | (not captured) | **0.94161** | First entry |
