# Playground Series S6E9 — Research & Design Doc

**Competition:** Kaggle Playground Series, Season 6 Episode 9
**Deadline:** 2026-09-30
**Reward:** Swag (leaderboard/knowledge)
**Status:** Scaffolded. BLOCKED on rules acceptance (data download 403s until the
account joins the competition on the website).

---

## 1. Domain Research

Playground Series competitions use synthetically generated tabular datasets
modeled on a real-world dataset. The exact target and features for S6E9 are not
yet known here because the data has not been downloaded (see blocker below).

**To fill in once data is available:**
- What is the real-world dataset this episode is based on? (check the competition
  overview + "Dataset Description" tab)
- Is it classification or regression? What is the evaluation metric?
- Class balance / target distribution.

## 2. Prior Art

- Playground episodes reward: clean CV, strong gradient boosting (LightGBM /
  XGBoost / CatBoost), light feature engineering, and careful blending. Deep
  feature engineering rarely beats a well-tuned boosting ensemble on these
  synthetic sets.
- Check the Code + Discussion tabs for S6E9 specifically once joined; early
  high-vote notebooks usually reveal the metric quirks and any leakage.
- Transfer from our own history: our tuned single LightGBM took S6E8 to 0.965
  AUC. Same playbook should port here.

## 3. Feature Engineering Plan (provisional)

1. Baseline: raw features + label/target encoding for categoricals.
2. Standard synthetic-data moves: count-encoding, interaction features between
   the top importance columns, and (if a known "original" dataset exists)
   concatenating the original data as extra training rows.
3. Keep it disciplined: add a feature only if it improves OOF CV.

## 4. Design Decisions (to be logged as they're made)

| Decision | Choice | Rationale |
|---|---|---|
| CV strategy | (TBD: StratifiedKFold for classification / KFold for regression) | Match the metric; 5 folds standard |
| Model family | LightGBM first, then XGBoost + CatBoost for the blend | Proven on prior Playground episodes |
| Metric | (TBD from overview) | Optimize CV on the exact competition metric |
| Ensemble | Rank-average or simple mean of boosters | Cheap, robust, standard for Playground |
| Excluded | Deep learning | Overkill for synthetic tabular; boosting dominates |

## 5. Hypotheses (provisional, refine after EDA)

1. A single tuned LightGBM lands in a competitive band; the blend adds a small
   but real margin.
2. One or two engineered interaction features move CV; most do nothing.
3. If Kaggle published the "original" source dataset, appending it to train is
   the single biggest lever.
4. The public/private split will punish overfit feature sets, so trust OOF CV
   over LB.

---

## Blocker / Next Action

`kaggle competitions download playground-series-s6e9` returns **403 Forbidden**
because the account has not accepted the competition rules. Manual step:

1. Open https://www.kaggle.com/competitions/playground-series-s6e9
2. Click **Join Competition** and accept the rules.
3. Then: `kaggle competitions download playground-series-s6e9 -p competitions/playground-s6e9/data`
4. Unzip, run EDA, fill in sections 1-2, then execute the standard pipeline.
