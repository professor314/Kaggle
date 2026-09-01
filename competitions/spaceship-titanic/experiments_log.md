# Spaceship Titanic — Experiment Log

**Competition:** Spaceship Titanic
**Metric:** Accuracy
**Dataset:** 8693 train rows, 14 features (raw), 4277 test rows

---

## Experiment 1: Arena Run (9 pipelines)

**Date:** 2026-08-20
**LB Score:** 0.79939
**CV Score:** 0.8073
**CV-LB Gap:** 0.8% (excellent — validates larger dataset = less overfitting)

### Winner: GBM_moderate
```python
model = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    min_samples_leaf=8, subsample=0.8, random_state=42
)
features = ['CryoSleep', 'TotalSpending', 'LogSpending', 'NoSpending', 'Age', 'IsChild',
            'VIP', 'GroupSize', 'IsAlone', 'DeckNum', 'IsPort',
            'FromEarth', 'FromEuropa', 'FromMars', 'ToTRAPPIST', 'To55Cancri',
            'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNum']
```

### Key Features Engineered
- **CryoSleep** — Binary flag (strongest predictor: cryo passengers almost always transported)
- **TotalSpending** — Sum of all luxury amenities (RoomService + FoodCourt + ShoppingMall + Spa + VRDeck)
- **NoSpending** — Binary flag for zero spending (strong signal — suggests cryo sleep)
- **LogSpending** — Log-transformed spending (right-skewed distribution)
- **GroupSize** — Number of people in travel group (from PassengerId parsing)
- **DeckNum** — Cabin deck as ordinal (from Cabin field parsing: deck/num/side)
- **IsPort** — Port side vs Starboard (from Cabin field)
- **HomePlanet / Destination** — One-hot encoded

### What Worked
- **CryoSleep is the dominant feature** — passengers in cryo were almost always transported
- **Spending features** — non-spenders correlate heavily with being transported (cryo = no spending)
- **22 features on 8700 rows** — appropriate complexity from strategies.md guidelines
- **Extended feature set** — 22 features consistently outperformed 11 core features (CV 0.80 vs 0.75)
- **GBM with moderate depth (4)** — enough capacity for the larger dataset
- **Tiny CV-LB gap (0.8%)** — confirms no overfitting on larger datasets

### What We Learned (New to Strategies)
- On 8700+ rows, you CAN use more features and more complex models without overfitting
- The CV-LB gap shrinks dramatically with dataset size: Titanic (891 rows) = 5-10%, House Prices (1460) = 3.5%, Spaceship (8700) = 0.8%
- Feature engineering from parsed string columns (PassengerId, Cabin) adds significant value
- CryoSleep alone would give ~72% accuracy — the rest is incremental improvement

---

## Submission History

| # | Model | Features | CV | LB | Gap | Notes |
|---|-------|----------|----|----|-----|-------|
| 1 | GBM Moderate (arena winner) | 22 | 0.8073 | 0.7994 | 0.8% | ✅ Excellent alignment |
| 2 | Research V2a (LEAKAGE) | 48 | 0.9524 | 0.4931 | 45.9% | ❌ Target leakage from GroupSurvivalRate |
| 3 | Research V2b (HGB, no leakage) | 50 | 0.8119 | 0.8024 | 0.9% | ✅ Best — spending ratios + interactions |
| 4 | LightGBM (boosting sweep) | 50 | 0.8120 | 0.8015 | 1.1% | Tied prior best; see Experiment 4 |

## Experiment 4: Boosting library sweep + stacking (Phase 3)

**Date:** 2026-08-27
**Models:** LightGBM, XGBoost, CatBoost, HistGB on the 50 no-leakage features,
plus a logistic-regression OOF stack and a simple-average blend.

### CV results
- LightGBM: **0.8120** (best individual) | HistGB: 0.8119 | XGBoost: 0.8093 | CatBoost: 0.8084
- Stacked ensemble: 0.8095 | Simple average: 0.8102
- **LB (LightGBM): 0.80149** — tied the prior 0.8024 best.

### Surprising finding
Stacking/averaging the four boosting libraries **did not beat the best single
model** — they are highly correlated and make the same errors, so combining them
adds meta-learner variance without diversity benefit. LightGBM alone was best.

### Conclusion: this competition has plateaued (~0.80-0.802) for tree models
The signal is dominated by CryoSleep + spending; extra model variety is noise.
Real gains would need **new signal**, not new models:
- Impute missing CryoSleep from zero-spending (they're near-equivalent)
- Name-based family/surname grouping features
- Group-level aggregates (group's total spend, group cabin consistency)
These are the next levers if we revisit.

---

## Experiment 2: Research V2a — Target Leakage (FAILED)

**Date:** 2026-08-21
**LB Score:** 0.49310 ❌
**CV Score:** 0.9524

### What Happened
- Computed GroupSurvivalRate from training Transported labels
- Same leakage issue as Titanic V2a — feature encodes the target
- CV was 95.2% — impossible without leakage on this dataset
- LB was 0.493 — basically random

---

## Experiment 3: Research V2b — No-Leakage Features

**Date:** 2026-08-21
**LB Score:** 0.80243 ✅ (new best!)
**CV Score:** 0.8119
**CV-LB Gap:** 0.9% (excellent)

### Strategy
- Removed GroupSurvivalRate (target leakage)
- Kept structural GroupSize and IsAlone
- Added per-amenity spending flags (5 binary flags)
- Added spending ratios (fraction per category)
- Added cabin number binning (CabinRegion: ship quarters)
- Added granular age groups (7 bins)
- Added HomePlanet × CryoSleep interactions
- Added Age × CryoSleep interaction
- Used HistGradientBoostingClassifier (300 iter, depth=5, lr=0.05, min_leaf=12)
- 50 features total

### What Worked
- HistGradientBoosting slightly outperformed standard GBM (0.8119 vs 0.8052 CV)
- Spending ratios add signal beyond raw amounts
- HomePlanet × CryoSleep captures planet-specific cryo patterns
- 50 features on 8693 rows is appropriate (no overfitting, 0.9% gap)
- CV-LB gap remained tiny (0.9%) — larger dataset = less overfitting risk

### Code
```python
model = HistGradientBoostingClassifier(
    max_iter=300, max_depth=5, learning_rate=0.05,
    min_samples_leaf=12, random_state=42
)
```

---

## ArenaGenerator V3 Sweep — 2026-08-21 21:45

**Arena Winner:** GradientBoosting_full_v5
**CV Score:** 0.79628
**Optimized Params:** {'min_samples_leaf': 15, 'max_leaf_nodes': 80, 'max_iter': 200, 'max_depth': 4, 'learning_rate': 0.05, 'l2_regularization': 0}
**Previous Best LB:** 0.80243
**Submission Status:** success
**File:** arena_v3_hgb_20260821_214459.csv
**Method:** ArenaGenerator (10-min budget, 2 feature sets) → ModelOptimizer (randomized, 50 iter)
