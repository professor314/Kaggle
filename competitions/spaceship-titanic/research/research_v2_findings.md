# Spaceship Titanic Research V2 — Findings

## Improvement: 0.79939 → 0.80243 (+0.3%)

---

## Critical Discovery: Target Leakage (Same as Titanic)

### The Problem
Computing "GroupSurvivalRate" from training Transported labels:
- CV: 95.24% (impossible without leakage)
- LB: 0.49310 (worse than random)
- The feature encodes the target through group membership

### Fix
Replace GroupSurvivalRate with structural GroupSize and IsAlone (no target info).

---

## What Worked

### 1. HistGradientBoostingClassifier
HGB outperformed standard GBM and RandomForest:
- HGB_strong: 0.8119 CV (best)
- GBM_moderate: 0.8052 CV
- RF_strong: 0.7996 CV

HGB advantages:
- Native NaN handling (no need to impute missing cabins)
- Built-in regularization
- Faster training than standard GBM (uses histograms)

### 2. Spending Ratios
Beyond total spending, the *fraction* spent on each amenity adds signal:
- RoomServiceRatio, SpaRatio, etc.
- Passengers who spend disproportionately on one service have different transport patterns

### 3. Per-Amenity Spending Flags
Binary flags for each spending type (SpentOnRoomService, SpentOnSpa, etc.):
- Captures "any vs none" signal that log-transform might miss
- NumServicesUsed (count of services used) is also informative

### 4. Cabin Region Binning
Binning cabin numbers into ship regions (0-300, 300-600, etc.):
- Different ship sections may have different transport rates
- Reduces the 1800+ unique cabin numbers to 6 regions

### 5. Granular Age Groups
7 age bins (0-5, 5-12, 12-18, 18-25, 25-40, 40-60, 60+):
- Children (5-12) have very different patterns from teenagers (12-18)
- More granular than simple IsChild flag

### 6. HomePlanet × CryoSleep Interaction
- EarthCryo, EuropaCryo, MarsCryo capture planet-specific cryo patterns
- Europa passengers in cryo have different transport rates than Earth passengers in cryo

---

## Arena Results

10 pipelines, 5-fold CV, prior_cv_lb_gap=0.01:

| Rank | Model | CV | Std | Worthiness |
|------|-------|----|-----|------------|
| 1 | HGB_strong | 0.8119 | 0.0068 | 0.7952 |
| 2 | HGB_conservative | 0.8084 | 0.0063 | 0.7921 |
| 3 | HGB_very_strong | 0.8086 | 0.0074 | 0.7913 |
| 4 | HGB_aggressive | 0.8096 | 0.0099 | 0.7899 |
| 5 | GBM_moderate | 0.8052 | 0.0071 | 0.7883 |

HGB dominated the top positions. The sweet spot was 300 iterations, depth=5.

---

## CV-LB Gap Analysis

| Experiment | CV | LB | Gap |
|-----------|----|----|-----|
| Exp 1 (22 features, GBM) | 0.8073 | 0.79939 | 0.8% |
| Exp 3 (50 features, HGB) | 0.8119 | 0.80243 | 0.9% |

The gap remains tiny (~1%) — confirming that 8700 rows is enough to prevent overfitting even with 50 features. This validates our strategies.md guideline that larger datasets can handle more features.

---

## Feature Importance (Inferred from V1 vs V2)

### Most Important
1. **CryoSleep** — dominant feature (~72% accuracy alone)
2. **Spending features** (Total, Log, NoSpending) — cryo passengers don't spend
3. **Cabin Deck** — certain decks transport more
4. **Age** — children have different patterns

### Added Value from V2
- **Spending ratios** — marginal but consistent improvement
- **Interaction features** — HomePlanet×CryoSleep helps distinguish subgroups
- **HGB model** — slightly better than GBM at capturing complex patterns

### Minimal Added Value
- CabinRegion — weak signal beyond DeckNum
- IsElder — too few elders in dataset
- PersonInGroup — mostly noise

---

## Next Steps for 0.81+
1. **CatBoost** — install and test (native categorical handling, ordered boosting)
2. **Feature selection** — remove weakest 15-20 features to reduce noise
3. **Tune HGB** — Bayesian optimization of learning_rate, max_depth, max_iter
4. **Ensemble** — blend HGB + GBM + RF (they make different errors)
5. **Out-of-fold group encoding** — properly encode group survival without leakage

---

*Generated: 2026-08-21*
