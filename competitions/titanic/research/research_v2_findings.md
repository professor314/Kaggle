# Titanic Research V2 — Findings

## Critical Discovery: Target Leakage

### The Problem
Computing "TicketGroupSurvival" and "FamilySurvival" from training labels introduces **target leakage**:
- These features encode the survival outcome of related passengers
- During CV, information leaks between folds (some family members in train, others in validation)
- Result: CV score of **98.88%** (impossible without leakage) but LB of **0.727** (worse than baseline)

### The Lesson
**Never compute features from the target variable unless using proper out-of-fold encoding.**

Red flags for leakage:
- CV score that seems "too good to be true" (>95% on a noisy classification problem)
- CV-LB gap > 20%
- Features that directly encode group survival rates

### Safe Alternatives
- **Group SIZE** features (TicketGroupSize, SurnameGroupSize) — structural, no leakage
- **Out-of-fold encoding** — compute survival rates in a leave-one-out manner
- **Train-only encoding for test** — use full training set to encode test passengers (but still leaks during CV)

---

## Feature Engineering Results

### Features That Help (vs baseline 0.77272)
| Feature | Type | Impact |
|---------|------|--------|
| SexPclass (Sex × Pclass) | Interaction | Captures "women in 1st class = 97% survival" |
| WomenChildren1st | Interaction | Combines women/children priority with class |
| FarePerPerson | Derived | Better fare signal (removes group ticket inflation) |
| FamilyBucket | Binned | Non-linear family size effect (2-4 is optimal) |
| Age_qt (quantile-transformed) | Transform | Reduces skewness impact |

### Features That Don't Help
| Feature | Reason |
|---------|--------|
| DeckOrdinal | 77% missing — too sparse to add signal |
| TicketGroupSize | Overlaps with FamilySize |
| SurnameGroupSize | Too noisy, many singletons |
| AgePclass | Redundant with Age + Pclass separately |

### Key Finding: Titanic Has a Hard Ceiling
- Our best score: **0.77272** (achieved with just 8 features + conservative GBM)
- Research V2b achieved: **0.76794** (slightly worse despite more engineering)
- The ceiling appears to be ~0.78 with standard tabular approaches
- Breaking 0.80 likely requires:
  - Proper out-of-fold survival propagation (the idea was right, implementation was wrong)
  - Manual rules for edge cases (women in 3rd class, children with specific titles)
  - Ensemble of hand-crafted rules + ML model

---

## Arena Results Summary

14 pipelines tested, 5-fold CV, prior_cv_lb_gap=0.06:

| Rank | Model | CV | Std | Features | Worthiness |
|------|-------|----|-----|----------|------------|
| 1 | GBM_conservative_v2 | 0.8406 | 0.0135 | 8 (minimal) | 0.7681 |
| 2 | LR_research | 0.8361 | 0.0127 | 18 (full) | 0.7644 |
| 3 | GBM_moderate_minimal | 0.8440 | 0.0224 | 8 (minimal) | 0.7632 |
| 4 | GBM_moderate_research | 0.8372 | 0.0165 | 18 (full) | 0.7619 |

**Key Insight:** On Titanic (891 rows), minimal features (8) consistently match or beat extended features (18). The 0.773 ceiling can't be broken by adding more features — it requires fundamentally different approaches (rules-based, out-of-fold group survival).

---

## Strategies Confirmed

1. **Simpler is better on small datasets** — 8 features beat 18 features
2. **Conservative regularization** — depth=2, min_leaf=15 remains optimal
3. **CV 0.83-0.84 is the sweet spot** — higher means overfitting, lower means underfitting
4. **The CV-LB gap diagnostic works** — 98.8% CV was an obvious red flag
5. **Feature engineering ceiling** — standard tabular features max out at ~0.773-0.78

---

*Generated: 2026-08-21*
