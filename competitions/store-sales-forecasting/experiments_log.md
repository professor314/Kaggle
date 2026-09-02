# Store Sales Forecasting — Experiment Log

**Competition:** Store Sales - Time Series Forecasting
**Metric:** RMSLE (Root Mean Squared Logarithmic Error)
**Dataset:** 3,000,888 train rows (54 stores × 33 families × ~1700 days)

---

## Experiment 1: Global LightGBM Baseline

**Date:** 2026-08-23
**Val RMSLE:** 0.4007
**LB RMSLE:** 0.47658
**CV-LB Gap:** 19% (expected for time series)

### Model
```python
LightGBM (490 iterations, num_leaves=127, max_depth=8, lr=0.05)
```

### Features (34 total)
- **Lag features (safe, >= 16 day offset):** lag_16, lag_17, lag_21, lag_28, lag_35, lag_42, lag_49, lag_56
- **Rolling features (shifted by 16):** roll_mean_7, roll_mean_14, roll_mean_28, roll_mean_56, roll_std_7, roll_std_14
- **Time features:** day_of_week, day_of_month, month, week_of_year, is_weekend, is_month_start, is_month_end, is_payday, quarter, year
- **External:** oil_price, oil_lag_14, is_holiday, onpromotion
- **Categorical:** store_nbr, family_encoded, store_type_encoded, cluster

### Top 5 Features (by gain)
1. log_sales_roll_mean_7 (dominant — 10x more important than #2)
2. log_sales_lag_21
3. log_sales_roll_mean_56
4. log_sales_lag_49
5. log_sales_lag_56

### Key Findings
- Weekly rolling mean is overwhelmingly the strongest signal
- Safe lags (16+) work well — no data leakage, still competitive
- Single global model across all 54×33 store-families achieves 0.47 LB
- Previous auto-pipeline submission was 1.29 — this is a 3x improvement
- Val RMSLE (0.40) is optimistic vs LB (0.48) — test period may have different patterns

### What to Try Next
1. **Per-family models** — 33 separate LightGBMs should capture family-specific patterns
2. **Add transaction lags** — lagged transaction counts as additional features
3. **Holiday proximity features** — days_until_holiday, days_since_holiday
4. **Recursive near-term lags** — predict day 1-7, use as features for day 8-16
5. **Reduce CV-LB gap** — use multiple validation windows to get more reliable estimates

---

## Experiment 2: Per-Family LightGBM (33 models)

**Date:** 2026-08-25
**Val RMSLE:** 0.3954
**LB RMSLE:** 0.42319
**CV-LB Gap:** 7% (improved from 19% with global model)

### Model
33 separate LightGBM models (one per product family), same params as baseline.

### Key Improvement
- Val: 0.401 → 0.395 (modest)
- LB: 0.477 → **0.423** (massive — 11% improvement!)
- The CV-LB gap collapsed from 19% to 7%, meaning per-family models generalize much better

### Why It Worked
Each family has different seasonality and promotion response:
- GROCERY I/II need more iterations (heavy patterns)
- BABY CARE/BOOKS need very few trees (sparse data)
- PRODUCE/BEVERAGES benefit from specific seasonal features
- Per-family training lets each model specialize

### Submission History

| # | Model | Val RMSLE | LB RMSLE | Gap | Notes |
|---|-------|-----------|----------|-----|-------|
| 1 | Auto-pipeline (broken) | — | 1.292 | — | Config was wrong |
| 2 | Global LightGBM | 0.401 | 0.477 | 19% | Baseline |
| 3 | Per-family LightGBM | 0.395 | 0.423 | 7% | Per-family specialization |
| 4 | Per-family v2 (+holiday+tx) | 0.391 | **0.421** | 8% | **New best** |

## Experiment 3: Per-Family v2 — holiday proximity + transaction lags

**Date:** 2026-08-27
**Val RMSLE:** 0.39129 | **LB RMSLE:** 0.42061 (new best, down from 0.42319)

### New features added on top of per-family v1
- **Holiday proximity:** `days_until_holiday`, `days_since_holiday`, `is_holiday_window`
  (within 3 days). Computed once on the ~1,700 unique dates and merged back.
- **Transaction lags:** `tx_lag_16` and `tx_roll_mean_7` (per store, shifted 16 days
  to stay leakage-safe, matching the sales-lag offset).

### Result
Small but real: LB 0.42319 → 0.42061. The holiday-window signal helps around the
big Ecuadorian holidays; lagged transactions add a store-traffic proxy. Both are
leakage-safe (16+ day offsets).

### Next levers (unchanged)
- Recursive near-term lags (predict days 1-7, feed as features for 8-16)
- Per-family hyperparameter tuning (families differ a lot in optimal depth/iters)
- Multiple validation windows to tighten the CV-LB gap

## Experiment 4: Per-Family v3 — big-iteration (more compute, NOT better)

**Date:** 2026-09-02
**Val RMSLE:** 0.38747 (best val yet) | **LB RMSLE:** 0.42231 (WORSE than v2's 0.42061)

After adding all-cores compute, spent it on more iterations per family:
lr 0.05->0.02, up to 6000 rounds (mean best_iter 1075, one family 5978),
num_leaves 63->95. Validation improved (0.395 -> 0.387) but the **leaderboard got
slightly worse** (0.42061 -> 0.42231).

### Lesson (important): more iterations != better here
This is textbook overfitting to the validation window (Aug 1-15). The deeper,
longer-trained per-family models fit that specific fortnight better but
generalized slightly worse to the actual test period. On this competition the
bottleneck is NOT compute/iterations — it's the CV-LB gap and feature signal.
**v2 (0.42061) remains the best submission.** Do not chase val RMSLE here.

### What would actually help (not more trees)
- Recursive near-term lags (the biggest untried lever).
- Multiple/rolling validation windows so val tracks LB (would have caught this).
- Per-family early-stopping tuned to each family's data size.

### Submission History (updated)

| # | Model | Val RMSLE | LB RMSLE | Notes |
|---|-------|-----------|----------|-------|
| 3 | Per-family v1 | 0.395 | 0.42319 | Per-family specialization |
| 4 | Per-family v2 (+holiday+tx) | 0.391 | **0.42061** | **Best** |
| 5 | Per-family v3 (big-iter) | 0.387 | 0.42231 | Overfit val; LB worse |
