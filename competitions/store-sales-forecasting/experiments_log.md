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
| 3 | Per-family LightGBM | 0.395 | **0.423** | 7% | **New best** |
