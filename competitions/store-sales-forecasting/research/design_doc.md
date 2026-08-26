# Store Sales Forecasting — Research & Design Document

## Competition Overview

**Goal:** Predict daily sales for 54 stores × 33 product families for Corporación Favorita (Ecuador grocery chain) for the 16-day window Aug 16-31, 2017.

**Metric:** RMSLE — Root Mean Squared Logarithmic Error. Penalizes under-predictions more than over-predictions. Appropriate for sales data where the scale varies by orders of magnitude across families.

**Dataset:** 3 million rows of training data (Jan 2013 – Aug 15, 2017), plus supplementary oil prices, holidays, store metadata, and transaction counts.

---

## Domain Research

### Ecuador Economy & Grocery Retail

- Ecuador's economy is **heavily petroleum-dependent**. Oil price drops directly affect consumer spending power.
- In **April 2016**, a 7.8 magnitude earthquake struck Ecuador. This disrupted supply chains, destroyed stores in coastal regions, and caused a government-mandated tax increase for reconstruction.
- Corporación Favorita is one of Ecuador's largest retailers, operating supermarkets across all provinces.

### Key Temporal Patterns (Hypotheses)

1. **Day of week effects** — Weekend sales differ from weekday (grocery shopping patterns)
2. **Payday effects** — Ecuador's common paydays are 15th and end-of-month
3. **Holiday effects** — National holidays (Christmas, New Year, Carnival) cause pre-holiday surges and holiday-day dips (store closures)
4. **Promotion effects** — The `onpromotion` column directly affects sales
5. **Oil price effects** — Declining oil → weaker economy → lower discretionary spending (but staples remain stable)
6. **Earthquake aftermath** — April 2016 onwards shows disruption and recovery patterns
7. **Product family seasonality** — School supplies spike in September (back-to-school), beverages spike in summer

### Product Families (33)

Key categories with distinct patterns:
- GROCERY I/II — staples, steady demand
- BEVERAGES — seasonal (summer peaks)
- PRODUCE — perishable, steady but weather-dependent
- BEAUTY/PERSONAL CARE — discretionary, promotion-sensitive
- SCHOOL/OFFICE SUPPLIES — highly seasonal (academic calendar)
- AUTOMOTIVE, HARDWARE — low volume, sporadic

---

## Approach Strategy

### Level 1: Global Model (baseline)

Train a single model on all store-family pairs with features encoding store/family identity. This captures shared patterns (holidays, oil, day-of-week) and leverages the full dataset.

**Model:** LightGBM or Gradient Boosting with lag features.

### Level 2: Per-Family Models

Train separate models per product family (33 models). Each captures family-specific seasonality and promotion response. Stores within a family share patterns.

### Level 3: Hierarchical / Grouped

Group by store cluster or store type. Train per-group models. Tests whether geographic/type segmentation improves predictions.

### Recommended Path: Start with Level 1, evaluate, then add Level 2 if needed.

---

## Feature Engineering Plan

### Time Features
- day_of_week (0-6)
- day_of_month (1-31)
- month (1-12)
- week_of_year
- is_weekend
- is_month_start, is_month_end
- is_payday (15th or last day of month)
- days_until_christmas, days_until_new_year (for seasonal ramps)

### Lag Features (critical for time series)
- sales_lag_1, sales_lag_7, sales_lag_14, sales_lag_28 (same store-family)
- sales_rolling_mean_7, sales_rolling_mean_14, sales_rolling_mean_28
- sales_rolling_std_7
- sales_diff_1 (day-over-day change)
- sales_same_day_last_week, sales_same_day_last_year

### External Features
- oil_price (daily, forward-filled for missing)
- oil_price_lag_7, oil_price_rolling_mean_14
- is_holiday (binary)
- holiday_type (national, regional, local, transfer)
- days_since_last_holiday, days_until_next_holiday
- onpromotion (given in test data — this is a known future value)

### Store/Family Features
- store_type (A-E)
- store_cluster (1-16)
- family_encoded (label or target encoding)
- store_avg_daily_sales (historical mean)
- family_avg_daily_sales (historical mean)

### Earthquake Feature
- is_post_earthquake (bool, after April 16, 2016)
- days_since_earthquake (for recovery curve)

---

## Cross-Validation Strategy

**Time-based split only** — never use random CV for time series.

Option A: **Expanding window**
- Train on all data up to date D, predict D+1 to D+16
- Shift D forward by 16 days
- Repeat 5-10 times

Option B: **Last-N-days holdout**
- Train: 2013-01-01 to 2017-07-31
- Valid: 2017-08-01 to 2017-08-15 (same length as test)
- This single split mirrors the competition exactly

**Recommendation:** Start with Option B (fast iteration), then validate with Option A.

---

## Baseline Approach

1. Load train.csv, join with stores/oil/holidays
2. Engineer time features + lag features
3. Train LightGBM on data up to 2017-07-31, validate on Aug 1-15
4. Evaluate RMSLE
5. Generate predictions for Aug 16-31 and submit

### Critical Notes

- **RMSLE requires non-negative predictions.** Clip predictions at 0.
- **Log transform target** — since metric is RMSLE, training on log1p(sales) with RMSE loss is equivalent.
- **Lag features in test set** — For the 16-day forecast, lag_1 through lag_15 won't have real values for later days. Need to use recursive prediction or only use lags ≥ 16.

---

## What We Need in the Codebase

### New toolkit capabilities (time series):
1. **TimeSeriesSplitter** — expanding window or temporal holdout CV
2. **Lag feature generator** — given a group (store, family) and target, compute lag/rolling features
3. **Holiday feature extractor** — parse the holidays_events.csv into usable features
4. **Recursive forecaster** — for predictions beyond lag availability

These become reusable toolkit components for any future time series competition.

---

## Success Criteria

- [ ] Working time-based CV pipeline
- [ ] Lag feature generator in toolkit
- [ ] Baseline RMSLE on validation set
- [ ] At least 2 iterations improving over baseline
- [ ] Final Kaggle submission
- [ ] Experiment log documenting learnings
- [ ] Blog post for imadestuff.com

---

## Prior Art (Public Solutions & Discussions)

### Key Findings from Top Solutions

**Scoring landscape:**
- Naïve/Moving Average baseline: ~1.05 RMSLE
- ARIMA: ~0.95 RMSLE
- Basic LightGBM with lags: ~0.79 RMSLE
- Top solutions: ~0.38-0.50 RMSLE
- Competitive goal: < 0.50 RMSLE

**What top performers did:**

1. **Lag/rolling features dominate** — lag_7, roll_mean_7, roll_mean_28 are consistently the top features. Weekly seasonality is the strongest signal.

2. **Per-family or per-store-family training** — Rather than one global model, top solutions train separate models per product family (or even per store×family). Each family has different seasonality and promotion response.

3. **Careful lag handling for the 16-day horizon** — Since we predict 16 days ahead, lag_1 through lag_15 won't exist for later days in the test set. Solutions either:
   - Use only lag_16+ features (safe but lose recent info)
   - Recursive prediction (predict day 1, use it as lag_1 for day 2, etc.)
   - Use lag_7 and lag_14 directly (they're available for the first 9 and 2 days respectively)

4. **Oil prices matter but with a lag** — Oil doesn't affect sales same-day. A 7-14 day lagged oil feature captures the delayed economic impact.

5. **Holiday encoding is nuanced** — "Transfer" holidays (where the holiday is moved to a different day) need special handling. Pre-holiday shopping surges are often stronger than the holiday itself.

6. **Zero-inflated families** — Some families (BABY CARE, BOOKS, HARDWARE) have many zero-sales days. These benefit from a two-stage model: first predict whether sales > 0, then predict the amount.

7. **The 2016 earthquake** — Stores in affected regions (coastal Ecuador, especially Esmeraldas, Guayas) show disrupted patterns. A binary is_post_earthquake feature or simply training on post-earthquake data only can help.

8. **Transactions as a proxy** — The transactions.csv file provides store-level foot traffic. It's a strong predictor but NOT available in the test set. It can be used as a feature if you first forecast transactions, or use its lagged values.

### What to Avoid

- Don't use transaction data as a direct feature (not in test set)
- Don't use random CV — always time-based splits
- Don't forget to clip predictions at 0 (RMSLE requires non-negative)
- Don't ignore the log transform — train on log1p(sales), predict, then expm1

### Realistic Target Score

Given our toolkit and approach:
- Baseline (global LightGBM, basic lags): ~0.5-0.6 RMSLE
- With per-family training + careful features: ~0.4-0.5 RMSLE
- Top 20% territory: < 0.45 RMSLE

---

## Updated Implementation Plan

Based on prior art research:

### Phase 1: Baseline (global model)
1. Log-transform target: `log1p(sales)`
2. Time features (dow, month, day, is_weekend, is_payday)
3. Lag features: lag_16, lag_21, lag_28 (all safe for 16-day horizon)
4. Rolling features: roll_mean_28, roll_mean_56 (shifted by 16 to avoid leakage)
5. Oil price (forward-filled, lagged by 14 days)
6. Holidays (binary + type)
7. Store type/cluster
8. Family as categorical
9. LightGBM, single model, time-based validation split
10. Target: < 0.60 RMSLE

### Phase 2: Per-family models
1. Train 33 separate LightGBM models
2. Each gets tuned lag/rolling windows
3. Add family-specific features (promotion response varies by family)
4. Target: < 0.50 RMSLE

### Phase 3: Advanced features
1. Recursive prediction for near-term lags
2. Two-stage model for zero-inflated families
3. Holiday proximity features (days until/since holiday)
4. Store clustering for transfer learning
5. Target: < 0.45 RMSLE

---

## Design Decisions

### 1. Global model first, per-family later

**Decision:** Train one model on all 54×33 store-family combinations rather than 33 (or 1782) separate models.

**Why:** A global model sees more data (3M rows vs ~90K per family), which means lag features are more reliable and the model generalizes better from the start. Per-family models are Phase 2 — once we understand the baseline behavior, we segment. Starting with segmentation risks overfitting low-volume families with sparse data.

### 2. Safe lags only (>= 16 days)

**Decision:** Only use lag features with offset >= 16 (the forecast horizon).

**Why:** We're predicting 16 days into the future. lag_1 through lag_15 won't exist for the later test dates. Using them would require recursive prediction (predicting day 1, using it as lag_1 for day 2, etc.), which compounds errors. Safe lags (16+) give real historical values for every test row without error propagation. This is conservative but reliable — we can add recursive near-term lags in a later iteration once the baseline is solid.

### 3. Log1p transform on target

**Decision:** Train on `log1p(sales)` with RMSE objective, not raw sales.

**Why:** The competition metric is RMSLE. Mathematically, RMSLE(y, ŷ) = RMSE(log1p(y), log1p(ŷ)). So training on log-transformed sales with standard RMSE loss directly optimizes the competition metric. This also handles the extreme scale differences between families — GROCERY I sells 10,000 units/day while BABY CARE sells 5.

### 4. LightGBM over other models

**Decision:** Use LightGBM as the primary model.

**Why:** (1) Handles 3M rows efficiently (~2 minutes training), (2) natively handles categorical features without encoding, (3) dominates Kaggle time series competitions, (4) early stopping prevents overfitting naturally, (5) feature importance output helps us understand what's working. Alternatives considered: XGBoost (slower on this data size), ARIMA (doesn't scale to 1782 series), Prophet (too slow per-series), LSTM (overkill for this granularity).

### 5. Rolling means shifted by 16

**Decision:** Rolling mean/std windows are computed after shifting the series by 16 positions.

**Why:** Without the shift, a 7-day rolling mean on the target would include future values relative to the prediction date. Shifting by 16 ensures the rolling window only uses data that's at least 16 days old — matching our safe lag constraint. This means roll_mean_7 is actually "the average of days 16-22 ago," not "the last 7 days."

### 6. Oil price as a feature (with 14-day lag)

**Decision:** Include both current oil price and a 14-day lagged version.

**Why:** Ecuador is petroleum-dependent — oil price shocks ripple through the economy. But the effect isn't immediate. A 14-day lag captures the delayed consumer spending response to price changes. The current price is included too since markets may react to expectations.

### 7. Transactions NOT used as a direct feature

**Decision:** Excluded the transactions.csv data from features.

**Why:** Transaction counts are not available in the test set (they're a concurrent measurement, not a forecast input). Using them during training but not test creates a feature mismatch. We could use lagged transactions (16+ days back), but decided to keep the baseline simple and add this in a later iteration.

### 8. National holidays only (simplified)

**Decision:** Used only national holidays as a binary flag, ignoring regional/local holidays and transfers.

**Why:** The holiday calendar is complex (transferred holidays, regional vs local, work days). For the baseline, a simple binary "is today a national holiday" captures the biggest effect. The nuanced holiday proximity features (days-until-holiday surges, regional holidays affecting specific stores) are Phase 3 improvements.

### 9. Validation = last 16 days of training

**Decision:** Validate on Aug 1-15, 2017 (the 16 days immediately before the test period).

**Why:** This mirrors the test setup exactly — same forecast horizon, same recency, same seasonal period. Using earlier validation windows would test different seasonal patterns. The tradeoff is that it's a single split (not cross-validated), but for time series, the most recent window is the most informative.

### 10. Clip predictions at zero

**Decision:** All predictions are clipped to max(0, prediction) before submission.

**Why:** RMSLE is undefined for negative values, and sales can't be negative. The expm1 transform can theoretically produce negative values from small log-space predictions. Clipping at zero is both mathematically necessary and domain-correct.
