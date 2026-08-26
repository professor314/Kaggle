# NLP Disaster Tweets — Research & Design Document

## Competition Overview

**Goal:** Classify tweets as describing a real disaster (1) or not (0).
**Metric:** F1 score (harmonic mean of precision and recall).
**Dataset:** 7613 training tweets, 3263 test tweets.

---

## Domain Research

### The Problem

People tweet about disasters using dramatic language, but they also use disaster-related words metaphorically:
- "The sky is ablaze with color" (not disaster)
- "Forest fire near La Ronge Sask. Canada" (disaster)
- "My heart is on fire for you" (not disaster)
- "WILDFIRE EVACUATIONS in California" (disaster)

The challenge is disambiguating literal disaster references from figurative/casual language.

### What Top Solutions Did (from Kaggle discussions)

1. **TF-IDF + Logistic Regression** — Simple baseline that scores ~0.79 F1
2. **TF-IDF + SVM/Naive Bayes** — Slightly better, ~0.80
3. **BERT fine-tuning** — Top solutions use BERT/DistilBERT, scoring 0.83-0.85
4. **Keyword as a feature** — The keyword column provides signal (disaster-related keywords)
5. **Text cleaning** — URL removal, @mention removal, hashtag splitting help
6. **Ensemble of TF-IDF + BERT** — Best results combine traditional and deep learning

### Realistic Targets

- TF-IDF + Logistic Regression: ~0.79 F1
- TF-IDF + tuned SVM: ~0.80-0.81 F1
- BERT fine-tuned: ~0.83-0.85 F1
- Our goal (no GPU/transformers): 0.80+ F1

---

## Design Decisions

### 1. TF-IDF over raw word counts

**Decision:** Use TF-IDF vectorization, not simple bag-of-words.

**Why:** TF-IDF downweights common words (the, is, a) and upweights discriminative words (earthquake, evacuation, wildfire). On short texts like tweets, this matters — every word carries more weight, and common words add noise.

### 2. Character n-grams + word n-grams

**Decision:** Use both word-level (1-2 grams) and character-level (2-5 grams) TF-IDF.

**Why:** Character n-grams capture subword patterns (hashtags, misspellings, URL fragments) that word-level misses. Tweets are messy text — abbreviations, hashtags, non-standard spelling. Character n-grams are robust to this. Combining both gives the model vocabulary-level AND morphological features.

### 3. Logistic Regression as primary model

**Decision:** Use Logistic Regression (with regularization) over SVM or tree-based models.

**Why:** (1) Fast to train on sparse TF-IDF matrices, (2) naturally outputs probabilities for threshold tuning, (3) L2 regularization handles the high-dimensional sparse features well, (4) competitive with SVM on this problem size, (5) more interpretable (can inspect coefficients).

### 4. Include keyword as a feature

**Decision:** Encode the keyword column and concatenate with TF-IDF features.

**Why:** Keywords like "earthquake", "wildfire", "flooding" have strong base rates for being actual disasters. The keyword alone isn't enough (many are used figuratively), but combined with text features it adds signal. Missing keywords are encoded as a separate "unknown" category.

### 5. Text cleaning (minimal)

**Decision:** Remove URLs and @mentions, lowercase, but keep hashtags and punctuation.

**Why:** URLs and @mentions are noise (they don't indicate disaster vs not). But hashtags (#earthquake) and punctuation (ALL CAPS, exclamation marks) carry signal — urgent disaster tweets use different stylistic patterns than casual metaphorical usage.

### 6. Stratified 5-fold CV

**Decision:** Use stratified K-fold cross-validation (not time-based).

**Why:** Unlike time series, tweet classification has no temporal ordering that matters. The dataset is mildly imbalanced (57/43), so stratification preserves the class ratio in each fold. 5 folds gives reliable estimates on 7613 rows.

### 7. No location feature

**Decision:** Exclude the location column entirely.

**Why:** 33% of locations are missing, and the remaining 3341 unique values are free-text garbage (people put joke locations, multiple formats for the same city, etc.). The noise-to-signal ratio is too poor. Not worth the engineering effort for marginal gain.

---

## Implementation Plan

1. Text cleaning (remove URLs, @mentions, lowercase)
2. TF-IDF vectorization (word 1-2 grams + char 3-5 grams)
3. Keyword encoding (one-hot or target-encoded)
4. Logistic Regression with C tuning via CV
5. Evaluate F1 on 5-fold stratified CV
6. Submit best model
