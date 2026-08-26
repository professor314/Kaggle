# NLP Disaster Tweets — Experiment Log

**Competition:** Natural Language Processing with Disaster Tweets
**Metric:** F1 Score
**Dataset:** 7613 train tweets, 3263 test tweets

---

## Experiment 1: TF-IDF + Logistic Regression

**Date:** 2026-08-23
**CV F1:** 0.7689
**LB F1:** 0.80110

### Model
```python
LogisticRegression(C=0.5, max_iter=1000, solver="liblinear")
```

### Features (50,230 total)
- Word TF-IDF (1-2 grams): 20,000 features
- Character TF-IDF (3-5 grams): 30,000 features
- Keyword encoding: 230 features

### Text Preprocessing
- Remove URLs (`http\S+`)
- Remove @mentions (`@\w+`)
- Lowercase

### Design Decisions
1. **Combined word + char n-grams** — char n-grams capture hashtags, misspellings, subword patterns
2. **sublinear_tf=True** — applies log(1+tf) which dampens high-frequency terms
3. **Keyword as a feature** — adds domain signal (disaster-related keywords)
4. **No location** — too noisy (3341 unique, 33% missing, free-text garbage)
5. **Logistic Regression over SVM** — LR scored 0.769 vs SVM's 0.765

### Results
- CV said 0.769, LB gave 0.801 — LB is better (unusual, likely slightly different test class balance)
- Previous auto-pipeline: 0.800 LB — marginal improvement
- Character n-grams are important for tweet-length text

### What to Try Next
- Add text length as a feature
- Add count of capitals, punctuation, hashtags
- TF-IDF with trigrams (word level)
- Ensemble: average LR + SVM predictions
- BERT/DistilBERT fine-tuning (would need GPU or smaller model)

---

## Submission History

| # | Model | CV F1 | LB F1 | Notes |
|---|-------|-------|-------|-------|
| 1 | Auto-pipeline (LR, basic) | 0.749 | 0.800 | Previous session |
| 2 | TF-IDF (word+char) + keyword + LR | 0.769 | **0.801** | Current best |
