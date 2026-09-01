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

## Experiment 2: DistilBERT fine-tuning (5-fold CV)

**Date:** 2026-08-26
**CV F1 (OOF):** 0.8042 (best threshold 0.55; 0.8036 at 0.50)
**Mean fold F1:** 0.8037 ± 0.0077
**LB F1:** pending

### Setup
- Model: `distilbert-base-uncased`, HuggingFace `transformers` 5.16
- 5-fold stratified CV, 3 epochs/fold, batch 16, max_len 128, lr 2e-5, fp16
- GPU: RTX 4080 Laptop (12GB usable), ~38s per fold train time
- Toolkit: new `TransformerClassifier` in `kaggle_ml_toolkit/deep_learning.py`
- Prediction: averaged test probabilities across the 5 folds

### Result — surprising
Vanilla DistilBERT fine-tuning essentially **tied** the TF-IDF baseline
(0.804 vs 0.801 OOF), NOT the 0.83+ predicted in PHASE3_PLAN. Fold spread
(±0.008) is larger than the gap, so this is a statistical tie.

### Why (hypotheses)
1. This dataset has known **mislabeled training examples**, capping easy gains.
2. Single-model, single-seed, minimal cleaning — no MLM pretraining on tweets,
   no external data, no ensembling with the TF-IDF model.
3. Tweets are short; TF-IDF char n-grams already capture much of the signal.

### What to Try Next (to actually beat 0.81)
- Ensemble DistilBERT probs with the TF-IDF LR probs (diverse errors)
- Try `bertweet-base` (pretrained on tweets) or `roberta-base`
- Clean known mislabeled rows; multi-seed bagging
- Submit this to Kaggle to measure the real CV→LB gap first

---

## Submission History

| # | Model | CV F1 | LB F1 | Notes |
|---|-------|-------|-------|-------|
| 1 | Auto-pipeline (LR, basic) | 0.749 | 0.800 | Previous session |
| 2 | TF-IDF (word+char) + keyword + LR | 0.769 | 0.801 | Prior best |
| 3 | DistilBERT 5-fold (avg probs) | 0.804 | 0.836 | +0.035 on LB! Big CV→LB gap |
| 4 | Ensemble 0.45*BERT + 0.55*TF-IDF | 0.808 | 0.838 | +0.002 over BERT alone |
| 5 | 3-way: DistilBERT+TF-IDF+BERTweet | 0.812 | **0.842** | Best. BERTweet added diversity |

## Experiment 4: 3-way ensemble with BERTweet (Phase 3)

**Date:** 2026-08-27
**Blend:** 0.3*DistilBERT + 0.4*TF-IDF + 0.3*BERTweet probs, threshold 0.44
**CV OOF F1:** 0.8120 | **LB F1:** 0.84216 (new best, +0.004 over 2-way)

### How
Trained `vinai/bertweet-base` (tweet-pretrained) on the same 5 folds, saved its
OOF/test probs, then grid-searched 3-way simplex weights + threshold on OOF F1.

### Diversity check
- BERTweet OOF F1: 0.8081 (on par with DistilBERT's 0.804)
- Correlation BERTweet vs DistilBERT: **0.910** (higher than hoped — both are
  transformers), vs TF-IDF: 0.815
- Despite high correlation, the 3-way average still improved OOF and LB, because
  averaging three models cancels more idiosyncratic noise than two.

### Takeaway
BERTweet was worth adding (+0.004 LB). The progression 0.801 → 0.836 → 0.838 →
0.842 shows diminishing but real returns from ensembling. Next lever for a bigger
jump would be roberta-large or a genuinely different modality, not more BERT variants.

## Experiment 3: DistilBERT + TF-IDF ensemble

**Date:** 2026-08-27
**Blend:** 0.45 * DistilBERT probs + 0.55 * TF-IDF LR probs, threshold 0.52
**CV OOF F1:** 0.8076 | **LB F1:** 0.83818 (new best)

### How
Generated OOF + test probabilities for both models on identical 5 folds
(no leakage), then tuned blend weight and threshold on OOF only. Probabilities
and labels saved to `artifacts/oof_probs.npz` for reuse.

### Diversity
Prob correlation between the two models: **0.83** — fairly high, which caps the
ensemble gain. More diverse bases (e.g. bertweet/roberta) would likely help more
than re-weighting these two.

### Takeaway
Ensemble is the best single submission (0.838), but the lift over DistilBERT
alone is marginal (+0.002). The bigger win was DistilBERT itself (+0.035 over
TF-IDF). Next lever for real gains: a tweet-pretrained model (bertweet) as a
third, more diverse base.

### Update after submission (2026-08-27)
DistilBERT scored **0.83634 on the LB** — a +0.035 jump over TF-IDF (0.801),
matching the plan's 0.83+ target after all. The earlier "ties baseline" read
was misleading because CV underestimated LB by ~0.032. Lesson: this comp's
public test set is cleaner than its noisy training labels, so CV is pessimistic
for transformers. Submitting was the right call over trusting CV.
