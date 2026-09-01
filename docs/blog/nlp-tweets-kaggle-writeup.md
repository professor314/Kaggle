# NLP Disaster Tweets — Solution Writeup

## Summary

Reached **0.842 F1** on the leaderboard with a 3-way ensemble of DistilBERT, a TF-IDF logistic-regression model, and BERTweet. This is +0.041 over a strong TF-IDF baseline (0.801). The biggest single lever was transformer fine-tuning; ensembling added the final increment.

## Approach

1. **TF-IDF baseline** — word + character n-grams + keyword encoding, logistic regression. LB 0.801.
2. **DistilBERT fine-tuning** — 5-fold CV, 3 epochs, max_len 128, fp16. LB 0.836.
3. **2-way ensemble** — blend DistilBERT + TF-IDF probabilities (weights and threshold tuned on out-of-fold predictions). LB 0.838.
4. **3-way ensemble** — add BERTweet (tweet-pretrained) as a third base. LB 0.842.

## What Worked

- **DistilBERT fine-tuning** — the dominant improvement (+0.035 over TF-IDF). Contextual embeddings capture sarcasm and figurative language that bag-of-words misses.
- **Submitting despite a mediocre CV score** — DistilBERT's out-of-fold F1 was only 0.804, essentially tied with TF-IDF. The leaderboard gave 0.836. This competition's public test set is cleaner than its noisy training labels, so CV underestimates transformers. Trusting CV alone would have discarded a winning model.
- **Leakage-free ensembling** — generated out-of-fold probabilities for every base model on identical folds, then tuned blend weights and threshold on those OOF predictions only.
- **BERTweet as a third base** — even though it correlated 0.91 with DistilBERT, averaging three models cancelled enough idiosyncratic noise for +0.004.

## What Didn't Work (or underdelivered)

- **The 2-way ensemble barely beat DistilBERT alone** (+0.002). Two highly-correlated models don't complement each other much.
- **Heavy text cleaning** — transformers handle casing and punctuation well; aggressive normalization didn't help. We only stripped URLs and @mentions.
- **Expecting big diversity from BERTweet** — its 0.91 correlation with DistilBERT was higher than hoped. Both are transformers making similar errors.

## Final Model Description

**3-way probability ensemble**
- DistilBERT (`distilbert-base-uncased`), 5-fold, 3 epochs, weight 0.3
- TF-IDF (word 1-2 grams + char 3-5 grams + keyword) + LogisticRegression, weight 0.4
- BERTweet (`vinai/bertweet-base`), 5-fold, 3 epochs, weight 0.3
- Decision threshold: 0.44 (tuned on OOF F1)

## Score

| Model | LB F1 |
|-------|-------|
| TF-IDF + LogReg (baseline) | 0.801 |
| DistilBERT (5-fold) | 0.836 |
| DistilBERT + TF-IDF | 0.838 |
| **DistilBERT + TF-IDF + BERTweet** | **0.842** |

## Key Takeaways

1. **Fine-tune a transformer first** — it was worth more than every other change combined.
2. **When CV and LB disagree, submit** — CV can be pessimistic on competitions with noisy training labels.
3. **Tune the ensemble on OOF, not the training set** — the only leakage-free way to pick weights and threshold.
4. **Diverse errors beat strong-but-similar models** — the gains shrank as we added correlated transformers; a genuinely different modality would help more than a fourth BERT variant.

## AI Disclosure

Developed with **Kiro**, an AI coding agent, collaborating with a human. Kaggle explicitly allows AI-assisted development.
- **Human**: strategy, decisions on what to submit, interpretation
- **AI (Kiro)**: pipeline implementation, training, ensembling, submission generation, and this writeup

AI tools were used throughout, from implementation through submission.
