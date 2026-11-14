# Name That LLM (identify the model behind a text) - Research & Design Doc

**Kaggle ref:** name-that-llm
**Deadline:** 2027-01-14 (VERIFIED 2026-11-12 via kaggle competitions list) | **Reward:** 50,000 USD | **Teams:** 5708
**Type:** Text classification (which LLM wrote this)
**Status:** First pass, 2026-11-12. BLOCKED on rules acceptance. Data download returns 403 until the account joins on kaggle.com (same blocker as S6E9). Accepting rules is left to the account owner.

## 1. What we know (file list only, no data downloaded)
Files: train.csv 2.1MB, test.csv 2.2KB, sample_submission.parquet 899B

## 2. Prior art and facts
Public LB around 0.68: ModernBERT + LoRA + sparse ensemble 0.680, Qwen3-Embedding LoRA 0.681, 'All Public + What Failed' 0.684 (VERIFIED: kernels list). Data is tiny, so iteration is fast.

## 3. Reuse from existing code
deep_learning.py + the XLM-R fine-tune/kernel flow from contradictory-watson (machine_shape must be NvidiaTeslaT4). TF-IDF + logistic regression baseline as in watson tfidf_simple.

## 4. Design decisions
| Decision | Choice | Status |
|---|---|---|
| Metric | Read from Overview/Evaluation tab | ASSUMED until read |
| Baseline | Reproduce a top public notebook before any custom work | planned |
| CV | Match metric, keep OOF to compare against LB | planned |

## 5. Next steps
Accept rules, run a TF-IDF + LR baseline locally (minutes), then a ModernBERT/embedding model on a Kaggle T4 kernel.
