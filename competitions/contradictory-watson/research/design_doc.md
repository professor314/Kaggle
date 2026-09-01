# Contradictory, My Dear Watson — Design Doc

**Competition:** Contradictory, My Dear Watson (multilingual NLI)
**Metric:** Accuracy | 3 classes (0=entailment, 1=neutral, 2=contradiction)
**Data:** 12,120 train / 5,195 test, 15 languages (~57% English)
**Best:** 0.69 val (kernel v8 COMPLETE); LB pending manual submit
**Full log:** see `../experiments_log.md`

## Domain research
Natural Language Inference across 15 languages. Chance = 0.33. TF-IDF can't
capture entailment semantics (barely beats chance), so a multilingual
transformer is required. The English-heavy split means cross-lingual transfer
matters for the non-English tail.

## Prior art
Standard route to 0.90+: multilingual transformer (XLM-R) with two-stage
training — pretrain on a large English NLI corpus (MNLI/SNLI), then fine-tune on
the competition data. Translate-train (MT English NLI into the 15 languages) is
another lever.

## Feature engineering plan (what shipped)
Sentence-pair encoding (premise + hypothesis) fed to XLM-RoBERTa via the
toolkit's `TransformerClassifier` (extended with NLI pair support). No hand
features.

## Design decisions
| Decision | Choice | Rationale |
|---|---|---|
| Model | xlm-roberta-base | Multilingual, fits Kaggle GPU |
| Compute | Kaggle **T4** kernel | P100 breaks shipped PyTorch (sm_60 vs sm_70+) |
| Submission | Notebook/code submission | Direct CSV submit returns HTTP 400 |
| Training | Single-stage (v8) | Two-stage MNLI pretrain OOM'd the kernel |

## Status & honest assessment
Kernel v8 run COMPLETE; submission.csv validated (5,195 rows, {0,1,2}). Model is
underfitting at ~0.69 val (target 0.85+). Submit is blocked on the manual
notebook-submit step (see experiments_log). To reach 0.85+: two-stage MNLI
pretrain that fits the T4 (MNLI subsample + lazy tokenization), longer training,
or xlm-roberta-large at short max_len.
