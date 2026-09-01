# Contradictory, My Dear Watson — Experiment Log

**Competition:** Contradictory, My Dear Watson (multilingual NLI)
**Metric:** Accuracy | **Classes:** 0=entailment, 1=neutral, 2=contradiction
**Data:** 12,120 train / 5,195 test, 15 languages (57% English)
**Submission flow:** Kaggle **notebook/kernel** (direct CSV API submit returns 400)

---

## Experiment 1: TF-IDF + overlap features (kernel)

**Approach:** TF-IDF on premise/hypothesis + hand-crafted overlap/negation
features + LogisticRegression. Chance for 3 classes is 0.33.
**Result:** modest, roughly 0.36-0.40 range (barely above chance — TF-IDF can't
capture entailment semantics well).

---

## Experiment 2: XLM-RoBERTa-base (Phase 3)

**Date:** 2026-08-27
**Local held-out val accuracy:** 0.6922 (English 0.716, non-English 0.663)
**Model:** xlm-roberta-base via toolkit `TransformerClassifier` (extended with
sentence-pair support for NLI). 3 epochs, max_len 96, lr 1e-5, fp16.

### Honest assessment
0.69 is a big jump over TF-IDF (~0.36) but **well below the 0.85+ target** the
plan predicted. The model is **underfitting**: train loss started at 1.10
(= log 3, pure chance) and only reached ~0.75 after 3 epochs. XLM-R needs more
training and/or NLI pretraining to converge.

### Submission mechanics — important finding
Direct `kaggle competitions submit` returns HTTP 400 for this competition; it
requires a **notebook submission**. Pushed `contradictory-watson-xlmr` kernel
(GPU + internet enabled) that trains XLM-R on Kaggle and writes the submission.
Kernel bumped to 4 epochs, lr 2e-5, batch 32 to reduce the underfitting seen
locally.

### Improvement path (to reach 0.85+)
1. **Two-stage training:** pretrain on MNLI/SNLI (large English NLI corpus),
   then fine-tune on competition data. This is the standard route to 0.90+.
2. Train longer (5-6 epochs) with a proper LR schedule.
3. Try `xlm-roberta-large` (Kaggle GPU can handle it at short max_len).
4. Translate-train: machine-translate English NLI data into the 15 languages.

---

## Experiment 3: XLM-R single-stage T4 kernel v8 (COMPLETE)

**Date:** 2026-08-31
**Kernel:** `seanconnolly/contradictory-watson-xlmr` v8, accelerator **T4**
(P100 broke the shipped PyTorch: sm_60 vs sm_70+). Two-stage MNLI pretraining
OOM'd the kernel, so v8 is single-stage.

**Status:** Kernel run **COMPLETE**. Output `submission.csv` pulled and verified:
- Header `id,prediction`, 5,195 rows, predictions in {0,1,2}. Format valid.

**Submission blocker (documented):** This is a notebook/code competition.
- Direct CSV submit → HTTP 400 (`CreateSubmission`), as expected.
- Code submit via `-k <kernel> -f submission.csv -v <version>` → HTTP 400
  (`CreateCodeSubmission`) when guessing the version number.
- **Manual step for the human:** open the kernel on kaggle.com, use
  "Submit to Competition" from the notebook's Output, OR run
  `kaggle competitions submit contradictory-my-dear-watson -k seanconnolly/contradictory-watson-xlmr -f submission.csv -v <correct_version>`
  after confirming the exact latest version number from the kernel's page.

### Improvement path (to reach 0.85+) — still open
1. **Two-stage training:** pretrain on MNLI/SNLI, then fine-tune. Memory-safe on
   T4 via MNLI subsample + lazy `.map` tokenization (first attempt OOM'd).
2. Train longer (5-6 epochs) with a proper LR schedule.
3. Try `xlm-roberta-large` at short max_len.
4. Translate-train: MT English NLI into the 15 languages.

---

## Submission History

| # | Model | Local/Val | LB Acc | Notes |
|---|-------|:---------:|:------:|-------|
| 1 | TF-IDF + overlap (LR) | ~0.40 | pending | Barely above chance |
| 2 | XLM-R base (kernel, 4 ep) | 0.69 val | pending | Superseded by v8 |
| 3 | XLM-R single-stage T4 v8 | 0.69 val | pending (manual submit) | Kernel COMPLETE, submission.csv valid, needs web/kernel submit |
