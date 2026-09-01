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

**Submission blocker — ROOT CAUSE FOUND (2026-08-31):** The 400 on
`CreateCodeSubmission` was NOT a version problem. The API error body said:

> "Submission not allowed: Your Notebook's runtime of 233 minutes exceeds this
> competition's GPU max of 120 minutes."

The v8 kernel ran **233 minutes** because it silently fell back to **CPU**
(GPU probe / accelerator not effective), and training xlm-roberta on CPU takes
~4 hours. Kaggle refuses to accept any submission from a run that violated the
time cap, so every existing version (5-8) 400'd (9+ 403'd = don't exist).

**Fix attempt (kernel v9, 2026-08-31):** Made the run fit under 120 min and
added `REQUIRE_GPU = True` (raise instead of the silent 4-hour CPU path).
Faster settings: EPOCHS 4→3, MAX_LEN 96→80, BATCH 32→64, fp16.
**v9 ERRORed** — and the log revealed the DEEPER root cause below.

**DEEPER ROOT CAUSE (v9 error log):** Kaggle assigned a **Tesla P100** (compute
capability sm_60), NOT the T4 the metadata requested. This is a documented
Kaggle-API bug: kernels pushed via the API/CLI always get a P100 regardless of
`"accelerator": "nvidiaTeslaT4"` (the T4 selection only sticks in the web UI).
The preinstalled PyTorch only ships sm_70+ kernels, so `torch.cuda` fails on the
P100 → our guard correctly raised instead of falling back to a 4-hour CPU run.

**Fix (kernel v10, 2026-08-31):** Added a P100-compatibility shim at the top of
the script: before importing torch, reinstall `torch==2.4.1` from the CUDA 12.1
index (`cu121` wheels include sm_60), which runs fine on the P100. Internet is
enabled for this kernel so the reinstall works. Costs ~5 min but keeps the whole
run well under the 120-min cap. Pushed as version 10.

**When v10 is COMPLETE, submit with (this now works — the earlier 400 was the
time-cap precondition, not a version/CLI problem):**
  `kaggle competitions submit contradictory-my-dear-watson -k seanconnolly/contradictory-watson-xlmr -f submission.csv -v 10`
(or `api.competition_submit_cli(file_name="submission.csv", message=..., competition="contradictory-my-dear-watson", kernel="seanconnolly/contradictory-watson-xlmr", version="10")`).

**v10 ERRORed** — the P100 torch-reinstall shim did not save the run.

**ACTUAL FIX FOUND via research (kernel-metadata field, VERIFIED by others
Aug 2026 in Kaggle/kaggle-api issue #490):** The GPU model is selected by the
`machine_shape` field in kernel-metadata.json, NOT `accelerator` (which the CLI
ignores) and NOT `enable_gpu` (which is only yes/no). Omitting `machine_shape`
or using the generic value defaults to **P100** — the exact silent trap we hit.

Fix (kernel v11, 2026-08-31): set `"machine_shape": "NvidiaTeslaT4"` in
kernel-metadata.json (maps to the UI's "GPU T4 x2", native sm_75 PyTorch, no
shim needed). Removed the P100 reinstall shim from script.py. This is a pure
CLI fix — no manual UI step required.

When the T4 run is COMPLETE, submit with:
  `kaggle competitions submit contradictory-my-dear-watson -k seanconnolly/contradictory-watson-xlmr -f submission.csv -v <version>`

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
