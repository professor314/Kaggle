# The RSNA Knee Project — Story Log

A running, plain-language narrative of how this model is being built: the
breakthroughs, the dead ends, the research, and the decisions behind them. This
file is the source of truth for the story; the public blog post
(`imadestuff-blog/.../rsna-knee-mri-baseline/index.md`) is drawn from it.

Keep it honest and chronological. Add a dated entry whenever something is learned
or decided. Numbers here are the real ones we measured.

---

## The task, in one paragraph
Predict 12 binary findings per knee MRI study (ACL, MCL, medial/lateral meniscus,
medial/lateral/PF osteoarthritis, effusion, synovitis, Baker's cyst, contusion,
fracture), scored by macro-averaged ROC AUC. $77,000 in prizes (including a
separate "most efficient model" award), closes 2026-10-22. It is a Code
Competition: 100GB+ of DICOM MRI you never download — your notebook runs on
Kaggle's GPU with the data mounted read-only.

## The two hard constraints that shape everything
1. **You never touch the data locally.** 4,407 studies, 24,371 series, ~820k
   DICOM slices, 100GB+. Everything runs on Kaggle. Your machine is the editor.
2. **Only 58 of 4,407 studies are expert-labeled.** Every other study ships with
   a free-text radiology report (English + ~12 other languages). The *test* set
   has NO reports. So reports can only ever be a way to *make training labels* —
   never a model input at scoring time.

---

## Chapter 1 — Prove the plumbing before building a model
Instinct says start with the impressive model. The failure modes here are almost
all plumbing: can't reach the data, reads time out, submission format wrong,
kernel gets the wrong GPU. So we de-risked first:

- **Explore kernel** confirmed the file layout, the 12 finding names (from
  `sample_submission.csv`), the 58-of-4,407 label scarcity, and that slices are
  16-bit MR needing percentile windowing (not a fixed 0–255 scale).
- **Lesson (expensive):** recursively listing 820k DICOM files *stalls the
  kernel*. Index images from the CSVs instead of walking the tree.
- **Smoke kernel** ran the full job shape with a dumb predictor (guess training
  prevalence per finding), read timed DICOMs, wrote a valid submission → **LB 0.500**.
  That proved data access + timed reads + submission format + the submit flow.
- **The GPU gotcha that cost real time:** Kaggle picks the GPU from
  `machine_shape` (`"NvidiaTeslaT4"`), NOT `accelerator`. Leave it unset → a
  P100 modern PyTorch won't run → a dead kernel with a confusing error.

## Chapter 2 — The from-scratch baseline, and why it stalled
A deliberately boring model: sample a few evenly-spaced slices, mean-pool them,
feed a pretrained EfficientNet-B0 with 12 sigmoid heads, train BCE on the 58
labeled studies, fall back to prevalence on any read failure.

- Result: **LB 0.533.** Barely above chance. Leaderboard leaders sit ~0.95.
- **Why it stalled (two root causes we later proved):**
  1. *Mean-pooling K slices into one blurry image destroys the local lesion
     signal.* A torn ACL is visible on a couple of slices; averaging washes it out.
  2. *Training on 58 labels is training on noise.* An image model on 58 labels
     scored ~0.52 gold macro-AUC — chance.

## Chapter 3 — The reports hold the signal (but the test set doesn't have them)
We tested where the signal actually lives:
- **Rule-based report miner** (bilingual keyword + negation): **0.607 gold.**
- **Zero-shot NLI** (mDeBERTa-XNLI): **0.556 gold** — worse than rules.
- **Image model on 58 labels:** ~0.52 gold — chance.

Conclusion: the radiologist's words are the signal. But the test set has no
reports, so text can only ever be a *label source*. The pipeline has to be:
**reports → good training labels → an image model that learns those labels and
scores the test images.**

## Chapter 4 — The breakthrough: label quality is the entire ceiling
Studying a public 0.903 / rank-18 write-up (and then measuring it ourselves)
made the key insight land: **every image-side lever tops out at the quality of
the report-derived labels.** The winning recipe:
- An **LLM report extractor** turns each report into 12 probabilities (a hedged
  "tear suspected" → ~0.8, not a hard 1). That reaches ~0.83–0.89 label accuracy
  vs the 58 gold, where regex tops out ~0.81.
- A **Multiple-Instance-Learning (MIL) image model**: a shared 2D backbone runs
  per slice, then the per-slice features are **pooled (max / attention), not
  mean-pooled into one image** — the exact mistake our baseline made.
- Clinical labeling rules that matter: *unmentioned finding = 0* (radiologists
  report positives, stay silent on negatives — we had wrongly treated blanks as
  "unlabeled"); explicit negation = 0; weight the Impression section; contusion
  needs a traumatic pattern; fracture includes avulsion/insufficiency; never
  horizontal-flip (5 of 12 labels are laterality-specific).
- Honest validation: GroupKFold(5) by study, **excluding the 58 gold** so the
  gold metric doesn't leak.

## Chapter 5 — The second breakthrough: don't redo what the community published
The realization that changed our plan: on a public competition, people have
already published nearly every reusable building block on Kaggle — and much of
it is CC0 (public domain). Redoing common training ourselves is wasted time.

What we found and license-verified (see `PUBLIC_ASSETS_LICENSES.md`, all checked
via the Kaggle API on 2026-09-02):
- **Labels:** `stevenleehans/rsna-knee-llm-report-labels` — CC0, **0.8927 vs
  gold** (its v4 blend), the best label set. Plus CC0 label sets from pilkwang,
  lixin73, yunusgmsoy, rayanbabur, good as decorrelated blend partners.
- **Preprocessed images:** `alenic/...jpeg-224x224` (MIT) skips the 100GB DICOM ETL.
- **A whole trained model:** `dreaddevelopment/raptor-knee-widedense` — CC0, a
  CoAtNet 12-finding model scoring **0.924 public LB / 0.9167 gold single-model**,
  with a companion inference notebook.

This reframes the plan: **assemble from public parts first** for a fast, real
score, then improve. Verify every license (done) so anything we lean on is
prize-eligible.

## Chapter 6 — The tiered plan we're now executing
- **Tier 0:** correctness modules + local tests (clinical rules, DICOM ETL,
  series selection, leak-free folds, submission assembler).
- **Tier A (reuse-first):** license audit → an **early submission from the CC0
  trained model** → then our own MIL head trained on the best public labels +
  preprocessed images + a medical backbone.
- **Tier 1:** our improved free labels + our MIL model.
- **Tier 2:** LLM labels (cost-gated; free-local first, else a ~$17 mid-API pass).
- **Tier 3:** full 5-fold ensemble + a robust internet-off inference kernel.
- **Tracks:** MRNet-pretrained backbone (P), multi-plane fusion (M),
  self-supervised pretraining on the 820k unlabeled slices (S), and a compact
  model for the separate **efficiency prize** (E).
- **Submit early and often** — a real LB score at each milestone, logged in
  `experiments_log.md`, each building on the last.

---

## Running scoreboard (honest numbers)
| Milestone | What | Gold macro-AUC | Public LB |
|---|---|---|---|
| Smoke | prevalence guess | — | 0.500 |
| Baseline | from-scratch EfficientNet-B0, mean-pool, 58 labels | ~0.52 | 0.533 |
| Rules | bilingual keyword + negation miner | 0.607 | — |
| NLI | mDeBERTa-XNLI zero-shot | 0.556 | — |
| Best public labels | stevenleehans v4 blend (CC0) | 0.8927 | — |
| A.1 (reuse) | CC0 raptor CoAtNet, our internet-off kernel | (0.9167 reported) | **0.924** ✅ |

## Principles this project runs on
- Prove the boring path before the fun part.
- The bottleneck is label quality, not model cleverness.
- Reuse published, license-cleared work instead of redoing common training.
- Validate honestly on held-out gold; don't overfit the public LB (prevalence
  differs from the private set).
- Keep the story current so the *how* is as clear as the *what*.
