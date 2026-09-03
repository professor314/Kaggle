# RSNA Knee Abnormality Detection — Experiment Log

**Metric:** macro-averaged ROC AUC | **Deadline:** 2026-10-22 | Code Competition

## Exp 0: Data exploration (2026-09-02)

Pushed a read-only, no-GPU kernel (`seanconnolly/rsna-knee-explore`) to learn the
layout on Kaggle (data is 100GB+, never downloaded).

**VERIFIED findings:**
- 12 findings: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA,
  PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture. Submission =
  StudyInstanceUID + those 12 (sample fills 0.5).
- train.csv: 4407 studies with a multilingual `Report`; labels on only **58/4407**.
- train_series.csv 24,371 rows; per-series Fluid_Sensitive/Fat_Suppression/Plane.
  ~5 series/study, ~33 slices/series.
- DICOM: 512x512 int16, MR, MONOCHROME2, value range e.g. -663..9999 (percentile
  windowing needed).
- **Gotcha:** recursively globbing all .dcm STALLS the kernel — index from CSVs.

**Lesson:** the hard part is a weak-label (58) multimodal problem; the baseline
will be a 2.5D EfficientNet on the labeled studies with a prevalence fallback,
targeting a valid in-time submission first.

## Exp 1: Pipeline smoke test (2026-09-02) — PROVED THE PATH

Kernel `seanconnolly/rsna-knee-smoke` (no GPU, internet off). Trivial predictor =
train prevalence for every study. Read 24 slices across 3 test studies in 2.2s,
wrote a valid submission, submitted. **LB 0.500** (chance, as expected). This
confirmed data access + timed DICOM reads + submission format + the code-comp
submit flow all work — before writing any model.

## Exp 2: 2.5D EfficientNet-B0 baseline (2026-09-02)

Kernel `seanconnolly/rsna-knee-baseline` on a T4 (machine_shape NvidiaTeslaT4).
Sampled K=8 slices/study, mean-pooled, EfficientNet-B0 with 12 sigmoids, BCE,
8 epochs on the 58 labeled studies; prevalence fallback for unreadable studies.

- v1 had internet ON (for pretrained weights) -> submit 400: "Your Notebook
  cannot use internet access in this competition" (read from the error body).
  VERIFIED gotcha: this competition forbids internet in submitted kernels.
- v2: internet OFF, `pretrained=False` (train from scratch). Ran in 69s, loss
  1.05 -> 0.34, model_used=True, valid submission, ACCEPTED. LB pending.

### Honest read
With only 58 labeled studies and no pretrained weights, the model overfits and
won't score well. But the GOAL of this spec was a valid, in-time, code-and-all
submission, and that is done.

## Exp 3: pretrained backbone + validation macro-AUC (2026-09-02)

Attached timm EfficientNet-B0 ImageNet weights as a Kaggle DATASET
(`seanconnolly/timm-efficientnet-b0-weights`) so we could load them with internet
off. Loaded fine (missing=2 = our 12-class head). K=12 slices, 20 epochs, 47/11
train/val split of the 58 labeled studies.

**Result (the honest signal): validation macro-AUC ~0.5 (0.53 / 0.50 / 0.54 /
0.45 at epochs 5/10/15/20), while train loss went to 0.006.** The model fully
memorizes the 47 training studies and generalizes to chance. Pretrained init did
not rescue it; more epochs made val WORSE (classic overfit).

### Conclusion (important, decides the next direction)
**58 labeled studies is not enough to learn 12 findings from raw MRI this way.**
This is the constraint the competition is built around. The reason there are 4,407
studies with radiology `Report` text but only 58 labels is that **the signal is in
the reports**. The productive path is NOT a better CNN, it is:
1. Mine the multilingual `Report` text -> weak labels for the other ~4,349 studies
   (keyword/negation rules per finding, or a multilingual zero-shot classifier).
2. Train the image model (or a text model) on THOUSANDS of labels instead of 58.
3. Optionally translate reports to English first, then a text classifier.

The image-only baseline has hit its ceiling at chance. Next spec iteration should
be text-driven.

## Exp 4: report-mining rules + test-report check (2026-09-02)

Built bilingual (EN/ES) keyword + negation rules mapping report text -> 12
findings, validated on the 58 labeled studies.

**Validation macro-AUC 0.592** (vs image model's 0.5). Per-finding:
- Works: MCL 0.78, Baker's 0.76, Lateral Meniscus 0.67, Contusion 0.66,
  Fracture 0.62, Synovitis 0.60, ACL 0.57.
- Still chance (rules wrong/crude): Effusion 0.47, Medial Meniscus 0.49,
  Medial OA 0.49, Lateral OA 0.49, PF OA 0.51.

**KEY STRUCTURAL FINDING: `test has Report column: False`.** The test set has NO
report text (reports exist for TRAIN only). So report-mining CANNOT directly
predict the hidden test set.

### The actual architecture this implies (the winning shape)
Reports are a LABEL SOURCE, not an inference input:
1. Mine reports -> weak labels for ALL 4,407 train studies (not just 58).
2. Train an IMAGE model on those thousands of weakly-labeled studies.
3. The image model predicts the test set (which only has images).
This is "distill the radiologist's words into an image model." The report rules
need improving first (the 5 chance-level findings), then use them as the label
generator for a much larger image-training set.

### Exp 4b: refined rules (presence vs structure+injury split)
Split cues into PRESENCE findings (positive if mentioned & not negated) and
STRUCTURE findings (positive only with injury language nearby). Macro-AUC
**0.592 -> 0.607**. Gains: PF OA 0.51->0.61, Medial OA 0.49->0.57, ACL held.
Still stuck: Effusion 0.49 (35/58 positive, little to separate), Medial Meniscus
0.50, Lateral OA 0.51. Good enough to use as the weak-label generator next.

### Plan locked in
1. Apply refined rules to ALL 4,407 train reports -> weak labels (as a dataset).
2. Train the image model on ~4,407 weakly-labeled studies (vs 58).
3. Image model predicts the test set. Validate against the 58 gold labels.

## Exp 5: v2 kernel (reports->weak-labels->image model) — IN FLIGHT

Kernel `seanconnolly/rsna-knee-v2` (T4, internet off, weights dataset attached).
Weak-labels a 1500-study subsample of the unlabeled pool from reports, trains
EfficientNet-B0 (K=6 slices, IMG=192, 6 epochs), and logs **gold_val_macroAUC**
against the 58 gold studies each epoch (the honest metric).

STATUS at session end (2026-09-02): kernel RUNNING (still reading DICOMs; the
~1500x6 reads off the mounted FS are the slow part).

### Exp 5 RESULT (2026-09-01): v2 ran, still chance
1500 studies weak-labeled from reports, EfficientNet trained (loss->0.02), but
**gold_val_macroAUC ~0.52** across all 6 epochs. More weak labels did NOT help the
image model learn. Best LB so far: **0.533** (from-scratch image baseline) vs
**smoke 0.500**. Leaderboard leaders are at **~0.95** — we're near the bottom.
Conclusion: the image path is stuck at chance; the signal is the reports. Our
rule-based report miner (0.607 gold) already beats every image attempt.

### Exp 6 (2026-09-01): zero-shot NLI report labeling (FREE, in flight)
Uploaded `seanconnolly/mdeberta-v3-base-mnli-xnli` (multilingual NLI, ~530MB) as
a dataset so it runs offline. Kernel `rsna-knee-nli-labels` scores each finding
as entailment(report -> hypothesis), validates on the 58 gold, and writes
nli_train_labels.csv for all 4,407 studies. RUNNING at session pause (labeling
4,407 reports x 12 hypotheses on the T4 is slow).
Goal: beat the 0.607 rules on gold. If it does, use nli_train_labels.csv as the
image model's training labels AND consider it may transfer better.

### NEXT SESSION - resume here
1. `kaggle kernels status seanconnolly/rsna-knee-v2`
   - COMPLETE -> pull output, read the gold_val_macroAUC per epoch:
       * >0.5 clearly -> the reports->images approach works. Submit v2:
         `kaggle competitions submit rsna-knee-abnormality-detection -k seanconnolly/rsna-knee-v2 -f submission.csv -v <ver> -m "reports weak-labels + EfficientNet"`
         then scale WEAK_TRAIN_N / K / EPOCHS within the GPU cap.
       * ~0.5 -> weak-labels-to-images isn't extracting yet; try per-plane slice
         selection + fix the 5 chance-level report rules (Effusion, Medial
         Meniscus, Lateral OA especially) before scaling.
   - ERROR/timeout -> shrink WEAK_TRAIN_N (e.g. 500) to prove it finishes, then grow.
2. Record gold-val + LB numbers here; update STATUS.md.
3. Blog post stays draft; finish offline + publish when the model is real.

## Submission History

| # | Model | LB (public) | Notes |
|---|-------|:-----------:|-------|
| 1 | Prevalence smoke test | 0.500 | Proved the pipeline |
| 2 | 2.5D EfficientNet-B0 (from scratch, 58 labels) | pending | Real model, in-time, accepted |
