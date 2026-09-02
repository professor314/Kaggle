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
submission, and that is done. Next levers (future): attach timm weights as a
Kaggle dataset to restore pretrained init; mine the report text (multimodal);
more slices; per-fold validation.

## Submission History

| # | Model | LB (public) | Notes |
|---|-------|:-----------:|-------|
| 1 | Prevalence smoke test | 0.500 | Proved the pipeline |
| 2 | 2.5D EfficientNet-B0 (from scratch, 58 labels) | pending | Real model, in-time, accepted |
