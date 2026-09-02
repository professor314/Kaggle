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

## Submission History

| # | Model | Local macro-AUC | LB | Notes |
|---|-------|:---------------:|:--:|-------|
| — | (baseline pending) | — | — | see spec `.kiro/specs/rsna-knee-baseline` |
