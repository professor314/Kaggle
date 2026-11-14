# Requirements — RSNA Knee Abnormality Detection (Conservative Baseline)

## Introduction

Build a **conservative, guaranteed-to-finish baseline** for the RSNA Knee
Abnormality Detection Kaggle competition ($77,000, closes 2026-10-22). The task:
predict **12 binary findings per knee MRI study**, scored by **macro-averaged ROC
AUC**. The dataset is ~4,407 train studies / 24,371 series / ~820,000 DICOM
slices (100GB+), and only 58 studies carry expert labels.

The overriding goal for THIS spec is **not** a high score. It is to establish a
complete, correct pipeline that **runs end-to-end inside a single Kaggle GPU
notebook within the time limit and produces a valid submission**. Score
improvements come in later iterations once the plumbing is proven.

### Verified facts (from the exploration kernel, 2026-09-02)
- VERIFIED: data at `/kaggle/input/competitions/rsna-knee-abnormality-detection`.
- VERIFIED: files: `sample_submission.csv`, `test.csv` (StudyInstanceUID only),
  `test_series.csv`, `train.csv`, `train_series.csv`, plus `train_series/` and
  `test_series/` DICOM trees (`<study>/<series>/<slice>.dcm`).
- VERIFIED: the **12 findings** (submission columns after StudyInstanceUID):
  `ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA,
  Effusion, Synovitis, Baker's, Contusion, Fracture`. sample_submission fills 0.5.
- VERIFIED: `train.csv` = 4407 studies, columns `StudyInstanceUID, Report` + the
  12 findings. Labels present for exactly **58/4407** studies (rest blank). The
  `Report` text is multilingual (e.g. Spanish) — the multimodal signal, out of
  scope for the baseline.
- VERIFIED: `train_series.csv` (24,371 rows) / `test_series.csv` give per-series
  metadata: `Fluid_Sensitive, Fat_Suppression, Anatomical_Plane` (Axial/Sagittal).
  One study ~= 5 series; one series ~= 33 slices.
- VERIFIED: DICOM slice = 512x512, int16, Modality MR, MONOCHROME2, value range
  well outside [0,255] (e.g. -663..9999) -> percentile windowing needed. PixelSpacing
  ~0.293mm, SliceThickness 4mm.
- VERIFIED: recursively globbing all `.dcm` STALLS the kernel — index images from
  the CSVs + shallow listdir, never a full-tree walk.

## Requirements

### Requirement 1: Runs within Kaggle limits
**User Story:** As the competitor, I want the notebook to finish within Kaggle's
GPU limits, so that it always produces a scored submission.

#### Acceptance Criteria
1. WHEN the kernel runs on a Kaggle T4 (via `machine_shape: "NvidiaTeslaT4"`) THE notebook SHALL complete within the competition's GPU time cap with margin.
2. THE pipeline SHALL index images from the provided CSVs, and SHALL NOT enumerate the DICOM tree with a recursive filesystem walk.
3. WHERE inference time is at risk, THE pipeline SHALL cap the number of slices read per study to a fixed budget.
4. THE notebook SHALL write `submission.csv` to `/kaggle/working/` in the exact `sample_submission.csv` format (same id column, same 12 finding columns, same row order/count).

### Requirement 2: Valid submission format
**User Story:** As the competitor, I want the output to match the sample exactly,
so that the submission is accepted and scored.

#### Acceptance Criteria
1. THE submission SHALL contain one row per test study (matching `sample_submission.csv`).
2. THE submission SHALL contain the 12 finding columns as probabilities in [0, 1].
3. IF a study cannot be read THEN THE pipeline SHALL emit a safe default (e.g. the per-finding train prevalence) rather than crash or leave blanks.
4. THE column names and order SHALL match `sample_submission.csv` exactly.

### Requirement 3: Conservative 2.5D image model
**User Story:** As the competitor, I want a simple, robust model, so that it is
reliable before I invest in a stronger one.

#### Acceptance Criteria
1. THE baseline SHALL use a pretrained 2D CNN (timm, e.g. EfficientNet-B0) applied to a small fixed number of sampled slices per study, aggregated (mean) to a per-study prediction. (No 3D conv, no multimodal text in the baseline.)
2. THE model SHALL output 12 independent sigmoids (multi-label), trained with binary cross-entropy.
3. WHERE GPU is available THE training SHALL use it via the toolkit `compute` config; WHERE not, it SHALL still run (smaller/skip train) so the submission step never depends on training succeeding.
4. THE image preprocessing SHALL read DICOM pixel data, apply intensity normalization/windowing, resize to a fixed square (e.g. 224), and handle single-channel MRI by repeating to 3 channels.

### Requirement 4: Weak-label-aware training
**User Story:** As the competitor, I want training to work despite very few
labels, so that the baseline is honest about what it learned.

#### Acceptance Criteria
1. THE pipeline SHALL train only on studies that have labels (expected ~58) and SHALL clearly log how many labeled studies were used.
2. IF the labeled set is too small to train a stable model THEN THE baseline SHALL fall back to predicting per-finding train prevalence for all test studies (a valid, honest baseline) and SHALL log that it did so.
3. THE pipeline SHALL compute a local validation macro-AUC on a held-out slice of the labeled studies when feasible, and log it.

### Requirement 5: Prove the path before the model (smoke test)
**User Story:** As the competitor, I want to confirm data access, timed DICOM
reading, submission format, and the submit flow BEFORE writing the model, so that
I don't sink time into code that can't be delivered.

#### Acceptance Criteria
1. THE first Kaggle push SHALL be a "pipeline smoke test" that exercises the full shape of the real job with a trivial predictor (train prevalence), NO CNN.
2. THE smoke test SHALL read a small number of slices per test study (proving timed multi-DICOM reads work) and SHALL write a valid `submission.csv`.
3. THE smoke test SHALL be submitted to the leaderboard to confirm the code-competition submit flow works end-to-end (a low-but-valid score is success).
4. Only AFTER the smoke test scores SHALL the real model be built.

### Requirement 6: Reproducible + documented
**User Story:** As the maintainer, I want the run documented, so that iterations
build on a known baseline.

#### Acceptance Criteria
1. THE kernel SHALL print a compute summary and all key counts (labeled studies, slices/study budget, epochs) to the log.
2. THE approach, decisions, and result SHALL be recorded in the competition `research/design_doc.md` and `experiments_log.md`.
3. Facts SHALL be tagged VERIFIED/ASSUMED per the toolkit's anti-loop steering.

## Out of scope (future iterations, NOT this baseline)
- 3D CNNs / full-volume models.
- Multimodal fusion with the radiology-report text.
- Heavy augmentation, TTA, multi-model ensembles.
- Pseudo-labeling the ~4,349 unlabeled studies.
- Chasing leaderboard rank. This spec targets "a valid scored submission that ran in time."
