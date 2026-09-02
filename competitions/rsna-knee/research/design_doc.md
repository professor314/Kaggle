# RSNA Knee Abnormality Detection — Research & Design Doc

**Competition:** RSNA Knee Abnormality Detection ($77,000) | **Deadline:** 2026-10-22
**Task:** 12 binary findings per knee MRI study | **Metric:** macro-averaged ROC AUC
**Type:** Code Competition (notebook submission, data on Kaggle, 100GB+)

## Domain research
Knee MRI multi-label abnormality detection. The 12 findings (VERIFIED from
sample_submission): ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA,
Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture. Studies are
multi-series MR (Axial/Sagittal planes, fluid-sensitive / fat-suppressed
variants), ~5 series/study, ~33 slices/series, 512x512 int16 MONOCHROME2.

The defining challenge: **4,407 train studies but only 58 are expert-labeled**
(the rest have only a free-text `Report`, multilingual). This is a weak/scarce
label problem. The intended winning path is multimodal (mine labels from the
reports, or pretrain on report-derived pseudo-labels), which is out of scope for
the baseline.

## Prior art
- Public writeups (e.g. an 18th-place pipeline) confirm: 2.5D per-slice CNNs
  aggregated per study are the standard backbone; report-text mining is the edge.
- Never download the data (100GB); train in a Kaggle notebook on their GPU.

## Feature/model plan (baseline)
- Index images from the CSVs (VERIFIED: full-tree glob stalls).
- 2.5D: sample K evenly-spaced slices/study, timm EfficientNet-B0, mean-pool
  logits, 12 sigmoids, BCE.
- Percentile intensity windowing (int16, no fixed scale) -> [0,1] -> 224 -> 3ch.
- Prevalence fallback so a valid submission is guaranteed.

## Design decisions
| Decision | Choice | Rationale |
|---|---|---|
| First goal | Valid, in-time submission (not score) | 100GB + GPU caps make "it runs" the hard part |
| Image index | CSVs, not filesystem walk | VERIFIED: recursive .dcm glob stalls the kernel |
| Model | timm EfficientNet-B0, 2D, mean-pool slices | Small/robust; reuses toolkit CNN/GPU patterns |
| Labels | train on the 58 labeled studies only | Rest are unlabeled (Report text only) |
| Fallback | per-finding train prevalence | Guarantees a legal submission if training fails/skips |
| GPU | Kaggle T4 via `machine_shape: NvidiaTeslaT4` | VERIFIED fix from the Watson saga |
| Excluded (baseline) | 3D conv, report-text multimodal, TTA, ensembles, pseudo-labels | Save for later iterations |

## Status (2026-09-02)
- Exploration + smoke + baseline kernels all COMPLETE. Two submissions accepted:
  smoke (LB 0.500) and the 2.5D EfficientNet baseline (accepted, LB pending).
- 8 local unit tests pass (`test_rsna_lib.py`).

## VERIFIED gotcha: no internet in submitted kernels
This competition rejects any submission whose kernel used internet
("Your Notebook cannot use internet access in this competition", HTTP 400).
So the kernel runs with `enable_internet: false`. Consequence: no runtime
download of pretrained weights -> baseline trains EfficientNet-B0 FROM SCRATCH.

## Data profiling (VERIFIED 2026-09-02) — decisions before modeling
- **Test has NO reports** (`test.csv` = StudyInstanceUID only; no other CSVs).
  Reports are train-only -> they are a LABEL SOURCE; the scored model is IMAGE-based.
- **Labels are clean**: values are exactly {0.0, 1.0}; blanks are truly UNLABELED
  (58/4407), not zeros. Do NOT treat blanks as negatives.
- **Reports clean**: 0 empty, 0 dup studies, length 52..4743 (median 977).
- **DICOM**: uint16, MIXED dimensions (416/512/560/768...) -> resize REQUIRED;
  per-series value ranges vary wildly -> per-image percentile windowing REQUIRED
  (no fixed scale).
- **Series**: 3-14 per study (median 5). Planes Sagittal 9864 / Coronal 8609 /
  Axial 5898. `Fat_Suppression` is a perfect duplicate of `Fluid_Sensitive`
  (drop one). Plane is informative: meniscus/ligaments best on sagittal,
  collaterals/OA on coronal, effusion on axial -> select slices per-plane, don't
  blend all series blindly.

## Next iterations (post-baseline)
- Attach timm EfficientNet weights as a Kaggle dataset -> restore pretrained init
  offline (the single biggest expected quality lever given 58 labels).
- Mine the multilingual `Report` text (multimodal) for weak labels on the other
  ~4,349 studies.
- More slices / per-fold macro-AUC validation / light augmentation.
