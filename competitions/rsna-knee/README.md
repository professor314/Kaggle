# RSNA Knee Abnormality Detection

A conservative, time-safe baseline for the [RSNA Knee Abnormality Detection]
(https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) Kaggle
competition ($77,000, closes 2026-10-22). Predict **12 binary findings per knee
MRI study**, scored by **macro-averaged ROC AUC**.

This is part of the [Kaggle ML Toolkit](../../README.md) project.

## The problem in one paragraph

4,407 MRI studies, 24,371 series, ~820,000 DICOM slices, 100GB+. You never
download it, everything runs in a Kaggle GPU notebook with the data mounted at
`/kaggle/input`. The catch: only **58 of the 4,407 studies are expert-labeled**.
The rest have only a free-text radiology `Report` (multilingual). So it is really
a scarce-label, multimodal medical-imaging problem.

## Approach (baseline)

The first goal is not a high score, it is a pipeline that **finishes inside the
GPU time limit and always produces a valid submission**. Strategy:

1. **Index images from the CSVs**, never a recursive DICOM-tree walk (that stalls
   the kernel).
2. **2.5D**: sample K evenly-spaced slices per study, mean-pool, feed a pretrained
   timm **EfficientNet-B0** with 12 sigmoid outputs (BCE loss), trained on the 58
   labeled studies.
3. **Prevalence fallback**: if training can't run or a study can't be read, emit
   the per-finding train prevalence, so the submission is always valid.

Explicitly out of scope for the baseline (future iterations): 3D CNNs, mining the
report text (multimodal), TTA, ensembles, pseudo-labeling the unlabeled studies.

## How it was de-risked

Before writing the model, two cheap Kaggle kernels proved the plumbing:

- `kernel/explore.py` — confirmed the data layout, the 12 finding names, the 58
  labeled studies, and DICOM shape (512x512 int16 MR). Found that recursively
  globbing all `.dcm` stalls the kernel.
- `kernel/smoke.py` — the full job shape with a trivial prevalence predictor. It
  read K slices per test study (timed), wrote a valid `submission.csv`, and was
  accepted by the leaderboard. Only after that did the real model get built.

## Files

```
rsna-knee/
├── rsna_lib.py            # reusable helpers (data dir, preprocessing, fallback, submission)
├── test_rsna_lib.py       # local unit tests (synthetic DICOMs, no Kaggle needed)
├── kernel/                # kernel scripts (explore, smoke, baseline)
├── kernel_smoke/          # smoke kernel push folder
├── kernel_baseline/       # baseline kernel push folder (GPU T4)
├── research/design_doc.md
└── experiments_log.md
```

## Run it

Everything runs on Kaggle (nothing local). Push a kernel and submit the completed
version (it is a code competition, CSV upload is rejected):

```bash
kaggle kernels push -p kernel_baseline
# when COMPLETE:
kaggle competitions submit rsna-knee-abnormality-detection \
  -k seanconnolly/rsna-knee-baseline -f submission.csv -v <version> -m "baseline"
```

Local unit tests:

```bash
pytest test_rsna_lib.py -q
```

## Gotchas (learned the hard way)

- **Kaggle GPU model** is selected by `machine_shape: "NvidiaTeslaT4"` in the
  kernel metadata, not `accelerator`. Omit it and you silently get a P100 that
  breaks modern PyTorch.
- **Do not walk the DICOM tree.** Index from the CSVs.
- **Notebook submission only.** Submit the kernel version, not a CSV upload.
