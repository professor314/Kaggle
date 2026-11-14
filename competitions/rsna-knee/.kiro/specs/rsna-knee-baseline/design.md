# Design — RSNA Knee Baseline (Conservative, Time-Safe)

## Overview

A single Kaggle GPU notebook that reads the competition CSVs as its file index,
trains a light 2.5D CNN on the small labeled set, and writes a valid
`submission.csv`. The design's first priority is **finishing in time and always
producing a valid submission**, with a prevalence fallback so the submission step
never depends on training succeeding.

This runs entirely on Kaggle (data mounted at `/kaggle/input`, trained on their
T4). Nothing is downloaded or trained locally. We push the kernel with
`kaggle kernels push` and submit the completed run.

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| File indexing | Use `*_series.csv` / `train.csv` as the index | VERIFIED: walking ~820k DICOMs is far too slow (explore kernel stalled on it) |
| Model | timm EfficientNet-B0, pretrained, 2D | Small, fast, robust; reuses our CNN/GPU patterns |
| Study aggregation | Sample K slices/study, mean-pool predictions | "2.5D" without 3D conv cost; caps runtime per study |
| Slice budget K | small fixed number (e.g. 8-16) per study | Bounds inference time regardless of series length |
| Output head | 12 independent sigmoids | Multi-label, matches macro-AUC scoring |
| Loss | BCEWithLogitsLoss | Standard multi-label |
| GPU/threads | via `kaggle_ml_toolkit.compute` | One knob, GPU when present |
| Fallback | per-finding train prevalence for all test studies | Guarantees a valid, honest submission even if training is skipped/fails |
| Precision | fp16 autocast on GPU | Faster, fits memory |

## Architecture / flow

```
Kaggle notebook (T4)
  1. Locate data dir under /kaggle/input (direct join, no deep walk)
  2. Load CSVs: sample_submission, test(_series), train(+labels)
  3. Identify the 12 finding columns from sample_submission header  [CONFIRM]
  4. Build study -> [slice paths] index from *_series.csv (cap K slices)
  5. Compute per-finding train prevalence (fallback vector)
  6. IF enough labeled studies:
        - DICOM -> normalized 224x224x3 tensor (windowing, repeat channel)
        - train EfficientNet-B0 (12 sigmoids, BCE, few epochs) on labeled studies
        - hold out a fold -> local macro-AUC (logged)
     ELSE:
        - skip training; use prevalence fallback
  7. Inference on every test study:
        - read up to K slices, predict, mean-pool -> 12 probs
        - on any read error -> prevalence fallback for that study
  8. Assemble submission.csv in exact sample format -> /kaggle/working
```

## DICOM preprocessing (the genuinely new code)

- `pydicom.dcmread(path).pixel_array` -> 2D array (VERIFIED shape/dtype pending
  from explore; treat as single-channel MRI).
- Intensity normalization: clip to a percentile window (e.g. 1st-99th) then scale
  to [0,1]; MRI has no fixed HU scale, so per-slice normalization is safest.
- Resize to 224x224 (cv2 or PIL); repeat to 3 channels for the pretrained net.
- Slice sampling: take K slices evenly spaced through the series (middle slices
  carry the most signal), not the whole stack.

## Correctness properties

1. **Valid submission shape** — output rows == sample_submission rows; columns and
   order identical; all 12 values in [0,1]; no NaN/blank.
2. **Never crashes on bad data** — any unreadable study/slice yields the fallback,
   not an exception.
3. **Bounded work** — slices read per study <= K; no full-tree glob.
4. **Fallback validity** — prevalence vector is itself a legal submission.

## Testing strategy

- Local unit tests (tiny synthetic DICOMs via pydicom) for: preprocessing shape,
  prevalence fallback vector, submission assembly matches a mock sample format.
- The Kaggle run is the integration test: it must reach COMPLETE and produce a
  submission within the cap. First push runs with tiny settings (few epochs,
  small K, maybe a subset of test) to prove it finishes, then scale up.

## Deployment

1. Push explore kernel (done) to confirm CSV/label layout + DICOM shape.
2. Push the baseline kernel with `machine_shape: "NvidiaTeslaT4"`, GPU on,
   internet on only if timm weights need downloading (else attach a weights dataset).
3. Confirm COMPLETE + valid submission.csv, submit the kernel version.
4. Record result; only then iterate (more slices, real training, augmentation).
