# Digit Recognizer (MNIST) — Experiment Log

**Competition:** Digit Recognizer
**Metric:** Accuracy
**Dataset:** 42,000 train / 28,000 test, 28×28 grayscale (pixel columns 0-255)

---

## Experiment 1: sklearn models (PCA+SVM / RF / MLP)

**Date:** 2026-08-22
**LB accuracy:** 0.98371 (PCA+SVM, val 0.9843)
**Rank:** ~435
Baseline established with classical ML — no deep learning.

---

## Experiment 2: CNN (Phase 3)

**Date:** 2026-08-27
**Val accuracy:** 0.9964 | **LB accuracy:** 0.99385 (+0.010 over sklearn)
**Model:** Toolkit `CNNClassifier` — SmallCNN (2×[conv-bn-relu ×2 + pool + dropout]
+ FC head), light RandomAffine augmentation, OneCycleLR, AdamW, fp16, 15 epochs.
Training time: ~1 min total on the RTX 4080 Laptop.

### Bug caught during development (worth remembering)
The first full run reported val_acc 0.9964 but produced a **wildly skewed test
prediction distribution** (class 1: 134 preds, class 8: 7735, vs. expected
~2800 each). Root cause: a **train/test normalization mismatch**. `fit` divided
X by 255 before computing mean/std, but `predict` (via `_to_tensor`) applied
mean/std to raw 0-255 test data without the /255 step — so test inputs were
~255× off-scale. Validation looked perfect only because it reused the correctly
scaled training tensors.

**Fix:** moved all scaling (÷255 + standardize) inside `_to_tensor`, driven by a
learned `_scale` factor, so train and inference are transformed identically. A
sane test-prediction distribution (~2500-3200/class) is now a sanity check we
verify before submitting.

### What to try next
- Test-time augmentation (TTA) and multi-model averaging → ~99.5%+
- 5-fold CNN ensemble
- Train on the full MNIST (70K) via torchvision, not just the 42K Kaggle subset

---

## Submission History

| # | Model | Val Acc | LB Acc | Notes |
|---|-------|:-------:|:------:|-------|
| 1 | sklearn PCA+SVM | 0.984 | 0.98371 | Classical ML baseline, rank ~435 |
| 2 | CNN + augmentation (15 ep) | 0.996 | 0.99385 | Phase 3 CNNClassifier |
| 3 | 5-fold CNN ensemble + TTA | 0.9937 | **0.99446** | +0.0006; ensembling + test-time aug |

## Experiment 3: 5-fold CNN ensemble + TTA (Phase 3)

**Date:** 2026-08-27
**Mean fold val acc:** 0.9937 | **LB acc:** 0.99446 (new best)
Trained 5 CNN folds and averaged predicted probabilities over the clean image
plus 4 lightly-augmented views per test image (test-time augmentation). Small
but real gain over the single model, as expected near the MNIST ceiling.
