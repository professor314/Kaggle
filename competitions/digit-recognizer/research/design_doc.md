# Digit Recognizer (MNIST) — Design Doc

**Competition:** Digit Recognizer
**Metric:** Accuracy | **Best LB:** 0.99446 (5-fold CNN + TTA)
**Full log:** see `../experiments_log.md`

## Domain research
Classic MNIST: 28×28 grayscale handwritten digits, 42K train / 28K test. A
well-known ceiling around 0.995+; the interesting work is the deep-learning
pipeline and avoiding train/test preprocessing bugs.

## Prior art
CNNs with light augmentation and test-time augmentation are the standard route
past 0.99. Ensembling folds gives small but real gains near the ceiling.

## Feature engineering plan (what shipped)
No hand features (raw pixels). SmallCNN: 2×[conv-bn-relu ×2 + pool + dropout]
+ FC head. Light RandomAffine augmentation, OneCycleLR, AdamW, fp16.

## Design decisions
| Decision | Choice | Rationale |
|---|---|---|
| Baseline | sklearn PCA+SVM (0.984) | Establish classical floor first |
| Model | Toolkit `CNNClassifier` (SmallCNN) | Fast (~1 min on RTX 4080 Laptop) |
| Augmentation | RandomAffine + TTA (5 views) | Robustness, small accuracy gain |
| Ensemble | 5-fold CNN, averaged probs | +0.0006 near the ceiling |
| Sanity check | Verify test pred distribution ~2500-3200/class | Catches scaling bugs |

## Key finding / bug caught
A train/test normalization mismatch (÷255 applied in `fit` but not `predict`)
produced a wildly skewed test distribution despite 0.996 val accuracy. Fixed by
moving all scaling inside `_to_tensor`. The prediction-distribution check is now
a required pre-submit sanity gate.
