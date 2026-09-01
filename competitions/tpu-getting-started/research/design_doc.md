# TPU Getting Started (Flower Classification) — Design Doc

**Competition:** Petals to the Metal – Flower Classification on TPU
**Metric:** Macro F1 | **Status:** kernel drafted, not yet scored
**Type:** Code Competition (notebook submission)

## Domain research
Fine-grained image classification: 104 flower species, TFRecord image data
designed for TPU. This is a Phase 3 (deep learning / vision) target and new
territory for the toolkit.

## Prior art
Winning approach: transfer learning from a strong pretrained backbone
(EfficientNet / ViT via `timm`), heavy image augmentation, and TTA. TPU or GPU
training required.

## Design decisions
| Decision | Choice | Rationale |
|---|---|---|
| Backbone | timm EfficientNet (transfer learning) | Strong, standard for fine-grained vision |
| Compute | Kaggle GPU (T4) kernel | P100 breaks shipped PyTorch; TPU path optional |
| Submission | Notebook/code submission | Code competition |
| Augmentation | Flips, rotation, color jitter + TTA | Fine-grained classes need robustness |

## Status / next
Kernel drafted. Next: finalize the timm pipeline, run on the GPU kernel, verify
prediction distribution, submit. Candidate for an `ImageClassifier` addition to
the toolkit for reuse on future natural-image competitions.
