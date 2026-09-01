# Phase 3 — Progress Tracker

## Sprint 1: PyTorch + NLP Transformers

### Setup (DONE)
- [x] Python 3.12 installed via `uv` (PyTorch doesn't support 3.14 yet)
- [x] Created `.venv-dl` virtual environment (Python 3.12)
- [x] PyTorch 2.6.0+cu124 installed in .venv-dl
- [x] torchvision installed
- [x] Installed transformers 5.16, datasets, accelerate
- [x] Installed pandas, scikit-learn, kaggle, matplotlib/seaborn/scipy/pyyaml
- [x] Verified GPU: RTX 4080 Laptop GPU (12GB usable) — NOTE: not the RTX 3090
      the plan assumed; VRAM budget is smaller, so cap model size accordingly.
- [x] Built `TransformerClassifier` in `kaggle_ml_toolkit/deep_learning.py`
      (sklearn-style fit/predict/predict_proba/save/load, fp16 auto on CUDA)
- [x] Wired lazy import in `__init__.py` so the main 3.14 env still imports the
      toolkit without torch.

### NLP Disaster Tweets (0.801 → 0.842 F1) ✅
- [x] Fine-tune DistilBERT (3 epochs/fold, ~38s/fold on GPU)
- [x] Evaluate on 5-fold CV → OOF F1 0.804
- [x] Submit DistilBERT → **LB 0.836** (+0.035; CV badly underestimated LB)
- [x] Build DistilBERT+TF-IDF ensemble → OOF 0.808, **LB 0.838**
- [x] Add BERTweet 3-way ensemble → OOF 0.812, **LB 0.842** (new best)
- [ ] Next lever: roberta-large or a non-transformer base (BERTweet corr was 0.91)

### Digit Recognizer (0.984 → 0.9945 acc) ✅
- [x] Built toolkit `CNNClassifier` (SmallCNN + augment + OneCycleLR + fp16)
- [x] Built `gpu_utils` (get_device, gpu_info, seed_everything, autocast)
- [x] Trained 15 epochs → val 0.9964, **LB 0.99385** (+0.010 over sklearn)
- [x] Caught & fixed a train/test normalization bug (see experiments_log)
- [x] 5-fold CNN ensemble + TTA → **LB 0.99445** (new best)
- [ ] Next: more TTA views / more folds for marginal gains

### Toolkit additions this sprint
- [x] `TransformerClassifier` — HF fine-tuning, sklearn-style API
- [x] `CNNClassifier` — grayscale image CNN, sklearn-style API
- [x] `gpu_utils` — device/seed/mixed-precision helpers
- [x] Lazy imports in `__init__.py` so the 3.14 tabular env still works

### Contradictory-Watson (multilingual NLI) 🟡 in flight
- [x] Extended `TransformerClassifier` with sentence-pair (premise/hypothesis) support
- [x] Fine-tuned XLM-RoBERTa locally → val ~0.69 (underfits at 3 epochs)
- [x] Diagnosed: notebook-only comp (direct CSV submit = 400 by design)
- [x] Diagnosed: Kaggle **P100 GPU incompatible** with shipped PyTorch (sm_60 vs
      sm_70+) — the empty-log ERROR. FIXED by switching kernel accelerator to **T4**.
- [x] Kernel v8 (single-stage, T4) running on Kaggle
- [ ] CONFIRM v8 completed + scored on LB (check `kernels status`)
- [ ] To hit 0.85+: two-stage MNLI pretrain then fine-tune. First attempt OOM'd
      the kernel (tokenized full 393K MNLI at once). Retry with subsample + lazy
      `.map()` tokenization, on the T4.

### LLM-Finetuning (not started)
- [ ] Investigate competition format
- [ ] LoRA fine-tuning if applicable

---

## How to Resume

Environment is fully set up. Use the DL venv's python directly (do NOT rely on
`activate` in scripts); always call it by absolute path:

```
& "c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Projects\Kaggle\.venv-dl\Scripts\python.exe" <script>
```

Installed in .venv-dl: torch 2.6.0+cu124, transformers 5.16, datasets, accelerate,
lightgbm, xgboost, catboost, timm-free CNN, pandas, sklearn, kaggle, matplotlib.

Verify GPU (local training):
```python
import torch
print(torch.cuda.is_available())            # True
print(torch.cuda.get_device_name(0))         # NVIDIA GeForce RTX 4080 Laptop GPU
```

First thing to check on resume: the Watson kernel status (see Contradictory-Watson
section above). Pipeline scripts already exist under each competition's `notebooks/`.

### Gotchas learned (don't re-discover these)
- Shell is PowerShell: call exes with `& "..."`, use `;` not `&&`.
- Kaggle notebook comps reject direct CSV submit (HTTP 400) — submit via kernel.
- Kaggle P100 GPU breaks current PyTorch (sm_60 vs sm_70+) — set kernel
  `"accelerator": "nvidiaTeslaT4"`.
- CNN train/test scaling must be identical — verify test prediction distribution
  looks sane (not skewed) before submitting.

---

## Note on Python versions
- **3.14** — main env, used for all tabular/scikit-learn/LightGBM work
- **3.12 (.venv-dl)** — deep learning only (PyTorch, transformers, timm)
- Both can coexist. Use the appropriate python for each task.
