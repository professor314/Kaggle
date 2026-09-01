# Phase 3: Deep Learning & GPU Compute

## Hardware Available

- **GPU:** NVIDIA RTX 3090 (24GB GDDR6X)
- **RAM:** 64GB DDR5
- **CPU:** Intel i9-12900K (16-core)

This is a serious ML rig. The 24GB VRAM can handle most models that fit on a single GPU, including large transformers.

---

## What to Install

```bash
# PyTorch with CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Hugging Face ecosystem
pip install transformers datasets accelerate tokenizers

# Image classification
pip install timm albumentations

# LightGBM GPU (rebuild from source with USE_GPU=ON, or use CUDA version)
pip install lightgbm --config-settings=cmake.define.USE_CUDA=ON

# Experiment tracking
pip install wandb
```

---

## Competitions Unlocked by GPU

### Tier 1: Immediate wins (TF-IDF → transformers)

| Competition | Current Score | GPU Approach | Expected Score |
|---|---|---|---|
| **NLP Disaster Tweets** | 0.801 F1 | DistilBERT fine-tuning (3 epochs, 5 min) | 0.83-0.85 |
| **Contradictory-Watson** | ~0.33 acc | XLM-RoBERTa multilingual NLI | 0.85+ |
| **LLM-Finetuning** | pending | LoRA fine-tuning on competition model | competitive |

### Tier 2: Image classification (new territory)

| Competition | Current Score | GPU Approach | Expected Score |
|---|---|---|---|
| **Digit Recognizer** | rank 435 | ResNet/EfficientNet, 5 epochs | top 50 |
| **TPU-Getting-Started** (flowers) | pending | EfficientNet-B4 or ViT | competitive |

### Tier 3: Advanced (bigger lift)

| Competition | Prize | GPU Approach |
|---|---|---|
| **Kaggriculture** | $50K | Neural agent (PPO/DQN for game strategy) |
| **ARC Prize** | $700K-$850K | Novel reasoning architectures |
| **Pokemon TCG** | $240K | RL agent with neural value function |

---

## Toolkit Additions Needed

### 1. `kaggle_ml_toolkit/text_classifier.py`
```python
class TransformerClassifier:
    """Fine-tune HuggingFace transformers for text classification."""
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2): ...
    def train(self, train_texts, train_labels, val_texts, val_labels, epochs=3): ...
    def predict(self, texts): ...
    def save(self, path): ...
    def load(self, path): ...
```

### 2. `kaggle_ml_toolkit/image_classifier.py`
```python
class ImageClassifier:
    """Train image classifiers using timm pretrained models."""
    def __init__(self, model_name="efficientnet_b0", num_classes=10): ...
    def train(self, train_loader, val_loader, epochs=10): ...
    def predict(self, test_loader): ...
    def get_transforms(self, train=True): ...
```

### 3. `kaggle_ml_toolkit/gpu_utils.py`
```python
def get_device(): ...  # auto-detect GPU/CPU
def gpu_info(): ...    # print GPU name, VRAM, utilization
def enable_mixed_precision(): ...  # fp16 training for 2x speed
```

### 4. Update `time_series.py`
- Add GPU-accelerated LightGBM params (`device: "gpu"`)
- Add neural time series (N-BEATS, TFT) as alternatives

### 5. Experiment tracking
- Weights & Biases integration for logging training curves
- Or simple CSV-based tracking if we want to stay lightweight

---

## Execution Order

### Sprint 1: Install + NLP (1 session)
1. Install PyTorch + transformers + CUDA
2. Verify GPU works: `torch.cuda.is_available()`
3. Fine-tune DistilBERT on NLP Disaster Tweets → submit
4. Fine-tune XLM-RoBERTa on Contradictory-Watson → push kernel

### Sprint 2: Vision (1 session)
1. Install timm + albumentations
2. Train EfficientNet on Digit Recognizer → submit
3. Train on TPU-Getting-Started flowers → push kernel

### Sprint 3: LightGBM GPU + Store Sales (1 session)
1. Install LightGBM with CUDA support
2. Rerun Store Sales with GPU acceleration (3-10x faster iterations)
3. Try more aggressive hyperparameter search with the speed gain
4. Rerun Playground S6E8 with GPU LightGBM

### Sprint 4: Advanced (ongoing)
1. Kaggriculture neural agent
2. LLM fine-tuning competition
3. Any new Featured competitions worth the prize money

---

## Design Decisions

### Why PyTorch over TensorFlow?
- Better debugging (eager mode by default)
- Hugging Face transformers are PyTorch-first
- timm library (best pretrained image models) is PyTorch
- More active community for cutting-edge research

### Why DistilBERT first?
- 6x faster than BERT, only 3% worse
- Fits easily on any GPU
- 3-5 minutes to fine-tune on small datasets (7K-12K rows)
- Good enough to jump from 0.80 → 0.83+ on tweet classification

### Why EfficientNet for images?
- Best accuracy/compute tradeoff
- Pretrained on ImageNet (transfer learning)
- Scales from B0 (small) to B7 (large) based on VRAM
- With 24GB VRAM we can use B4 or even B5

### Mixed precision (fp16)?
- Yes, always. RTX 3090 has excellent fp16 performance
- 2x speed, half the VRAM usage, no accuracy loss in practice
- PyTorch makes this trivial: `torch.amp.autocast`
