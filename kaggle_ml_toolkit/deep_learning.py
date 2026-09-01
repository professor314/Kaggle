"""Deep learning model support for the Kaggle ML Toolkit (Phase 3).

Provides a scikit-learn-style wrapper around HuggingFace transformers for
text classification, so fine-tuning a model fits the same mental model as the
rest of the toolkit (fit / predict / predict_proba / save / load).

Requirements (installed in the dedicated `.venv-dl` Python 3.12 environment):
- torch >= 2.0 with CUDA
- transformers, datasets, accelerate
- scikit-learn, pandas, numpy

Design notes:
- PyTorch does not yet support Python 3.14, so all deep learning work runs in
  the separate `.venv-dl` (Python 3.12) environment. The tabular toolkit keeps
  running on the main 3.14 env.
- Mixed precision (fp16) is enabled automatically on CUDA for ~2x speed and
  lower VRAM use, which matters on the 16GB RTX 4080 Laptop GPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


def get_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_info() -> dict:
    """Return basic GPU information, or an empty dict if no GPU is present."""
    import torch

    if not torch.cuda.is_available():
        return {}
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "name": props.name,
        "total_memory_gb": round(props.total_memory / 1024**3, 1),
        "capability": f"{props.major}.{props.minor}",
    }


@dataclass
class TransformerClassifier:
    """Fine-tune a HuggingFace transformer for text classification.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (e.g. ``distilbert-base-uncased``).
    num_labels:
        Number of target classes.
    max_length:
        Token sequence length for truncation/padding.
    epochs, batch_size, learning_rate, weight_decay, warmup_ratio:
        Standard fine-tuning hyperparameters.
    fp16:
        Use mixed precision. Defaults to True on CUDA, ignored on CPU.
    output_dir:
        Working directory for checkpoints/logs.
    seed:
        Random seed for reproducibility.
    """

    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2
    max_length: int = 128
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    fp16: Optional[bool] = None
    output_dir: str = "./_transformer_out"
    seed: int = 42

    _tokenizer: object = field(default=None, init=False, repr=False)
    _model: object = field(default=None, init=False, repr=False)

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def _tokenize(self, texts: Sequence[str], text_pairs: Optional[Sequence[str]] = None):
        """Tokenize single texts, or sentence pairs when text_pairs is given.

        Passing a second sequence enables sentence-pair tasks like NLI, where the
        tokenizer joins the two with a separator token (e.g. premise [SEP] hypothesis).
        """
        tok = self._ensure_tokenizer()
        if text_pairs is not None:
            return tok(
                list(texts),
                list(text_pairs),
                truncation=True,
                padding=False,
                max_length=self.max_length,
            )
        return tok(
            list(texts),
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )

    def fit(
        self,
        train_texts: Sequence[str],
        train_labels: Sequence[int],
        val_texts: Optional[Sequence[str]] = None,
        val_labels: Optional[Sequence[int]] = None,
        train_text_pairs: Optional[Sequence[str]] = None,
        val_text_pairs: Optional[Sequence[str]] = None,
    ) -> "TransformerClassifier":
        """Fine-tune the model. Validation data and sentence pairs are optional."""
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )

        set_seed(self.seed)
        use_fp16 = self.fp16 if self.fp16 is not None else torch.cuda.is_available()

        tok = self._ensure_tokenizer()
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )

        train_enc = self._tokenize(train_texts, train_text_pairs)
        train_ds = Dataset.from_dict({**train_enc, "labels": list(train_labels)})

        eval_ds = None
        if val_texts is not None and val_labels is not None:
            val_enc = self._tokenize(val_texts, val_text_pairs)
            eval_ds = Dataset.from_dict({**val_enc, "labels": list(val_labels)})

        collator = DataCollatorWithPadding(tokenizer=tok)

        # transformers 5.x uses warmup_steps (not warmup_ratio). Derive steps
        # from the requested ratio so the API stays stable across versions.
        n_train = len(train_ds)
        steps_per_epoch = max(1, -(-n_train // self.batch_size))  # ceil division
        total_steps = steps_per_epoch * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=warmup_steps,
            fp16=use_fp16,
            logging_steps=50,
            save_strategy="no",
            report_to=[],
            seed=self.seed,
            disable_tqdm=False,
        )

        trainer = Trainer(
            model=self._model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
        )
        trainer.train()
        self._model.eval()
        return self

    def predict_proba(self, texts: Sequence[str], batch_size: int = 64,
                      text_pairs: Optional[Sequence[str]] = None) -> np.ndarray:
        """Return class probabilities of shape (n_samples, num_labels).

        Pass text_pairs for sentence-pair tasks (e.g. NLI premise/hypothesis).
        """
        import torch

        if self._model is None:
            raise RuntimeError("Model is not trained. Call fit() or load() first.")

        device = get_device()
        self._model.to(device)
        self._model.eval()
        tok = self._ensure_tokenizer()

        probs = []
        texts = list(texts)
        pairs = list(text_pairs) if text_pairs is not None else None
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                tok_args = [batch]
                if pairs is not None:
                    tok_args.append(pairs[start : start + batch_size])
                enc = tok(
                    *tok_args,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(device)
                logits = self._model(**enc).logits
                probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(probs)

    def predict(self, texts: Sequence[str], batch_size: int = 64,
                text_pairs: Optional[Sequence[str]] = None) -> np.ndarray:
        """Return hard class predictions."""
        return self.predict_proba(texts, batch_size=batch_size,
                                  text_pairs=text_pairs).argmax(axis=1)

    def save(self, path: str) -> None:
        """Persist the fine-tuned model and tokenizer to a directory."""
        if self._model is None:
            raise RuntimeError("Nothing to save; model is not trained.")
        os.makedirs(path, exist_ok=True)
        self._model.save_pretrained(path)
        self._ensure_tokenizer().save_pretrained(path)

    def load(self, path: str) -> "TransformerClassifier":
        """Load a fine-tuned model and tokenizer from a directory."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForSequenceClassification.from_pretrained(path)
        self._model.eval()
        return self


# Backwards-compatible alias for the originally-planned name.
DeepLearningTrainer = TransformerClassifier


@dataclass
class CNNClassifier:
    """Train a compact CNN on grayscale image arrays (e.g. MNIST-style data).

    Expects inputs shaped ``(N, H, W)`` with pixel values that will be scaled
    internally. Designed for single-channel image classification where a small
    purpose-built CNN outperforms large pretrained natural-image models.

    Parameters
    ----------
    num_classes:
        Number of target classes.
    image_size:
        (H, W) of the input images.
    epochs, batch_size, learning_rate, weight_decay:
        Training hyperparameters.
    augment:
        Apply light random affine augmentation (shift/rotation/scale) during
        training. Helps generalization on handwritten digits.
    fp16:
        Mixed precision on CUDA. Defaults to True on GPU.
    val_split:
        Fraction of training data held out for monitoring (0 disables).
    seed:
        Random seed.
    """

    num_classes: int = 10
    image_size: tuple = (28, 28)
    epochs: int = 12
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    augment: bool = True
    fp16: Optional[bool] = None
    val_split: float = 0.1
    seed: int = 42

    _model: object = field(default=None, init=False, repr=False)
    _mean: float = field(default=0.0, init=False, repr=False)
    _std: float = field(default=1.0, init=False, repr=False)
    _scale: float = field(default=1.0, init=False, repr=False)

    def _build_model(self):
        import torch.nn as nn

        c = self.num_classes

        class SmallCNN(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2), nn.Dropout(0.25),
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2), nn.Dropout(0.25),
                )
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(),
                    nn.Linear(64 * 3 * 3, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
                    nn.Linear(256, num_classes),
                )

            def forward(self, x):
                return self.classifier(self.features(x))

        return SmallCNN(c)

    def _to_tensor(self, X):
        """Scale, normalize, and reshape arrays to (N,1,H,W) tensors.

        Applies the SAME pipeline to train and inference data: divide by the
        learned scale factor (e.g. 255), then standardize with train mean/std.
        Keeping scaling here (not in fit) prevents any train/test mismatch.
        """
        import numpy as np
        import torch

        X = np.asarray(X, dtype=np.float32)
        h, w = self.image_size
        X = X.reshape(-1, h, w)
        X = X / self._scale
        X = (X - self._mean) / (self._std + 1e-6)
        return torch.from_numpy(X).unsqueeze(1)

    def fit(self, X, y):
        """Train the CNN on image arrays X with integer labels y."""
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from torchvision import transforms

        from kaggle_ml_toolkit.gpu_utils import get_device, seed_everything

        seed_everything(self.seed)
        device = get_device()
        use_fp16 = self.fp16 if self.fp16 is not None else torch.cuda.is_available()

        X = np.asarray(X, dtype=np.float32)
        # Learn the scale factor (255 if data looks like 0-255, else 1) and the
        # standardization stats on the SCALED data. _to_tensor reuses all three
        # so train and inference are transformed identically.
        self._scale = 255.0 if X.max() > 1.5 else 1.0
        scaled = X / self._scale
        self._mean = float(scaled.mean())
        self._std = float(scaled.std())

        Xt = self._to_tensor(X)
        yt = torch.tensor(np.asarray(y), dtype=torch.long)

        n = len(Xt)
        idx = np.arange(n)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(idx)
        n_val = int(n * self.val_split)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        train_ds = TensorDataset(Xt[tr_idx], yt[tr_idx])
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=torch.cuda.is_available())
        val_loader = None
        if n_val > 0:
            val_ds = TensorDataset(Xt[val_idx], yt[val_idx])
            val_loader = DataLoader(val_ds, batch_size=self.batch_size * 2, shuffle=False)

        aug = None
        if self.augment:
            aug = transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))

        self._model = self._build_model().to(device)
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.learning_rate,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.learning_rate, epochs=self.epochs,
            steps_per_epoch=max(1, len(train_loader)))
        loss_fn = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

        for epoch in range(1, self.epochs + 1):
            self._model.train()
            running = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                if aug is not None:
                    xb = aug(xb)
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_fp16):
                    out = self._model(xb)
                    loss = loss_fn(out, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                running += loss.item() * xb.size(0)
            msg = f"  epoch {epoch}/{self.epochs} train_loss {running/len(tr_idx):.4f}"
            if val_loader is not None:
                acc = self._evaluate(val_loader, device)
                msg += f" val_acc {acc:.4f}"
            print(msg)

        self._model.eval()
        return self

    def _evaluate(self, loader, device) -> float:
        import torch

        self._model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                pred = self._model(xb).argmax(1).cpu()
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        return correct / max(1, total)

    def predict_proba(self, X, batch_size: int = 512):
        """Return class probabilities of shape (n_samples, num_classes)."""
        import numpy as np
        import torch

        from kaggle_ml_toolkit.gpu_utils import get_device

        if self._model is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        device = get_device()
        self._model.to(device).eval()
        Xt = self._to_tensor(X)
        probs = []
        with torch.no_grad():
            for start in range(0, len(Xt), batch_size):
                xb = Xt[start : start + batch_size].to(device)
                logits = self._model(xb)
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(probs)

    def predict(self, X, batch_size: int = 512):
        """Return hard class predictions."""
        return self.predict_proba(X, batch_size=batch_size).argmax(axis=1)

    def save(self, path: str) -> None:
        import torch

        if self._model is None:
            raise RuntimeError("Nothing to save; model is not trained.")
        torch.save(
            {"state_dict": self._model.state_dict(), "mean": self._mean,
             "std": self._std, "scale": self._scale,
             "num_classes": self.num_classes, "image_size": self.image_size},
            path,
        )

    def load(self, path: str) -> "CNNClassifier":
        import torch

        ckpt = torch.load(path, map_location="cpu")
        self.num_classes = ckpt["num_classes"]
        self.image_size = tuple(ckpt["image_size"])
        self._mean = ckpt["mean"]
        self._std = ckpt["std"]
        self._scale = ckpt.get("scale", 1.0)
        self._model = self._build_model()
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()
        return self
