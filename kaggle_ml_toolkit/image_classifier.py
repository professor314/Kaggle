"""timm-based image classifier for natural RGB images (Phase 3 extension).

Complements the small-footprint CNNClassifier (grayscale, MNIST-style) with a
transfer-learning classifier for natural color images such as the TPU Flowers
competition. Wraps a pretrained `timm` backbone (default EfficientNet-B0) in the
same scikit-learn-style API as the rest of the toolkit
(fit / predict / predict_proba / save / load).

Requirements (in the `.venv-dl` env): torch, torchvision, timm, numpy.

Design notes:
- Inputs are arrays shaped (N, H, W, 3) with uint8 or float pixels; scaling and
  ImageNet normalization are applied internally and identically to train and
  inference (the same discipline that fixed the CNNClassifier normalization bug).
- Mixed precision on CUDA. Backbone is fine-tuned end-to-end (small LR) since
  flower species differ from ImageNet classes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ImageNet statistics used by timm pretrained backbones.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class ImageClassifier:
    """Fine-tune a pretrained timm backbone for RGB image classification.

    Parameters
    ----------
    num_classes:
        Number of target classes.
    model_name:
        Any timm model id (default ``efficientnet_b0``).
    image_size:
        (H, W) the images are resized to before the backbone.
    epochs, batch_size, learning_rate, weight_decay:
        Training hyperparameters.
    augment:
        Light random flips/affine during training.
    fp16:
        Mixed precision on CUDA; defaults to True on GPU.
    val_split:
        Fraction held out for monitoring (0 disables).
    seed:
        Random seed.
    """

    num_classes: int = 104
    model_name: str = "efficientnet_b0"
    image_size: tuple = (224, 224)
    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    augment: bool = True
    fp16: Optional[bool] = None
    val_split: float = 0.1
    seed: int = 42

    _model: object = field(default=None, init=False, repr=False)

    def _build_model(self):
        import timm

        return timm.create_model(
            self.model_name, pretrained=True, num_classes=self.num_classes
        )

    def _to_tensor(self, X):
        """Scale to [0,1], ImageNet-normalize, resize, return (N,3,H,W) tensor.

        Applied identically to train and inference. Accepts (N,H,W,3) arrays in
        0-255 or 0-1 range (auto-detected).
        """
        import torch
        import torch.nn.functional as F

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 4 or X.shape[-1] != 3:
            raise ValueError(f"Expected (N, H, W, 3) RGB array, got shape {X.shape}")
        if X.max() > 1.5:
            X = X / 255.0
        X = (X - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(X).permute(0, 3, 1, 2)  # (N,3,H,W)
        if tuple(t.shape[-2:]) != tuple(self.image_size):
            t = F.interpolate(t, size=self.image_size, mode="bilinear",
                              align_corners=False)
        return t

    def fit(self, X, y):
        """Fine-tune on RGB image arrays X (N,H,W,3) with integer labels y."""
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        from kaggle_ml_toolkit.gpu_utils import get_device, seed_everything

        seed_everything(self.seed)
        device = get_device()
        use_fp16 = self.fp16 if self.fp16 is not None else torch.cuda.is_available()

        Xt = self._to_tensor(X)
        yt = torch.tensor(np.asarray(y), dtype=torch.long)

        n = len(Xt)
        idx = np.arange(n)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(idx)
        n_val = int(n * self.val_split)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        train_loader = DataLoader(
            TensorDataset(Xt[tr_idx], yt[tr_idx]),
            batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=torch.cuda.is_available(),
        )
        val_loader = None
        if n_val > 0:
            val_loader = DataLoader(TensorDataset(Xt[val_idx], yt[val_idx]),
                                    batch_size=self.batch_size * 2, shuffle=False)

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
                if self.augment:
                    if torch.rand(1).item() < 0.5:
                        xb = torch.flip(xb, dims=[3])  # horizontal flip
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_fp16):
                    loss = loss_fn(self._model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                running += loss.item() * xb.size(0)
            msg = f"  epoch {epoch}/{self.epochs} train_loss {running/len(tr_idx):.4f}"
            if val_loader is not None:
                msg += f" val_acc {self._evaluate(val_loader, device):.4f}"
            print(msg)

        self._model.eval()
        return self

    def _evaluate(self, loader, device) -> float:
        import torch

        self._model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in loader:
                pred = self._model(xb.to(device)).argmax(1).cpu()
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        return correct / max(1, total)

    def predict_proba(self, X, batch_size: int = 128):
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
                xb = Xt[start:start + batch_size].to(device)
                probs.append(torch.softmax(self._model(xb), dim=1).cpu().numpy())
        return np.vstack(probs)

    def predict(self, X, batch_size: int = 128):
        """Return hard class predictions."""
        return self.predict_proba(X, batch_size=batch_size).argmax(axis=1)

    def save(self, path: str) -> None:
        import torch

        if self._model is None:
            raise RuntimeError("Nothing to save; model is not trained.")
        torch.save(
            {"state_dict": self._model.state_dict(), "num_classes": self.num_classes,
             "model_name": self.model_name, "image_size": self.image_size},
            path,
        )

    def load(self, path: str) -> "ImageClassifier":
        import torch

        ckpt = torch.load(path, map_location="cpu")
        self.num_classes = ckpt["num_classes"]
        self.model_name = ckpt["model_name"]
        self.image_size = tuple(ckpt["image_size"])
        self._model = self._build_model()
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()
        return self
