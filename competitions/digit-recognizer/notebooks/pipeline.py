"""
Digit Recognizer (MNIST) Pipeline
==================================
Classifies handwritten digits (0-9) from 28x28 grayscale images.
Uses sklearn-only approaches (no deep learning frameworks required).

Approaches:
  A: PCA + SVM (RBF kernel) — fast and accurate
  B: Random Forest (200 trees)
  C: MLP Neural Network (256, 128 hidden layers)
  D: Raw SVM (no PCA) — slowest but potentially best

Picks the best model by validation accuracy, generates submission.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSION_DIR = Path(__file__).parent.parent / "submissions"
SUBMISSION_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
VAL_SIZE = 0.10  # 10% stratified validation split

# =============================================================================
# 1. Load Data
# =============================================================================
print("=" * 60)
print("DIGIT RECOGNIZER PIPELINE")
print("=" * 60)

print("\n[1/6] Loading data...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

print(f"  Train shape: {train_df.shape}")
print(f"  Test shape:  {test_df.shape}")

# Separate features and labels
y = train_df["label"].values
X = train_df.drop("label", axis=1).values.astype(np.float64)
X_test = test_df.values.astype(np.float64)

# =============================================================================
# 2. Normalize pixels to [0, 1]
# =============================================================================
print("\n[2/6] Normalizing pixels (divide by 255)...")
X = X / 255.0
X_test = X_test / 255.0

print(f"  Pixel range: [{X.min():.2f}, {X.max():.2f}]")
print(f"  Label distribution: {np.bincount(y)}")

# =============================================================================
# 3. Train/Validation Split
# =============================================================================
print("\n[3/6] Splitting validation set (10% stratified)...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train: {X_train.shape[0]} samples")
print(f"  Val:   {X_val.shape[0]} samples")

# =============================================================================
# 4. Train Multiple Approaches
# =============================================================================
print("\n[4/6] Training models...")
print("=" * 60)

results = {}


# --- Approach A: PCA (100 components) + SVM ---
print("\n[A] PCA (100 components) + SVM (RBF, C=10)...")
t0 = time.time()

pca = PCA(n_components=100, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)

svm_pca = SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE)
svm_pca.fit(X_train_pca, y_train)

y_pred_a = svm_pca.predict(X_val_pca)
acc_a = accuracy_score(y_val, y_pred_a)
time_a = time.time() - t0

results["PCA+SVM"] = {"accuracy": acc_a, "time": time_a}
print(f"    Accuracy: {acc_a:.4f} | Time: {time_a:.1f}s")


# --- Approach B: Random Forest ---
print("\n[B] Random Forest (200 trees, max_depth=None)...")
t0 = time.time()

rf = RandomForestClassifier(
    n_estimators=200, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE
)
rf.fit(X_train, y_train)

y_pred_b = rf.predict(X_val)
acc_b = accuracy_score(y_val, y_pred_b)
time_b = time.time() - t0

results["RandomForest"] = {"accuracy": acc_b, "time": time_b}
print(f"    Accuracy: {acc_b:.4f} | Time: {time_b:.1f}s")


# --- Approach C: MLP Neural Network ---
print("\n[C] MLP Neural Network (256, 128 hidden layers)...")
t0 = time.time()

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    max_iter=30,
    batch_size=256,
    learning_rate_init=0.001,
    random_state=RANDOM_STATE,
    verbose=False,
)
mlp.fit(X_train, y_train)

y_pred_c = mlp.predict(X_val)
acc_c = accuracy_score(y_val, y_pred_c)
time_c = time.time() - t0

results["MLP"] = {"accuracy": acc_c, "time": time_c}
print(f"    Accuracy: {acc_c:.4f} | Time: {time_c:.1f}s")


# --- Approach D: Raw SVM (no PCA) — only if PCA+SVM < 98% ---
print("\n[D] Raw SVM (RBF, C=10, no dimensionality reduction)...")
print("    (This may take 3-5 minutes on 37800 samples × 784 features)")
t0 = time.time()

svm_raw = SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE)
svm_raw.fit(X_train, y_train)

y_pred_d = svm_raw.predict(X_val)
acc_d = accuracy_score(y_val, y_pred_d)
time_d = time.time() - t0

results["SVM_raw"] = {"accuracy": acc_d, "time": time_d}
print(f"    Accuracy: {acc_d:.4f} | Time: {time_d:.1f}s")


# =============================================================================
# 5. Compare Results & Pick Best
# =============================================================================
print("\n\n[5/6] Results Comparison")
print("=" * 60)
print(f"{'Model':<15} {'Accuracy':<12} {'Time (s)':<10}")
print("-" * 37)

for name, res in sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True):
    print(f"{name:<15} {res['accuracy']:<12.4f} {res['time']:<10.1f}")

best_name = max(results, key=lambda k: results[k]["accuracy"])
best_acc = results[best_name]["accuracy"]
print(f"\n>>> BEST: {best_name} with accuracy {best_acc:.4f}")

# =============================================================================
# 6. Generate Submission with Best Model
# =============================================================================
print(f"\n[6/6] Generating submission using {best_name}...")

# Retrain best model on FULL training data (all 42K samples)
print("  Retraining on full training data (42000 samples)...")

if best_name == "PCA+SVM":
    pca_full = PCA(n_components=100, random_state=RANDOM_STATE)
    X_full_pca = pca_full.fit_transform(X)
    X_test_pca = pca_full.transform(X_test)
    
    best_model = SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE)
    best_model.fit(X_full_pca, y)
    predictions = best_model.predict(X_test_pca)

elif best_name == "RandomForest":
    best_model = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE
    )
    best_model.fit(X, y)
    predictions = best_model.predict(X_test)

elif best_name == "MLP":
    best_model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        max_iter=30,
        batch_size=256,
        learning_rate_init=0.001,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    best_model.fit(X, y)
    predictions = best_model.predict(X_test)

elif best_name == "SVM_raw":
    best_model = SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE)
    best_model.fit(X, y)
    predictions = best_model.predict(X_test)

# Create submission DataFrame
submission = pd.DataFrame({
    "ImageId": np.arange(1, len(predictions) + 1),
    "Label": predictions.astype(int),
})

# Save submission
submission_path = SUBMISSION_DIR / "submission_best_model.csv"
submission.to_csv(submission_path, index=False)

print(f"  Submission saved: {submission_path}")
print(f"  Shape: {submission.shape}")
print(f"  Label distribution:\n{submission['Label'].value_counts().sort_index().to_string()}")
print(f"  ImageId range: [{submission['ImageId'].min()}, {submission['ImageId'].max()}]")

# =============================================================================
# 7. Submit via Kaggle CLI
# =============================================================================
print("\n" + "=" * 60)
print("SUBMITTING TO KAGGLE...")
print("=" * 60)

import subprocess

submit_cmd = [
    "kaggle", "competitions", "submit",
    "-c", "digit-recognizer",
    "-f", str(submission_path),
    "-m", f"sklearn {best_name} | val_acc={best_acc:.4f}",
]

print(f"  Command: {' '.join(submit_cmd)}")
result = subprocess.run(submit_cmd, capture_output=True, text=True)

if result.returncode == 0:
    print(f"  SUCCESS: {result.stdout.strip()}")
else:
    print(f"  STDOUT: {result.stdout.strip()}")
    print(f"  STDERR: {result.stderr.strip()}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print(f"Best model: {best_name} (val accuracy: {best_acc:.4f})")
print("=" * 60)
