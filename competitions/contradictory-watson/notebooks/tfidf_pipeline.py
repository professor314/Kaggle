"""Contradictory My Dear Watson — TF-IDF NLI Pipeline.

Natural Language Inference: given premise + hypothesis, predict:
  0 = entailment, 1 = neutral, 2 = contradiction
Multi-class text classification with TF-IDF features.
"""
import sys
sys.path.insert(0, r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.sparse import hstack

DATA_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle\competitions\contradictory-watson\data")
SUB_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle\competitions\contradictory-watson\submissions")
SUB_DIR.mkdir(exist_ok=True)

# Load
print("Loading data...")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
print(f"  Train: {train.shape}, Test: {test.shape}")
print(f"  Columns: {list(train.columns)}")
print(f"  Target distribution: {train['label'].value_counts().to_dict()}")
print(f"  Languages: {train['language'].nunique()} ({train['language'].value_counts().head(5).to_dict()})")

# Combine premise + hypothesis into one text
train["combined_text"] = train["premise"] + " [SEP] " + train["hypothesis"]
test["combined_text"] = test["premise"] + " [SEP] " + test["hypothesis"]

# TF-IDF on combined text (word + char)
print("\nVectorizing...")
all_text = pd.concat([train["combined_text"], test["combined_text"]])

word_tfidf = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 2),
    max_features=50000, min_df=2, max_df=0.95, sublinear_tf=True
)
char_tfidf = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 5),
    max_features=50000, min_df=2, max_df=0.95, sublinear_tf=True
)

word_tfidf.fit(all_text)
char_tfidf.fit(all_text)

X_train_w = word_tfidf.transform(train["combined_text"])
X_train_c = char_tfidf.transform(train["combined_text"])
X_test_w = word_tfidf.transform(test["combined_text"])
X_test_c = char_tfidf.transform(test["combined_text"])

# Language as a feature
lang_tfidf = TfidfVectorizer(analyzer="word")
lang_tfidf.fit(pd.concat([train["language"], test["language"]]))
X_train_lang = lang_tfidf.transform(train["language"])
X_test_lang = lang_tfidf.transform(test["language"])

X_train = hstack([X_train_w, X_train_c, X_train_lang])
X_test = hstack([X_test_w, X_test_c, X_test_lang])
y_train = train["label"].values

print(f"  Total features: {X_train.shape[1]}")

# Cross-validation
print("\nCross-validating...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for c in [0.5, 1.0, 2.0, 5.0]:
    model = LogisticRegression(C=c, max_iter=1000, solver="lbfgs", random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"  C={c}: Acc={scores.mean():.4f} +/- {scores.std():.4f}")

# Best model
print("\nTraining final model...")
best_model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
best_model.fit(X_train, y_train)

# Predict
test_preds = best_model.predict(X_test)

submission = pd.DataFrame({
    "id": test["id"],
    "prediction": test_preds,
})
sub_file = SUB_DIR / "tfidf_logreg_nli.csv"
submission.to_csv(sub_file, index=False)
print(f"  Saved: {sub_file}")
print(f"  Predictions: {pd.Series(test_preds).value_counts().to_dict()}")
print(f"\n=== DONE ===")
