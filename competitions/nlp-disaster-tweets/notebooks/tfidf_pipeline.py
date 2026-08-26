"""NLP Disaster Tweets — TF-IDF + Logistic Regression Pipeline.

Binary text classification using TF-IDF features (word + character n-grams)
combined with keyword encoding. Evaluated with stratified 5-fold CV on F1.
"""

import sys
sys.path.insert(0, r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle")

import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack

DATA_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle\competitions\nlp-disaster-tweets\data")
SUB_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle\competitions\nlp-disaster-tweets\submissions")

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
print(f"  Train: {train.shape[0]} rows, Test: {test.shape[0]} rows")
print(f"  Target balance: {train['target'].value_counts().to_dict()}")

# ============================================================
# 2. TEXT CLEANING
# ============================================================
def clean_text(text: str) -> str:
    """Remove URLs, @mentions, lowercase. Keep hashtags and punctuation."""
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove @mentions
    text = text.lower().strip()
    return text

train["clean_text"] = train["text"].apply(clean_text)
test["clean_text"] = test["text"].apply(clean_text)
print("  Text cleaned.")

# ============================================================
# 3. TF-IDF VECTORIZATION
# ============================================================
print("Vectorizing...")

# Word-level TF-IDF (unigrams + bigrams)
word_tfidf = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=20000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)

# Character-level TF-IDF (3-5 grams)
char_tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=30000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)

# Fit on combined train+test text for consistent vocabulary
all_text = pd.concat([train["clean_text"], test["clean_text"]])
word_tfidf.fit(all_text)
char_tfidf.fit(all_text)

# Transform
X_train_word = word_tfidf.transform(train["clean_text"])
X_train_char = char_tfidf.transform(train["clean_text"])
X_test_word = word_tfidf.transform(test["clean_text"])
X_test_char = char_tfidf.transform(test["clean_text"])

print(f"  Word features: {X_train_word.shape[1]}")
print(f"  Char features: {X_train_char.shape[1]}")

# ============================================================
# 4. KEYWORD ENCODING
# ============================================================
# Fill missing keywords with "unknown"
train["keyword_clean"] = train["keyword"].fillna("unknown").str.replace("%20", " ")
test["keyword_clean"] = test["keyword"].fillna("unknown").str.replace("%20", " ")

# One-hot encode keywords
keyword_tfidf = TfidfVectorizer(analyzer="word", max_features=300)
keyword_tfidf.fit(pd.concat([train["keyword_clean"], test["keyword_clean"]]))
X_train_kw = keyword_tfidf.transform(train["keyword_clean"])
X_test_kw = keyword_tfidf.transform(test["keyword_clean"])
print(f"  Keyword features: {X_train_kw.shape[1]}")

# ============================================================
# 5. COMBINE FEATURES
# ============================================================
X_train = hstack([X_train_word, X_train_char, X_train_kw])
X_test = hstack([X_test_word, X_test_char, X_test_kw])
y_train = train["target"].values

print(f"  Total features: {X_train.shape[1]}")

# ============================================================
# 6. CROSS-VALIDATION
# ============================================================
print("\nCross-validating...")

model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="liblinear",
    random_state=42,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

print(f"  F1 scores per fold: {cv_scores.round(4)}")
print(f"  Mean F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Try different C values
print("\n  Tuning regularization (C)...")
best_c = 1.0
best_f1 = cv_scores.mean()

for c in [0.1, 0.5, 2.0, 5.0, 10.0]:
    m = LogisticRegression(C=c, max_iter=1000, solver="liblinear", random_state=42)
    scores = cross_val_score(m, X_train, y_train, cv=cv, scoring="f1")
    mean_f1 = scores.mean()
    print(f"    C={c}: F1={mean_f1:.4f}")
    if mean_f1 > best_f1:
        best_f1 = mean_f1
        best_c = c

print(f"\n  Best C: {best_c}, Best F1: {best_f1:.4f}")

# ============================================================
# 7. TRAIN FINAL MODEL
# ============================================================
print(f"\nTraining final model (C={best_c})...")
final_model = LogisticRegression(
    C=best_c, max_iter=1000, solver="liblinear", random_state=42
)
final_model.fit(X_train, y_train)

# Training metrics
train_preds = final_model.predict(X_train)
print(f"  Train F1: {f1_score(y_train, train_preds):.4f}")
print(f"\n  Classification Report (train):")
print(classification_report(y_train, train_preds, target_names=["Not Disaster", "Disaster"]))

# ============================================================
# 8. GENERATE SUBMISSION
# ============================================================
print("Generating submission...")
test_preds = final_model.predict(X_test)

submission = pd.DataFrame({
    "id": test["id"],
    "target": test_preds,
})

sub_file = SUB_DIR / "tfidf_logreg_baseline.csv"
submission.to_csv(sub_file, index=False)
print(f"  Saved: {sub_file}")
print(f"  Predictions: {pd.Series(test_preds).value_counts().to_dict()}")
print(f"\n=== DONE === CV F1: {best_f1:.4f}")
