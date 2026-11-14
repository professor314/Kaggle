"""LLM Classification Finetuning - TF-IDF + length features baseline.

Task: given a prompt and two chatbot responses (A and B), predict the probability that
a human preferred A, preferred B, or called it a tie. Metric: multi-class log loss.

Approach (deliberately simple, a non-transformer baseline to beat):
  * TF-IDF on each response, then use the DIFFERENCE (A - B) so the model learns
    "which response looks more like preferred responses".
  * Length features (chars, words, lines, newlines), A vs B and their difference.
    Verbosity bias is a well-known signal in this task.
  * Multinomial logistic regression, 3-fold CV log loss reported.
A fine-tuned transformer (e.g. Gemma/LoRA) is the real next step; this is the floor.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold


def _find_input(slug):
    """Locate a competition's input dir across Kaggle mount layouts."""
    for p in (f"/kaggle/input/{slug}", f"/kaggle/input/competitions/{slug}"):
        if os.path.isdir(p):
            return p
    hits = glob.glob(f"/kaggle/input/**/{slug}", recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{slug} not found. Inputs: {glob.glob('/kaggle/input/*')}")


DATA = _find_input("llm-classification-finetuning")
train = pd.read_csv(os.path.join(DATA, "train.csv"))
test = pd.read_csv(os.path.join(DATA, "test.csv"))
sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
print("train", train.shape, "test", test.shape)
print("train columns:", train.columns.tolist())
print("submission columns:", sample.columns.tolist())

# Label: argmax over the winner_* indicator columns (order matches the submission columns).
label_cols = [c for c in sample.columns if c != "id"]
y = train[label_cols].values.argmax(axis=1)
print("class balance:", np.bincount(y) / len(y))


def length_feats(df):
    out = {}
    for side in ("a", "b"):
        s = df[f"response_{side}"].fillna("").astype(str)
        out[f"chars_{side}"] = np.log1p(s.str.len())
        out[f"words_{side}"] = np.log1p(s.str.split().str.len())
        out[f"lines_{side}"] = np.log1p(s.str.count("\n"))
    out["chars_diff"] = out["chars_a"] - out["chars_b"]
    out["words_diff"] = out["words_a"] - out["words_b"]
    out["lines_diff"] = out["lines_a"] - out["lines_b"]
    return pd.DataFrame(out).values


tfidf = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), sublinear_tf=True, min_df=3)
tfidf.fit(pd.concat([train["response_a"], train["response_b"]]).fillna("").astype(str))


def featurize(df):
    xa = tfidf.transform(df["response_a"].fillna("").astype(str))
    xb = tfidf.transform(df["response_b"].fillna("").astype(str))
    return hstack([xa - xb, csr_matrix(length_feats(df))]).tocsr()


X = featurize(train)
Xt = featurize(test)

oof = np.zeros((len(train), 3))
pred = np.zeros((len(test), 3))
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
for k, (tr, va) in enumerate(cv.split(X, y)):
    m = LogisticRegression(C=1.0, max_iter=300)
    m.fit(X[tr], y[tr])
    oof[va] = m.predict_proba(X[va])
    pred += m.predict_proba(Xt) / 3
    print(f"fold {k}: log loss {log_loss(y[va], oof[va]):.4f}")

print(f"OOF log loss: {log_loss(y, oof):.4f}  (uniform prior baseline ~1.098)")

sub = pd.DataFrame(pred, columns=label_cols)
sub.insert(0, "id", test["id"].values)
sub.to_csv("submission.csv", index=False)
print(sub.head())
