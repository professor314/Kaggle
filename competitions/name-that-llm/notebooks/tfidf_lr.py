"""Name That LLM - TF-IDF (word + char) + logistic regression baseline.

Task: given a text, predict which of 18 LLMs wrote it. Code competition: the visible
test.csv has 3 rows; the hidden test is swapped in at scoring time, so everything is
computed from whatever test file is mounted.

CV is GroupKFold on prompt_id: every model answered the same prompts, so a random split
would leak prompt-specific wording between train and validation.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold


def _find_input(slug):
    for p in (f"/kaggle/input/{slug}", f"/kaggle/input/competitions/{slug}"):
        if os.path.isdir(p):
            return p
    hits = glob.glob(f"/kaggle/input/**/{slug}", recursive=True)
    if hits:
        return hits[0]
    if os.path.isdir("data"):  # local run
        return "data"
    raise FileNotFoundError(f"{slug} not found. Inputs: {glob.glob('/kaggle/input/*')}")


DATA = _find_input("name-that-llm")
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "submissions"
os.makedirs(OUT, exist_ok=True)

train = pd.read_csv(os.path.join(DATA, "train.csv"))
test = pd.read_csv(os.path.join(DATA, "test.csv"))
print("train", train.shape, "test", test.shape, "classes", train["model"].nunique())

word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000, sublinear_tf=True)
texts = pd.concat([train["text"], test["text"]]).fillna("").astype(str)
word.fit(texts)
char.fit(texts)


def feats(s):
    s = s.fillna("").astype(str)
    return hstack([word.transform(s), char.transform(s)]).tocsr()


X, Xt = feats(train["text"]), feats(test["text"])
y = train["model"].values


def model():
    return LogisticRegression(C=10.0, max_iter=2000)


RUN_CV = os.environ.get("RUN_CV", "1") == "1"
if RUN_CV:
    oof = np.empty(len(y), dtype=object)
    for k, (tr, va) in enumerate(GroupKFold(n_splits=5).split(X, y, train["prompt_id"])):
        m = model().fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
        print(f"fold {k}: acc {accuracy_score(y[va], oof[va]):.4f}")
    print(f"OOF accuracy (GroupKFold by prompt_id): {accuracy_score(y, oof):.4f}  (chance {1/len(set(y)):.4f})")

final = model().fit(X, y)
sub = pd.DataFrame({"id": test["id"], "model": final.predict(Xt)})
sub.to_parquet(os.path.join(OUT, "submission.parquet"), index=False)
sub.to_csv(os.path.join(OUT, "submission.csv"), index=False)
print(sub.head())
