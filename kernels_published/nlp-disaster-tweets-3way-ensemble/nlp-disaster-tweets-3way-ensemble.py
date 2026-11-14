"""NLP Disaster Tweets - DistilBERT + TF-IDF + BERTweet 3-way ensemble.

Three diverse bases, blended for a strong F1:
  1. TF-IDF (word + char + keyword) -> Logistic Regression
  2. DistilBERT (distilbert-base-uncased), fine-tuned
  3. BERTweet (vinai/bertweet-base), pretrained on tweets so it makes different
     errors and adds genuine diversity to the ensemble

All three produce out-of-fold (OOF) probabilities on identical 5 folds. The blend
weights and decision threshold are tuned on OOF predictions only (no leakage),
then applied to averaged test probabilities.

Progression: TF-IDF LB 0.801 -> DistilBERT 0.836 -> 2-way 0.838 -> 3-way 0.842
(rank ~49, top 10%).

Self-contained Kaggle version: reads from /kaggle/input/nlp-getting-started/.
Enable the GPU accelerator AND internet (the transformer weights download from
the HuggingFace hub at runtime). BERTweet also needs the `emoji` package.
"""
def _find_input(slug):
    """Locate a competition's input dir across Kaggle mount layouts."""
    import os, glob
    for p in (f"/kaggle/input/{slug}", f"/kaggle/input/competitions/{slug}"):
        if os.path.isdir(p):
            return p
    hits = glob.glob(f"/kaggle/input/**/{slug}", recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{slug} not found. Inputs: {glob.glob('/kaggle/input/*')}")


import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# BERTweet's tokenizer needs the emoji package.
try:
    import emoji  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "emoji"], check=False)

DATA = Path(_find_input("nlp-getting-started"))
N_SPLITS = 5
EPOCHS = 3
MAX_LEN = 128
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clean_light(text):
    """Light cleaning for transformers (keep casing/punctuation)."""
    text = re.sub(r"http\S+|www\.\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_tfidf(text):
    """Lowercasing clean for the TF-IDF model."""
    text = re.sub(r"http\S+|www\.\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    return text.lower().strip()


def build_tfidf_features(train, test):
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=20000,
                           min_df=2, max_df=0.95, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30000,
                           min_df=2, max_df=0.95, sublinear_tf=True)
    kw = TfidfVectorizer(analyzer="word", max_features=300)

    all_word = pd.concat([train["tfidf_text"], test["tfidf_text"]])
    word.fit(all_word); char.fit(all_word)
    train["kw"] = train["keyword"].fillna("unknown").str.replace("%20", " ")
    test["kw"] = test["keyword"].fillna("unknown").str.replace("%20", " ")
    kw.fit(pd.concat([train["kw"], test["kw"]]))

    Xtr = hstack([word.transform(train["tfidf_text"]),
                  char.transform(train["tfidf_text"]),
                  kw.transform(train["kw"])]).tocsr()
    Xte = hstack([word.transform(test["tfidf_text"]),
                  char.transform(test["tfidf_text"]),
                  kw.transform(test["kw"])]).tocsr()
    return Xtr, Xte


def train_transformer(model_name, tr_texts, tr_y, va_texts, te_texts, fold):
    """Fine-tune one transformer; return (val_proba_pos, test_proba_pos)."""
    set_seed(SEED)
    use_fp16 = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def encode(texts):
        return tok(list(texts), truncation=True, padding=False, max_length=MAX_LEN)

    train_ds = Dataset.from_dict({**encode(tr_texts), "labels": list(tr_y)})
    steps_per_epoch = max(1, -(-len(train_ds) // 16))
    warmup = int(steps_per_epoch * EPOCHS * 0.06)

    args = TrainingArguments(
        output_dir=f"./_hf_{fold}", num_train_epochs=EPOCHS,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        learning_rate=2e-5, weight_decay=0.01, warmup_steps=warmup,
        fp16=use_fp16, logging_steps=50, save_strategy="no", report_to=[], seed=SEED,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      data_collator=DataCollatorWithPadding(tokenizer=tok))
    trainer.train()
    model.eval()

    def proba(texts):
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), 64):
                batch = list(texts[start:start + 64])
                enc = tok(batch, truncation=True, padding=True, max_length=MAX_LEN,
                          return_tensors="pt").to(DEVICE)
                logits = model.to(DEVICE)(**enc).logits
                out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(out)[:, 1]

    return proba(va_texts), proba(te_texts)


def main():
    print("Device:", DEVICE)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    train["bert_text"] = train["text"].apply(clean_light)
    test["bert_text"] = test["text"].apply(clean_light)
    train["tfidf_text"] = train["text"].apply(clean_tfidf)
    test["tfidf_text"] = test["text"].apply(clean_tfidf)

    y = train["target"].values
    Xtr_tfidf, Xte_tfidf = build_tfidf_features(train, test)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_bert = np.zeros(len(train)); test_bert = np.zeros(len(test))
    oof_tfidf = np.zeros(len(train)); test_tfidf = np.zeros(len(test))
    oof_bt = np.zeros(len(train)); test_bt = np.zeros(len(test))

    for fold, (tri, vai) in enumerate(cv.split(train, y), start=1):
        print(f"\n===== Fold {fold}/{N_SPLITS} =====")

        # TF-IDF + Logistic Regression
        lr = LogisticRegression(C=0.5, max_iter=1000, solver="liblinear", random_state=SEED)
        lr.fit(Xtr_tfidf[tri], y[tri])
        oof_tfidf[vai] = lr.predict_proba(Xtr_tfidf[vai])[:, 1]
        test_tfidf += lr.predict_proba(Xte_tfidf)[:, 1] / N_SPLITS

        # DistilBERT
        v, t = train_transformer("distilbert-base-uncased", train["bert_text"].values[tri],
                                 y[tri], train["bert_text"].values[vai],
                                 test["bert_text"].values, f"db{fold}")
        oof_bert[vai] = v; test_bert += t / N_SPLITS

        # BERTweet (diverse, tweet-pretrained base)
        v, t = train_transformer("vinai/bertweet-base", train["bert_text"].values[tri],
                                 y[tri], train["bert_text"].values[vai],
                                 test["bert_text"].values, f"bt{fold}")
        oof_bt[vai] = v; test_bt += t / N_SPLITS

    print("\n===== OOF F1 =====")
    for name, oof in [("DistilBERT", oof_bert), ("TF-IDF", oof_tfidf), ("BERTweet", oof_bt)]:
        print(f"  {name}: {f1_score(y, (oof >= 0.5).astype(int)):.4f}")

    # Grid over 3-way simplex weights + threshold, tuned on OOF F1.
    best = {"f1": 0.0}
    for wb in np.arange(0, 1.01, 0.1):
        for wt in np.arange(0, 1.01 - wb + 1e-9, 0.1):
            wbt = 1 - wb - wt
            if wbt < -1e-9:
                continue
            blend = wb * oof_bert + wt * oof_tfidf + wbt * oof_bt
            for thr in np.arange(0.35, 0.66, 0.01):
                f1 = f1_score(y, (blend >= thr).astype(int))
                if f1 > best["f1"]:
                    best = {"f1": f1, "wb": round(wb, 2), "wt": round(wt, 2),
                            "wbt": round(wbt, 2), "thr": round(thr, 2)}
    print(f"\nBest 3-way blend: DistilBERT={best['wb']} TFIDF={best['wt']} "
          f"BERTweet={best['wbt']} thr={best['thr']} -> OOF F1 {best['f1']:.4f}")

    test_blend = best["wb"] * test_bert + best["wt"] * test_tfidf + best["wbt"] * test_bt
    preds = (test_blend >= best["thr"]).astype(int)
    sub = pd.DataFrame({"id": test["id"], "target": preds})
    sub.to_csv("submission.csv", index=False)
    print(f"\nSaved: submission.csv  balance {pd.Series(preds).value_counts().to_dict()}")
    print(f"=== DONE === 3-way OOF F1 {best['f1']:.4f} (best public LB 0.842, rank ~49)")


if __name__ == "__main__":
    main()
