"""
Contradictory, My Dear Watson - XLM-RoBERTa NLI (Kaggle kernel, single-stage).

Multilingual NLI: classify premise/hypothesis pairs as entailment (0),
neutral (1), or contradiction (2). Fine-tunes xlm-roberta-base on Kaggle's GPU.

This is the memory-safe single-stage version. A two-stage variant (pretrain on
MNLI first) repeatedly OOM'd Kaggle's kernel, so we ship the version that runs
and get a confirmed leaderboard score first. Local held-out val acc ~0.69.
"""

import glob

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 96
EPOCHS = 4
BATCH = 32
LR = 2e-5
SEED = 42

set_seed(SEED)


def find(name):
    m = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not m:
        raise FileNotFoundError(f"{name} not found. Inputs: {glob.glob('/kaggle/input/*')}")
    return m[0]


train = pd.read_csv(find("train.csv"))
test = pd.read_csv(find("test.csv"))

# Guard against the Kaggle P100 issue where the shipped PyTorch build lacks
# compiled kernels for the GPU's compute capability (sm_60). We probe the GPU
# with a tiny op; if it raises, fall back to CPU so the run still completes.
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    try:
        _ = (torch.zeros(1, device="cuda") + 1).item()
    except Exception as e:
        print(f"GPU probe failed ({type(e).__name__}); falling back to CPU.")
        USE_CUDA = False
print(f"Train {train.shape} | Test {test.shape} | using {'cuda' if USE_CUDA else 'cpu'}")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
collator = DataCollatorWithPadding(tokenizer=tok)


def encode(premise, hypothesis):
    return tok(list(premise), list(hypothesis), truncation=True,
               padding=False, max_length=MAX_LEN)


train_enc = encode(train["premise"], train["hypothesis"])
train_ds = Dataset.from_dict({**train_enc, "labels": train["label"].tolist()})

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
steps_per_epoch = max(1, -(-len(train_ds) // BATCH))
args = TrainingArguments(
    output_dir="/kaggle/working/_out",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_steps=int(steps_per_epoch * EPOCHS * 0.06),
    fp16=USE_CUDA,
    logging_steps=100,
    save_strategy="no",
    report_to=[],
    seed=SEED,
    use_cpu=not USE_CUDA,
)
Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator).train()
model.eval()

device = "cuda" if USE_CUDA else "cpu"
model.to(device)
preds = []
prem, hyp = test["premise"].tolist(), test["hypothesis"].tolist()
with torch.no_grad():
    for start in range(0, len(prem), 128):
        enc = tok(prem[start:start + 128], hyp[start:start + 128], truncation=True,
                  padding=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        preds.append(model(**enc).logits.argmax(-1).cpu().numpy())
preds = np.concatenate(preds)

sub = pd.DataFrame({"id": test["id"], "prediction": preds.astype(int)})
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(f"Saved submission {sub.shape} | dist {pd.Series(preds).value_counts().sort_index().to_dict()}")
