"""
Contradictory, My Dear Watson - XLM-RoBERTa NLI, TWO-STAGE (Kaggle kernel).

Multilingual NLI: classify premise/hypothesis as entailment (0), neutral (1),
contradiction (2). Two-stage training (the documented path to 0.85+):

  Stage 1 - pretrain xlm-roberta-base on a SUBSAMPLE of MNLI (large English NLI
            corpus) so the head learns the NLI task.
  Stage 2 - fine-tune on the competition's 12k multilingual pairs.

Memory safety (earlier two-stage attempts OOM'd Kaggle's kernel):
  - MNLI is subsampled (MNLI_N) and tokenized lazily via Dataset.map, not held
    as giant python lists.
  - Short MAX_LEN, fp16, modest batch.
  - Must finish under the competition's 120-min GPU cap -> REQUIRE_GPU.

MNLI label convention matches the competition: 0=entailment, 1=neutral,
2=contradiction (HF `nyu-mll/glue`/`multi_nli` uses the same mapping).
Internet is enabled so the MNLI dataset + model download work.
"""
import glob

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 80
MNLI_N = 150_000          # subsample of MNLI (~393k total) to fit the time cap
STAGE1_EPOCHS = 1         # one pass over 150k MNLI pairs
STAGE2_EPOCHS = 4         # fine-tune on the 12k competition pairs
BATCH = 64
LR = 2e-5
SEED = 42
REQUIRE_GPU = True

set_seed(SEED)


def find(name):
    m = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not m:
        raise FileNotFoundError(f"{name} not found. Inputs: {glob.glob('/kaggle/input/*')}")
    return m[0]


train = pd.read_csv(find("train.csv"))
test = pd.read_csv(find("test.csv"))

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    try:
        _ = (torch.zeros(1, device="cuda") + 1).item()
    except Exception as e:
        print(f"GPU probe failed ({type(e).__name__}).")
        USE_CUDA = False
if REQUIRE_GPU and not USE_CUDA:
    raise RuntimeError(
        "No usable CUDA GPU. Two-stage training must finish under the 120-min "
        "cap; the CPU path cannot. Attach a T4 accelerator and re-run."
    )
print(f"Train {train.shape} | Test {test.shape} | using {'cuda' if USE_CUDA else 'cpu'}")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
collator = DataCollatorWithPadding(tokenizer=tok)


def tok_pair(batch):
    return tok(batch["premise"], batch["hypothesis"], truncation=True,
               padding=False, max_length=MAX_LEN)


# ---- Stage 1: MNLI pretraining (lazy tokenization, subsampled) ----
print(f"Loading MNLI subsample (n={MNLI_N}) ...")
mnli = load_dataset("nyu-mll/multi_nli", split="train")
mnli = mnli.filter(lambda ex: ex["label"] in (0, 1, 2))   # drop -1 (no gold label)
if len(mnli) > MNLI_N:
    mnli = mnli.shuffle(seed=SEED).select(range(MNLI_N))
mnli = mnli.rename_columns({"premise": "premise", "hypothesis": "hypothesis"})
mnli_ds = mnli.map(tok_pair, batched=True,
                   remove_columns=[c for c in mnli.column_names if c not in ("label",)])
mnli_ds = mnli_ds.rename_column("label", "labels")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

s1_steps = max(1, -(-len(mnli_ds) // BATCH))
args1 = TrainingArguments(
    output_dir="/kaggle/working/_s1",
    num_train_epochs=STAGE1_EPOCHS,
    per_device_train_batch_size=BATCH,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_steps=int(s1_steps * STAGE1_EPOCHS * 0.06),
    fp16=USE_CUDA,
    logging_steps=200,
    save_strategy="no",
    report_to=[],
    seed=SEED,
    use_cpu=not USE_CUDA,
)
print("Stage 1: pretraining on MNLI ...")
Trainer(model=model, args=args1, train_dataset=mnli_ds, data_collator=collator).train()

# ---- Stage 2: fine-tune on competition data ----
comp_enc = tok(list(train["premise"]), list(train["hypothesis"]),
               truncation=True, padding=False, max_length=MAX_LEN)
comp_ds = Dataset.from_dict({**comp_enc, "labels": train["label"].tolist()})

s2_steps = max(1, -(-len(comp_ds) // BATCH))
args2 = TrainingArguments(
    output_dir="/kaggle/working/_s2",
    num_train_epochs=STAGE2_EPOCHS,
    per_device_train_batch_size=BATCH,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_steps=int(s2_steps * STAGE2_EPOCHS * 0.06),
    fp16=USE_CUDA,
    logging_steps=100,
    save_strategy="no",
    report_to=[],
    seed=SEED,
    use_cpu=not USE_CUDA,
)
print("Stage 2: fine-tuning on competition data ...")
Trainer(model=model, args=args2, train_dataset=comp_ds, data_collator=collator).train()
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
