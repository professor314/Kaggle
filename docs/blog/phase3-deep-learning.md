# Phase 3: Teaching the Toolkit to Do Deep Learning

*How we added transformers and CNNs to an AI-assisted Kaggle toolkit — and the honest results, including the parts that didn't go as planned.*

## Where we started

The toolkit had carried us through tabular competitions with scikit-learn and LightGBM. But three competitions were stuck against a ceiling only deep learning could break:

- **NLP Disaster Tweets** — TF-IDF + logistic regression plateaued at 0.801 F1
- **Digit Recognizer** — classical ML (PCA + SVM) sat at 0.984, rank ~435
- **Contradictory, My Dear Watson** — multilingual NLI, barely above chance with bag-of-words

Phase 3 was about adding PyTorch, transformers, and CNNs to the toolkit — then actually applying them.

## The environment split nobody warns you about

The first lesson had nothing to do with models. PyTorch didn't support our main Python 3.14 environment, so we set up a **separate Python 3.12 virtual environment** (`.venv-dl`) exclusively for deep learning. The tabular stack stayed on 3.14.

This meant the toolkit had to import cleanly in *both* environments — but the DL modules depend on torch, which the tabular env intentionally lacks. The fix was a lazy import in the package's `__init__.py`: deep-learning classes are only imported when actually requested, so `import kaggle_ml_toolkit` still works everywhere.

We also corrected a hardware assumption in our own planning docs: the machine had an **RTX 4080 Laptop (16GB)**, not the RTX 3090 (24GB) we'd written down. Small thing, but it changes what models fit in VRAM — worth getting right before you try to load something that won't fit.

## Building the pieces

Three additions, all matching the toolkit's existing scikit-learn-style API (`fit` / `predict` / `predict_proba` / `save` / `load`):

- **`TransformerClassifier`** — a HuggingFace fine-tuning wrapper, later extended to handle sentence pairs for NLI
- **`CNNClassifier`** — a compact conv net for grayscale image arrays like MNIST
- **`gpu_utils`** — device detection, seeding, mixed-precision helpers

One compatibility snag: `transformers` 5.x renamed `warmup_ratio` to `warmup_steps`. We derive steps from the ratio internally so the API stays stable across versions.

## Result 1: Tweets — a surprise about CV vs. leaderboard

We fine-tuned DistilBERT with 5-fold cross-validation. The out-of-fold F1 came back at **0.804** — essentially tied with the TF-IDF baseline. Discouraging.

But we submitted anyway, and the leaderboard said **0.836** — a jump of +0.035.

That gap between CV (0.804) and LB (0.836) is the interesting part. This competition's public test set is cleaner than its noisy training labels, so cross-validation *underestimated* the transformer's real-world performance. The lesson: when CV and the leaderboard disagree, don't assume CV is right — submit and find out.

We then built ensembles:
- DistilBERT + TF-IDF → **0.838**
- Adding BERTweet (a tweet-pretrained model) for a 3-way blend → **0.842**

The 3-way ensemble was our best. Interesting note: BERTweet correlated 0.91 with DistilBERT — more similar than we hoped — yet averaging three models still cancelled enough noise to gain another +0.004.

Progression: **0.801 → 0.836 → 0.838 → 0.842**.

## Result 2: Digits — and a bug that looked like success

The CNN trained to **99.64% validation accuracy** on the first run. Then we looked at the prediction distribution on the 28,000 test digits:

```
{0: 1742, 1: 134, 2: 3975, ..., 8: 7735}
```

Class 1 had 134 predictions; class 8 had 7,735. For MNIST, every class should be roughly ~2,800. A model at 99.6% validation accuracy would never produce that.

The root cause was a **train/test normalization mismatch**. During training, images were scaled by /255 *before* computing the normalization statistics — but at prediction time, the scaling step was skipped, so test inputs were ~255× off. Validation looked perfect only because it reused the already-scaled training tensors.

The fix was to move all scaling into a single shared transform used by both paths. A sane prediction distribution (~2,800 per class) is now a sanity check we run before every submission. The corrected model scored **0.99385**, and a 5-fold ensemble with test-time augmentation pushed it to **0.9945**.

The takeaway: a great validation score is not proof of correctness. Check the shape of your outputs, not just the metric.

## Result 3: Watson — a known Kaggle bug, and honest debugging

This one was a slog worth documenting, because the failure was external and the debugging path was instructive.

First, direct CSV submission returned HTTP 400. That turned out to be **by design** — Contradictory-Watson is a notebook/code competition, so Kaggle rejects direct file uploads and requires submission from a notebook that writes `submission.csv`.

So we moved to a Kaggle notebook. It failed at startup with an empty log. We fixed the data-attach (rules acceptance), tried again — still failing at startup, still no log. We guessed out-of-memory and shrank the workload. Still failing.

The breakthrough came from running the notebook *interactively* in Kaggle's UI, which surfaced the real error the batch log was hiding:

> If you want to use the Tesla P100-PCIE-16GB GPU with PyTorch, please check the instructions... `AcceleratorError`

This is a [documented Kaggle bug](https://github.com/Kaggle/docker-python/issues/1546): Kaggle's default **P100 GPU has compute capability sm_60**, but the PyTorch build in their current image is compiled for **sm_70+**. PyTorch has zero runnable kernels on the P100, so it dies the instant a tensor touches the GPU — at the C++/CUDA level, before Python prints anything, which is why the log was empty.

The fix: switch the notebook's accelerator from **P100 to T4** (compute capability sm_75, compatible), plus a defensive GPU probe that falls back to CPU if the mismatch ever recurs. The notebook then ran past the point that had killed every prior attempt.

An honest note on process: we initially misattributed the empty-log failures to out-of-memory. That was a wrong guess — the empty log genuinely looked like an OOM kill. Getting the *real* error required the interactive run. When a failure produces no diagnostic output, stop guessing and find a way to see the actual error before changing more code.

## What worked, what didn't

**Worked:**
- Lazy imports to keep one toolkit usable across two Python environments
- Transformer fine-tuning: the single biggest lever on text (+0.035 on tweets)
- A purpose-built small CNN beating classical ML on MNIST
- Ensembling and TTA for the last fraction of a percent

**Didn't (and why it's worth saying):**
- CV underestimated the leaderboard on tweets — nearly made us discard a winning model
- A normalization bug produced a perfect validation score with garbage predictions
- Two wrong diagnoses on Watson (data-attach, then OOM) before finding the real GPU-architecture bug

## AI Disclosure

This work was done with **Kiro**, an AI coding agent, collaborating with a human. The division of labor:

- **Human**: direction, accepting Kaggle competition rules, running the interactive notebook that surfaced the hidden GPU error, final judgment calls
- **AI (Kiro)**: implementing the toolkit modules, writing and debugging pipelines, running training, generating submissions, and drafting this write-up

The most useful thing the collaboration produced wasn't a score — it was a documented fix for a real Kaggle GPU bug, and a couple of honest reminders that a good metric isn't the same as a correct model.

*Content was rephrased from the linked sources for compliance with licensing restrictions.*
