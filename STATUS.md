# Kaggle ML Toolkit — Project Status

*Last updated: 2026-08-31*

## Toolkit Status

| Phase | Focus | Status |
|-------|-------|--------|
| Phase 1 | Core pipeline + Titanic | ✅ Complete |
| Phase 2 | Interpretability, augmentation, code competitions | ✅ Complete |
| Phase 3 | Deep learning (PyTorch/TF), cloud/GPU | 🚧 In progress — NLP + vision done; multilingual NLI kernel in flight |
| Time Series | time_series.py module (lags, rolling, temporal CV) | ✅ Complete |

---

## Competition Status

| Competition | Best LB | Rank | Submissions | Research | Status |
|------------|---------|------|:-----------:|:--------:|--------|
| **Titanic** | 0.773 (acc) | 6493 | 11 | ✅ Full | ✅ Complete |
| **House-Prices** | **0.1235 (RMSLE)** | 713 | 6 | ✅ Full | ✅ Boosting blend + outlier removal |
| **Spaceship-Titanic** | 0.802 (acc) | 639 | 6 | ✅ Full | ✅ Plateaued ~0.80 (tree models) |
| **Store-Sales** | **0.421 (RMSLE)** | 119 | 4 | ✅ Full | ✅ Per-family + holiday/tx features |
| **Playground-S6E8** | **0.965 (AUC)** | 1610 | 7 | — | ✅ Iterated |
| **NLP-Disaster-Tweets** | **0.842 (F1)** | **49** | 6 | ✅ Full | ✅ 3-way ensemble (top 10%) |
| **Home-Data-ML** | **13202 (MAE)** | **111** | 4 | — | ✅ Boosting stack (top 3%) |
| **ConnectX** | rank 64 | 64 | 1 | — | ⚠️ Minimal |
| **Digit-Recognizer** | **0.9945 (acc)** | 136 | 3 | — | ✅ 5-fold CNN + TTA (Phase 3) |
| **Contradictory-Watson** | pending | — | kernel v8 | ✅ Full | 🟡 Kernel COMPLETE (0.69 val); submission.csv valid, needs manual web/kernel submit |
| **LLM-Finetuning** | pending | — | kernel v3 | — | ⚠️ Code Competition |
| **TPU-Getting-Started** | pending | — | kernel v3 | — | ⚠️ Code Competition |
| **Kaggriculture** | pending | — | agent | — | ⚠️ Code Competition ($50K) |
| **RSNA-Knee** | 0.500 (smoke) | — | 2 | ✅ Full | 🟡 Pipeline proven; 2.5D EfficientNet baseline submitted (from scratch, 58 labels) |

---

## Key Scores & Improvements

| Competition | First Attempt | Current Best | Method |
|---|---|---|---|
| Store-Sales | 1.292 → 0.423 → **0.421** | Per-family LightGBM + holiday proximity + transaction lags |
| Playground-S6E8 | 0.961 → 0.965 → **0.965** | Tuned LightGBM (21 features, lr=0.03) |
| NLP-Tweets | 0.801 → 0.838 → **0.842** | DistilBERT → +TF-IDF → +BERTweet 3-way ensemble |
| Digit-Recognizer | 0.984 → 0.994 → **0.9945** | sklearn → CNN → 5-fold CNN + TTA |
| Spaceship-Titanic | 0.799 → **0.802** | Boosting sweep confirmed plateau (LightGBM best) |
| Titanic | 0.761 → **0.773** | Conservative GBM (8 features) |
| House-Prices | 0.145 → 0.127 → **0.1235** | Boosting blend (LGB+XGB+Cat+Ridge+Lasso) + outlier removal |
| Home-Data-ML | 14184 → 13499 → **13202** | Boosting stack (LGB+XGB+Cat+Ridge+Lasso, Ridge meta) |

---

## Blog Posts Published to imadestuff.com

- ✅ "Building an AI-Assisted Kaggle Toolkit" (toolkit overview)
- ✅ "Research First: Beating Kaggle's Titanic with History Books" (Titanic deep dive)
- ✅ "Building a 24/7 Go Bot with KataGo on KGS" (KataGo bot)

## Drafts ready in docs/blog (Phase 3)

- 📝 "Phase 3: Teaching the Toolkit to Do Deep Learning" (phase3-deep-learning.md)
- 📝 "NLP Disaster Tweets — Solution Writeup" (nlp-tweets-kaggle-writeup.md)

---

## What's Next (as of 2026-08-27)

### In flight — check first next session
- **Contradictory-Watson kernel** (`seanconnolly/contradictory-watson-xlmr`, v8):
  run **COMPLETE** on T4. submission.csv pulled + validated (5,195 rows, {0,1,2}).
  Blocked only on the submit step: notebook comp rejects CSV submit (400), and
  code submit needs the exact kernel version number (guessing 400'd). **Manual:**
  submit from the kernel's Output page on kaggle.com, or CLI with the correct
  `-v <version>`. To reach 0.85+, retry two-stage MNLI pretrain (subsample +
  lazy tokenization) on the T4.

### New competition opportunity (2026-08-31)
- **Playground S6E9** (deadline 2026-09-30, only ~26 teams so far): wide open,
  early-mover advantage, fits the tabular toolkit perfectly. **Blocked on rules
  acceptance:** data download 403s until the human clicks "Join Competition" at
  kaggle.com/competitions/playground-series-s6e9. Folder scaffolded at
  `competitions/playground-s6e9/`. Once joined: download data, run the standard
  research → EDA → per-model → ensemble pipeline.
- **Playground S6E8** closes **tonight** (2026-08-31 23:59). Currently 0.965 AUC,
  rank ~1904. Low ROI for a last-minute single submission; leaving as-is.

### Manual tasks left for the human (badges + publishing)
- Utility Scripter badge: open `kaggle-workflow-utils` → File → Set as Utility Script
- Code Forker badge: Copy & Edit any public notebook (API can't fork)
- Review + publish the two Phase 3 blog drafts in docs/blog/ to imadestuff.com

### Improvement opportunities (documented per competition in experiments_log.md)
- **Store-Sales** (0.421) → recursive near-term lags; per-family hyperparameter tuning
- **NLP Tweets** (0.842) → roberta-large or a non-transformer base for real diversity
- **Digit Recognizer** (0.9945) → more TTA views / more folds for marginal gains
- **Spaceship-Titanic** (0.802, PLATEAUED) → needs NEW signal (family/surname groups,
  CryoSleep imputation), not more models — stacking boosters didn't help
- **Home-Data-ML** → only 1 submission; run through the full pipeline
- **House-Prices** → try the boosting-ensemble approach built for Spaceship

### Deadline pressure
- **Playground S6E8** (Aug 31) — ensemble or feature engineering
- **Kaggriculture** (Sep 30, $50K) — needs proper agent strategy, Code Competition

### Phase 3 capabilities — DONE
- ✅ Transformer fine-tuning (`TransformerClassifier`, incl. NLI sentence pairs)
- ✅ Image CNN (`CNNClassifier`) + `gpu_utils`
- ✅ GPU training on RTX 4080 Laptop; Kaggle T4 kernels for code competitions
- Still open: LLM-Finetuning (LoRA), TPU flowers (vision kernel), cloud scaling
