# Name That LLM (identify the model behind a text) - Experiment Log

**Kaggle ref:** name-that-llm | **Deadline:** 2027-01-14

## 2026-11-12 - First pass (research only)
- Verified deadline, reward, team count, file list via kaggle CLI.
- Listed top public notebooks via kaggle kernels list (see design_doc.md section 2).
- Tried data download: not possible, rules not accepted (403 on DownloadDataFiles) - for S6E11 confirmed directly; the others assumed the same until the account joins.
- No submissions made. No experiments run.
- Priority: 2c (small data, reuses Watson code)

## 2026-11-13 - Submission 1: TF-IDF (word 1-2 + char_wb 2-5) + logistic regression
**Data (VERIFIED):** train 2,699 rows (id, prompt_id, text, model), 18 models x ~150. Visible test only 3 rows, so it's a code competition with a hidden test. Submission = submission.parquet (id, model).
**CV:** GroupKFold(5) on prompt_id (all models answer the same prompts; a random split would leak). Local OOF acc 0.3164; on Kaggle 0.3123. Chance 0.0556.
**Kernel:** seanconnolly/name-that-llm-tf-idf-lr-baseline (slug got the hyphen from 'TF-IDF'). Script in notebooks/tfidf_lr.py.
**Submission:** 58331950, PENDING.
**Gap:** public notebooks ~0.68 with ModernBERT/Qwen3-embedding LoRA. Next: fine-tune an encoder on a T4 (reuse Watson XLM-R kernel), style features (markdown, list usage, length).
