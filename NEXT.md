# Next Session — Prioritized Checklist

*Updated: 2026-08-27. Delete/reorder items as they're done. Full context lives in
STATUS.md, PHASE3_PROGRESS.md, and each competition's experiments_log.md.*

## 1. Check things in flight (do first)
- [x] Watson SUBMITTED 2026-09-01 → **LB 0.66159** acc (single-stage XLM-R).
      Root cause of the long saga: `machine_shape` must be `NvidiaTeslaT4`.
- [x] Joined Playground S6E9; data downloaded.
- [ ] **S6E9 pipeline**: `lgbm_pipeline.py` written + dtype-fixed. Run it in the
      BACKGROUND (see interrupt gotcha) to get OOF AUC, then submit. Deadline 2026-09-30.
- [ ] Watson to 0.85+: two-stage MNLI pretrain → fine-tune (subsample + lazy
      tokenization on the T4).

## 2. Quick wins / low effort
- [ ] Publish the 2 Phase 3 blog drafts (docs/blog/phase3-deep-learning.md,
      nlp-tweets-kaggle-writeup.md) to imadestuff.com — review first
- [ ] Manual Kaggle badges: Utility Scripter (File → Set as Utility Script on
      kaggle-workflow-utils), Code Forker (Copy & Edit any public notebook)

## 3. Phase 3 — deep learning (main focus)
- [ ] TPU-Getting-Started (flower classification): vision kernel with timm
      EfficientNet — new territory, uses the GPU
- [ ] Watson to 0.85+: two-stage MNLI pretrain → fine-tune, memory-safe on T4
      (subsample MNLI + lazy .map tokenization; first attempt OOM'd)
- [ ] LLM-Finetuning: investigate format, LoRA if applicable
- [ ] Consider adding an ImageClassifier (timm) to the toolkit for natural-image comps

## 4. Modeling improvements (tabular, submit directly — no kernel hassle)
- [ ] Home-Data-ML: only 1 submission; run full pipeline
- [ ] House-Prices: try the boosting-ensemble approach built for Spaceship
- [ ] Store-Sales (0.421): recursive near-term lags; per-family hyperparameter tuning
- [ ] Playground S6E8 (deadline Aug 31): ensemble / feature engineering

## Known gotchas (don't re-discover)
- **Long commands get interrupted by the terminal (~returns `^C`, exit -1).**
  Anything over ~30s (Kaggle API, git push, model training) can be killed
  mid-run, and it can take the child process down with it. FIX: launch as a
  DETACHED background process writing to a log file, return immediately, then
  poll the log in separate short calls. Do NOT block with `-Wait` on long jobs.
- **Kaggle GPU model is set by `machine_shape`, NOT `accelerator`.** Use
  `"machine_shape": "NvidiaTeslaT4"` in kernel-metadata.json. Omitting it (or the
  `accelerator` field, which the CLI ignores) silently gives a P100 (sm_60) that
  breaks modern PyTorch (needs sm_70+) → dead kernel. Set it on every push.
- **Kaggle GPU batch-session limit = 2.** `kernels push` fails with "Maximum
  batch GPU session count of 2 reached"; wait for prior runs or cancel via the
  Active Events page (API can't cancel a session).
- **pandas 3: text columns report dtype `str`, not `object`.** Detect categoricals
  with `not pd.api.types.is_numeric_dtype(col)`, not `== "object"`, or the
  encoder silently skips them and LightGBM errors on str dtypes.
- Notebook comps reject direct CSV submit (400) — submit via kernel + version.
  For code submit, the earlier 400s were the 120-min GPU cap, not the version.
- Shell is PowerShell: call exes with `& "..."`, use `;` not `&&`.
- Verify test prediction distribution looks sane before submitting (caught a CNN bug).
- Always call the DL venv python by absolute path.
- On ANY Kaggle 4xx, print `e.response.text` — the body states the real cause.
