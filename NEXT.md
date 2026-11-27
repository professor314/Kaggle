# Next Session — Prioritized Checklist

*Updated: 2026-10-24. Delete/reorder items as they're done. Full context lives in
STATUS.md, PHASE3_PROGRESS.md, and each competition's experiments_log.md.*

> **Passed deadlines (2026-10-24):** kaggriculture (9/30), Playground S6E8/S6E9
> (9/30), rsna-knee (~10/22) are closed — archived unless re-verified. Focus is
> now on submit-anytime tabular comps + Contradictory-Watson.

## 0. Done this session (2026-10-24, offline)
- [x] House-Prices: added OOF target encoding of Neighborhood + 5 other
      high-card categoricals (`notebooks/boosting_stack_te.py`). OOF RMSLE
      0.11282 → **0.11218**. SUBMITTED 2026-10-24 → **LB 0.12162 (NEW BEST**,
      was 0.12345). Plateau broken. Next lever: per-model TE smoothing / weighted stack.

## 1. Check things in flight (do first)
- [x] Watson SUBMITTED 2026-09-01 → **LB 0.66159** acc (single-stage XLM-R).
      Root cause of the long saga: `machine_shape` must be `NvidiaTeslaT4`.
- [x] Joined Playground S6E9; data downloaded.
- [x] **S6E9 pipeline**: DONE + PLATEAUED (2026-09-27). Ran LGBM baseline, booster
      blend, big-iteration, and OOF target-encoding + Logistic base. Best LB
      **0.94169**. Nothing beats ~0.9417 (bases rank-corr 0.9864). See
      `competitions/playground-s6e9/experiments_log.md` Exp 4. Deadline 2026-09-30.
- [ ] **Kaggriculture (deadline 2026-09-30!)**: live agent = Herd engine v4
      (`evo/herd_engine.py`). To improve, recreate `.venv-dl` with
      `kaggle_environments`, benchmark any candidate head-to-head vs herd_engine
      (NOT the GA champion), and only submit a VERIFIED win. See STRATEGY.md §3 for
      the meta-fit-seed idea. NOT done this session (needs the DL env rebuilt).
- [ ] Watson to 0.85+: two-stage MNLI pretrain → fine-tune (subsample + lazy
      tokenization on the T4).

## Env note (2026-09-27)
- `.venv-dl` is gone. The **anaconda3** python
  (`C:\Users\profe\anaconda3\python.exe`) has lightgbm/xgboost/catboost/sklearn/
  pandas/scipy and ran the S6E9 pipeline fine. It does NOT have
  `kaggle_environments` (needed for kaggriculture) — rebuild a venv for that.

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
- **Set `n_jobs=-1` (LGBM/XGB) and `thread_count=-1` (CatBoost).** Default is
  single-core; on big data (600K+ rows) that wastes a multi-core machine and a
  blend can take 10+ min. With all cores it's several times faster. For CatBoost
  on huge data consider `task_type="GPU"` (RTX 3090 available).

## 2026-11-12 Kaggle first pass (8 unentered competitions)
Scaffolded research/design_doc.md + experiments_log.md for: playground-s6e11, gemma-4-developer-agent, physical-engineerings-last-exam, name-that-llm, squares-in-squares, pokemon-ptcg-playground, enveda-casmi26, quantum-circuit-compilation.
ALL BLOCKED on accepting rules on kaggle.com (data download 403 confirmed for S6E11). Suggested order: S6E11 (closes 11/30), gemma-4-developer-agent (12/2), PLE (19 teams), name-that-llm (reuses Watson code).
S6E11 baseline script is ready: competitions/playground-s6e11/notebooks/lgbm_pipeline.py (compiles, not run).
Skipped: gemma-4-developer-agent-paper (writing track, closes 2026-11-12 23:59).

## 2026-11-26 S6E11 status
Best LB 0.94719 (3-way blend). Boosters saturated. Next: look for original source dataset to augment, seed-average. Closes 11/30. Then Enveda (12/14, LB 0.000 vs ~0.56 leaders), Gemma (12/2).

## 2026-11-27 S6E11
Best LB 0.94843 (shallow TE + original rows). See competitions/playground-s6e11/README.md section 6. Closes 11/30. After: Enveda (12/14), Gemma (12/2).

## 2026-11-27 late
S6E11 best LB 0.94849. Pairs/GLM rejected. Remaining: neural nets on Kaggle GPU. Then Enveda, Gemma.
