# Next Session — Prioritized Checklist

*Updated: 2026-08-27. Delete/reorder items as they're done. Full context lives in
STATUS.md, PHASE3_PROGRESS.md, and each competition's experiments_log.md.*

## 1. Check things in flight (do first)
- [x] Watson kernel COMPLETE (2026-08-31). submission.csv pulled + validated.
- [ ] **SUBMIT Watson** (manual/web): notebook comp, CSV submit 400s. Submit from
      the kernel Output page on kaggle.com, or CLI with the exact latest version:
      `kaggle competitions submit contradictory-my-dear-watson -k seanconnolly/contradictory-watson-xlmr -f submission.csv -v <version>`
- [ ] **Join Playground S6E9** at kaggle.com (accept rules) so data download stops
      403ing, then run the tabular pipeline. Deadline 2026-09-30, wide open.

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
- Shell is PowerShell: call exes with `& "..."`, use `;` not `&&`
- Notebook comps reject direct CSV submit (400) — submit via kernel
- Kaggle P100 GPU breaks current PyTorch (sm_60 vs sm_70+) → kernel accelerator T4
- Verify test prediction distribution looks sane before submitting (caught a CNN bug)
- Always call the DL venv python by absolute path
