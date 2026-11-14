# Kaggle portfolio checklist (updated 2026-11-12)

## Done (via CLI)
- All 24 notebooks public (23 existing + new S6E11 baseline).
- Fixed input-path failures (Kaggle now mounts competitions under a different path; scripts
  use `_find_input(slug)`): watson-tfidf, llm-finetuning, house-prices, home-data, store-sales,
  digit-recognizer, tpu-getting-started. All completed on re-run.
- llm-finetuning-tfidf-baseline rewritten from an exploratory script to a clean baseline
  (OOF log loss 1.0868 vs 1.098 uniform).
- nlp-disaster-tweets-3way-ensemble: needed GPU + internet (HF weights); metadata fixed, re-run pushed.
- Retitled CPU scratch notebooks (retitling renames the URL slug). GPU notebooks were NOT retitled
  because every push re-runs them and spends GPU quota:
  rsna-knee-a1, rsna-knee-v2, rsna-knee-baseline, rsna-knee-nli-labels, contradictory-watson-xlmr.
- New: s6e11-credit-card-fraud-lightgbm-baseline (OOF AUC 0.94594 reproduced on Kaggle, LB 0.94673).

## Manual steps (web UI only)
1. Rename the 4 GPU RSNA notebooks (Edit > title) so they read like a series, e.g.
   "RSNA Knee MRI - EfficientNet Baseline", "... v2 (Weak Labels)", "... Wide-Dense A1", "... NLI Weak Labels".
2. Pin your best 4-6 on the profile: S6E11 baseline, contradictory-watson-xlmr, house-prices stack,
   digit-recognizer CNN ensemble, one RSNA notebook.
3. Profile: bio, location, links (imadestuff.com, GitHub).
4. Kaggle badges: open `kaggle-workflow-utils` > File > Set as Utility Script; Copy & Edit any public notebook (Code Forker).
5. Post on S6E11 (closes 11/30) once there is a notable improvement; comment on discussion threads.
6. Optionally publish the Watson XLM-R weights as a Kaggle Model/Dataset.

## Notes
- Script notebooks do not render markdown. For the showcase notebooks, converting to .ipynb with
  markdown intro cells is a worthwhile follow-up (not done).
- Kaggle only accepts tags from its fixed list; custom keywords (e.g. 'fraud') are dropped.

## Blog drafts waiting for review (kaggle/docs/blog/)
phase3-deep-learning.md, nlp-tweets-kaggle-writeup.md, titanic-kaggle-writeup.md, toolkit-intro.md.
Blog style: no em/en dashes. Not published.
