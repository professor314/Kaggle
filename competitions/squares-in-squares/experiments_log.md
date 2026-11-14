# Squares in Squares (packing optimization) - Experiment Log

**Kaggle ref:** squares-in-squares | **Deadline:** 2027-07-18

## 2026-11-12 - First pass (research only)
- Verified deadline, reward, team count, file list via kaggle CLI.
- Listed top public notebooks via kaggle kernels list (see design_doc.md section 2).
- Tried data download: not possible, rules not accepted (403 on DownloadDataFiles) - for S6E11 confirmed directly; the others assumed the same until the account joins.
- No submissions made. No experiments run.
- Priority: 3 (long deadline, no ML reuse)

## 2026-11-13 - Submission 1: grid baseline
**Format (VERIFIED):** CSV id,n,square_id,x,y,angle for n=1..1000 (500,500 rows). Score = sum_n side(n)/sqrt(n) - 1, lower is better (from the public Square Packing Lab notebook's total_score; ASSUMED identical to the official metric).
**Approach:** notebooks/grid_baseline.py, k x k grid, k=ceil(sqrt n). Local score 28.9377. Same as the sample submission.
**Submission:** 58331996, PENDING.
**Gap:** best public portfolio ~22.285 (merges known best packings, Erich Friedman's tables). Next: import known-optimal packings for small n (Friedman's 'Squares in Squares' page), then local search with overlap repair.
