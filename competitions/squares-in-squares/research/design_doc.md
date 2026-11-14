# Squares in Squares (packing optimization) - Research & Design Doc

**Kaggle ref:** squares-in-squares
**Deadline:** 2027-07-18 (VERIFIED 2026-11-12 via kaggle competitions list) | **Reward:** 50,000 USD | **Teams:** 2002
**Type:** Geometric packing / combinatorial optimization, per-puzzle scoring
**Status:** First pass, 2026-11-12. BLOCKED on rules acceptance. Data download returns 403 until the account joins on kaggle.com (same blocker as S6E9). Accepting rules is left to the account owner.

## 1. What we know (file list only, no data downloaded)
Files: sample_submission.csv 12.9MB only (no train/test)

## 2. Prior art and facts
Public notebooks: per-puzzle leaderboard (148 votes), 'Living Atlas of Tight Packings', CPU/CUDA packing search (VERIFIED: kernels list). No tabular data, so this is an optimization problem, not ML.

## 3. Reuse from existing code
Little. numpy/scipy for local search; GPU search ideas from public CUDA notebooks. Toolkit is not a fit; keep scripts standalone.

## 4. Design decisions
| Decision | Choice | Status |
|---|---|---|
| Metric | Read from Overview/Evaluation tab | ASSUMED until read |
| Baseline | Reproduce a top public notebook before any custom work | planned |
| CV | Match metric, keep OOF to compare against LB | planned |

## 5. Next steps
Accept rules, read the Evaluation tab to learn how a submission row is scored, study the per-puzzle leaderboard notebook, then pick a few puzzles where simple local search beats the baseline.
