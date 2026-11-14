# Pokemon TCG AI Battle Challenge (Playground) - Research & Design Doc

**Kaggle ref:** the-pokemon-company-ptcg-ai-battle-challenge-playground
**Deadline:** 2027-01-08 (VERIFIED 2026-11-12 via kaggle competitions list) | **Reward:** Swag | **Teams:** 718
**Type:** Game-playing agent
**Status:** First pass, 2026-11-12. BLOCKED on rules acceptance. Data download returns 403 until the account joins on kaggle.com (same blocker as S6E9). Accepting rules is left to the account owner.

## 1. What we know (file list only, no data downloaded)
Files: Card_ID_List_EN/JP.pdf, EN/JP_Card_Data_R2_full.csv, more (paginated)

## 2. Prior art and facts
Public notebooks: Ultra 900+, hierarchical policy with Bradley-Terry modeling, meta analysis (VERIFIED: kernels list). Closest in spirit to the connectx / kaggriculture work.

## 3. Reuse from existing code
kaggriculture agent and kaggle_environments setup (note: .venv-dl needs rebuilding per NEXT.md), arena_generator.py / model_arena.py ideas.

## 4. Design decisions
| Decision | Choice | Status |
|---|---|---|
| Metric | Read from Overview/Evaluation tab | ASSUMED until read |
| Baseline | Reproduce a top public notebook before any custom work | planned |
| CV | Match metric, keep OOF to compare against LB | planned |

## 5. Next steps
Accept rules, download card data, run the Ultra 900+ notebook as baseline, then try a heuristic agent first.
