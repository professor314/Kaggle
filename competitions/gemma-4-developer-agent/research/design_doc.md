# Gemma 4 Developer Agent (agent harness) - Research & Design Doc

**Kaggle ref:** gemma-4-developer-agent
**Deadline:** 2026-12-02 (VERIFIED 2026-11-12 via kaggle competitions list) | **Reward:** 75,000 USD | **Teams:** 5583
**Type:** LLM coding-agent harness (code competition)
**Status:** First pass, 2026-11-12. BLOCKED on rules acceptance. Data download returns 403 until the account joins on kaggle.com (same blocker as S6E9). Accepting rules is left to the account owner.

## 1. What we know (file list only, no data downloaded)
Files: HARNESS_README.md 56KB, docker/Dockerfile.public, Dockerfile.sandbox, imp.py, more (paginated list)

## 2. Prior art and facts
Top public notebook: 'Getting Started - Gemma 4 Developer Agent' by Ryan Holbrook, 2641 votes (VERIFIED). Others focus on budget-aware agents. Harness README is the key document; needs rules acceptance to download.

## 3. Reuse from existing code
kaggle_ml_toolkit/code_competition.py (ASSUMED relevant, not yet read), kernel push flow from contradictory-watson/artifacts/kernel_meta.

## 4. Design decisions
| Decision | Choice | Status |
|---|---|---|
| Metric | Read from Overview/Evaluation tab | ASSUMED until read |
| Baseline | Reproduce a top public notebook before any custom work | planned |
| CV | Match metric, keep OOF to compare against LB | planned |

## 5. Next steps
Accept rules, download HARNESS_README.md, fork the Getting Started notebook, run unmodified for a baseline score before changing anything.
