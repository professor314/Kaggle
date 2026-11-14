# Physical Engineerings Last Exam (agent harness) - Research & Design Doc

**Kaggle ref:** physical-engineerings-last-exam
**Deadline:** 2027-01-05 (VERIFIED 2026-11-12 via kaggle competitions list) | **Reward:** 400,000 USD | **Teams:** 19
**Type:** LLM agent harness for engineering problems
**Status:** First pass, 2026-11-12. BLOCKED on rules acceptance. Data download returns 403 until the account joins on kaggle.com (same blocker as S6E9). Accepting rules is left to the account owner.

## 1. What we know (file list only, no data downloaded)
Files: NOTE.md (55 bytes) only

## 2. Prior art and facts
Only 19 teams (VERIFIED), so the field is tiny for a 400k prize. Public starter notebooks and agent harnesses exist (code-gated harness, dual-tier harness, field-lab starter). Data is likely hidden, only a note is visible.

## 3. Reuse from existing code
Same harness approach as gemma-4-developer-agent; reuse whatever works there.

## 4. Design decisions
| Decision | Choice | Status |
|---|---|---|
| Metric | Read from Overview/Evaluation tab | ASSUMED until read |
| Baseline | Reproduce a top public notebook before any custom work | planned |
| CV | Match metric, keep OOF to compare against LB | planned |

## 5. Next steps
Accept rules, read Overview/Evaluation carefully, run the starter notebook unmodified, then compare with the code-gated harness notebook.
