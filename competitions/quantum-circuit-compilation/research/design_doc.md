# Google Verifiable Quantum Circuit Compilation - Research & Design Doc

**Kaggle ref:** google-verifiable-quantum-circuit-compilation
**Deadline:** 2027-01-20 (VERIFIED 2026-11-12 via kaggle competitions list) | **Reward:** 150,000 USD | **Teams:** 2951
**Type:** Optimization with a verifier (circuit compilation)
**Status:** First pass, 2026-11-12. BLOCKED on rules acceptance. Data download returns 403 until the account joins on kaggle.com (same blocker as S6E9). Accepting rules is left to the account owner.

## 1. What we know (file list only, no data downloaded)
Files: kickmix/kickmix.so (Linux binary), sample_submission/*.kmb, more (paginated)

## 2. Prior art and facts
Top public notebook: 'Verified Three-Task Portfolio' at LB 24.94; other notebooks mention DF-THC group compression and a 24.65 local GM refit (VERIFIED: kernels list). kickmix.so is a Linux shared library, so local Windows runs need WSL or Kaggle notebooks.

## 3. Reuse from existing code
None directly. Likely run on Kaggle notebooks.

## 4. Design decisions
| Decision | Choice | Status |
|---|---|---|
| Metric | Read from Overview/Evaluation tab | ASSUMED until read |
| Baseline | Reproduce a top public notebook before any custom work | planned |
| CV | Match metric, keep OOF to compare against LB | planned |

## 5. Next steps
Accept rules, read evaluation, run the 24.94 portfolio notebook for a baseline, check WSL availability.
