# Enveda CASMI26 (molecule ID from mass spectra) - Research & Design Doc

**Kaggle ref:** enveda-CASMI26-molecule-id-mass-spectra
**Deadline:** 2026-12-14 (VERIFIED 2026-11-12 via kaggle competitions list) | **Reward:** 50,000 USD | **Teams:** 5211
**Type:** Cheminformatics: structure from MS/MS spectra
**Status:** First pass, 2026-11-12. BLOCKED on rules acceptance. Data download returns 403 until the account joins on kaggle.com (same blocker as S6E9). Accepting rules is left to the account owner.

## 1. What we know (file list only, no data downloaded)
Files: train.parquet 3.03GB, test.parquet 4.8MB, sample_submission.csv 44KB

## 2. Prior art and facts
Public LB 0.487 notebook 'Peaks to Structures + Scorer'; a fast spectral cosine baseline exists; an official denovo tutorial notebook by inversion (VERIFIED: kernels list). 3GB train is heavy for local RAM.

## 3. Reuse from existing code
Little. compute.py / cloud_scaling.py for remote runs. Needs rdkit (not yet confirmed installed).

## 4. Design decisions
| Decision | Choice | Status |
|---|---|---|
| Metric | Read from Overview/Evaluation tab | ASSUMED until read |
| Baseline | Reproduce a top public notebook before any custom work | planned |
| CV | Match metric, keep OOF to compare against LB | planned |

## 5. Next steps
Accept rules, start from inversion's denovo tutorial and the cosine baseline on Kaggle notebooks (avoid downloading 3GB), check rdkit availability.
