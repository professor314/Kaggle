# Enveda CASMI26: molecule ID from MS/MS spectra

## 1. Problem and metric
Given MS/MS spectra for a test molecule, return up to 25 candidate SMILES, best first. Scored by MRR@25 (reciprocal rank of the true structure). Test molecules are not in train, so this is candidate generation plus ranking, not lookup. Closes 2026-12-14.

## 2. Results
| # | Approach | Public LB | Status |
|---|---|---|---|
| 1 | Cosine nearest neighbour over 600k train spectra | 0.000 | VERIFIED. Cannot work: answers are not in train. |
| 2 | Reproduction of public 0.488 pipeline (own account) | pending | kernel v1 running |

## 3. Design decisions
- **Reuse-first.** The public field is one lineage of pipelines (library search + PubChem/natural-product candidates + neural scorer + fragment check). Rebuilding it alone would take weeks. We run it on our account with credits intact, then look for our own gain.
- **Why 0.000 first:** we assumed library search. Evidence it fails: train and test molecules are disjoint. Alternative rejected: using more train spectra (still disjoint).
- **Own contribution plan:** ensemble our own scorer with the public one. Public notes show swapping in a self-trained scorer lost on all 7 attempts; averaging z-scores gave +0.001.

## 4. Toolkit reuse
None yet. Candidate: a kernel-pull-and-repush helper (done by hand here).

## 5. Reproduce
`kaggle kernels push -p competitions/enveda-casmi26/notebooks/kernel-v2` (T4, internet off, needs the listed public datasets; runs for hours).

## 6. What's next
1. Submit kernel v1 output when complete.
2. Train own scorer and ensemble.
3. Check for mass-accuracy / adduct priors.