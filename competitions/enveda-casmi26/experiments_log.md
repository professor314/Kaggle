# Enveda CASMI26 (molecule ID from mass spectra) - Experiment Log

**Kaggle ref:** enveda-CASMI26-molecule-id-mass-spectra | **Deadline:** 2026-12-14

## 2026-11-12 - First pass (research only)
- Verified deadline, reward, team count, file list via kaggle CLI.
- Listed top public notebooks via kaggle kernels list (see design_doc.md section 2).
- Tried data download: not possible, rules not accepted (403 on DownloadDataFiles) - for S6E11 confirmed directly; the others assumed the same until the account joins.
- No submissions made. No experiments run.
- Priority: 4 (domain heavy, big data)

## 2026-11-13 - Submission 1: spectral nearest-neighbour library search
**Format (VERIFIED):** submission.csv molecule_id,smiles with up to 25 ';'-separated SMILES ranked best first. 400 test molecules / 1,213 test spectra. train.parquet 2,539,608 spectra (21 row groups), target column `normalized_smiles`.
**Approach:** Kaggle CPU kernel seanconnolly/enveda-casmi26-spectral-nn-baseline (notebooks/kernel-v1/). 0.01 Da bins to 1000 m/z, sqrt intensity, L2 norm, cosine vs first 600k train spectra; +0.5 bonus for precursor within 10 ppm and same adduct; per molecule take max over its spectra, top 25 unique SMILES.
**Submission:** 58332240, PENDING. Didn't download the 3 GB train locally.
**Next:** use all 2.5M spectra (sparse matmul in chunks), match on precursor mass first (formula-level filter), then the public 'Peaks to Structures + Scorer' (LB 0.487) ideas.

## 2026-11-27 - Diagnosis of 0.000 and reuse-first plan
**Why submission 1 scored 0.000 (VERIFIED by reading public notebooks):** test molecules are NOT in train, so searching train SMILES cannot return the answer. Winning pipelines generate candidates from natural-product databases (COCONUT, LOTUS, PubChem, spectral libraries), then rank them with a spectrum-to-fingerprint neural scorer and fragment checks. Metric is MRR@25.
**Field (from alexchilton's notebook):** 974 public notebooks, most are one lineage (dmitriigluzdov v4g assets) scoring 0.46-0.49; gold ~0.477 on 10/27, leader 0.513 then.
**Reuse-first (workflow 00):** pulled alexchilton's public 0.488 notebook (credits: analyticaobscura, popwenman, romanrozen, sanpier, dmitriigluzdov, ozertuu and others in the notebook), pushed to our account as seanconnolly/enveda-casmi26-p2s-reproduction (private, T4, version 1). Notebook is kept unmodified with its credits. Input datasets are public.
**Status:** v1 RUNNING. Next: on completion submit its output; then try an original contribution (own scorer ensembled, not swapped: public notes say swapping own net LOST every time, ensembling +0.001).
