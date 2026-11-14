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
