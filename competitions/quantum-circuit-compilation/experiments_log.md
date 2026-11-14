# Google Verifiable Quantum Circuit Compilation - Experiment Log

**Kaggle ref:** google-verifiable-quantum-circuit-compilation | **Deadline:** 2027-01-20

## 2026-11-12 - First pass (research only)
- Verified deadline, reward, team count, file list via kaggle CLI.
- Listed top public notebooks via kaggle kernels list (see design_doc.md section 2).
- Tried data download: not possible, rules not accepted (403 on DownloadDataFiles) - for S6E11 confirmed directly; the others assumed the same until the account joins.
- No submissions made. No experiments run.
- Priority: 5

## 2026-11-13 - Submission 1: official sample submission
**Format (VERIFIED):** submission.zip with invsqrt.kmb, dqi.kmb, dfthcsos/angle_k_*.kmb + dfthcsos/submission.npz. Score = geometric mean of C_ref / C_task over the 3 tasks (higher is better per public notebooks; ASSUMED until LB shows it). Verifiers + kickmix.so (Linux) in the data.
**Approach:** Kaggle kernel seanconnolly/qcc-sample-submission-baseline zips sample_submission/ as is (13 members, 73.8 MB zipped; dqi.kmb is 886 MB raw). Reference floor.
**Submission:** 58332242, PENDING. The local data copy (1 GB+) was deleted to keep it out of Google Drive; the paper PDF was kept in research/.
**Next:** the public portfolios (~24.9) combine improved circuits from many credited notebooks/datasets; start by reading verifiable_circuit_compilation.pdf and one task's verifier (invsqrt is the smallest).
