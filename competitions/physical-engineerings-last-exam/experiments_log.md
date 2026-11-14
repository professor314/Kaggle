# Physical Engineerings Last Exam (agent harness) - Experiment Log

**Kaggle ref:** physical-engineerings-last-exam | **Deadline:** 2027-01-05

## 2026-11-12 - First pass (research only)
- Verified deadline, reward, team count, file list via kaggle CLI.
- Listed top public notebooks via kaggle kernels list (see design_doc.md section 2).
- Tried data download: not possible, rules not accepted (403 on DownloadDataFiles) - for S6E11 confirmed directly; the others assumed the same until the account joins.
- No submissions made. No experiments run.
- Priority: 2b (low competition, check rules first)

## 2026-11-13 - Submission format check (BLOCKED on user)
VERIFIED (NOTE.md + starter notebook by pablodiegoo): hackathon, no Kaggle dataset. An entry is a GitHub commit with `submission.yaml`, a WebSocket `serve.py` (protocol v1, google-research/ple), one of 8 Google-sponsored `google-cloud/<id>` models, and `artifacts/` from running the 29 public dev tasks (up to 6h each, offline container). Template repo: physical-engineerings-last-exam/submission-template.
Blocked: needs the account owner to link/create the GitHub repo and get access to the sponsored cloud models. Added to USER_TODO.md. Starter notebook saved in notebooks/ref/.
