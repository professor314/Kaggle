# Tasks — RSNA Knee Baseline

Guiding rule: get a VALID, IN-TIME submission first. Each Kaggle push starts
tiny (few epochs, small slice budget, maybe a test subset) to prove it finishes,
then scales. Read the actual error body on any failure; tag findings VERIFIED.

## Task 0: Confirm data layout (exploration) — DONE
- [x] Push read-only explore kernel (no GPU) — confirmed CSVs + DICOM hierarchy
- [x] Fixed the explore kernel to avoid the stalling recursive glob (read CSVs + shallow listdir). Captured: 12 finding columns, train labels 58/4407, DICOM 512x512 int16 MR, series metadata.
- [x] Recorded VERIFIED facts into research/design_doc.md + requirements.md

## Task 1: Local scaffolding + tests (no Kaggle needed)
- [ ] `competitions/rsna-knee/rsna_lib.py` reusable helpers: locate data dir, load CSVs, read 12 finding columns from sample_submission header
- [ ] DICOM preprocessing fn: dcmread -> percentile window -> [0,1] -> resize 224 -> 3ch tensor
- [ ] Prevalence fallback: per-finding mean from train labels -> vector
- [ ] Submission assembler: build df matching sample_submission exactly
- [ ] Slice-sampling fn: K evenly-spaced slice paths for a study from *_series.csv
- [ ] (optional test) pytest with tiny synthetic DICOMs: preprocessing shape/range
- [ ] (optional test) pytest: prevalence fallback vector correct from a mock train.csv
- [ ] (optional test) pytest: submission assembler columns/order match sample, no NaN, values in [0,1]
- [ ] (optional test) pytest: slice sampler returns <= K and is deterministic

## Task 1b: Pipeline smoke test (PROVE THE PATH) — DONE
- [x] `kernel/smoke.py`: full job shape, prevalence predictor, reads K slices/study
- [x] smoke kernel (GPU off, internet off), pushed + ran (24 slices in 2.2s)
- [x] valid submission; SUBMITTED; **LB 0.500** (chance, as expected)
- [x] path proven -> proceeded to the model

## Task 2-3: Model + inference + submission — DONE
- [x] `kernel/baseline.py`: EfficientNet-B0, 12 sigmoids, BCE, K=8 slices mean-pooled
- [x] train on 58 labeled studies; prevalence fallback for unreadable studies
- [x] kernel_baseline metadata: machine_shape NvidiaTeslaT4, GPU on
- [x] VERIFIED gotcha: this comp FORBIDS internet in kernels (submit 400) -> set
      internet off + pretrained=False (train from scratch)
- [x] ran in 69s on T4, model_used=True, valid submission, ACCEPTED (LB pending)

## Task 2: Model + training (conservative)
- [ ] timm EfficientNet-B0, 12-sigmoid head, BCEWithLogitsLoss
- [ ] Study dataset: sample K evenly-spaced slices/study, mean-pool logits
- [ ] Train loop on labeled studies only, fp16 on GPU via `compute`, few epochs
- [ ] Held-out fold -> local macro-AUC, logged
- [ ] IF too few labels or any train failure -> skip to prevalence fallback (logged)

## Task 3: Inference + submission
- [ ] For each test study: read <= K slices, predict, mean-pool -> 12 probs
- [ ] Per-study try/except -> prevalence fallback on error (never crash)
- [ ] Write /kaggle/working/submission.csv in exact sample format
- [ ] Assert row count == sample, columns == sample, values in [0,1]

## Task 4: Kaggle kernel config + tiny proof run
- [ ] kernel-metadata.json: `machine_shape: "NvidiaTeslaT4"`, enable_gpu true, competition_sources set, internet as needed for timm weights (or attach weights dataset)
- [ ] Push with TINY settings (e.g. 1 epoch, K=4, small test subset) to prove end-to-end finishes and writes a valid submission
- [ ] Read the run log; fix any errors (dump error bodies)

## Task 5: Scale within the time budget
- [ ] Increase K and epochs while watching runtime vs the GPU cap; keep margin
- [ ] Full test set inference
- [ ] Submit the completed kernel version; record LB macro-AUC

## Task 6: Document
- [ ] research/design_doc.md + experiments_log.md updated with method, decisions (VERIFIED/ASSUMED), local val, LB
- [ ] STATUS.md row for rsna-knee
- [ ] Commit + push to professor314/Kaggle (exclude .kiro, data, artifacts, weights)

## Notes / guardrails
- Do NOT walk the DICOM tree; index from CSVs (VERIFIED: walking stalls the kernel).
- Everything runs on Kaggle; local box only edits + pushes.
- GPU sessions limited to 2; wait/cancel before re-push if "session count" errors.
- Notebook submission only (like Watson): submit the kernel version, not a CSV.
