---
phase: 7
title: Submission
triggers: kaggle_ml_toolkit.submission_generator
---

# Submission

## Context
The submission phase converts our best model's predictions into Kaggle-format output and submits to the leaderboard. We track submissions carefully — Kaggle limits most competitions to 10 submissions per day. Each submission is an experiment we can learn from.

## Verify before you conclude (anti-loop rules)

These rules exist because of a real failure: on Watson we stated several wrong
"root causes" as fact, acted on each, and looped for multiple sessions without a
submitted score. See `docs/COE-watson-submission-loop.md`. Follow these to avoid
repeating it.

1. **Dump the error body on ANY 4xx.** A bare "400 Bad Request" is not a
   diagnosis. Print the full HTTP response body before concluding anything. For
   the Kaggle API this means catching the exception and printing `e.response.text`.
   The body usually states the exact cause (e.g. "runtime exceeds the 120-minute
   GPU max").
2. **Tag every submission fact VERIFIED or ASSUMED.** In the experiments log,
   mark each claim as `VERIFIED (command + evidence)` or `ASSUMED (needs check)`.
   An ASSUMED item must never drive more than one action before being verified.
3. **A new conclusion that contradicts a recorded fact = STOP and re-test.**
   Do not tell the user or change direction on a claim that conflicts with the
   log until you have re-run the check live and updated the log.
4. **Two failures on the same step = diagnose, don't patch.** After the second
   failed attempt at the same operation, stop trying variations. Read the actual
   error, find the root cause, and only then attempt a fundamentally different
   fix. Do not burn daily submission attempts guessing (e.g. version numbers).
5. **Re-verify inherited "facts" at the start of a session.** Before acting on a
   submission note written in a previous session, confirm it still holds with one
   quick check. Environments and competition rules change.

## Known submission facts (keep VERIFIED, with evidence + date)

- **Contradictory, My Dear Watson: notebook-submission ONLY.** Direct CSV submit
  returns HTTP 400 `CreateSubmission`. VERIFIED 2026-08-31. Getting-Started label
  does NOT imply CSV upload is allowed.
- **Kaggle kernel GPU model is chosen by `machine_shape`, NOT `accelerator`.**
  The `accelerator` field is ignored by the CLI, and `enable_gpu: true` is only
  a yes/no. Omit `machine_shape` (or use the generic "Gpu") and you silently get
  a **P100 (sm_60)**, which breaks modern PyTorch (needs sm_70+) and produces
  dead kernels. Set `"machine_shape": "NvidiaTeslaT4"` in kernel-metadata.json
  to get the T4 (native sm_75, no torch reinstall). Must be set on every push;
  a UI selection does not persist to CLI pushes. VERIFIED 2026-08-31 (Kaggle
  kaggle-api issue #490, confirmed fix; our v9/v10 P100 failures).
- **GPU batch-session limit is 2.** `kernels push` fails with "Maximum batch GPU
  session count of 2 reached" if two GPU kernels are already running/queued.
  Wait for prior runs to finish or delete stale ones before re-pushing.

## Decision Points

### Model to Submit
**Question:** Which model should generate the submission predictions?
**Options:**
- Best single model: Use the top individual optimized model. Simple and interpretable.
- Ensemble: Use the ensemble from the previous phase. Usually the strongest predictions.
- Both: Submit the ensemble as primary and best single model as backup. Uses 2 daily submissions.
- Experimental: Submit a model with a specific hypothesis to test against the leaderboard.
**Recommendation:** Ensemble as primary submission. Submit best single model too if daily limit allows.

### Submission Message
**Question:** What information should the submission message contain?
**Options:**
- Minimal: Model name and date only.
- Descriptive: Model name, key parameters, CV score, and feature set description.
- Full context: Everything above plus hypothesis being tested and expected rank range.
**Recommendation:** Descriptive — enough to identify what changed between submissions without being verbose.

### Submission Method
**Question:** How should we submit to Kaggle?
**Options:**
- CLI automated: Use `kaggle competitions submit` via the toolkit. Fastest, scriptable.
- Manual upload: Generate the CSV, then upload through Kaggle web UI. More control, visual confirmation.
- CLI with confirmation: Generate file, show preview, ask for confirmation, then submit via CLI.
**Recommendation:** CLI with confirmation — automation with a human checkpoint prevents accidental submissions.

### Pre-Submission Validation
**Question:** What checks should we run before submitting?
**Options:**
- Format check: Verify column names, row count, no missing values. Minimum viable check.
- Format + sanity: Add prediction distribution check, compare against training target distribution.
- Full validation: Add leakage scan, submission diff against previous best, CV-vs-LB gap analysis.
**Recommendation:** Format + sanity — catches most issues. Full validation for high-stakes submissions.

### Post-Submission Actions
**Question:** What should we do after getting the leaderboard score?
**Options:**
- Record only: Log the LB score in experiment tracker. Minimum bookkeeping.
- Record + analyze: Compare LB score vs. CV score, flag large gaps for investigation.
- Record + analyze + plan: All above plus decide next experiment based on LB feedback.
**Recommendation:** Record + analyze — CV-LB gap is the most important signal for diagnosing overfitting or data shift.

## Actions
Based on decisions above, Kiro will:
1. Generate predictions on test data using chosen model
2. Execute `SubmissionGenerator.generate()` to create formatted CSV
3. Run pre-submission validation checks
4. Present file preview (first/last rows, shape, distribution)
5. If CLI: execute `SubmissionGenerator.submit()` after user confirmation
6. If manual: provide file path and instructions for web upload
7. Record submission details in experiment tracker
8. After LB score is known: analyze CV-LB gap and record in design doc
