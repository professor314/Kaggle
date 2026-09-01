# Correction of Error: The Watson Submission Loop

**Date:** 2026-08-31
**Competition:** Contradictory, My Dear Watson
**Impact:** Multiple sessions of churn; several confidently-stated conclusions
that were wrong or unverified; direction changed repeatedly; still no submitted
score.

## Summary

We tried to submit the Watson competition many times over multiple sessions.
The agent stated several different "root causes" as fact, acted on each, and
kept looping instead of converging. No leaderboard score was ever recorded.

## Timeline of wrong / unverified claims

1. **"Notebook comp, direct CSV submit rejected (400)."** Written into the log
   as a fact early on. It was *true*, but it was recorded from a single failed
   attempt without capturing the error body, so we never actually knew *why*.
2. **"The 400 is a wrong kernel-version number."** Guessed versions, burned
   attempts, never read the error body. Wrong.
3. **"Root cause is the 233-min CPU run exceeding the 120-min cap."** This time
   the error body was finally read and this was correct — but only after the
   loop had already gone on for a while.
4. **"Kernel needs to run on T4; set accelerator."** Partly right, but missed
   the documented API bug (pushed kernels always get a P100).
5. **"It accepts a CSV, so we're in the spirit either way / just upload."**
   Stated confidently to the user. **Wrong** — re-tested live and it 400'd.
   Directly contradicted claim #1 which was in our own notes.

## The single root cause of the LOOP (not the bug)

**Assumptions were written into the docs as facts without being verified, then
inherited by every later step.** The technical bug (P100 vs time cap) was real,
but the *looping* was caused by process, not by Kaggle:

- A one-off observation ("CSV submit 400'd") became a permanent "fact" in the
  log without the evidence (the error body) that would let us reason about it.
- Later reasoning trusted the note instead of re-checking, so we solved the
  wrong problem repeatedly.
- When memory of the note faded across sessions, the agent re-derived a
  *contradictory* claim ("it accepts CSV") and stated it confidently, because
  nothing in the process forced a re-verification against the recorded fact.
- Error messages were treated as opaque ("400 Bad Request") instead of being
  opened up (the body literally explained the cause the whole time).

## What we should have done

1. On the FIRST 400, capture and record the full error **body**, not just the
   status code. The body said exactly what was wrong once we finally read it.
2. Label everything in the log as **VERIFIED** (with the command + evidence) or
   **ASSUMED** (needs checking). Never let an ASSUMED item drive multiple steps.
3. When a new conclusion **contradicts** a recorded fact, stop and re-test
   before acting or telling the user.
4. Two failed attempts at the same thing = stop patching, read the actual error,
   diagnose the root cause before the third attempt.

## Corrective actions (built into steering / process)

- Added a **"Verify before you conclude"** section to
  `.kiro/steering/07-submission.md` and `.kiro/steering/01-research.md`.
- Standing rule: submission facts must be tagged VERIFIED/ASSUMED with the
  evidence (command + response body) in the experiments log.
- Standing rule: always dump the HTTP error **body** on any 4xx from the Kaggle
  API before drawing a conclusion.

## Verified facts for Watson (as of 2026-08-31, with evidence)

- **VERIFIED:** Direct CSV submit → HTTP 400 `CreateSubmission`
  (re-tested 2026-08-31 with `kaggle competitions submit ... -f submission.csv`).
  This competition is **notebook-submission only**.
- **VERIFIED:** Code submit 400 earlier was the message
  "runtime of 233 minutes exceeds this competition's GPU max of 120 minutes"
  (read from the HTTP error body).
- **VERIFIED:** v9 error log showed a **Tesla P100 (sm_60)**, not the requested
  T4; preinstalled PyTorch supports sm_70+ only.
- **ASSUMED (needs confirming):** v10's runtime torch reinstall makes the P100
  work and finishes < 120 min. Confirm from the v10 run log before concluding.
