---
inclusion: always
---

# 00 - Kaggle Operating Workflow (read first, every Kaggle session)

Sean's Kaggle work is a **learning project built on our own platform**, not a pile of one-off
scripts. Every competition feeds the toolkit, the lessons file, git, and the blog. Nothing runs
in a vacuum. The phase docs (01-research ... 08-content-creation) say HOW to do each phase.
This doc says what must happen every session.

## Before starting work
1. Read the phase steering doc for the phase you are in (01-08 in this folder).
2. Check the toolkit before writing code: `kaggle_ml_toolkit/` (see the module index the hook
   prints). Use the module if it fits; extend it if it almost fits; only write standalone code
   for problems the toolkit can't express (agents, packing, circuits). Write down why.
3. Read `docs/LESSONS.md` (cross-competition gotchas) and the competition's
   `experiments_log.md` + `research/design_doc.md`. Re-verify inherited facts (07-submission).
4. Research docs come from `ResearchDocumentGenerator` (01-research), not hand-written.

## While working
- **Heavily commented code, written to teach.** Module docstring: problem, metric, approach,
  results. Comments explain *why* (the ML idea, the trade-off, the gotcha), not just what.
  A reader learning ML should be able to follow every non-trivial line.
- **Invented something reusable? Put it in the toolkit.** E.g. a Kaggle input-path resolver, a
  kernel-metadata writer, a submission packager. Add it to `kaggle_ml_toolkit/`, export it in
  `__init__.py`, add a unit test under `tests/unit/`, run `pytest`, then use the toolkit version
  from the competition code. Don't leave copies pasted into each script.
- **Log as you go** in `experiments_log.md`: approach, CV, LB, VERIFIED/ASSUMED tags, next levers.
- **Cross-pollinate.** When a lesson applies beyond one competition, add it to `docs/LESSONS.md`.
  When starting a competition, look for a similar past one and reuse its approach.
- Code that runs on Kaggle goes in the repo too: `competitions/<c>/notebooks/` or
  `kernels_published/<slug>/`. Never leave it only in a temp folder.

## Before ending a session (the hook's stop check nags if these are missing)
1. **Git:** commit with a descriptive message and `git push` (repo: github.com/professor314/Kaggle,
   PUBLIC). Never commit competition data (`competitions/*/data/` is ignored; check `git status`).
2. **Blog:** write or extend today's draft `docs/blog/YYYY-MM-DD-<topic>.md` with `draft: true`:
   what we tried, what worked, what failed and why, scores, what we learned. Follow
   08-content-creation and the imadestuff.com style (no em or en dashes). Do NOT publish to the
   site without Sean's approval.
3. **Toolkit:** anything new and reusable is in the package with a passing test.
4. Update `NEXT.md` (what's next) and `USER_TODO.md` (anything waiting on Sean).
