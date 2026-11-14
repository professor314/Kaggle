# NFL Big Data Bowl 2027

NFL analytics **hackathon** (writeup, not leaderboard). Link 10 Hz Combine
tracking → regular-season NFL performance. Deadline 2027-01-06.

## Submission: Re-Acceleration Burst (RAB) for wide receivers
How explosively a WR re-accelerates out of a route-drill break.
- RAB → career offensive snaps: **r=+0.39, p=0.0001** (n=98)
- **Survives draft-capital control**: snaps~draft+RAB R² 0.29→0.40 (ΔR²=+0.11, RAB p=0.0001);
  partial r=+0.39; 5-fold CV R² 0.21→0.34 out of sample
- Robust: bootstrap 95% CI [0.20,0.56], permutation p<1e-4, stable across 16 break-window settings
- Contrast: in-game separation is a *speed* story (10-yd split r=-0.32), not a cut-mechanics story
- RAB beats the 40 (which doesn't predict snaps) and survives controlling for it (partial r=+0.36)
- **Value board** (research-driven): RAB flags late-round hits as value — Puka Nacua (pick 177, 2211 snaps),
  Andrei Iosivas (206, 2006), Demario Douglas (210, 1448), Xavier Hutchinson (205, 1445)
- Lit-grounded: agility>speed for WRs (Harvard SAC 2021); turn-angle primitive (CMU arXiv:2507.06122)
- Honest null (limitation, not in submission): naive in-game break re-accel ~ Combine RAB r=+0.12 (n.s.)

## Files
- `writeup/WRITEUP.md` — the submission draft (1981 prose words, 7 figures fig1-7 + 1 table)
- `break_burst_score.py` — RAB pipeline + fig1-4
- `diagnose_components.py` — full component search behind the metric choice
- `robustness.py` — draft control, bootstrap, permutation, CV + fig5-6
- `enhancements.py` — turn-angle test, RAB+3cone composite, value board + fig7
- `sensitivity.py` — break-window parameter sweep
- `tracking_to_tracking.py` — exploratory in-game test (null; figA1, repo-only)
- `RESEARCH_IMPROVEMENTS.md` — researched improvement plan + sources
- `explore_schema.py`, `profile_drills.py`, `probe_wr_metric.py`, `probe_game_tracking.py` — EDA
- `STRATEGY.md`, `STORY.md`, `experiments_log.md`, `competition_config.yaml`
- `figures/`, `artifacts/` — outputs
- Data (gitignored): `C:\Users\profe\kaggle-data\competitions\nfl-big-data-bowl-2027\data`

## Reproduce
```powershell
$py = "C:\Users\profe\My Drive\Programming\kaggle\.venv-dl\Scripts\python.exe"
& $py break_burst_score.py      # metric, stats, figures
& $py diagnose_components.py     # component-level correlation search
```

## Status
Initial submission drafted; NOT yet posted to Kaggle (pending review). To submit:
post the writeup as a Kaggle Writeup on the competition page, attaching the four
figures and linking a notebook version of `break_burst_score.py`.
