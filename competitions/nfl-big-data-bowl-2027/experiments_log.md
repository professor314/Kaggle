# Experiments Log — nfl-big-data-bowl-2027

| date | experiment | result | notes |
|------|-----------|--------|-------|
| 2026-10-29 | BBS v1: break "sharpness" -> separation | r=0.006 (null) | absolute speed deltas; no within-drill norm. Abandoned. |
| 2026-10-29 | Component diagnostic (within-drill z, 8 features x 5 outcomes) | reaccel/recovery -> snaps sig | separation = speed only; break mechanics n.s. for separation |
| 2026-10-29 | RAB = z(reaccel_rate)+z(recovery_pct) -> career snaps | **r=+0.39, p=1e-4** | headline metric |
| 2026-10-29 | RAB -> draft pick | r=-0.25, p=0.036 | higher RAB, earlier pick |
| 2026-10-29 | RAB -> separation | r=+0.07, n.s. | deliberate contrast in writeup |
| 2026-10-29 | 40-time -> snaps | r=-0.10, n.s. | stopwatch doesn't predict usage |
| 2026-10-29 | partial RAB~snaps \| 40 | r=+0.36, p=6e-4 | RAB adds signal beyond stopwatch |
| 2026-10-29 | **draft control**: snaps~draft+RAB (OLS) | R2 0.29->0.40, ΔR2=+0.11, RAB β=+599 p=1e-4 | RAB is NOT a draft proxy |
| 2026-10-29 | partial RAB~snaps \| draft+undrafted | r=+0.39, p=1e-4 | key robustness result |
| 2026-10-29 | bootstrap RAB~snaps (5000) | r=0.39, 95% CI [0.20,0.56] | excludes 0 |
| 2026-10-29 | permutation test RAB~snaps | p<1e-4 | |
| 2026-10-29 | 5-fold CV snaps: draft vs draft+RAB | R2 0.21 -> 0.34 (+0.13 OOS) | generalizes |
| 2026-10-29 | break-window sensitivity (16 settings) | r 0.25-0.39, all p<0.05 | not an artifact |
| 2026-10-29 | tracking-to-tracking: Combine RAB ~ in-game RAB | r=+0.12, p=0.23 (NULL) | naive in-game break detector; reported as limitation, NOT in submission |
| 2026-10-29 | turn-angle-at-break (CMU-style) ~ snaps | r=+0.18-0.20, p~0.06 | tested; weaker than reaccel, kept secondary |
| 2026-10-29 | RAB + 3-cone composite -> snaps | R2 0.28 vs 3cone 0.19 / RAB 0.06 (n=29) | complementary (RAB~3cone r=0.11) |
| 2026-10-29 | **value board**: RAB pctile vs draft slot | Puka Nacua (177,2211 snaps), Iosivas, Douglas, Hutchinson flagged as value | named-player validation, fig7 |

## Reproduce
```
P=.venv-dl/Scripts/python
$P competitions/nfl-big-data-bowl-2027/break_burst_score.py   # RAB + fig1-4
$P competitions/nfl-big-data-bowl-2027/diagnose_components.py  # component search
$P competitions/nfl-big-data-bowl-2027/robustness.py           # draft control, bootstrap, CV, fig5-6
$P competitions/nfl-big-data-bowl-2027/sensitivity.py          # break-window sweep
$P competitions/nfl-big-data-bowl-2027/tracking_to_tracking.py # exploratory in-game (null)
```
Data: `C:\Users\profe\kaggle-data\competitions\nfl-big-data-bowl-2027\data`.
Outputs: `artifacts/` (tables), `figures/` (fig1-4).
