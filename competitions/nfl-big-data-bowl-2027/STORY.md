# NFL Big Data Bowl 2027 — Story

## 2026-10-29 — Entry + initial submission draft

Joined the competition (rules accepted by Sean in-browser; CLI download then
worked — 489 MB zip, ~2.3 GB extracted to the kaggle-data mirror).

### Picking the angle
The comp is a writeup hackathon: link Combine 10 Hz tracking to regular-season
NFL performance, one small idea done well. Profiled the data:
- 510 players across 2023-25 draft classes; combine_tracking has 9 drill types.
- WRs are the biggest clean group: 106 with route-drill tracking, 105 with
  in-game `separation_at_pass_forward`. Obvious candidate.

### First hypothesis (failed, honestly)
"Break Burst Score": sharper/harder cuts in route drills -> more in-game
separation. Built it, ran it: **r=0.006, dead null.** Resisted the urge to
patch. Ran a proper component-level diagnostic with within-drill normalization
and multiple outcomes instead of fishing.

### What the data actually said
- Separation is a **speed** story: 10-yd split r=-0.32, 40 r=-0.27, 3-cone
  r=-0.37. None of the break-mechanics features move separation.
- But **re-acceleration out of the break** tracks career usage: career
  offensive snaps r=+0.32..0.36, draft pick r=-0.25. Consistent and significant.

### The metric we shipped: Re-Acceleration Burst (RAB)
Mean (within-drill z) of reaccel-rate + speed-recovery out of the break.
- RAB ~ career offensive snaps: **r=+0.39, p=0.0001 (n=98)**
- RAB ~ draft overall pick: r=-0.25, p=0.036 (n=73)
- RAB ~ separation: r=+0.07 (n.s.) — the deliberate contrast
- 40-time ~ snaps: r=-0.10 (n.s.); **partial RAB~snaps | 40: r=+0.36, p=0.0006**
  -> RAB adds signal the stopwatch misses.

Headline for judges: *scouts time the sprint; the signal is in the
re-acceleration.* Four figures: break anatomy, the separation null, RAB vs
snaps, top/bottom receivers.

Writeup draft at `writeup/WRITEUP.md` (<2000 words, 4 figures). Not yet posted
to Kaggle — pending Sean's review.
