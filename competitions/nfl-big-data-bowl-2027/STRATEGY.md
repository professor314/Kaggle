# NFL Big Data Bowl 2027 — Strategy

## Competition type
Featured **Hackathon** (Analytics writeup), sponsored by the NFL.
- **No leaderboard metric.** Judged by NFL team analytics staff against a rubric.
- **Deliverable:** a writeup, <= 2000 words, < 10 tables/figures. Code in an
  appendix / linked notebook.
- **PASS/FAIL gate:** the writeup MUST explicitly link Combine *tracking* data to
  regular-season NFL game performance. Writeups that don't use tracking data are
  not scored.
- **Rubric weights:** Combine-to-NFL linkage (PASS/FAIL), insight quality,
  data visualization (20%), writing clarity.
- **Deadline:** final submission 2027-01-06. Lots of runway.
- **Tracks:** Open ($45k) and University ($45k, undergrad only). We target Open.

## Winning shape (from past Big Data Bowls)
Winners pick ONE small, legible idea — a single position group / drill — define a
clean new **metric**, validate it against on-field outcomes, and present it with
a few excellent visuals and tight prose. Judges are practitioners; they reward
actionable, non-obvious insight over model complexity.

## Our angle: receiver break mechanics -> NFL outcomes
Population: 105 WRs (2023-2025 draft classes) with BOTH Combine route-drill
tracking AND in-game tracking-derived separation.

We extract movement features at the **break** (the speed trough) of each route
drill: deceleration in, re-acceleration out, speed kept through the cut, speed
recovery. We then test them against in-game separation, target rate, draft
capital, and career snaps.

### Two validated findings (the submission)
1. **Null (and useful):** In-game separation is explained by straight-line speed
   (10-yd split r=-0.32, 40 r=-0.27) and short-area agility (3-cone r=-0.37),
   NOT by any break-mechanics feature (all n.s.). The drill that *looks* most
   like route running doesn't transfer to separation.
2. **Signal:** The **Re-Acceleration Burst (RAB)** — how explosively a WR
   regains speed out of the break — predicts career offensive snaps
   (r=+0.39, p=0.0001) and draft position (r=-0.25, p=0.036). The 40-time does
   NOT predict snaps; RAB survives controlling for the 40 (partial r=+0.36,
   p=0.0006). RAB adds signal the stopwatch misses.

Headline: *"Scouts time the sprint; the tape is in the re-acceleration."*

## Method notes
- Break = central speed trough (trim first/last 15% of each attempt).
- Within-drill z-scoring cancels drill geometry/scale before aggregating per
  player, so RAB compares a receiver to peers on the same drill.
- Join key is `nfl_id` throughout.

## Risks / caveats to disclose in the writeup
- Small n (98-105); 2023-25 classes, careers still young -> snaps are a noisy
  proxy and partly reflect draft capital/opportunity.
- Separation metric is itself tracking-derived (`separation_at_pass_forward`).
- Observational; RAB is a scouting signal, not a causal claim.

## Next iterations (post initial submission)
- Repeat RAB for DBs (mirror drills) vs coverage separation allowed.
- Add game_tracking_* to measure in-game re-accel out of breaks and compare to
  the Combine RAB (true tracking-to-tracking linkage).
- Control for draft capital explicitly (partial corr / simple regression).
