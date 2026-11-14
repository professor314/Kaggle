# The Re-Acceleration Burst: What the Combine Clock Misses About Receivers

**NFL Big Data Bowl 2027 — Open Track**

## Motivation

Every receiver prospect in Indianapolis gets reduced to a handful of numbers: a
40-yard dash, a 3-cone, a short shuttle. Those times measure how fast an athlete
covers ground in a straight line or through a fixed cone pattern. But the moment
that actually wins a route is the **break** — the instant a receiver sheds speed,
changes direction, and re-accelerates to pull away from a defender. A stopwatch
cannot see inside that moment. The 10 Hz Combine tracking data can.

This analysis asks a narrow, testable question for wide receivers: **do the
movement mechanics of the break, measured during Combine route drills, tell us
anything about NFL performance that the traditional times do not?** The answer
turns out to be a useful two-part story — one negative, one positive — and it
produces a single new scouting metric we call the **Re-Acceleration Burst (RAB)**.

This builds on a known but blunt result in the public literature: for receivers,
*agility* tends to matter more than straight-line speed. The Harvard Sports
Analysis Collective found WRs in the top 20% of agility scores were roughly twice
as likely to post a 1,000-yard season within three years, and the 40-yard dash is
a weak predictor of receiver value. Those studies lean on stopwatch agility drills
(3-cone, shuttle). Our contribution goes inside the movement itself with tracking
data — isolating the one phase (the break) and the one quality (re-acceleration)
that carry the signal.

## Data and population

We use `combine_tracking.csv` (10 Hz x/y/speed/acceleration/direction during
Combine drills), `player_play.csv` (per-play, tracking-derived outcomes including
`separation_at_pass_forward`), `player_career_successes.csv` (career snaps,
games), `combine_results.csv` (traditional measurements), and `players.csv`.

We restrict to wide receivers who ran the position route drills and later played
NFL snaps: **105 WRs from the 2023-2025 draft classes**, each with both Combine
route-drill tracking and in-game separation (37,525 route-level separation
measurements). We analyze the 12 route drills with a genuine change-of-direction
break (comeback, curl, speed-out, slant, dagger, in, hook, post-corner, slot
sail/strike), excluding pure straight-line and contested-catch drills.

## Measuring the break

Within each drill attempt (a 10 Hz trajectory), we locate the **break** as the
central speed trough — the slowest frame in the middle 70% of the attempt, which
cleanly isolates the cut from the starting stance and the coast-down at the end.
Around that frame we compute four movement features:

- **deceleration in** — how fast the receiver brakes approaching the break;
- **speed kept** — minimum speed at the break relative to entry speed (a sharper
  cut sheds more speed);
- **re-acceleration out** — how fast speed is regained leaving the break;
- **speed recovery** — peak speed out of the break relative to peak speed into it.

*Figure 1* shows one comeback-route attempt: speed falls from roughly 8 to under
1 unit into the break, then climbs back out. The shaded regions mark the decel-in
and re-accel-out windows the metric reads.

Because drills differ in geometry and length, we **z-score every feature within
drill** before averaging to a per-player value. Each receiver is therefore graded
against peers who ran the *same* drill, which removes drill-to-drill scale and
lets us pool across routes.

## Finding 1: separation is a speed story, not a cut-mechanics story

Our first hypothesis was the intuitive one: receivers who cut harder and sharper
in drills should create more separation in games. **They do not.** Across 98
receivers with at least 20 in-game routes, no break-mechanics feature shows any
meaningful relationship with mean in-game separation (all |r| < 0.17, all
non-significant).

What *does* explain separation is plain speed. Mean in-game separation correlates
with the **10-yard split (r = -0.32, p = 0.002)**, the **40-yard dash
(r = -0.27, p = 0.010)**, and the **3-cone (r = -0.37)** — faster, more agile
receivers get open by more yards. *Figure 2* contrasts the flat Re-Acceleration
Burst-vs-separation cloud (left) with the clear downward speed-vs-separation
trend (right).

This is worth stating plainly because it is counterintuitive and corrects a
natural assumption: **the drill that looks most like route running (cutting
through routes) does not predict the route-running outcome everyone cares about
(separation). Straight-line speed does.** Separation, at least as captured by
`separation_at_pass_forward`, is largely a function of how fast you are, not how
prettily you cut in shorts.

## Finding 2: re-acceleration predicts who actually plays

If break mechanics don't buy separation, do they buy anything? Yes — but against
a different outcome. When we test the break features against **career
involvement**, the re-acceleration components separate from the pack. How
explosively a receiver regains speed out of the break predicts **career offensive
snaps** (re-acceleration rate r = +0.32, p = 0.001; speed recovery r = +0.36,
p = 0.0004) and **draft capital** (earlier pick; r = -0.25, p = 0.03).

We combine the two re-acceleration components into one standardized score:

> **Re-Acceleration Burst (RAB)** = mean of the within-drill z-scored
> re-acceleration rate and speed recovery out of the break.

RAB relates to NFL usage as follows (n = 98 unless noted):

| Relationship | Pearson r | p | n |
|---|---|---|---|
| RAB → career offensive snaps | **+0.39** | **0.0001** | 98 |
| RAB → draft overall pick (earlier = better) | -0.25 | 0.036 | 73 |
| RAB → mean in-game separation | +0.07 | 0.52 (n.s.) | 98 |
| 40-yard dash → career snaps | -0.10 | 0.34 (n.s.) | 89 |

*Figure 3* plots RAB against career offensive snaps: the receivers who
re-accelerate hardest out of their breaks are the ones who end up on the field.

The decisive comparison is the last two rows. **The 40-yard dash does not
predict career snaps** — the stopwatch tells you who is fast, not who plays. RAB
does. And RAB is not merely re-measuring speed: a **partial correlation of RAB
with career snaps, controlling for the 40-yard dash, is r = +0.36 (p = 0.0006)**.
The re-acceleration signal survives almost untouched after removing straight-line
speed. It is information the clock does not contain.

### Is RAB just a proxy for draft position?

The obvious objection: higher-RAB receivers were drafted earlier, and early picks
play more regardless, so maybe RAB only re-states draft capital. It does not. In a
regression of career snaps on draft position (and an undrafted flag), adding RAB
raises explained variance from **R² = 0.29 to R² = 0.40** — a **+0.11** gain — and
RAB's own coefficient is large and significant (**+599 snaps per unit RAB,
p = 0.0001**). The partial correlation of RAB with snaps, holding draft capital
fixed, is **r = +0.39 (p = 0.0001)** (*Figure 5*, which plots RAB against snaps
after both are residualized on draft capital). Out of sample, a 5-fold
cross-validation tells the same story: draft position alone predicts snaps at
R² = 0.21, and adding RAB lifts that to **R² = 0.34** — a genuine out-of-sample
gain, not in-sample overfitting. RAB carries information that draft rooms, on
average, did not fully price in.

The finding is also stable. The RAB-snaps correlation holds across every
combination of break-window parameters we tried (16 settings; r from 0.25 to 0.39,
all p < 0.05), its bootstrap 95% confidence interval is **[0.20, 0.56]** (5,000
resamples, excludes zero; *Figure 6*), and a label-permutation test returns
p < 0.0001. The
result does not hinge on one arbitrary choice.

## Reconciling the two findings

Why would re-acceleration predict playing time but not separation? Because the two
outcomes reward different things. **Separation** is a geometry-and-speed outcome
on a single snapshot — the fastest athletes simply run away from coverage.
**Career snaps** reflect a coaching staff's cumulative judgment that a receiver
can be trusted in a route tree: come out of breaks ready to work back to the ball,
adjust, and threaten after the catch. Re-acceleration out of the break is a
movement signature of exactly that quality — the ability to re-enter top-end
effort immediately after decelerating — and it shows up in who earns snaps even
when it doesn't widen the separation window on average.

*Figure 4* lists the highest- and lowest-RAB receivers in the 2023-2025 classes.
The top of the board is populated by early-round producers (Jordan Addison, Rome
Odunze, Quentin Johnston — all first-round picks with 1,600+ career snaps),
providing face validity: the metric is surfacing receivers the league already
valued, from drill movement alone.

## The value board: where RAB and the draft disagree

The most useful test of a scouting metric is what it flags that the draft did
not. *Figure 7* plots every receiver's RAB against their draft slot, colored by
career snaps. The interesting receivers are off the diagonal: high RAB taken late
(potential value), or early picks with low RAB (burst-light bets).

The late-round, high-RAB group is a murderers' row of recent hits. **Puka Nacua**
— pick 177, now at 2,211 career snaps and one of the most productive late-round
receivers in the league — sits well above where he was drafted on the RAB board.
He is joined by Andrei Iosivas (pick 206, 2,006 snaps), Demario Douglas (210,
1,448), and Xavier Hutchinson (205, 1,445). RAB, computed only from Combine drill
movement, would have flagged all four as better movers than their draft slots
implied — and the field has since borne that out. Conversely, several early picks
with below-median RAB (e.g. Jermaine Burton, pick 80, 131 snaps; Jack Bech) have
been slower to earn snaps. This is exactly the shape of signal a front office
wants: a cheap, tracking-derived flag that points at receivers the room is
systematically under- or over-rating.

## Why this is actionable

For an evaluator, RAB is a cheap, concrete addition to the Combine report. It is
computed entirely from tracking a prospect already generates, requires no new
testing, and is **orthogonal to the 40** — it tells you something the headline
time does not. The clearest use is tie-breaking similar-timed receivers: when two
prospects post the same 40, the one with higher RAB has historically been the one
who accrues snaps, and the value board (*Figure 7*) turns that into a concrete
watch-list of under- and over-rated movers.

Equally important is the negative result: teams should be wary of inferring
*separation ability* from how clean a receiver's cuts look in drills. In this
sample, that inference has no support — separation tracks speed, full stop.

## Limitations

This is an honest, early-career snapshot. The sample is modest (98-105 WRs) and
spans only the 2023-2025 classes, so career snaps are a noisy, opportunity-laden
proxy that partly reflects draft capital — though, as shown above, RAB predicts
snaps even after draft capital is controlled. The separation measure is itself
tracking-derived. The analysis is observational: RAB is a scouting signal, not a
causal lever.

We also ran the strongest linkage test we could — a tracking-to-tracking check,
measuring re-acceleration out of breaks directly in regular-season game tracking
(26,000+ breaking-route plays) and correlating it with Combine RAB. That link was
weak and non-significant (r = +0.12). We report this candidly: a naive
speed-trough break detector that works on clean, isolated Combine drills is a poor
instrument in live games, where press releases, varied route depths, and
quarterback timing scramble the signal. A coverage-aware in-game break detector,
and extending the method to defensive backs' mirror drills, are the clear next
steps.

## Conclusion

Measuring *how* receivers move through the break, rather than *how fast* they run
in a straight line, yields a metric the Combine clock misses. Separation is a
speed story; playing time is a re-acceleration story. The Re-Acceleration Burst
captures the latter from drill tracking alone, predicts career snaps better than
the 40-yard dash, and does so with information the stopwatch does not hold.
**Scouts time the sprint — but the signal is in the re-acceleration.**

---

*Appendix: code to reproduce every number and figure is in the linked notebook.
`break_burst_score.py` builds the RAB pipeline and Figures 1-4;
`diagnose_components.py` runs the full component search behind the metric choice;
`robustness.py` produces the draft-capital regression, bootstrap, permutation and
cross-validation results plus Figures 5-6; `enhancements.py` builds the value
board (Figure 7) and the turn-angle and composite tests; `sensitivity.py` sweeps
the break-window parameters; `tracking_to_tracking.py` runs the exploratory
in-game check. Analysis uses only the competition's public data, joined on
`nfl_id`.*

*Selected references: Harvard Sports Analysis Collective (2021), agility and
receiver success; Nguyen & Yurko (2025), "A Bayesian circular mixed-effects model
for directional movement in American football," arXiv:2507.06122 (turn-angle
movement primitive); NFL Big Data Bowl 2022 winning methodology (Ritchie,
Kumagai, Moreau, Cavan).*
