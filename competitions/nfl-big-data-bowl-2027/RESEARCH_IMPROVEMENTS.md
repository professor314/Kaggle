# Research-Driven Improvement Plan (2026-10-29)

Sources reviewed: BDB 2022 winners (RStudio), CMU turn-angle paper
(arXiv:2507.06122), Harvard Sports Analysis Collective 2021 agility study,
multiple combine-vs-WR-success analyses (Pro Football Network, JSCR 2016,
PubMed 30358695).

## What the research says
1. **Agility > speed for WR success.** Harvard SAC 2021: WRs with top-20% agility
   scores were ~2x as likely to post a 1,000-yd season within 3 yrs. Multiple
   analyses find the 40 is a weak predictor of WR value; 3-cone/shuttle/vertical
   carry more. => Our RAB (tracking-based agility/burst) is a refinement of a
   thesis the literature already supports. CITE THIS to frame the metric.
2. **Turn angle is a validated public movement primitive** (CMU). Angle between
   successive displacement vectors; model its *variability*. Faster players turn
   less sharply (speed up, concentration up). Acceleration <-> more variable
   turning (cuts). => add a turn-angle-based "cut sharpness" feature to the break
   and the metric, grounded in published method.
3. **Winners name players.** CMU validates by eye test (Pickens shifty, Metcalf
   straight-line); BDB 2022 winners named Diontae Spencer as undervalued. =>
   produce a ranked prospect board AND call out specific over/under-valued WRs.
4. **Winners present visually and make a tool.** Clean leaderboard, intuitive
   figures, ideally an interactive/animated artifact.

## Improvements executed (results)
- [x] A. Turn-angle-at-break (CMU-style) feature: tested; r=+0.18-0.20 vs snaps
      (p~0.06), weaker than re-acceleration -> kept as a secondary/reported
      feature, NOT folded into RAB (keeps metric clean). In enhancements.py.
- [x] B. Value board (fig7): RAB flags Puka Nacua (pick 177, 2211 snaps),
      Iosivas (206/2006), Douglas (210/1448), Hutchinson (205/1445) as value;
      Jermaine Burton / Jack Bech as burst-light early picks. Named-player
      validation now in writeup.
- [x] C. Literature framing added to intro (Harvard SAC agility>speed; CMU
      turn-angle; BDB22 winners) + references block in appendix.
- [x] D. Composite RAB+3-cone -> snaps R2=0.28 vs 3cone 0.19 / RAB 0.06 (n=29);
      RAB~3cone r=0.11 => complementary. Noted in experiments_log.

## Deferred (bigger than one session)
- Coverage-aware in-game break detector (to rescue the tracking-to-tracking null).
- Extend to DBs (mirror drills vs coverage separation allowed).
- Interactive Shiny/Plotly app + route animations.
