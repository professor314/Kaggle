# Implementation Plan — Kaggriculture agent

Grounded in `data/README.md` + `data/AGENTS.md` (game rules) and our results so far.
Goal: climb the leaderboard from the bottom 8% by fixing the core problem —
**we overfit to the weak built-in `starter`.** Each tier is a shippable milestone;
submit + record the LB after each. All local iteration runs in parallel across
CPU cores (the agent itself is single-threaded on Kaggle — no GPU/threads).

## Where we are (VERIFIED 2026-09-05)
- **Herd engine submitted** (pending LB). Beats prior farm-operator champion 12/0/0
  (avg **$41,168 vs $6,080**, ~7× money) and `starter` 8/0/0 ($37.8k vs $3.6k),
  verified over 12 games alternating seats.
- This is the animal-herd strategy the top-player replays revealed (net $63k-$104k).
  The breakthrough was two hidden logistics mechanics (see STORY.md ch.13):
  bought animals must be PICKUP'd from the shed and carried to a structure to PLACE;
  FEED consumes wheat from the *feeding unit's* inventory, so units must carry wheat.
- Prior milestones: farm-operator LB 251.8, GA genome 238.9, loader-fix 83.2.
- Remaining gap to the top (~$80k): sheep/wool, land for a bigger herd, CARE +
  fertilizer cadence (fertilizer was the top players' #1 income stream).

## Tier 0 — Diagnose the gap (cheap, unblocks everything) — DONE
- [x] 0.1 Replay analysis: `evo/inspect_replay.py` + `evo/analyze_top.py` on 5
      top-player replays; `evo/trace_herd.py` for per-day money/herd/shed tracing.
- [x] 0.2 Leaks found: the whole prior line of agents never PICKUP'd animals from
      the shed and never carried wheat to FEED — so animals starved and money bled.
      Recorded in STORY.md ch.12-13.
- [x] 0.3 Scoring: live skill rating that moves as the agent plays the field.

## Tier H — Herd engine (the winning strategy) 
- [x] H.1 Reverse-engineer the top-player strategy signature (`evo/analyze_top.py`):
      ~15 animals, ~290 hires/game, heavy CARE + fertilizer, near-broke→explode.
- [x] H.2 Build `evo/herd_engine.py` with the PICKUP+carry+PLACE animal flow and
      the wheat-carry FEED loop; grow herd 1-at-a-time only while fully fed.
- [x] H.3 Verify: 12/0/0 vs farm-operator (~$41k vs $6k), 8/0/0 vs starter.
      **CHECKPOINT: submitted 2026-09-05** (LB pending).
- [ ] H.4 Add SHEEP (wool $200) alongside cows; tune the cow/sheep/goose mix.
- [ ] H.5 CARE + COLLECT_FERTILIZER cadence discipline (fertilizer = top #1 income).
- [ ] H.6 BUY_LAND for a bigger herd once the home quadrant is full + flush.
- [ ] H.7 Re-verify vs farm-operator + herd_engine-v1; **submit** if it beats them.

## Tier 1 — Fix the training signal: competitive co-evolution (biggest lever)
Design: population plays EACH OTHER (sampled round-robin) as the main fitness
signal, PLUS a small stable anchor set so fitness stays comparable across gens.
- [x] 1.1 Opponent anchor set (`evo/opponents.py`): `starter` + two sharp
      hand-written presets (aggressive-expand, market-timer). DONE.
- [ ] 1.2 Co-evolution GA (`evo/coevo.py`): each candidate plays K sampled PEERS
      (both seats) + the anchors + a Hall of Fame of past champions. Fitness = avg
      money across all matches. HoF grows with the best genome each generation
      (self-play), so the target chases itself upward. Cost = K·pop, not pop².
- [ ] 1.3 Run co-evolution seeded from `evo/state/best.json`; verify the winner
      beats `starter`, the sharp presets, AND our previous champion head-to-head.
      **CHECKPOINT: submit + log LB + rank.**

## Tier 2 — Deepen the strategy space (more genes / better actions)
- [ ] 2.1 Expand the genome: land-expansion timing, farm-hand hiring (more
      actions/turn = more throughput), per-crop planting mix, animal economics
      (feed wheat, CARE bonus, coop/pasture ramp), fertilizer usage windows.
- [ ] 2.2 Better market logic: sell into scarcity/town demand, respect the price
      curves (hold premium goods off gluts), spread sells across turns. Gene-gated.
- [ ] 2.3 Better unit routing: reduce idle farmer/hand turns (the replay from 0.1
      will show how many turns are wasted moving vs acting).
- [ ] 2.4 Re-run GA vs the pool with the deeper genome. **CHECKPOINT: submit + log LB.**

## Tier 3 — Scale the search
- [ ] 3.1 Larger/longer GA run (bigger pop, more gens, more games/eval for less
      noise) once the opponent pool + genome are solid. Record RUN_STATS.
- [ ] 3.2 Optional: lightweight lookahead / rollout at decision time IF it fits the
      per-turn time budget (measure turn time first; agent must stay fast on Kaggle).
- [ ] 3.3 **CHECKPOINT: submit the best; log LB + rank.**

## Cross-cutting
- [ ] X.1 Keep `evo/RUN_STATS.md` + `evo/state/run_stats.jsonl` current (auto-recorded).
- [ ] X.2 Keep `STORY.md` iteration table + LB rows current at every milestone;
      the draft blog post is drawn from it (stays `draft: true`).
- [ ] X.3 Commit + push after each tier (exclude the large lb/ and *.zip artifacts).

## Submission checkpoints (submit + record rank each time)
1. **1.3** — GA vs opponent pool + Hall of Fame (fix the overfit).
2. **2.4** — deeper strategy genome.
3. **3.3** — scaled search / optional lookahead.

## Notes / guardrails
- The agent runs single-threaded on Kaggle with a per-turn time cap — keep it fast,
  no GPU, no threads. Put all compute into local parallel self-play + evolution.
- `agent()` MUST be the last top-level callable in the submitted file (loader picks
  the last callable). This already bit us once (ERROR → fixed).
- Measure every change over MANY games (both seats) before trusting it; one game is noise.
- The market punishes dumping — timing/restraint is real strategy.
