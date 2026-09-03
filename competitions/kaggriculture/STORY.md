# Kaggriculture — Iteration Story Log

A running, plain-language record of how this game agent is being built and
improved, iteration by iteration. This is the source of truth for the story; the
draft blog post is drawn from it. Add a dated entry (and an iteration-table row)
whenever we change the agent or learn something.

---

## The task, in one paragraph
Kaggriculture is a two-player farming sim on Kaggle ($50K, closes 2026-09-30).
Over a 30-day season (720 turns), you buy seeds/animals, plant, water, harvest,
raise livestock, and trade on a **dynamic market** where prices fall as you sell
(premium goods crash to a $1 floor on gluts). You submit an **agent** — a Python
function that gets the game state each turn and returns actions. Most money in the
bank at the end wins. Scored by playing your agent against others.

## How we work on it
- **It's an agent competition, not a dataset one.** There's no data to download.
  The "data" is *self-play games* we generate locally.
- **The agent runs single-threaded on Kaggle's hardware** with a per-turn time
  budget. So the agent code itself should stay fast and simple — no GPU, no
  threads (those would be wasted or cause timeouts).
- **Our 24 cores go into the iteration loop:** running many independent games in
  parallel to measure a change, and to power a genetic algorithm that evolves the
  strategy. That's where the compute helps.

---

## Iteration table (honest numbers)
Fitness = avg final money vs the built-in `starter` + a 1000×winrate bonus.
Money is the real objective (we start each game with $3,000).

| # | Date | What changed | vs starter (money / winrate) | LB / status |
|---|---|---|---|---|
| 0 | 2026-08-25 | First greedy agent submitted | — | **ERROR** (crashed) |
| 1 | 2026-09-02 | Fix: make `agent()` the last top-level callable | — | COMPLETE (LB rating ~94–133, moves as it plays) |
| 2 | 2026-09-02 | Local benchmark of the greedy agent | ~1,168 / **0%** (loses money!) | measured |
| 3 | 2026-09-02 | Parameterized "genome" agent (animals, melon, throttled selling) | ~2,300 / 0% | measured |
| 4 | 2026-09-02 | GA run 1 (23 gens, 6,624 games) — best genome baked + submitted | ~3,700–3,994 / 88–100% vs starter | **LB 243.7** ✅ |
| 5 | 2026-09-03 | Co-evolution (peers+anchors+HoF, ~24k games) | ties incumbent (6/12) | not submitted (no real gain) |
| 6 | 2026-09-03 | Ratchet (champion-challenger) | gate too loose → ties | not submitted |
| 7 | 2026-09-03 | **Farm-operator** (units route to nearest work; keep farm full+watered) | **16/16 vs champ, ~$5,524 vs $3,549** | **submitted** ✅ |

**Note on LB scores:** Kaggriculture uses a live skill rating that changes as your
agent keeps playing the field. The greedy fix and the GA agent were rated ~94 and
**243.7** respectively — the evolved agent is clearly rated well above the greedy one.

## Chapter 5 — The reality check: we overfit to a weak opponent
The GA agent's 243.7 put us at **rank ~6,940 / 7,491** (median 761.9, top 2,988.6).
The lesson: we trained the GA to beat the built-in `starter`, maxed out at ~100%
winrate, but `starter` is a weak sparring partner. Beating it perfectly only banks
~$3,994, while the leaderboard rates you against real submitted agents that net far
more. Running the same GA longer wouldn't help — it was optimizing the wrong exam.

## Chapter 6 — Competitive co-evolution (let them play each other)
The fix separates two ideas that had been collapsed together:
- The **genetic pool** = the agents we're evolving (the students).
- The **opponent pool** = who they play to earn fitness (the exam).

Previously the exam was one weak agent. Now fitness comes from **competitive
co-evolution**: each candidate plays a sample of its **peers** (both seats), plus a
small **anchor set** (`starter` + two sharp hand-written presets) for a stable
yardstick, plus a **Hall of Fame** of our own past champions that grows each
generation (self-play). The opposition scales with the population, so the target
keeps getting harder instead of sitting at `starter`'s level — while the anchors +
HoF keep fitness comparable across generations and prevent cycling/drift. Cost is
K·pop games per generation, not pop². (`evo/coevo.py`, `evo/opponents.py`.)

### Co-evolution run (gens 0–34, ~24k games) — the verification that saved a bad submit
Ran co-evolution to gen 34 (Hall of Fame grew to 19). Best fitness held ~3,750–3,885
(avg money vs the tough pool), mean fitness climbed and held ~3,450–3,565 — a healthy,
non-collapsing population. BUT a head-to-head check before submitting found the
co-evolved champion **loses 1/12 to our already-submitted GA champion** and only ties
`starter` (while beating both sharp anchors 100%). Lesson: **optimizing average money
across a pool rewards robustness, not beating a specific strong opponent.** We did NOT
submit it (it would have lowered our rank). FIX for next session: add the submitted GA
champion to the anchor set so evolution is explicitly forced to beat the incumbent;
only submit a candidate that provably wins head-to-head vs the current champion. State
is checkpointed (`evo/state_coevo/`), resume with `--resume-state`.

## Chapter 7 — Champion-challenger ratchet (start from best, only accept real gains)
Built `evo/ratchet.py`: seed the population from the current champion + mutations,
score candidates mainly on win rate vs the champion, and only PROMOTE a challenger
if it passes a head-to-head gate. A modest run promoted 7× over 12 gens — but an
independent, larger head-to-head found the final champion only TIES the incumbent
(6/12) and loses to `starter` (4/10). Lesson: **the 20-game promotion gate was too
loose for this high-variance game — 55–85% over 20 games was mostly noise.** Did NOT
submit. FIX: tighten the gate (more games + higher win-rate bar, e.g. 40 games @ 60%)
so a promotion reflects a statistically real edge, and always confirm with a separate
verification run before submitting.

## Chapter 8 — Economic theory + the real binding constraint (action budget)
Worked the game's economics: melon is highest gross ($250, but crashes to $1 on
gluts), ANIMALS are the perpetual engine (one-time cost -> milk/wool/eggs forever),
wheat is infrastructure (cash + animal feed), the market is an exhaustible commons
(meter premium sales, sell into town demand), and actions/turn is the constraint
(hire hands + buy land to scale). Built three theory-driven agents
(`evo/strategy_agents.py`): animal_engine, melon_baron, market_maker.

**Result: all three lost 0/16 to BOTH the champion and `starter`, banking only
$613–$1,500 (animal_engine ended BELOW its $3,000 start).** The diagnosis is the
real insight: **the binding constraint is the ACTION BUDGET, not the strategy.**
Our timid champion wins because it only does things it can COMPLETE (plant→water→
harvest→sell wheat). The aggressive agents acquired more tiles/animals than one
farmer can maintain, so animals starved (a starved animal = total loss of a $400+
investment) and crops weeded from missed watering. Correct theory-driven agent must:
feed/water everything it owns BEFORE acquiring more, hire hands BEFORE expanding,
and never let an owned animal miss a feed. (Fixing this is the next step, not
seeding the ratchet with the broken versions.)

## Chapter 9 — Trace the game: the real problem is EXECUTION, not economics
`disciplined_engine` (maintenance-first) improved $613 -> $2,310 but still lost 0/16
and ended below the $3,000 start. So we TRACED an actual game (`evo/trace_game.py`),
which overturned the theory:
- The champion WINS with an almost-empty farm: final tiles = 20 empty, 4 weeds,
  **1 carrot**. It spends $220 on day 1, sits flat at ~$2,780 until day 16, then
  drifts up to $3,817. It barely farms — it mostly holds cash and avoids losses.
- disciplined_engine over-spends to $1,920 on day 1 (land/seeds/feed it never
  converts) and never recovers; final tiles = 43 empty, 6 weeds, 1 wheat.
- **Neither agent actually farms.** Tiles sit EMPTY and rot into WEEDS because unit
  routing is weak (units wander instead of planting/watering/harvesting).

Conclusion (data, not theory): the binding problem right now is **operational
competence** — an agent that reliably keeps every tile planted + watered and
harvests on time should crush everything we have (there are 20–43 idle tiles of
pure upside). Economics (animals, melon timing, market) only matters AFTER we can
actually run a full farm. Next: build a "keep the farm full and watered" operator
and benchmark it before any fancy strategy.

## Chapter 10 — Operational competence wins (farm_operator)
Built `farm_operator`: units MOVE to the nearest tile that needs work (water >
harvest > dig weed > plant), so the farm stays full and watered instead of idling.
Pure high-velocity wheat loop, expand land only from surplus. **Result: beats the
prior GA champion 16/16 head-to-head (~$5,524 vs ~$3,549, a ~56% money gain) and
`starter` 16/16.** First agent to end well ABOVE the $3,000 start with real
production. The whole prior 30k-game evolution was tuning a strategy that couldn't
run a farm; one routing fix (units go to the work) beat all of it. Submitted as the
new agent. **Big lesson: fix execution before optimizing strategy.** Next headroom:
NOW the economics (animals as perpetual engine, melon timing, market selling into
town demand) can be layered on TOP of a farm that actually operates — and the GA/
ratchet can tune farm_operator's few thresholds against a proper opponent pool.

### GA generation log (best per generation)
| Gen | best money | best winrate | mean fitness | secs |
|---|---|---|---|---|
| 0 | 3,253 | 33% | 2,242 | 50 |

## Chapter 1 — The submission that wouldn't run
The first greedy agent errored on Kaggle. The cause was a loader quirk, not the
strategy: kaggle-environments picks the **last callable defined in your file** as
the agent. Our file ended with a helper (`_move_toward`), so the runner called
that instead of `agent()` and crashed with "missing 2 required positional
arguments." Fix: define all helpers first and `agent()` last. That alone took us
from ERROR to a real leaderboard score of **600**.

## Chapter 2 — Running games locally to see the truth
With the environment installed locally, we built a parallel benchmark: play N
games against the built-in `starter` agent, alternating seats, across CPU cores.
The result was humbling — the greedy agent **lost every game and ended below its
$3,000 starting money** (~1,168 avg). It was buying seeds and land it couldn't
recoup, never raising animals (the best steady income), never growing melon (the
highest-value crop), and dumping its whole shed to market every turn, which
crashes prices to the $1 floor. So the bottleneck was clearly *strategy*, not
speed.

## Chapter 3 — A genome the computer can tune
Rather than hand-tune a dozen thresholds, we made the agent **parameterized**: its
behavior is driven by a set of numeric "genes" — when to buy animals, land, which
crops to grow, what fraction of a stack to sell, a minimum sell price, whether to
fertilize and care for animals, and so on. A richer default genome (with animals,
melon, and throttled selling) already roughly doubled the greedy agent to ~2,300,
though still short of beating `starter`.

## Chapter 4 — Evolving a strategy (genetic algorithm)
Now the computer searches for us. A genetic algorithm holds a **population** of
genomes, scores each by playing games vs `starter` (in parallel across cores),
keeps the best (elitism), and breeds the next generation via **tournament
selection + uniform crossover + gaussian mutation**. Each generation should climb
on average; we checkpoint the best genome so we can stop anytime and bake the
winner into a standalone `main.py` to submit. *(Run in progress — generations and
best fitness logged in `evo/state/history.csv`.)*

---

## Principles this project runs on
- Optimize the agent for *decision quality per millisecond*, not throughput.
- Put the hardware into the *search* (parallel self-play + evolution), not the agent.
- Measure every change with many games before trusting it; one game is noise.
- The market punishes greedy dumping — timing and restraint are strategy.
- Keep the story current so the *how* is as clear as the *what*.
