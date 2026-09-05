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
| 7 | 2026-09-03 | **Farm-operator** (units route to nearest work; keep farm full+watered) | **16/16 vs champ, ~$5,524 vs $3,549** | **submitted** ✅ (LB 251.8) |
| 8 | 2026-09-05 | **Herd engine** (animal-herd strategy from replays; PICKUP+FEED logistics) | **12/0/0 vs farm-operator, ~$41,168 vs $6,080; 8/0/0 vs starter** | **submitted** ✅ (LB 472.5) |
| 9 | 2026-09-05 | **Herd engine v2** — emergency at-risk feeding (zero escapes) | **14/6 (70%) vs v1; 8/0/0 vs starter, avg $45,320** | **submitted** ✅ |
| 10 | 2026-09-05 | **Herd engine v3** — price-aware selling (read the live price curve) | **23/1/6 (77%) vs v2; 8/0/0 vs starter** | **submitted** ✅ |
| 11 | 2026-09-05 | **Herd engine v4** — balanced herd (cows+sheep+geese) | **8/0/0 vs starter (avg $49k) & vs strategy agents (avg $52k)** | **submitted** ✅ (pending) |

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

## Chapter 11 — Investor (high-risk) tested: capital strategies underperform here
Built an `investor` agent (bet capital on land + animals + melon). First version
front-loaded all spending on turn 1 → stranded assets in the shed → deterministic
$520 loss. Fixed it to STAGE investments (deploy one asset before buying the next,
keep a cash cushion) → improved to $2,660, but still loses 0/16 to farm_operator
(~$5,700) with ZERO variance. Combined with animal_engine ($613) and
disciplined_engine ($2,310), three independent experiments agree: **in a 30-day
season with a premium market that crashes on gluts, capital-intensive play
underperforms the simple high-velocity wheat operator.** Land/animal ROI doesn't
pay back in time; every build/place/feed turn is a turn not running the wheat loop.
Keeping investor as a GA contender so selection can confirm empirically — but the
expectation is it gets selected out.

## Chapter 12 — Watching the winners: the animal-herd engine (~$80k, 15x us)
Got real replays of TOP players (Dmytro Maliarenko vs keiz) — the signal we were
missing. They finish with **$63k–$88k**; our best (farm_operator) nets ~$5.5k. A
**~15x gap.** How they do it (consistent across 3 replays):
- **Big HERD of animals**: 7–14 cows + sheep (+ a goose) by end game. This is the
  whole engine — milk ($160) + wool ($200) produced continuously.
- **HIRE ~270–300 times**: hire hands EVERY day to buy the actions needed to feed/
  care/harvest the herd. Actions are the constraint; they buy labor to beat it.
- Heavy **CARE** (yield bonus) + **COLLECT_FERTILIZER** (extra sellable) + **BUY_PRODUCT**
  (buy wheat to feed animals) + constant **SELL**.
- Money curve: near-broke until ~day 9 (all-in on setup), then EXPLODES ~$2.5k ->
  ~$16k at day 11 and compounds to $80k. Classic invest-then-exponential.
- They barely grow crops (a few tomatoes); crops/wheat exist only to bootstrap cash
  and FEED the animals.

**This overturns Chapter 11.** Capital strategy doesn't "underperform" — it's THE
game; our earlier attempts just executed it badly (starved animals, no daily hiring).
farm_operator's tidy wheat loop is a local optimum ~15x below the real ceiling.
NEW PLAN: build the herd engine properly — bootstrap fast, buy cows aggressively,
HIRE a hand every day, CARE + collect fertilizer, buy wheat as feed, sell milk/wool
continuously. Validate against these replays, not our own weak agents.

### GA generation log (best per generation)
| Gen | best money | best winrate | mean fitness | secs |
|---|---|---|---|---|
| 0 | 3,253 | 33% | 2,242 | 50 |

## Chapter 13 — Building the herd engine: the mechanic everyone missed
We set out to build the animal-herd engine the replays revealed (ch.12). The first
two attempts busted the exact way every prior animal agent had — down to $2,072 and
$558, deterministically. Tracing the game exposed the real reason, and it wasn't
strategy at all. It was two **logistics mechanics** hidden in the engine source that
none of our agents (or the theory in ch.8/11) had ever handled:

1. **A bought animal lands in the SHED, not on a tile.** To place it, a unit has to
   be shed-adjacent, `PICKUP` the animal into *its own inventory*, carry it to an
   empty matching structure (pasture for cow/sheep, coop for goose), then `PLACE`.
   Our agents bought cows and built pastures but never picked the cows up — so $400
   animals sat in the shed forever while the money was gone.
2. **`FEED` consumes 1 wheat from the FEEDING UNIT'S inventory — not the shed.** So a
   unit must be *carrying* wheat to feed an animal. Our agents had wheat in the shed
   and animals in the field and never connected them, so every animal starved in two
   days (a total loss) and the agent bought replacements until it went broke.

The winning loop is therefore a **carry problem**: units ferry wheat OUT to feed the
herd and ferry animals OUT to populate it. Once we routed units to (a) grab a wheat
buffer at the shed and carry it to hungry animals, and (b) grow the herd ONE animal
at a time and only while the current herd is fully fed (never starve a $400 asset),
the money curve finally matched the replays: near-broke through the ~day-3-10 setup,
then it EXPLODES — $300 at day 10 → $2.3k day 11 → $5.2k day 12 → $20k+ by day 30.

**Result: 12/0/0 vs the farm-operator champion, avg $41,168 vs $6,080 (a ~7× money
edge), and 8/0/0 vs `starter` ($37.8k vs $3.6k).** Verified over 12 games alternating
seats before submitting. This is the first agent that runs the actual winning
strategy instead of a local optimum. Submitted as the new agent.

**Big lesson (again): read the engine, not just the rules doc.** The rules *described*
PICKUP/PLACE and "feed with wheat," but only the source made it unambiguous that both
flow through per-unit inventory. Three prior chapters of "capital strategies
underperform" were really "we never fed or placed our animals." Next headroom toward
the top players' $80k: add sheep (wool $200), buy land for a bigger herd, and add
CARE discipline + fertilizer collection cadence (fertilizer was their #1 income).

## Chapter 14 — Stopping the leak: no animal ever starves
The herd engine (ch.13) hit LB 472.5 (up from 251.8), but profiling it against the
top players (`evo/profile_agent.py`, `evo/analyze_top.py`) showed we still trailed
their $80k. We chased two hypotheses and let the data pick the winner:

- **Metered selling** (trickle premium goods to protect the price): looked great in
  a single trace ($7.6k → $47.5k) but LOST head-to-head (25-44% vs v1). Against a
  real opponent, whoever sells FIRST gets the high price before the glut — hoarding
  just hands them the good prices. Reverted.
- **Land + sheep expansion** (bigger herd): made it WORSE. A herd spread across two
  quadrants outran the units' ability to feed it, so animals starved on the far
  tiles — the ch.8 "action budget" failure returning. Reverted.

The actual leak was subtler and measurable: `evo/count_escapes.py` showed the herd
**peaks at 10-12 but ends at 7-8 — we lose 2-4 animals per game to starvation.** Each
escaped cow is a $400 asset plus all its future milk gone, and game-to-game money
variance tracked almost exactly with how many animals we kept alive.

Fix: **emergency at-risk feeding.** Any animal with `consecutive_unfed >= 1` (one
miss already banked) escapes tonight if not fed, so the moment a unit is carrying
wheat it drops all other work and runs to the nearest at-risk animal; units with no
wheat urgently fetch a buffer from the shed for them. Result: **zero escapes** — every
game now ends with the full herd alive — and **14/6 (70%) head-to-head vs the
submitted v1**, 8/0/0 vs `starter` at avg $45,320 (max $67,864, into the top-player
range). Verified over 20 games alternating seats before submitting.

**Lesson: the highest-value fix was defensive, not offensive.** We kept trying to earn
more (sell smarter, grow bigger) when the real money was in not LOSING assets we'd
already paid for. Protect the compounding engine first; scale it second. (Next, once
feeding a bigger herd is solved, land + sheep can come back — but only with the labor
logistics to service it.)

## Chapter 15 — Selling on the price curve, not a guessed distribution
Question: can we use economic signals to sell before the glut? The instinct was to
"learn the price distribution from past games," but the game's own rules make that
unnecessary — **price is a DETERMINISTIC function of market inventory** (README price
table), and we can READ `market.prices` + `market.inventory` live every turn. So we
probed a real game (`evo/market_probe.py`) and each good showed a distinct, exploitable
pattern:

- **FERTILIZER decays all season** ($100 → ~$10): everyone dumps this free byproduct,
  so inventory gluts and price collapses. → **sell it ALL immediately; never hold.**
- **WHEAT & WOOL APPRECIATE** (wheat 25→54, wool 200→249): town shops constantly drain
  them below equilibrium, so they get *more* valuable over time. → **sell only
  surplus, prefer selling near the recent high.** (Also: wool holds ~$245 while we
  barely produce it — confirms sheep are worth adding.)
- **MILK is the volatile one** (dump → ~$7, recover → ~$78): premium goods crash on
  gluts but town/shops also consume them, so the price oscillates. → **the one good
  where timing pays: sell a share proportional to how close the price is to its
  recent high; hold some when it's depressed.**

Implementation: a small module-level price memory (the agent is otherwise stateless
between turns; module scope persists for the episode) tracks a slowly-decaying
recent-high per item. Premium goods sell in full near their high, half at a mediocre
price, and hold when depressed — unless the shed is filling toward its 100-item cap,
in which case we dump regardless (a discarded unit is worth $0, worse than any sale).

**Result: 23/1/6 (77%) head-to-head over 30 games vs the escape-fix version, with a
real money edge ($22.3k vs $17.8k — the earlier naive sell-tweak had been dead-even),
and still 8/0/0 vs `starter`.** The lesson: we didn't need machine learning on price
history — the market is a known curve, so the win came from *reading* it (recent-high
relative timing + per-good behavior) rather than modeling it. The distribution idea
would only matter for the parts that ARE stochastic (which shops unlock, and the
opponent's sell timing); those remain future work if we want to push further.

## Chapter 16 — Doing the deterministic research: what the code actually rewards
We read the engine source and listed every deterministic edge (`RESEARCH_IDEAS.md`),
then implemented them and let benchmarks decide. Two clear lessons:

**B1 — exact next-price selling: TRIED, REJECTED.** We ported the engine's exact
`market_price(item, inventory)` and wrote `optimal_sell_qty` to sell each good only
down to a min-price floor (holding the rest for when town demand lifts the price).
It LOST head-to-head (10% at a 55%-base floor, 35% even tuned aggressive). The reason
is the same two-player truth we keep meeting: **against an opponent, selling FIRST
beats holding for a better price — they take the price you wait for.** So the simple
recent-high heuristic (v3) is what we ship; the exact price model is kept in the file
for the income-side math but not used to gate sells. (A good negative result: it
answers "shouldn't we hold for higher prices?" with a measured no.)

**C3 — balanced herd (cows + sheep + geese): SHIPPED.** The engine's constants make
the case: milk (T=122) and wool (T=105) crash INDEPENDENTLY, so a herd that produces
both spreads sales across two premium markets and neither floods as fast. We buy a
couple geese to bootstrap, then keep cows and sheep roughly balanced. Result: against
DIVERSE opponents (starter, farm_operator) it jumps to **8/0/0 at ~$49-52k** (pure-cow
v3 was ~$37-45k) with a much higher money floor ($35k vs $12k). Against its own mirror
(another herd bot flooding the same milk) it's only ~35% — but both agents' money
RISES, and the real leaderboard is the diverse field, not a self-mirror. Verified
across starter + strategy agents before submitting.

Fertilizer discipline (C1) and CARE cadence (C2) are already exercised heavily by the
routing (the profiler showed ~500 CARE + ~500 COLLECT_FERTILIZER/game, well above the
top players' ~50 — if anything we over-do them, so there's no cheap win there; the
open lever is feeding logistics C4 to enable a genuinely BIGGER herd, next).

## Chapter 17 — Tournament: the remaining ideas didn't beat v4
Built the rest of the deterministic research list as candidates and ran a proper
ROUND-ROBIN tournament (`evo/tournament.py`) — each candidate vs a fixed DIVERSE pool
(starter, farm_operator, and our own v3/v4 champions), 12-14 games/pairing, both
seats, ranked by aggregate win rate then money. We do this because we learned
mirror-match results mislead; the field is diverse.

Results:
- **C4 — bigger herd + land expansion: REJECTED.** Aggregate 71% (worst). It crushes
  weak agents but the land buy + bigger herd + longer feeding walks don't pay back in
  30 days vs strong opponents (17% vs v4). This is the THIRD independent time
  expansion has lost — the tight, fully-utilized single-quadrant herd is the right
  scale for a 30-day season. We fixed the land trigger so it DID expand (herd 11->13),
  and it was still worse. Filed as settled: don't expand.
- **B2 — opponent-flood detection (sell before a glut using inventory deltas):
  REJECTED.** Aggregate 70% vs v4's 81%. The +2/turn inventory-rise trigger fires on
  our own prior sells and dumps premium goods prematurely, underperforming v4's
  metered selling. The signal is real but the threshold is too crude; a better version
  would subtract our own known sells first (future work).
- **v4 (balanced herd) — WINNER, 81-89% aggregate, ~$44-49k vs the field.**

Conclusion: after building B1 (rejected), C4 (rejected), B2 (rejected), and C3
(shipped as v4), **v4 remains the objective best** and is already the submitted agent.
Nothing new to submit — re-submitting would be the identical v4. The clean balanced
herd beats every "more sophisticated" variant. Remaining open ideas (A2 demand-aware
production, a refined B2 that nets out our own sells, D1 GA-tuning vs a proper pool)
are for a future session.

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
