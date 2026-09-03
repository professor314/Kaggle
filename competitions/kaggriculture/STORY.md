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

**Note on LB scores:** Kaggriculture uses a live skill rating that changes as your
agent keeps playing the field. The greedy fix and the GA agent were rated ~94 and
**243.7** respectively — the evolved agent is clearly rated well above the greedy one.

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
