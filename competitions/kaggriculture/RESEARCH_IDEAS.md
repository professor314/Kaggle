# Kaggriculture — Research Projects to Improve the Agent

Everything below is grounded in the actual game engine source
(`.venv-dl/.../envs/kaggriculture/kaggriculture.py`), not just the rules doc.
The goal is to close the gap from our ~$22-45k herd engine to the top players' $80k+.

Legend: **[det]** deterministic/deducible from code + observation · **[opp]** driven
by opponent actions · **[rng]** stochastic (RNG-driven).

---

## Part A — What is stochastic, and how predictable is it?

The entire game has **exactly one source of randomness**, and it is *pseudo*-random
with a known, reproducible construction:

```python
# _end_of_day(): the ONLY rng in the game
seed = env.info.get("seed", 0)
rng = random.Random((seed * 1_000_003) ^ day)   # re-seeded EVERY day
# order of consumption each day:
#   1) _spawn_weeds: rng.random() < weedSpawnChance for EACH empty tile (row-major)
#   2) shop unlock (every 3 days): rng.choice(sorted(SHOPS))
```

Consequences (deduced from code):

1. **Weed spawns** — each empty unlocked tile has a 0.5% chance/day. **[rng but tiny]**
   Irrelevant to strategy: with a full farm there are few empty tiles, and our
   operator already digs weeds. Not worth modeling.

2. **Town shop unlocks** — one new shop every 3 days, drawn uniformly with
   replacement from 8 shop types, capped at 8 instances (so unlocks stop ~day 24).
   **[rng, but the schedule is fixed and the effect is monotonic]**
   - We can't see the seed (it's cleared from config before agents observe it), so
     we can't predict *which* shop unlocks *before* it happens.
   - BUT once a shop unlocks it is in `town.unlocked_shops` **forever** and visible.
     Demand only ever grows. So the actionable signal is simply: *read the current
     shop list and lean production toward whatever the town consumes most.*

### Research A1 — Seed fingerprinting (speculative, high-effort) **[rng→det]**
The daily RNG is `Random((seed*1000003) ^ day)`. The **weed pattern + shop unlock on
early days is a fingerprint of the seed.** In principle we could brute-force the seed
from the first few days' observations, then predict *all* future shop unlocks and
weed spawns for the rest of the game.
- Feasibility: the seed space may be large and we only get weak observations (weeds
  are rare; one shop/3 days). Likely NOT invertible from in-game observations alone.
- Verdict: **low priority** — high effort, uncertain payoff, and the benefit
  (knowing shop unlocks ~a few days early) is small vs. just reacting to the visible
  shop list. Document the idea; don't build it yet.

### Research A2 — Demand-aware production & selling **[det, HIGH VALUE]**
`unlocked_shops` is fully observable and demand is monotonic. Each shop instance
consumes 1 of each of its products every `townShopSellInterval` (4 turns); the town
center consumes 1 of every non-fertilizer product/day. This *removes* inventory →
*raises* price. So:
- Count current per-product town demand from `unlocked_shops` + SHOPS table (we have
  this table exactly). Steer the herd/crop mix toward high-demand products (esp. the
  ones with 2× single-product shops: YARN_STORE→wool, PET_CAFE→carrot).
- Sell each product **in step with the rate the town drains it**, so we ride the
  demand-driven price recovery instead of gluttoning ahead of it.
- **This is the highest-value near-term project** and purely deterministic.

---

## Part B — What is opponent-driven, and can we predict it?

Only ONE mechanic couples the two players: **the shared market.** Everything else
(your farm, shed, units) is independent — the opponent cannot touch your tiles.

Market coupling (from `_commit_unit` + the concurrent order loop):
- Sells are processed **one unit at a time, concurrently** across both players. When
  both sell the same good, each unit sold adds 1 to market inventory and re-quotes
  the price, so **the opponent selling the same good as you accelerates the glut.**
- `market.inventory` is **fully visible** to both players every turn. So we can *see*
  the combined effect of both players' trades in the inventory number, even though we
  can't see the opponent's shed/plan.
- **Key rule: a sale at the $1 floor does NOT add to inventory** (`if price > 1`).
  So at the floor there is zero marginal price impact — dump freely; the price only
  recovers via town consumption, never falls further.

### Research B1 — Exact next-price computation (replace the recent-high heuristic) **[det, HIGH VALUE]**
We currently time sells against a rolling recent-high *heuristic*. But the price is a
**closed-form function of `market.inventory`** and we have the exact `MARKET_PARAMS`
(base, I0, T, shape funcs, targets) + the `_shape`/`price` formula from the source.
So we can compute, precisely:
- the price we'd get for the Nth unit of any sale (marginal revenue),
- the point where selling more drives price below some threshold,
- the optimal quantity to sell this turn to maximize revenue without over-crashing.
Turn selling from a heuristic into an **optimization**: sell up to the quantity where
marginal price ≥ a floor we set (e.g. don't sell the unit that would drop milk below
$X). Port `MARKET_PARAMS` + `_shape` + `price()` into our agent (they're constants).

### Research B2 — Opponent modeling from market inventory deltas **[opp, MEDIUM]**
We can't see the opponent's shed, but we can watch `market.inventory[item]` turn over
turn and **subtract the known town consumption + our own trades** to infer the
opponent's net sells of each good. That tells us:
- what the opponent is producing (herd? crops? which?),
- when they're about to glut a market we also sell into (so we sell FIRST or switch
  goods),
- whether they're a weak agent (little market activity) or a strong herd player.
Adaptive response: if the opponent floods milk, pivot our sells to wool/eggs/wheat
where we still get full price. **This is the main "beat a specific opponent" lever**
and directly addresses the co-evolution lesson (robustness ≠ beating the incumbent).

### Research B3 — First-mover selling within a turn **[opp/det, LOW-MED]**
Orders are concurrent and one-unit-at-a-time, capped at 10/turn. Selling earlier in
the season / earlier in the day (before the opponent's daily harvest hits the market)
captures higher prices. Study whether front-loading premium sells to specific hours
(e.g. right after town consumption ticks refresh prices upward) beats uniform selling.

---

## Part C — What else the code reveals (deterministic edges we're not using)

### Research C1 — Fertilizer is the top players' #1 income; we under-exploit it **[det, HIGH]**
Every surviving animal makes 1 fertilizer/day **whether or not it's fed or cared for**
(`_daily_refresh_animals`). It's free money. Top players sold ~327/game; we collect it
but should (a) collect from EVERY animal every day without fail, (b) sell it EARLY
(fertilizer price decays $100→$10 over the season — confirmed in market_probe). A
dedicated "fertilizer round" each morning could add a large, reliable income stream.

### Research C2 — CARE yield-banking math **[det, MEDIUM]**
`CARE` banks +1 to the next scheduled production if the animal was fed AND cared that
day (`pending_care_bonus`), capped by `max_held`. For a cow (milk every 2 days,
max_held 6) consistent CARE roughly doubles milk yield. Quantify the exact ROI of a
CARE action (1 unit-action) vs its milk payoff and make CARE a first-class priority
when a unit is idle and standing on a fed animal. Cheap, deterministic, compounding.

### Research C3 — Optimal herd composition from the yield/price tables **[det, HIGH]**
We have every constant: costs (goose 300 / cow 400 / sheep 500), intervals (1/2/3
days), first-yield days (4/8/6), max_held (4/6/6), and base prices (egg 50 / milk 160
/ wool 200) + the T values that set how fast each market crashes (egg T=332 forgiving,
milk T=122, wool T=105 both crash fast). Compute steady-state $/animal/day AND
$/market-crash for each, then solve the mix that maximizes total revenue given the
market can only absorb so much of each before flooring. Likely answer: a *balanced*
herd (cows+sheep+some geese) beats all-cow because it spreads sales across 3 markets
that each crash independently. This is the structural upgrade that stalled earlier —
now do it with the labor-logistics fix so a bigger/spread herd doesn't starve.

### Research C4 — Feeding logistics / unit routing efficiency **[det, HIGH — enables C3]**
The escape fix stopped losses, but a bigger herd needs units to reach every animal
daily. Research: cluster animals near the shed and near each other to minimize walk
distance; pre-position wheat-carrying units; compute the max herd size N units can
service in a 24-turn day. This is the true cap on herd size and thus on income.
(Recall: land expansion FAILED last time purely because feeding logistics didn't
scale — solve this before re-expanding.)

### Research C5 — Bootstrap optimization **[det, MEDIUM]**
Games where we bust trace to a weak day 0-9 bootstrap (min money seen: $38). Research
the fastest safe path from $3,000 to a self-sustaining herd: exact sequencing of
wheat-loop cash vs first animal purchase vs first hires, minimizing the near-broke
window and its bust risk. A more robust bootstrap raises our floor (lowest games) more
than raising the ceiling.

### Research C6 — Land ROI in a 30-day season **[det, MEDIUM]**
LAND_PRICES = [1000, 2000, 4000]. Compute whether a 2nd/3rd quadrant pays back within
the remaining days given herd income/tile. Pair strictly with C4 (only expand if we
can feed it). Likely: 2nd quadrant yes (if bought ~day 10-14), 3rd rarely.

---

## Part D — Search / learning methods (once the above hand-built edges are in)

### Research D1 — GA/tournament tuning of the herd engine's thresholds **[method]**
The herd engine now has real knobs (herd target, hire target formula, sell ratios,
feed reserve, bootstrap timing). Evolve them — but against a PROPER opponent pool
(our own champions + herd variants + the replay-derived strategy), not `starter`.
Use the ratchet's tightened promotion gate (large-sample head-to-head) so we only
ship real gains. (Reuses evo/coevo.py, evo/ratchet.py, evo/tourney.py.)

### Research D2 — Decision-time lookahead / rollout **[method, budget-gated]**
The agent is stateless and fast; there may be per-turn budget for a shallow rollout
(simulate the next few turns of the market given known town consumption + our sells)
to pick the revenue-maximizing sell quantity. MUST measure per-turn time first — the
agent has a hard cap on Kaggle. Only if B1's closed-form isn't already sufficient.

### Research D3 — Self-play RL (long shot) **[method, high-effort]**
PHASE3_PLAN lists a neural agent. Given the game is fully simulable locally and fast,
a PPO/DQN over a compact state could in principle exceed hand-built play. High effort,
uncertain vs. the strong hand-built + GA baseline; park it unless C/B plateau.

---

## Suggested order (value / effort)
1. **B1** exact next-price selling (port the price formula) — turns our best current
   lever into math. Deterministic, high value, low risk.
2. **C1 + C2** fertilizer discipline + CARE ROI — cheap, compounding, deterministic.
3. **A2** demand-aware production/selling from `unlocked_shops`.
4. **C4 → C3** feeding logistics, THEN balanced herd + land (the structural ceiling).
5. **B2** opponent modeling from inventory deltas (beat specific strong opponents).
6. **D1** GA-tune the knobs vs a proper pool; verify large-sample before shipping.
7. Park A1 (seed fingerprinting), D2/D3 unless the above plateau.

Each project: implement behind a benchmark, verify head-to-head vs the current
champion over ≥20-30 games (both seats) BEFORE submitting, and log the result in
STORY.md. Submit only verified gains.
