# Kaggriculture — Top-Player Meta Analysis

Source: 30 valid top-player replays in `replays/top_*.json` (5 more were truncated
downloads, quarantined in `replays/corrupt/`). Regenerate with:

```powershell
$env:PYTHONIOENCODING='utf-8'
$py = '..\..\.venv-dl\Scripts\python.exe'
& $py evo\analyze_top.py                 # strategy-signature table
& $py evo\analyze_ops.py > evo\analyze_ops_out.txt 2>&1   # operational deep-dive
```

## Headline

- **Top players finish ~$124k on average** (best entries: keiz, Giulio Ravasio,
  Andrey Tikhomirov ~$150k-190k).
- **Our GA champion finishes ~$4k**; the ratchet champion is similar. The gap is a
  *strategy* gap, not a search-quality gap — see root cause in `STRATEGY.md` §3.

## Strategy signature (from `analyze_top.py`, aggregate over all entries)

| Metric | Meta value |
|---|---|
| avg final money | ~$123,864 |
| avg peak herd | ~13.7 animals |
| avg HIRE orders | ~285 |
| avg CARE | ~52 |
| avg COLLECT_FERTILIZER | ~51 |
| avg BUY feed | ~96 (top-tier 160-275) |

**Most-sold items (total qty across all games):**
FERTILIZER 6,119 » WHEAT 3,057 > MILK 2,719 > WOOL 1,453 > STRAWBERRY 1,071 > MELON 8.

Reading: **fertilizer is the #1 revenue commodity**, wheat is the staple crop,
milk (cows) + wool (sheep) are the animal products. Geese/eggs and melon are
effectively ignored.

## Operational patterns (from `analyze_ops.py`)

### Opening (nearly identical across strong players, turns d0.01-d0.02)
1. `BUY_PRODUCT WHEAT ~43` (feed stockpile)
2. `BUY_ANIMAL COW 2` + `BUY_ANIMAL SHEEP 2`
3. **5x `HIRE` immediately** on turn 1
4. `BUY_SEED WHEAT ~6-7` + `BUY_SEED MELON 12`
5. then steady `BUY_PRODUCT WHEAT 1` roughly every turn (ongoing feed)

### Hiring (the biggest lever)
- Ramp: ~5 hires/day early, climbing to a **hard cap of ~12/day by ~day 11**,
  sustained to the end. Total ~290-301 hires per game.
- Hires are placed at hour 1-2 of each day.
- **Labor is the whole engine** — more hands = more watering/harvest/care/collect.

### Land
- **Two BUY_LAND events:** ~day 6 (cheap, ~$77-1,200) and ~day 11 (~$6.7k-16.7k).

### Unit task mix (share of all unit-turns, top scorer keiz)
MOVE ~49% · WATER ~11% · FEED ~6% · CARE ~6% · COLLECT_FERTILIZER ~6% ·
HARVEST ~5% · then PICKUP/PLACE/PLANT/BUILD/FERTILIZE smaller.

### Herd build order (keiz, representative)
COW 2 / SHEEP 2 at day 0 → COW 8 / SHEEP 11 / GOOSE 1 by day 12, then held.

### Money curve
Inflects around **day 9-10** (invest → compound).

## Implications for our genome (what to change)

| Gene | Champion (GA / ratchet) | Meta implies | Change |
|---|---|---|---|
| `hire_gate` | 80,875 / 100,000 | hire from turn 1 | drop to ~0-500 |
| per-day hire cap (code: `hires_today < 2`) | 2 | up to ~12/day | raise the cap |
| `care_animals` | 0.47 / **0** | ~52 CARE/game | set to 1.0 |
| `fertilize` | 0.09 / 0.08 | fertilizer = #1 sell | set to 1.0 |
| `land2_gate` | 8,000 | buy ~day 6 cheap | lower to ~1,200 |
| `land3_gate` | 10,951 / 7,785 | buy ~day 11 | lower to ~6,000 |
| `animal_gate` / `cow_gate` | 6-8k | cows+sheep early | lower to ~900 / ~2,500 |
| `plant_priority` | ~0.7-0.75 | wheat staple + some strawberry | keep-ish |
| MELON / GOOSE focus | — | ignored by meta | don't invest |

Note: the existing `AGGRESSIVE_EXPAND` opponent in `opponents.py` already encodes
most of this (`hire_gate=1500, care_animals=1, fertilize=1`). The meta-fit genome
should start from that and be tuned/tested against our champion.

## Caveat

`final_money` here is read from the replay's per-player money, and our agent's
per-day hire cap is enforced in `genome_agent.py` code (`hires_today < 2`), not just
by `hire_gate` — so fully matching the meta's ~12 hires/day needs a code change to
that cap, not only a gene change.

## CORRECTION (2026-09-05): this analysis compared against the WRONG champion

The sections below compare the meta to `evo/champion.json` (an OLD ~$4k GA
champion). The agent actually LIVE on the leaderboard is the **Herd engine v4**
(`evo/herd_engine.py`), which already implements movement + a balanced herd +
price-curve selling and scores ~$44-49k (12/12 vs starter). See STORY.md ch.8-17.

So the "movement gap" below is real for the `genome_agent.py` branch, but that
branch was already superseded. The meta-fit genome built this session loses 0/12
to the herd engine and must NOT be submitted. The genome/GA branch is kept only
for GA tooling; real improvements should build on `herd_engine.py`. Open levers
from STORY.md ch.17 not yet cracked: A2 demand-aware production, a refined
opponent-flood sell that nets out our own sells, and GA-tuning vs a proper pool.

## ROOT CAUSE of the plateau (found 2026-09-05) — the agent has NO movement

The bigger problem is not the genome; it's the agent code. In `genome_agent.py`,
the farmer and every hand are given an action based ONLY on the tile they are
**currently standing on** (`_decide(tiles[hy][hx], ...)`). The agent never emits a
movement action (`NORTH/SOUTH/EAST/WEST`).

Why that makes the meta impossible for us:
- Hired hands spawn on the four shed-access tiles (`(4,4),(5,4),(4,5),(5,5)` on a
  10x10 board). At game start only NW is unlocked, so **3 of those 4 tiles are
  LOCKED** and the 4th holds at most one plant.
- Every productive tile op (`PLANT/WATER/HARVEST/FEED/CARE/COLLECT_FERTILIZER`)
  no-ops on a LOCKED tile and acts only on the unit's current tile.
- So hiring 12 hands just stacks 12 idle units on 4 tiles. Labor scales but there
  is no way to distribute it across the ~25-100 farmable tiles. Money plateaus.

Evidence: meta_seed v2 (hires_per_day=5) reached only ~$1,100 flat and LOST to
`starter` ($1,158 vs $3,509). Hands = 0 productive work.

The engine clearly INTENDS movement: `FARMER_MOVES` exists, moving onto LOCKED
tiles is explicitly allowed (so hands can leave the shed corner), and the built-in
`random_agent` moves its hands. Our agent simply never did.

### The fix (required before any hire-heavy genome can work)
Add a movement + task-assignment layer to `genome_agent.py`:
1. Each turn scan `me["tiles"]` for pending work (empty owned tile -> PLANT;
   PLANT needing WATER/HARVEST/FERTILIZE; WEED -> DIG; animal needing
   FEED/CARE/COLLECT/HARVEST).
2. Assign each unit (farmer + hands) a DISTINCT target tile (greedy nearest).
3. If the unit isn't on its target, emit one `FARMER_MOVES` step toward it
   (Manhattan; LOCKED tiles are passable). If it IS on target, emit the tile op
   via the existing `_decide`.
4. Re-runs every day because hands reset/re-spawn each day.

This turns hiring from a money sink into productive parallel labor — the basis of
the $100k+ human strategies.
