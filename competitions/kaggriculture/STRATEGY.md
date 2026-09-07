# Kaggriculture — Training Strategy & Runbook

Persistent notes so any session can pick up where we left off. If something here
disagrees with the code, trust the code and update this file.

## 0. Environment (IMPORTANT — this is easy to lose)

- **Interpreter:** `Projects/Kaggle/.venv-dl/Scripts/python.exe` (Python 3.12.13)
- **Key package:** `kaggle_environments` 1.32.7, with the `kaggriculture` env registered.
- **Game engine source:** `Projects/Kaggle/.venv-dl/Lib/site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.py`
  (and its `README.md` — the authoritative rules).
- The repo has **no** venv of its own; the default `python` on PATH (3.14.x) does
  **not** have `kaggle_environments`. Always use the `.venv-dl` interpreter.

### Gotchas when running
- Importing `kaggle_environments` prints harmless OpenSpiel `Unknown game ...`
  chatter to **stderr** and can make the process exit non-zero **even on success**.
  Read **stdout** for the real result; redirect stderr (`2>$null`) to de-noise.
- Set `PYTHONIOENCODING=utf-8` — some opponent/team names contain CJK characters
  and Windows cp1252 will crash on printing them.

### Canonical run prefix (from `competitions/kaggriculture/`)
```powershell
$env:PYTHONIOENCODING='utf-8'
$py = '..\..\.venv-dl\Scripts\python.exe'
& $py evo\<script>.py   # 2>$null to hide OpenSpiel chatter
```

## 1. The agent

- **Parameterized "genome" agent:** `evo/genome_agent.py`. Behavior is driven by a
  dict of numeric genes (see `DEFAULT_GENES` and `GENE_BOUNDS`). A winning genome
  is baked into a standalone `submissions/main.py` for upload.
- Genes of note (full list in the file): `hire_gate`, `care_animals`, `fertilize`,
  `land2_gate/land3_gate/land4_gate`, `animal_gate`, `cow_gate`, `sell_frac`,
  `sell_min_price`, `hoard_days`, `plant_priority`, `wheat_seed_target`.

## 2. Training design (this DOES exist — we do NOT train only vs `starter`)

### Opponent pool — `evo/opponents.py`
The fixed "exam" the population must beat:
- `starter` — the built-in baseline (by name).
- `AGGRESSIVE_EXPAND` — a hand-written genome that encodes the meta:
  `hire_gate=1500, care_animals=1.0, fertilize=1.0`, early land + animals.
- `MARKET_TIMER` — patient, staple-heavy, sells only at good prices.

### Champion–challenger ratchet — `evo/ratchet.py`
- Seeds the population from the **current champion** + gaussian mutations (local search).
- **Fitness = win_rate_vs_champion (PRIMARY) + `SECONDARY_W` * avg money vs a
  robustness pool.** `SECONDARY_W = 0.0002`.
- **Promotion gate:** the best challenger replaces the champion only if it beats it
  over `--promote-games` with win rate `>= --promote-winrate` (default 0.55).
- **Hall of Fame:** each beaten champion is added to the robustness pool so we never
  forget how to beat it. Guarantees monotonic improvement.
- State: `evo/state_ratchet/champion.json`, `history.csv`, `run_stats.jsonl`.

### Other trainers present
- `evo/ga.py` — plain GA vs `starter` (the original run; see RUN_STATS.md).
- `evo/coevo.py` — co-evolution (optimizes avg money; produced a robust-but-worse
  agent, which is WHY we switched to the ratchet).
- `evo/tourney.py`, `tournament.py` — round-robin evaluation harnesses.

## 3. KNOWN WEAKNESS in the current setup (root cause of the plateau)

Both saved champions drifted to **`hire_gate ~80,875` (GA) / `100,000` (ratchet)**
and **`care_animals = 0` (ratchet)** — i.e. they barely hire and don't care for
animals. The real leaderboard meta is the opposite (see `META_ANALYSIS.md`).

Why the drift happened: the ratchet's **primary** signal is "beat our own
champion," and the meta-opponents (`aggressive_expand`, `market_timer`) only enter
through the **secondary** money term weighted at `0.0002`. Beating a weak champion
never *required* hiring, so hiring was never selected for.

**Fixes (in priority order):**
1. Directly seed a meta-fit genome (open the hire gate, `care_animals=1`,
   `fertilize=1`, earlier land) and gate it against the champion.
2. Raise the weight of the meta opponents, or make one of them the PRIMARY
   fitness target, so the ratchet is forced to learn to out-hire them.
3. Re-run the ratchet from the meta-fit seed.

## 4. Analysis tooling

- `evo/analyze_top.py` — strategy-signature table across all replays.
- `evo/analyze_ops.py` — deep operational patterns (opening, hire ramp, sells, herd).
- Both default `FOLDER` to `../replays` (override with `REPLAY_DIR` env var) and
  skip unreadable/corrupt JSON.
- `evo/verify_champion.py` — head-to-head: ratchet champ vs GA champ vs anchors.
- `evo/inspect_replay.py "<path>"` — dump one replay.

## 5. Replays (opponent data)

- `replays/top_*.json` — downloaded top-player episodes (kaggle-environments format:
  720 turns, 2 players; keys `steps / info.TeamNames / configuration / rewards`).
- `replays/corrupt/` — truncated downloads (interrupted mid-write). Re-download to fix.
- `replays/download.ps1` — resumable downloader (skips files already > 1 KB). It reads
  `replays/episode_ids.txt`. NOTE: this has hung the IDE twice; run it manually /
  in the background rather than blocking a session on it.

## 6. Submission

### READ STORY.md FIRST — the current champion is NOT evo/champion.json
`evo/champion.json` is an OLD GA champion (~$4k, entry #4 in STORY.md). The agent
actually LIVE on the leaderboard is the **Herd engine v4** (`evo/herd_engine.py`),
which scores **~$44-49k vs a diverse field and 12/12 vs starter (~$47k)**. Any new
candidate must be benchmarked head-to-head against `evo/herd_engine.py`, NOT the
GA champion. Confirmed 2026-09-05: the genome/`genome_agent.py` branch (even with a
movement layer) loses 0/12 to the herd engine — it is a dead branch, kept only for
the GA tooling. Do not submit anything that doesn't beat the herd engine.

### How submission works (from data/AGENTS.md)
- Your upload must have a `main.py` at the root exposing an `agent(obs)` function,
  and it MUST be the **last callable defined at module scope** (the kaggle-
  environments loader picks the last callable — a trailing helper will crash it).
- Single file:
  ```bash
  kaggle competitions submit kaggriculture -f main.py -m "message"
  ```
- Multi-file (tar the entry + helpers/weights):
  ```bash
  tar -czf submission.tar.gz main.py helper.py model_weights.pkl
  kaggle competitions submit kaggriculture -f submission.tar.gz -m "message"
  ```
- You must have accepted the rules ("Join Competition") at
  https://www.kaggle.com/competitions/kaggriculture before submitting.
- Competition slug: `kaggriculture`. LB uses a live skill rating that drifts as your
  agent plays, so only submit VERIFIED head-to-head gains vs the current champion.

### Current artifacts
- `submissions/submission.py` — the older "farm-operator" agent.
- `submissions/main.py`, `submissions/submission.tar.gz` — packaged upload.
- The live champion strategy is `evo/herd_engine.py` (herd v4).
