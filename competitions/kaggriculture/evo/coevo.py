"""Competitive co-evolution for the Kaggriculture genome agent (Tier 1.2).

Difference from ga.py: fitness is not "money vs a fixed weak starter". Instead each
candidate plays:
  - K sampled PEERS from the current population (both seats), and
  - the ANCHOR set (starter + sharp presets), and
  - a HALL OF FAME of past champions (grows each generation).
Fitness = average final money across all those matches.

Why: the opposition scales with the population (self-play), so the target keeps
getting harder instead of sitting at the weak starter's level — while the anchors
+ HoF keep fitness comparable across generations (a stable yardstick), avoiding
the cycling/drift that pure population-vs-population coevolution suffers.

Cost is K*pop games per generation (plus anchors/HoF), NOT pop^2.

Run:
  python evo/coevo.py --pop 40 --gens 40 --peers 4 --games 2 -j 12 \
      --resume evo/state/best.json
Outputs (separate dir so it doesn't clobber the ga.py run):
  evo/state_coevo/best.json, history.csv, gen_XXX.json, hof.json, run_stats.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genome_agent import GENE_BOUNDS, DEFAULT_GENES, make_agent  # noqa: E402
from opponents import base_opponents, resolve  # noqa: E402

STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state_coevo")


# ---- one match (worker) ----------------------------------------------------
def _play_match(payload):
    """payload = (cand_genome, opp_spec, steps, as_p1). Returns candidate money.

    opp_spec is "starter" (built-in name) or a genome dict.
    """
    cand_genome, opp_spec, steps, as_p1 = payload
    from kaggle_environments import make
    me = make_agent(cand_genome)
    opp = resolve(opp_spec)
    my_seat = 1 if as_p1 else 0
    players = [opp, me] if as_p1 else [me, opp]
    env = make("kaggriculture", configuration={"episodeSteps": steps})
    env.run(players)
    final = env.steps[-1]
    rewards = [s["reward"] if isinstance(s, dict) else s.reward for s in final]
    mine = rewards[my_seat] or 0
    theirs = rewards[1 - my_seat] or 0
    return mine, theirs


# ---- GA operators (same as ga.py) ------------------------------------------
def random_genome(rng):
    return {k: rng.uniform(lo, hi) for k, (lo, hi) in GENE_BOUNDS.items()}


def mutate(genome, rng, rate=0.3, scale=0.25):
    child = dict(genome)
    for k, (lo, hi) in GENE_BOUNDS.items():
        if rng.random() < rate:
            child[k] = min(hi, max(lo, child[k] + rng.gauss(0, scale) * (hi - lo)))
    return child


def crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in GENE_BOUNDS}


def tournament(pop, fits, rng, k=3):
    idxs = rng.sample(range(len(pop)), min(k, len(pop)))
    return pop[max(idxs, key=lambda i: fits[i])]


def build_jobs(pop, anchors, hof, peers, games, steps, rng):
    """Build the full match list for a generation.

    For each candidate i:
      - `peers` random distinct opponents from the population (both seats each),
      - every anchor (both seats),
      - up to 3 most-recent HoF champions (both seats).
    Returns (jobs, index_map) where index_map[i] = list of job indices for cand i.
    """
    jobs = []
    idx_map = [[] for _ in range(len(pop))]
    hof_recent = hof[-3:]
    for i, cand in enumerate(pop):
        opps = []
        # sampled peers (exclude self)
        others = [j for j in range(len(pop)) if j != i]
        for j in rng.sample(others, min(peers, len(others))):
            opps.append(pop[j])
        # anchors + HoF
        opps.extend(spec for _, spec in anchors)
        opps.extend(hof_recent)
        for opp in opps:
            for as_p1 in range(games):          # games=2 -> both seats
                idx_map[i].append(len(jobs))
                jobs.append((cand, opp, steps, bool(as_p1 % 2)))
    return jobs, idx_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--peers", type=int, default=4, help="sampled peer opponents per candidate")
    ap.add_argument("--games", type=int, default=2, help="games per opponent (2 -> both seats)")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--hof-every", type=int, default=2, help="add best to Hall of Fame every N gens")
    ap.add_argument("-j", "--workers", type=int, default=min(12, os.cpu_count() or 4))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", default=None, help="best.json to seed the population + HoF")
    ap.add_argument("--resume-state", action="store_true",
                    help="FULL resume: restore the last full population (latest gen_XXX.json) "
                         "AND the entire Hall of Fame (hof.json) from STATE_DIR, so we continue "
                         "exactly where a stopped run left off.")
    args = ap.parse_args()

    os.makedirs(STATE_DIR, exist_ok=True)
    rng = random.Random(args.seed)
    anchors = base_opponents()

    pop = [random_genome(rng) for _ in range(args.pop)]
    pop[0] = dict(DEFAULT_GENES)
    hof = []
    if args.resume_state:
        gen_files = sorted(f for f in os.listdir(STATE_DIR)
                           if f.startswith("gen_") and f.endswith(".json"))
        if not gen_files:
            print("--resume-state: no gen_*.json found; starting fresh", flush=True)
        else:
            last = gen_files[-1]
            saved = json.load(open(os.path.join(STATE_DIR, last)))
            restored = [entry["genome"] for entry in saved]
            if len(restored) >= args.pop:
                pop = restored[:args.pop]
            else:
                pop = restored + [mutate(random.choice(restored), rng)
                                  for _ in range(args.pop - len(restored))]
            hof_path = os.path.join(STATE_DIR, "hof.json")
            if os.path.exists(hof_path):
                hof = json.load(open(hof_path))
            print(f"FULL resume from {last}: {len(pop)} genomes, HoF {len(hof)} champions",
                  flush=True)
    elif args.resume and os.path.exists(args.resume):
        seed_best = json.load(open(args.resume))["genome"]
        pop[1] = dict(seed_best)
        for i in range(2, min(6, args.pop)):
            pop[i] = mutate(seed_best, rng)
        hof.append(dict(seed_best))          # our current champion is the first HoF member
        print(f"seeded from {args.resume}; HoF starts with 1 champion", flush=True)

    hist_path = os.path.join(STATE_DIR, "history.csv")
    if not (args.resume_state and os.path.exists(hist_path)):
        with open(hist_path, "w", newline="") as f:
            csv.writer(f).writerow(["gen", "best_fit", "mean_fit", "hof_size", "secs"])

    best_ever = None
    if args.resume_state and os.path.exists(os.path.join(STATE_DIR, "best.json")):
        best_ever = json.load(open(os.path.join(STATE_DIR, "best.json")))
    gens_done = 0
    total_secs = 0.0
    total_games = 0
    # continue generation numbering after any restored snapshots
    gen_offset = 0
    if args.resume_state:
        existing = [f for f in os.listdir(STATE_DIR)
                    if f.startswith("gen_") and f.endswith(".json")]
        gen_offset = len(existing)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for gi in range(args.gens):
            gen = gen_offset + gi
            t0 = time.time()
            jobs, idx_map = build_jobs(pop, anchors, hof, args.peers, args.games,
                                       args.steps, rng)
            results = list(ex.map(_play_match, jobs, chunksize=8))
            total_games += len(jobs)

            # fitness = average candidate money across all its matches
            fits = []
            for i in range(len(pop)):
                monies = [results[k][0] for k in idx_map[i]]
                fits.append(statistics.mean(monies) if monies else 0.0)

            order = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
            best_i = order[0]
            best = {"genome": pop[best_i], "fitness": fits[best_i], "gen": gen}
            if best_ever is None or best["fitness"] > best_ever["fitness"]:
                best_ever = best
                json.dump(best_ever, open(os.path.join(STATE_DIR, "best.json"), "w"), indent=2)

            # Hall of Fame: periodically enshrine the current best (self-play target)
            if gen % args.hof_every == 0:
                hof.append(dict(pop[best_i]))
                json.dump(hof, open(os.path.join(STATE_DIR, "hof.json"), "w"))

            secs = time.time() - t0
            gens_done += 1
            total_secs += secs
            print(f"gen {gen:02d} | best_fit {fits[best_i]:8.0f} | "
                  f"mean {statistics.mean(fits):8.0f} | HoF {len(hof)} | "
                  f"{len(jobs)} games | {secs:.0f}s", flush=True)
            with open(hist_path, "a", newline="") as f:
                csv.writer(f).writerow([gen, f"{fits[best_i]:.0f}",
                                        f"{statistics.mean(fits):.0f}", len(hof), f"{secs:.0f}"])
            json.dump([{"genome": pop[i], "fitness": fits[i]} for i in order],
                      open(os.path.join(STATE_DIR, f"gen_{gen:03d}.json"), "w"))

            new_pop = [pop[order[i]] for i in range(min(args.elite, len(pop)))]
            while len(new_pop) < args.pop:
                a = tournament(pop, fits, rng)
                b = tournament(pop, fits, rng)
                new_pop.append(mutate(crossover(a, b, rng), rng))
            pop = new_pop

    stats = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "mode": "coevolution (sampled peers + anchors + Hall of Fame)",
        "command": f"coevo.py --pop {args.pop} --gens {args.gens} --peers {args.peers} "
                   f"--games {args.games} -j {args.workers} --seed {args.seed}",
        "generations_completed": gens_done,
        "population": args.pop,
        "peers_per_candidate": args.peers,
        "total_games": total_games,
        "turns_per_game": args.steps,
        "total_game_turns": total_games * args.steps,
        "total_agent_decisions": total_games * args.steps * 2,
        "hof_size": len(hof),
        "workers": args.workers,
        "wall_clock_secs": round(total_secs, 1),
        "best_fitness": best_ever["fitness"] if best_ever else None,
        "best_gen": best_ever["gen"] if best_ever else None,
    }
    print("\n=== RUN TOTALS ===", flush=True)
    for k, v in stats.items():
        print(f"  {k}: {v}", flush=True)
    with open(os.path.join(STATE_DIR, "run_stats.jsonl"), "a") as f:
        f.write(json.dumps(stats) + "\n")
    print("\nsaved best to", os.path.join(STATE_DIR, "best.json"), flush=True)


if __name__ == "__main__":
    main()
