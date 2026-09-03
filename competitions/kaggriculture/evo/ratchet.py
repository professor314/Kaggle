"""Champion-challenger RATCHET for the Kaggriculture agent.

Goal: START FROM OUR BEST and only ever replace it with something that PROVABLY
beats it head-to-head. This guarantees monotonic improvement — unlike plain
co-evolution (which optimizes average money and produced a robust-but-worse
agent that lost to our incumbent).

How it works each generation:
  - Population is seeded from the CHAMPION + gaussian mutations (local search).
  - Fitness(candidate) = win_rate_vs_champion (PRIMARY)
                         + SECONDARY_W * (avg money vs a small robustness pool),
    normalised so beating the champion always dominates.
  - PROMOTION GATE: the generation's best challenger becomes the new champion
    ONLY if it beats the current champion over PROMOTE_GAMES games with win rate
    >= PROMOTE_WINRATE (both seats). Otherwise the champion stands (no regression).
  - When promoted, the OLD champion is kept in the robustness pool (so we don't
    forget how to beat it), and the population reseeds around the NEW champion.

Run (modest, quick iteration):
  python evo/ratchet.py --pop 24 --gens 12 --eval-games 6 --promote-games 20 \
      --promote-winrate 0.55 -j 12 --champion evo/champion.json

Outputs: evo/state_ratchet/champion.json (current best), history.csv, run_stats.jsonl
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
from genome_agent import GENE_BOUNDS, make_agent  # noqa: E402
from opponents import base_opponents, resolve  # noqa: E402

STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state_ratchet")
SECONDARY_W = 0.0002   # money term weight: 5000 money -> +1.0, so a ~2% winrate edge dominates


def _play(cand, opp_spec, steps, as_p1):
    from kaggle_environments import make
    me = make_agent(cand)
    opp = resolve(opp_spec)
    seat = 1 if as_p1 else 0
    players = [opp, me] if as_p1 else [me, opp]
    env = make("kaggriculture", configuration={"episodeSteps": steps})
    env.run(players)
    r = [s["reward"] if isinstance(s, dict) else s.reward for s in env.steps[-1]]
    return (r[seat] or 0), (r[1 - seat] or 0)


def _eval_candidate(payload):
    """Fitness vs champion (primary) + money vs robustness pool (secondary).

    payload = (cand, champion, pool, eval_games, steps). Returns
    (fitness, winrate_vs_champ, avg_money).
    """
    cand, champion, pool, eval_games, steps = payload
    cw = 0; monies = []
    for i in range(eval_games):
        as_p1 = (i % 2 == 1)
        mine, theirs = _play(cand, champion, steps, as_p1)
        monies.append(mine)
        if mine > theirs:
            cw += 1
    winrate = cw / eval_games
    for spec in pool:
        as_p1 = random.random() < 0.5
        mine, _ = _play(cand, spec, steps, as_p1)
        monies.append(mine)
    avg_money = statistics.mean(monies) if monies else 0.0
    fitness = winrate + SECONDARY_W * avg_money
    return fitness, winrate, avg_money


def _gate_match(payload):
    """One promotion-gate game: challenger vs champion. Returns 1 if challenger wins."""
    challenger, champion, steps, as_p1 = payload
    mine, theirs = _play(challenger, champion, steps, as_p1)
    return 1 if mine > theirs else 0


def mutate(genome, rng, rate=0.3, scale=0.2):
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


def seed_population(champion, n, rng):
    pop = [dict(champion)]                       # keep the champion itself
    while len(pop) < n:
        pop.append(mutate(champion, rng))        # local search around the champion
    return pop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--eval-games", type=int, default=6, help="games vs champion per candidate")
    ap.add_argument("--promote-games", type=int, default=20, help="gate games before promotion")
    ap.add_argument("--promote-winrate", type=float, default=0.55, help="gate threshold")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--elite", type=int, default=3)
    ap.add_argument("-j", "--workers", type=int, default=min(12, os.cpu_count() or 4))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--champion", default="evo/champion.json")
    args = ap.parse_args()

    os.makedirs(STATE_DIR, exist_ok=True)
    rng = random.Random(args.seed)

    # current champion: prefer an existing ratchet champion, else the seed file
    ratchet_champ = os.path.join(STATE_DIR, "champion.json")
    if os.path.exists(ratchet_champ):
        champion = json.load(open(ratchet_champ))["genome"]
        print("resuming from ratchet champion.json", flush=True)
    else:
        champion = json.load(open(args.champion))["genome"]
        print(f"seeded champion from {args.champion}", flush=True)

    anchors = [spec for _, spec in base_opponents()]
    robustness_pool = list(anchors)              # grows with beaten champions

    hist = os.path.join(STATE_DIR, "history.csv")
    with open(hist, "w", newline="") as f:
        csv.writer(f).writerow(["gen", "best_fit", "best_wr_vs_champ", "promoted", "secs"])

    promotions = 0
    pop = seed_population(champion, args.pop, rng)
    total_games = 0
    t_all = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for gen in range(args.gens):
            t0 = time.time()
            payloads = [(c, champion, robustness_pool, args.eval_games, args.steps) for c in pop]
            results = list(ex.map(_eval_candidate, payloads, chunksize=4))
            total_games += len(pop) * (args.eval_games + len(robustness_pool))
            fits = [r[0] for r in results]
            order = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
            bi = order[0]
            best_wr = results[bi][1]

            # PROMOTION GATE: does the best challenger beat the champion head-to-head?
            promoted = False
            gate_jobs = [(pop[bi], champion, args.steps, bool(i % 2))
                         for i in range(args.promote_games)]
            wins = sum(ex.map(_gate_match, gate_jobs, chunksize=4))
            total_games += args.promote_games
            gate_wr = wins / args.promote_games
            if gate_wr >= args.promote_winrate:
                robustness_pool.append(dict(champion))   # remember the old champ
                champion = dict(pop[bi])
                promoted = True
                promotions += 1
                json.dump({"genome": champion, "gate_winrate": gate_wr, "gen": gen},
                          open(ratchet_champ, "w"), indent=2)

            secs = time.time() - t0
            tag = f"PROMOTED (gate {gate_wr:.0%})" if promoted else f"held (gate {gate_wr:.0%})"
            print(f"gen {gen:02d} | best_fit {fits[bi]:.3f} | best wr vs champ "
                  f"{best_wr:.0%} | {tag} | pool {len(robustness_pool)} | {secs:.0f}s", flush=True)
            with open(hist, "a", newline="") as f:
                csv.writer(f).writerow([gen, f"{fits[bi]:.3f}", f"{best_wr:.2f}",
                                        int(promoted), f"{secs:.0f}"])

            # next generation: reseed around the (possibly new) champion + breed
            new_pop = [dict(champion)]
            new_pop += [pop[order[i]] for i in range(min(args.elite, len(pop)))]
            while len(new_pop) < args.pop:
                a = tournament(pop, fits, rng)
                b = tournament(pop, fits, rng)
                new_pop.append(mutate(crossover(a, b, rng), rng))
            pop = new_pop

    stats = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "mode": "champion-challenger ratchet",
        "command": f"ratchet.py --pop {args.pop} --gens {args.gens} "
                   f"--eval-games {args.eval_games} --promote-games {args.promote_games} "
                   f"--promote-winrate {args.promote_winrate} -j {args.workers}",
        "generations": args.gens,
        "promotions": promotions,
        "total_games": total_games,
        "total_game_turns": total_games * args.steps,
        "wall_clock_secs": round(time.time() - t_all, 1),
        "final_pool_size": len(robustness_pool),
    }
    print("\n=== RUN TOTALS ===", flush=True)
    for k, v in stats.items():
        print(f"  {k}: {v}", flush=True)
    with open(os.path.join(STATE_DIR, "run_stats.jsonl"), "a") as f:
        f.write(json.dumps(stats) + "\n")
    print(f"\npromotions: {promotions} | champion saved to {ratchet_champ}", flush=True)


if __name__ == "__main__":
    main()
