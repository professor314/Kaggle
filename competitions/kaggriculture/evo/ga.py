"""Genetic algorithm to evolve the Kaggriculture genome agent.

Each individual is a genome (dict of genes). Fitness = average final money over
`games_per_eval` games vs the built-in `starter` (alternating seats), with a win
bonus. Evaluation is parallel across CPU cores. Elitism + tournament selection +
uniform crossover + gaussian mutation. Checkpoints the best genome every gen so
you can stop anytime and keep the winner.

Run:
  python evo/ga.py --pop 40 --gens 30 --games 6 -j 12
  python evo/ga.py --resume evo/state/best.json --gens 20     # continue from a saved best

Outputs:
  evo/state/best.json        best genome so far (+ its fitness)
  evo/state/history.csv      per-generation best/mean fitness
  evo/state/gen_XXX.json     full population snapshot per generation
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

# genome_agent lives next to this file; make import work when run from repo root
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genome_agent import GENE_BOUNDS, DEFAULT_GENES  # noqa: E402

STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")


# ---- fitness (runs in a worker process) ------------------------------------
def _eval_genome(payload):
    """payload = (genome, games, steps, seed). Returns (fitness, avg_money, winrate)."""
    genome, games, steps, seed = payload
    from kaggle_environments import make
    from genome_agent import make_agent
    me = make_agent(genome)
    monies, wins = [], 0
    for gi in range(games):
        as_p1 = (gi % 2 == 1)
        my_seat = 1 if as_p1 else 0
        players = ["starter", me] if as_p1 else [me, "starter"]
        env = make("kaggriculture", configuration={"episodeSteps": steps})
        env.run(players)
        final = env.steps[-1]
        rewards = [s["reward"] if isinstance(s, dict) else s.reward for s in final]
        mine = rewards[my_seat] or 0
        theirs = rewards[1 - my_seat] or 0
        monies.append(mine)
        if mine > theirs:
            wins += 1
    avg_money = statistics.mean(monies)
    winrate = wins / games
    # fitness: money is the true objective; add a bonus for actually beating starter
    fitness = avg_money + 1000.0 * winrate
    return (fitness, avg_money, winrate)


# ---- GA operators ----------------------------------------------------------
def random_genome(rng):
    return {k: rng.uniform(lo, hi) for k, (lo, hi) in GENE_BOUNDS.items()}


def mutate(genome, rng, rate=0.3, scale=0.25):
    child = dict(genome)
    for k, (lo, hi) in GENE_BOUNDS.items():
        if rng.random() < rate:
            span = (hi - lo)
            child[k] = min(hi, max(lo, child[k] + rng.gauss(0, scale) * span))
    return child


def crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in GENE_BOUNDS}


def tournament(pop, fits, rng, k=3):
    idxs = rng.sample(range(len(pop)), min(k, len(pop)))
    return pop[max(idxs, key=lambda i: fits[i])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--games", type=int, default=6, help="games per genome eval (even -> both seats)")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("-j", "--workers", type=int, default=min(12, os.cpu_count() or 4))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", default=None, help="path to a best.json to seed the population")
    args = ap.parse_args()

    os.makedirs(STATE_DIR, exist_ok=True)
    rng = random.Random(args.seed)

    # init population
    pop = [random_genome(rng) for _ in range(args.pop)]
    pop[0] = dict(DEFAULT_GENES)  # keep the hand-tuned default as one seed
    if args.resume and os.path.exists(args.resume):
        seed_best = json.load(open(args.resume))["genome"]
        pop[1] = dict(seed_best)
        for i in range(2, min(6, args.pop)):     # a few mutated copies of it
            pop[i] = mutate(seed_best, rng)
        print(f"seeded population from {args.resume}", flush=True)

    hist_path = os.path.join(STATE_DIR, "history.csv")
    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(["gen", "best_fit", "mean_fit", "best_money", "best_winrate", "secs"])

    best_ever = None
    gens_done = 0
    total_secs = 0.0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for gen in range(args.gens):
            t0 = time.time()
            payloads = [(gm, args.games, args.steps, args.seed + gen) for gm in pop]
            results = list(ex.map(_eval_genome, payloads))
            fits = [r[0] for r in results]

            order = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
            best_i = order[0]
            best = {"genome": pop[best_i], "fitness": fits[best_i],
                    "avg_money": results[best_i][1], "winrate": results[best_i][2], "gen": gen}
            if best_ever is None or best["fitness"] > best_ever["fitness"]:
                best_ever = best
                json.dump(best_ever, open(os.path.join(STATE_DIR, "best.json"), "w"), indent=2)

            secs = time.time() - t0
            gens_done += 1
            total_secs += secs
            print(f"gen {gen:02d} | best_fit {fits[best_i]:8.0f} "
                  f"(money {results[best_i][1]:7.0f}, win {results[best_i][2]:.0%}) "
                  f"| mean {statistics.mean(fits):8.0f} | {secs:.0f}s", flush=True)
            with open(hist_path, "a", newline="") as f:
                csv.writer(f).writerow([gen, f"{fits[best_i]:.0f}", f"{statistics.mean(fits):.0f}",
                                        f"{results[best_i][1]:.0f}", f"{results[best_i][2]:.2f}", f"{secs:.0f}"])
            json.dump([{"genome": pop[i], "fitness": fits[i]} for i in order],
                      open(os.path.join(STATE_DIR, f"gen_{gen:03d}.json"), "w"))

            # next generation: elitism + bred children
            new_pop = [pop[order[i]] for i in range(min(args.elite, len(pop)))]
            while len(new_pop) < args.pop:
                a = tournament(pop, fits, rng)
                b = tournament(pop, fits, rng)
                new_pop.append(mutate(crossover(a, b, rng), rng))
            pop = new_pop

    # --- run totals (auto-recorded) ---
    total_games = gens_done * args.pop * args.games
    total_turns = total_games * args.steps
    stats = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "command": f"ga.py --pop {args.pop} --gens {args.gens} --games {args.games} "
                   f"-j {args.workers} --seed {args.seed}",
        "generations_completed": gens_done,
        "population": args.pop,
        "games_per_eval": args.games,
        "total_games": total_games,
        "turns_per_game": args.steps,
        "total_game_turns": total_turns,
        "total_agent_decisions": total_turns * 2,
        "workers": args.workers,
        "wall_clock_secs": round(total_secs, 1),
        "best_fitness": best_ever["fitness"] if best_ever else None,
        "best_money": best_ever["avg_money"] if best_ever else None,
        "best_winrate": best_ever["winrate"] if best_ever else None,
        "best_gen": best_ever["gen"] if best_ever else None,
    }
    print("\n=== RUN TOTALS ===", flush=True)
    for k, v in stats.items():
        print(f"  {k}: {v}", flush=True)
    # machine-readable append + human table row
    with open(os.path.join(STATE_DIR, "run_stats.jsonl"), "a") as f:
        f.write(json.dumps(stats) + "\n")

    print("\nBEST EVER:", json.dumps(best_ever, indent=2), flush=True)
    print("saved to", os.path.join(STATE_DIR, "best.json"), flush=True)


if __name__ == "__main__":
    main()
