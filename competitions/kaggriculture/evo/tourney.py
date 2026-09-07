"""Tournament GA over the FARM-OPERATOR genome, with periodic fresh mutants.

- Substrate: operator_genome.make_operator (winning routing + tunable knobs +
  optional melon/animal layers). Seeded from farm_operator's defaults (our proven
  ~$5.7k winner), so evolution can only improve on a strategy that already works.
- Fitness: population plays EACH OTHER (sampled peers, both seats) + a small anchor
  set (starter + our champion). Fitness = avg money across matches.
- Small tweaks: elitism + tournament select + low-scale gaussian mutation.
- PERIODIC FRESH MUTANTS (immigrants): each generation, replace the weakest few
  with brand-new random / heavily-mutated genomes, so the search keeps exploring
  and doesn't collapse into one basin.

Run:
  python evo/tourney.py --pop 30 --gens 15 --peers 4 --games 2 --immigrants 4 -j 12
Outputs: evo/state_tourney/best.json, history.csv, run_stats.jsonl
"""
from __future__ import annotations

import argparse, csv, json, os, random, statistics, time
from concurrent.futures import ProcessPoolExecutor

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from operator_genome import OP_GENE_BOUNDS, DEFAULT_OP_GENES, make_operator  # noqa: E402

STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state_tourney")


def _champion_genome():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "champion.json")
    return json.load(open(p))["genome"] if os.path.exists(p) else None


def _play(cand, opp_spec, steps, as_p1):
    from kaggle_environments import make
    from genome_agent import make_agent
    me = make_operator(cand)
    if opp_spec == "starter":
        opp = "starter"
    elif isinstance(opp_spec, dict) and "__champ__" in opp_spec:
        opp = make_agent(opp_spec["__champ__"])
    else:
        opp = make_operator(opp_spec)
    seat = 1 if as_p1 else 0
    players = [opp, me] if as_p1 else [me, opp]
    env = make("kaggriculture", configuration={"episodeSteps": steps})
    env.run(players)
    r = [s["reward"] if isinstance(s, dict) else s.reward for s in env.steps[-1]]
    return (r[seat] or 0)


def _eval(payload):
    cand, opponents, steps = payload
    monies = []
    for opp, as_p1 in opponents:
        monies.append(_play(cand, opp, steps, as_p1))
    return statistics.mean(monies) if monies else 0.0


def random_genome(rng):
    return {k: rng.uniform(lo, hi) for k, (lo, hi) in OP_GENE_BOUNDS.items()}


def mutate(g, rng, rate=0.3, scale=0.15):
    c = dict(g)
    for k, (lo, hi) in OP_GENE_BOUNDS.items():
        if rng.random() < rate:
            c[k] = min(hi, max(lo, c[k] + rng.gauss(0, scale) * (hi - lo)))
    return c


def crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in OP_GENE_BOUNDS}


def tournament(pop, fits, rng, k=3):
    idxs = rng.sample(range(len(pop)), min(k, len(pop)))
    return pop[max(idxs, key=lambda i: fits[i])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--gens", type=int, default=15)
    ap.add_argument("--peers", type=int, default=4)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--immigrants", type=int, default=4, help="fresh mutants injected per gen")
    ap.add_argument("--elite", type=int, default=3)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("-j", "--workers", type=int, default=min(12, os.cpu_count() or 4))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume-state", action="store_true")
    args = ap.parse_args()

    os.makedirs(STATE_DIR, exist_ok=True)
    rng = random.Random(args.seed)
    champ = _champion_genome()
    anchors = [("starter", None)]
    if champ is not None:
        anchors.append(({"__champ__": champ}, None))

    # init: mostly small mutations of the proven operator defaults + a few randoms
    pop = [dict(DEFAULT_OP_GENES)]
    while len(pop) < args.pop:
        if len(pop) <= args.pop * 0.7:
            pop.append(mutate(DEFAULT_OP_GENES, rng))
        else:
            pop.append(random_genome(rng))
    gen_offset = 0
    if args.resume_state:
        gens = sorted(f for f in os.listdir(STATE_DIR) if f.startswith("gen_") and f.endswith(".json"))
        if gens:
            saved = json.load(open(os.path.join(STATE_DIR, gens[-1])))
            pop = [e["genome"] for e in saved][:args.pop]
            while len(pop) < args.pop:
                pop.append(mutate(DEFAULT_OP_GENES, rng))
            gen_offset = len(gens)
            print(f"resumed from {gens[-1]} ({len(pop)} genomes)", flush=True)

    hist = os.path.join(STATE_DIR, "history.csv")
    if not (args.resume_state and os.path.exists(hist)):
        with open(hist, "w", newline="") as f:
            csv.writer(f).writerow(["gen", "best_fit", "mean_fit", "secs"])

    best_ever = None
    total_games = 0
    t_all = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for gi in range(args.gens):
            gen = gen_offset + gi
            t0 = time.time()
            # build each candidate's opponent list: sampled peers (both seats) + anchors
            payloads = []
            for i, cand in enumerate(pop):
                opps = []
                others = [j for j in range(len(pop)) if j != i]
                for j in rng.sample(others, min(args.peers, len(others))):
                    for s in range(args.games):
                        opps.append((pop[j], bool(s % 2)))
                for a_spec, _ in anchors:
                    for s in range(args.games):
                        opps.append((a_spec, bool(s % 2)))
                payloads.append((cand, opps, args.steps))
                total_games += len(opps)
            fits = list(ex.map(_eval, payloads, chunksize=2))

            order = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
            bi = order[0]
            if best_ever is None or fits[bi] > best_ever["fitness"]:
                best_ever = {"genome": pop[bi], "fitness": fits[bi], "gen": gen}
                json.dump(best_ever, open(os.path.join(STATE_DIR, "best.json"), "w"), indent=2)

            secs = time.time() - t0
            print(f"gen {gen:02d} | best_fit {fits[bi]:7.0f} | mean {statistics.mean(fits):7.0f} "
                  f"| {secs:.0f}s", flush=True)
            with open(hist, "a", newline="") as f:
                csv.writer(f).writerow([gen, f"{fits[bi]:.0f}", f"{statistics.mean(fits):.0f}", f"{secs:.0f}"])
            json.dump([{"genome": pop[i], "fitness": fits[i]} for i in order],
                      open(os.path.join(STATE_DIR, f"gen_{gen:03d}.json"), "w"))

            # next gen: elites + bred children + PERIODIC FRESH MUTANTS (immigrants)
            new_pop = [pop[order[i]] for i in range(min(args.elite, len(pop)))]
            n_children = args.pop - args.elite - args.immigrants
            for _ in range(max(0, n_children)):
                a = tournament(pop, fits, rng); b = tournament(pop, fits, rng)
                new_pop.append(mutate(crossover(a, b, rng), rng))
            for _ in range(args.immigrants):                       # explorers
                if rng.random() < 0.5:
                    new_pop.append(random_genome(rng))             # brand new
                else:
                    new_pop.append(mutate(DEFAULT_OP_GENES, rng, rate=0.6, scale=0.4))  # big jump
            pop = new_pop[:args.pop]

    stats = {"date": time.strftime("%Y-%m-%d %H:%M"), "mode": "tourney (peers+anchors+immigrants)",
             "generations": args.gens, "population": args.pop, "immigrants_per_gen": args.immigrants,
             "total_games": total_games, "wall_clock_secs": round(time.time() - t_all, 1),
             "best_fitness": best_ever["fitness"] if best_ever else None}
    print("\n=== RUN TOTALS ===", flush=True)
    for k, v in stats.items():
        print(f"  {k}: {v}", flush=True)
    with open(os.path.join(STATE_DIR, "run_stats.jsonl"), "a") as f:
        f.write(json.dumps(stats) + "\n")
    print(f"\nbest saved to {os.path.join(STATE_DIR, 'best.json')}", flush=True)


if __name__ == "__main__":
    main()
