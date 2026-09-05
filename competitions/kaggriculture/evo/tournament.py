"""Round-robin evaluation: score each CANDIDATE agent against a fixed DIVERSE pool
of opponents (not a self-mirror, which we learned is misleading). Reports per-opponent
win rate + money and an aggregate, so we can pick the objectively best agent to submit.

Usage:
  python evo/tournament.py --games 12
"""
import argparse, os, statistics, sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))

# candidates to compare (label -> path). All must define agent() as the last callable.
# NOTE: this harness compares agent .py files against a fixed DIVERSE opponent pool.
# Point CANDIDATES/OPPONENTS at whatever agent files you're comparing. Reference
# versions (past submissions) can be extracted from git, e.g.:
#   git show <commit>:competitions/kaggriculture/submissions/main.py > evo/ref_vN.py
# The current champion (balanced herd v4) lives at submissions/main.py.
CANDIDATES = {
    "current_main":  os.path.join(HERE, "..", "submissions", "main.py"),
    "herd_engine":   os.path.join(HERE, "herd_engine.py"),
}
# fixed diverse opponent pool (the "field"). starter is built-in; add ref files as needed.
OPPONENTS = {
    "starter":       "starter",
    "herd_engine":   os.path.join(HERE, "herd_engine.py"),
}


def load_agent(path):
    if not path.endswith(".py"):
        return path
    # ensure sibling modules (e.g. strategy_agents) import inside worker processes
    d = os.path.dirname(os.path.abspath(path))
    if d not in sys.path:
        sys.path.insert(0, d)
    env = {}
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), env)
    return [v for v in env.values() if callable(v)][-1]


def _play(args):
    cand_path, opp_spec, steps, as_p1 = args
    from kaggle_environments import make
    me = load_agent(cand_path)
    opp = load_agent(opp_spec)
    seat = 1 if as_p1 else 0
    players = [opp, me] if as_p1 else [me, opp]
    env = make("kaggriculture", configuration={"episodeSteps": steps})
    env.run(players)
    final = env.steps[-1]
    r = [s["reward"] if isinstance(s, dict) else s.reward for s in final]
    return (r[seat] or 0, r[1 - seat] or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12, help="games per candidate-opponent pairing")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("-j", "--workers", type=int, default=min(12, os.cpu_count() or 4))
    args = ap.parse_args()

    summary = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for cname, cpath in CANDIDATES.items():
            agg_w = agg_n = 0
            agg_money = []
            print(f"\n=== candidate: {cname} ===")
            for oname, ospec in OPPONENTS.items():
                # don't play a candidate against its own identical reference
                if os.path.abspath(str(ospec)) == os.path.abspath(cpath):
                    continue
                jobs = [(cpath, ospec, args.steps, g % 2 == 1) for g in range(args.games)]
                w = t = l = 0
                mine = []
                for my_m, op_m in ex.map(_play, jobs):
                    mine.append(my_m)
                    if my_m > op_m: w += 1
                    elif my_m == op_m: t += 1
                    else: l += 1
                agg_w += w; agg_n += args.games; agg_money += mine
                print(f"  vs {oname:14s}: W/T/L {w:2d}/{t:2d}/{l:2d}  "
                      f"({w/args.games:3.0%})  my$ avg {statistics.mean(mine):7.0f}")
            summary[cname] = (agg_w / agg_n if agg_n else 0, statistics.mean(agg_money))
            print(f"  AGGREGATE: winrate {summary[cname][0]:.0%}  avg$ {summary[cname][1]:.0f}")

    print("\n===== FINAL RANKING (by aggregate winrate, then money) =====")
    for name, (wr, mn) in sorted(summary.items(), key=lambda kv: (-kv[1][0], -kv[1][1])):
        print(f"  {name:16s}  winrate {wr:5.0%}   avg$ {mn:8.0f}")


if __name__ == "__main__":
    main()
