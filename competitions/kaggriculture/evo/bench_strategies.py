"""Benchmark the theory-driven strategy agents vs our current champion + starter.

Uses common random seeds where possible (same episodeSteps; kaggle-environments
seeds internally) and alternates seats to cut variance. Reports win rate + money.
"""
import json, os, sys, statistics
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genome_agent import make_agent
from strategy_agents import STRATEGIES

CHAMP = json.load(open("evo/champion.json"))["genome"]
N = 16   # games per matchup (both seats)


def _resolve(spec):
    if spec == "starter":
        return "starter"
    if spec == "champion":
        return make_agent(CHAMP)
    return STRATEGIES[spec]


def _one(payload):
    a_name, b_name, as_p1, steps = payload
    from kaggle_environments import make
    A = _resolve(a_name); B = _resolve(b_name)
    seat = 1 if as_p1 else 0
    players = [B, A] if as_p1 else [A, B]
    env = make("kaggriculture", configuration={"episodeSteps": steps})
    env.run(players)
    r = [s["reward"] if isinstance(s, dict) else s.reward for s in env.steps[-1]]
    return (r[seat] or 0), (r[1 - seat] or 0)


def matchup(a_name, b_name, ex, n=N, steps=720):
    jobs = [(a_name, b_name, bool(i % 2), steps) for i in range(n)]
    res = list(ex.map(_one, jobs, chunksize=2))
    wins = sum(1 for am, bm in res if am > bm)
    a_money = statistics.mean(am for am, _ in res)
    b_money = statistics.mean(bm for _, bm in res)
    return wins, n, a_money, b_money


def main():
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 4)) as ex:
        print(f"=== theory-driven strategies vs our champion ({N} games each) ===")
        for name in STRATEGIES:
            w, n, am, bm = matchup(name, "champion", ex)
            print(f"  {name:14s} vs champion : {w:2d}/{n} ({w/n:3.0%})  "
                  f"money {am:.0f} vs {bm:.0f}")
        print("  --- sanity: vs starter ---")
        for name in STRATEGIES:
            w, n, am, bm = matchup(name, "starter", ex)
            print(f"  {name:14s} vs starter  : {w:2d}/{n} ({w/n:3.0%})  "
                  f"money {am:.0f} vs {bm:.0f}")
        # and champion vs starter for reference
        w, n, am, bm = matchup("champion", "starter", ex)
        print(f"  {'champion':14s} vs starter  : {w:2d}/{n} ({w/n:3.0%})  "
              f"money {am:.0f} vs {bm:.0f}")


if __name__ == "__main__":
    main()
