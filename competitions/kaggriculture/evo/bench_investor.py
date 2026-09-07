"""Benchmark investor vs champion + farm_operator, and report VARIANCE (risk)."""
import json, os, sys, statistics
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genome_agent import make_agent
from strategy_agents import investor, farm_operator

CHAMP = json.load(open("evo/champion.json"))["genome"]


def _resolve(name):
    if name == "champion":
        return make_agent(CHAMP)
    if name == "farm_operator":
        return farm_operator
    if name == "investor":
        return investor
    return name  # "starter"


def _one(payload):
    a, b, as_p1, steps = payload
    from kaggle_environments import make
    A = _resolve(a); B = _resolve(b)
    seat = 1 if as_p1 else 0
    players = [B, A] if as_p1 else [A, B]
    env = make("kaggriculture", configuration={"episodeSteps": steps})
    env.run(players)
    r = [s["reward"] if isinstance(s, dict) else s.reward for s in env.steps[-1]]
    return (r[seat] or 0), (r[1 - seat] or 0)


def matchup(a, b, ex, n=16):
    res = list(ex.map(_one, [(a, b, bool(i % 2), 720) for i in range(n)], chunksize=2))
    wins = sum(1 for am, bm in res if am > bm)
    monies = [am for am, _ in res]
    return wins, n, statistics.mean(monies), min(monies), max(monies), statistics.pstdev(monies)


def main():
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 4)) as ex:
        for opp in ("champion", "farm_operator", "starter"):
            w, n, avg, lo, hi, sd = matchup("investor", opp, ex)
            print(f"investor vs {opp:13s}: {w:2d}/{n} ({w/n:3.0%})  "
                  f"money avg {avg:.0f} [min {lo:.0f}, max {hi:.0f}, sd {sd:.0f}]")


if __name__ == "__main__":
    main()
