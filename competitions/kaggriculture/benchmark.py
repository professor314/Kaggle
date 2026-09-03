"""Local self-play benchmark for the Kaggriculture agent.

Runs our agent against the built-in baselines over N games and reports
win rate + average final money. Use this to measure any change before submitting.

Usage:
  python benchmark.py                      # our agent vs starter, 10 games
  python benchmark.py --opp random -n 20
  python benchmark.py --agent submissions/main.py --opp starter -n 10
"""
import argparse
import os
import statistics
from concurrent.futures import ProcessPoolExecutor


def load_agent(path):
    """Load the `agent` callable from a .py file (same rule Kaggle uses: last callable)."""
    raw = open(path, encoding="utf-8").read()
    env = {}
    exec(compile(raw, path, "exec"), env)
    return [v for v in env.values() if callable(v)][-1]


def _play_one(args_tuple):
    """Run a single game in a worker process. Returns (my_money, opp_money, my_seat).

    Agents are loaded inside the worker (functions aren't picklable across procs;
    built-in opponents are passed by name string).
    """
    agent_path, opp_spec, steps, as_p1 = args_tuple
    from kaggle_environments import make
    me = load_agent(agent_path)
    opp = load_agent(opp_spec) if opp_spec.endswith(".py") else opp_spec
    my_seat = 1 if as_p1 else 0
    players = [opp, me] if as_p1 else [me, opp]
    env = make("kaggriculture", configuration={"episodeSteps": steps})
    env.run(players)
    final = env.steps[-1]
    rewards = [s["reward"] if isinstance(s, dict) else s.reward for s in final]
    return (rewards[my_seat] or 0, rewards[1 - my_seat] or 0, my_seat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="submissions/main.py")
    ap.add_argument("--opp", default="starter", help="starter | random | pass | path to .py")
    ap.add_argument("-n", "--games", type=int, default=10)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--swap", action="store_true", help="also play as player 1 (alternate seats)")
    ap.add_argument("-j", "--workers", type=int, default=min(12, os.cpu_count() or 4),
                    help="parallel worker processes (games are independent)")
    args = ap.parse_args()

    # build the job list; alternate seats so seat bias doesn't skew results
    jobs = [(args.agent, args.opp, args.steps, args.swap and (g % 2 == 1))
            for g in range(args.games)]

    my_money, opp_money = [], []
    wins = ties = losses = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (mine, theirs, seat) in enumerate(ex.map(_play_one, jobs)):
            my_money.append(mine)
            opp_money.append(theirs)
            if mine > theirs:
                wins += 1
            elif mine == theirs:
                ties += 1
            else:
                losses += 1
            print(f"  game {i+1}/{args.games} (seat {seat}): me={mine:.0f} opp={theirs:.0f}", flush=True)

    print("\n=== RESULT vs", args.opp, "over", args.games, "games "
          f"({args.workers} workers) ===")
    print(f"  W/T/L: {wins}/{ties}/{losses}  (win rate {wins/args.games:.0%})")
    print(f"  my money   avg {statistics.mean(my_money):.0f}  "
          f"min {min(my_money):.0f}  max {max(my_money):.0f}")
    print(f"  opp money  avg {statistics.mean(opp_money):.0f}")


if __name__ == "__main__":
    main()
