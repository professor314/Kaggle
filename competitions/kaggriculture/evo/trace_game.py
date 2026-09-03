"""Trace one game: print each player's money by day + final tile composition.

Shows WHY an agent wins/loses (money trajectory, what it built). Usage:
  python evo/trace_game.py champion disciplined_engine
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genome_agent import make_agent
from strategy_agents import STRATEGIES
from kaggle_environments import make

CHAMP = json.load(open("evo/champion.json"))["genome"]


def resolve(name):
    if name == "champion":
        return make_agent(CHAMP)
    if name == "starter":
        return "starter"
    return STRATEGIES[name]


def tile_summary(farm):
    counts = {}
    for row in farm["tiles"]:
        for t in row:
            if t is None:
                counts["empty"] = counts.get("empty", 0) + 1
            elif t == "LOCKED":
                counts["locked"] = counts.get("locked", 0) + 1
            elif isinstance(t, dict):
                k = t.get("kind")
                key = t.get("crop") if k == "PLANT" else (t.get("animal") or k)
                counts[key] = counts.get(key, 0) + 1
    return counts


def main():
    a, b = sys.argv[1], sys.argv[2]
    env = make("kaggriculture", configuration={"episodeSteps": 720})
    env.run([resolve(a), resolve(b)])
    steps = env.steps
    print(f"trace: P0={a}  P1={b}")
    print("day |    P0 money |    P1 money")
    turns_per_day = 24
    for i, state in enumerate(steps):
        if i % turns_per_day != 0 and i != len(steps) - 1:
            continue
        obs0 = state[0]["observation"]
        farms = obs0.get("farms")
        if not farms:
            continue
        day = obs0.get("day", i // turns_per_day)
        print(f"{day:3d} | {farms[0]['money']:11.0f} | {farms[1]['money']:11.0f}")
    final = steps[-1][0]["observation"]["farms"]
    print("\nfinal P0 tiles:", tile_summary(final[0]))
    print("final P1 tiles:", tile_summary(final[1]))
    r = [s["reward"] if isinstance(s, dict) else s.reward for s in steps[-1]]
    print(f"final reward: P0={r[0]}  P1={r[1]}")


if __name__ == "__main__":
    main()
