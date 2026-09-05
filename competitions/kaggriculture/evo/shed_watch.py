"""Watch the shed contents + market-order usage over a game to find selling leaks."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_environments import make


def load(path):
    env = {}
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), env)
    return [v for v in env.values() if callable(v)][-1]


here = os.path.dirname(__file__)
me = load(os.path.join(here, "herd_engine.py"))
opp = load(os.path.join(here, "..", "submissions", "main.py"))
env = make("kaggriculture", configuration={"episodeSteps": 720})
env.run([me, opp])
steps = env.steps
print("day | money | shed (non-seed) | #mkt-orders(last turn) | dropped?")
for i, state in enumerate(steps):
    if i % 24 != 0 and i != len(steps) - 1:
        continue
    cell = state[0]
    obs = cell.get("observation", {})
    priv = obs.get("private", {})
    shed = {k: v for k, v in (priv.get("shed", {}) or {}).items() if v > 0 and k not in ("WHEAT",)}
    total_nonseed = sum((priv.get("shed", {}) or {}).values())
    money = obs["farms"][0]["money"]
    act = cell.get("action") or {}
    nmkt = len(act.get("market") or [])
    day = obs.get("day", i // 24)
    flag = "  SHED FULL(>=100)" if total_nonseed >= 95 else ""
    print(f"{day:3d} | {money:6.0f} | {shed} tot={total_nonseed} | mkt={nmkt}{flag}")
