"""Probe the live market: how do prices + inventory move over a game? Establishes
which economic signals are actually observable and exploitable for sell timing."""
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

WATCH = ["WHEAT", "MILK", "WOOL", "FERTILIZER", "MELON"]
print("day | " + " | ".join(f"{w[:4]:>10s}(price/inv)" for w in WATCH) + " | shops")
for i, state in enumerate(steps):
    if i % 24 != 0 and i != len(steps) - 1:
        continue
    obs = state[0]["observation"]
    mk = obs.get("market", {})
    pr, inv = mk.get("prices", {}), mk.get("inventory", {})
    day = obs.get("day", i // 24)
    shops = len(obs.get("town", {}).get("unlocked_shops", []))
    cells = []
    for w in WATCH:
        cells.append(f"{pr.get(w,0):5.0f}/{inv.get(w,0)-10000:+5d}")
    print(f"{day:3d} | " + " | ".join(f"{c:>16s}" for c in cells) + f" | {shops}")
