"""Trace herd_engine vs an opponent: money by day + herd/tile composition + shed."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_environments import make

def load(path):
    env = {}
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), env)
    return [v for v in env.values() if callable(v)][-1]

herd = load(os.path.join(os.path.dirname(__file__), "herd_engine.py"))
opp = load(os.path.join(os.path.dirname(__file__), "..", "submissions", "main.py"))

def tile_summary(farm):
    c = {}
    for row in farm["tiles"]:
        for t in row:
            if t is None: c["empty"] = c.get("empty", 0) + 1
            elif t == "LOCKED": c["locked"] = c.get("locked", 0) + 1
            elif isinstance(t, dict):
                k = t.get("kind")
                key = t.get("crop") if k == "PLANT" else (t.get("animal") or k)
                c[key] = c.get(key, 0) + 1
    return c

env = make("kaggriculture", configuration={"episodeSteps": 720})
env.run([herd, opp])
steps = env.steps
print("day |  P0(herd) money | herd | shed(feed/fert)")
for i, state in enumerate(steps):
    if i % 24 != 0 and i != len(steps) - 1:
        continue
    obs0 = state[0]["observation"]
    farms = obs0.get("farms")
    if not farms: continue
    day = obs0.get("day", i // 24)
    ts = tile_summary(farms[0])
    herd_n = ts.get("COW", 0) + ts.get("SHEEP", 0) + ts.get("GOOSE", 0)
    # shed only visible in P0's own observation.private
    priv = state[0]["observation"].get("private", {})
    shed = priv.get("shed", {}) if priv else {}
    feed = shed.get("WHEAT", 0); fert = shed.get("FERTILIZER", 0)
    print(f"{day:3d} | {farms[0]['money']:15.0f} | {herd_n:4d} | wheat={feed} fert={fert} {ts}")
r = [s["reward"] if isinstance(s, dict) else s.reward for s in steps[-1]]
print(f"\nfinal reward: P0(herd)={r[0]}  P1(opp)={r[1]}")
