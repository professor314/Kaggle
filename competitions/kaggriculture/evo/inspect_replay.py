"""Inspect a Kaggriculture replay JSON: players, final money, config, and a peek
at each player's actions/tiles so we can learn what strong opponents do.

Usage: python evo/inspect_replay.py "<path to episode.json>"
"""
import json, sys, collections

path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))

print("top-level keys:", list(data.keys())[:20])

# kaggle-environments replay: 'steps' = list of turns; each turn = list per player
steps = data.get("steps") or data.get("Steps")
info = data.get("info", {})
conf = data.get("configuration", {})
print("configuration:", {k: conf.get(k) for k in ("episodeSteps", "boardSize", "startingMoney", "seed")})
if isinstance(info, dict):
    tn = info.get("TeamNames") or info.get("teamNames")
    if tn:
        print("teams:", tn)

if not steps:
    print("no steps found; keys were:", list(data.keys()))
    sys.exit()

print("num turns:", len(steps), "| players:", len(steps[0]))

# final rewards + money trajectory (sampled daily)
final = steps[-1]
rewards = [p.get("reward") for p in final]
print("final rewards:", rewards)

# money by day for each player
def money_of(state_p):
    obs = state_p.get("observation", {})
    farms = obs.get("farms")
    return farms

first_obs = steps[0][0].get("observation", {})
nplayers = len(steps[0])
print("\nday | " + " | ".join(f"P{i} money" for i in range(nplayers)))
for i in range(0, len(steps), 24):
    obs0 = steps[i][0].get("observation", {})
    farms = obs0.get("farms")
    if not farms:
        continue
    day = obs0.get("day", i // 24)
    print(f"{day:3d} | " + " | ".join(f"{f['money']:9.0f}" for f in farms))

# action histogram per player (what do they actually DO?)
print("\n=== farmer action histogram per player ===")
for pi in range(nplayers):
    hist = collections.Counter()
    for st in steps:
        act = st[pi].get("action")
        if isinstance(act, dict):
            fa = act.get("farmer")
            if fa:
                hist[fa[0] if isinstance(fa, list) else fa] += 1
            for m in (act.get("market") or []):
                if isinstance(m, list) and m:
                    hist["MKT:" + m[0]] += 1
    print(f"P{pi}:", dict(hist.most_common(15)))

# final tile composition
print("\n=== final tiles ===")
for pi, f in enumerate(steps[-1][0].get("observation", {}).get("farms", [])):
    comp = collections.Counter()
    for row in f.get("tiles", []):
        for t in row:
            if t is None: comp["empty"] += 1
            elif t == "LOCKED": comp["locked"] += 1
            elif isinstance(t, dict):
                comp[t.get("crop") or t.get("animal") or t.get("kind")] += 1
    print(f"P{pi}:", dict(comp))
