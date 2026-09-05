"""Count how many animals escape (starve) across games — the compounding leak.
An escape = a structure that HAD an animal and later is empty (animal gone, not sold).
We approximate by tracking, per game, peak animals vs final animals, and the number
of empty structures at end (each empty pasture/coop once held or was built for one)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_environments import make

def load(path):
    env = {}
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), env)
    return [v for v in env.values() if callable(v)][-1]

here = os.path.dirname(__file__)
me = load(os.path.join(here, "herd_engine.py"))
opp = load(os.path.join(here, "herd_v1.py"))

def counts(farm):
    animals = empty_struct = 0
    for row in farm["tiles"]:
        for t in row:
            if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE"):
                if t.get("animal"): animals += 1
                else: empty_struct += 1
    return animals, empty_struct

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
for g in range(N):
    env = make("kaggriculture", configuration={"episodeSteps": 720})
    env.run([me, opp])
    peak = 0
    for st in env.steps:
        f = st[0]["observation"].get("farms")
        if f: peak = max(peak, counts(f[0])[0])
    final = env.steps[-1][0]["observation"]["farms"][0]
    fa, es = counts(final)
    money = final["money"]
    print(f"game {g+1}: peak_herd={peak:2d}  final_herd={fa:2d}  empty_structs={es:2d}  "
          f"lost>={max(0,peak-fa)}  money={money:.0f}")
