"""Head-to-head: co-evolved champion vs the previous GA champion + anchors."""
import json, os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genome_agent import make_agent
from opponents import base_opponents, resolve
from kaggle_environments import make

coevo = json.load(open("evo/state_ratchet/champion.json"))["genome"]
ga = json.load(open("evo/champion.json"))["genome"]

def match(a_spec, b_spec, n=10, steps=720):
    a_wins = 0; a_money = []
    for i in range(n):
        as_p1 = (i % 2 == 1)
        a_seat = 1 if as_p1 else 0
        A = make_agent(a_spec) if isinstance(a_spec, dict) else resolve(a_spec)
        B = make_agent(b_spec) if isinstance(b_spec, dict) else resolve(b_spec)
        players = [B, A] if as_p1 else [A, B]
        env = make("kaggriculture", configuration={"episodeSteps": steps})
        env.run(players)
        r = [s["reward"] if isinstance(s, dict) else s.reward for s in env.steps[-1]]
        am = r[a_seat] or 0; bm = r[1-a_seat] or 0
        a_money.append(am)
        if am > bm: a_wins += 1
    return a_wins, n, statistics.mean(a_money)

print("=== co-evolved champion vs previous GA champion ===")
w, n, m = match(coevo, ga, n=12)
print(f"  coevo beats GA champ: {w}/{n} ({w/n:.0%})  coevo avg money {m:.0f}")

for name, spec in base_opponents():
    w, n, m = match(coevo, spec, n=10)
    print(f"  coevo vs {name}: {w}/{n} ({w/n:.0%})  avg money {m:.0f}")
