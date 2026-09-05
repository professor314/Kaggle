"""Profile OUR agent the same way analyze_top.py profiles the top players, so we
can compare signatures side-by-side and see what we're leaving on the table.

Runs herd_engine vs the current submission, then prints our strategy signature:
peak herd, hires, care, fertilizer collected, feed buys, sells-by-item, inflection.
"""
import os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_environments import make

ANIMALS = ("COW", "SHEEP", "GOOSE")


def load(path):
    env = {}
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), env)
    return [v for v in env.values() if callable(v)][-1]


def profile(steps, pi):
    hires = cares = collect = buyfeed = sells = plants = 0
    sell_items = collections.Counter()
    peak_animals = 0
    money_by_day = {}
    for si, st in enumerate(steps):
        cell = st[pi]
        obs = cell.get("observation", {})
        act = cell.get("action")
        farms = obs.get("farms")
        if farms and pi < len(farms):
            day = obs.get("day", si // 24)
            money_by_day[day] = farms[pi]["money"]
            a = sum(1 for row in farms[pi].get("tiles", [])
                    for t in row if isinstance(t, dict) and t.get("animal") in ANIMALS)
            peak_animals = max(peak_animals, a)
        if isinstance(act, dict):
            fa = act.get("farmer")
            allops = [fa] + list(act.get("hands") or [])
            for op in allops:
                if isinstance(op, list) and op:
                    if op[0] == "CARE": cares += 1
                    elif op[0] == "COLLECT_FERTILIZER": collect += 1
                    elif op[0] == "PLANT": plants += 1
            for m in (act.get("market") or []):
                if isinstance(m, list) and m:
                    if m[0] == "HIRE": hires += 1
                    elif m[0] == "SELL":
                        sells += 1
                        if len(m) > 1: sell_items[m[1]] += 1
                    elif m[0] == "BUY_PRODUCT": buyfeed += 1
    days = sorted(money_by_day)
    lo = min(money_by_day[d] for d in days[:8]) if days else 0
    inflect = next((d for d in days if money_by_day[d] > max(3000, lo * 4)), None)
    return {
        "final_money": money_by_day.get(max(days)) if days else 0,
        "peak_animals": peak_animals, "hires": hires, "cares": cares,
        "collect_fert": collect, "buy_feed": buyfeed, "sells": sells,
        "plants": plants, "top_sells": dict(sell_items.most_common(5)),
        "inflect_day": inflect,
    }


def main():
    here = os.path.dirname(__file__)
    me = load(os.path.join(here, "herd_engine.py"))
    opp = load(os.path.join(here, "..", "submissions", "main.py"))
    env = make("kaggriculture", configuration={"episodeSteps": 720})
    env.run([me, opp])
    r = profile(env.steps, 0)
    print("=== OUR herd_engine signature ===")
    for k, v in r.items():
        print(f"  {k:14s}: {v}")
    print("\n--- top players (from analyze_top.py aggregate) ---")
    print("  final_money   : ~86,756")
    print("  peak_animals  : ~15.5")
    print("  hires         : ~289  (~10/day)")
    print("  cares         : ~50")
    print("  collect_fert  : ~50")
    print("  buy_feed      : ~176")
    print("  top_sells     : FERTILIZER >> WHEAT > MILK > STRAWBERRY > WOOL")


if __name__ == "__main__":
    main()
