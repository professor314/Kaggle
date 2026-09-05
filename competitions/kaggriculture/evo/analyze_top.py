"""Deep strategy analysis across all top-player replays in the desktop folder.

For each player in each game, quantify the STRATEGY SIGNATURE:
  - final money
  - peak animal count (cows/sheep/geese) and when the herd is built
  - # HIRE orders, # CARE, # COLLECT_FERTILIZER, # BUY_PRODUCT(feed)
  - what they SELL (by item) and total sell orders
  - day the money curve inflects (invest -> compound)
Then summarize whether the top players converge on one strategy.
"""
import json, glob, os, collections, statistics

FOLDER = r"C:\Users\profe\OneDrive\Desktop\kaggle agent competition"
ANIMALS = ("COW", "SHEEP", "GOOSE")


def analyze_player(steps, pi):
    hires = cares = collect = buyfeed = sells = plants = 0
    sell_items = collections.Counter()
    peak_animals = 0
    herd_by_day = {}
    money_by_day = {}
    for si, st in enumerate(steps):
        cell = st[pi]
        obs = cell.get("observation", {})
        act = cell.get("action")
        # money + herd sampled daily
        farms = obs.get("farms")
        if farms and pi < len(farms):
            day = obs.get("day", si // 24)
            money_by_day[day] = farms[pi]["money"]
            a = 0
            for row in farms[pi].get("tiles", []):
                for t in row:
                    if isinstance(t, dict) and t.get("animal") in ANIMALS:
                        a += 1
            herd_by_day[day] = max(herd_by_day.get(day, 0), a)
            peak_animals = max(peak_animals, a)
        if isinstance(act, dict):
            fa = act.get("farmer")
            if isinstance(fa, list) and fa:
                op = fa[0]
                if op == "CARE": cares += 1
                elif op == "COLLECT_FERTILIZER": collect += 1
                elif op == "PLANT": plants += 1
            for m in (act.get("market") or []):
                if isinstance(m, list) and m:
                    if m[0] == "HIRE": hires += 1
                    elif m[0] == "SELL":
                        sells += 1
                        if len(m) > 1: sell_items[m[1]] += 1
                    elif m[0] == "BUY_PRODUCT": buyfeed += 1
    # inflection day: first day money doubles off its early low
    days = sorted(money_by_day)
    lo = min(money_by_day[d] for d in days[:8]) if days else 0
    inflect = next((d for d in days if money_by_day[d] > max(3000, lo * 4)), None)
    return {
        "final_money": money_by_day.get(max(days)) if days else 0,
        "peak_animals": peak_animals,
        "hires": hires, "cares": cares, "collect_fert": collect,
        "buy_feed": buyfeed, "sells": sells, "plants": plants,
        "top_sells": dict(sell_items.most_common(4)),
        "inflect_day": inflect,
    }


def main():
    files = sorted(glob.glob(os.path.join(FOLDER, "*.json")))
    rows = []
    for f in files:
        data = json.load(open(f, encoding="utf-8"))
        steps = data.get("steps")
        teams = data.get("info", {}).get("TeamNames", [])
        for pi in range(len(steps[0])):
            r = analyze_player(steps, pi)
            r["team"] = teams[pi] if pi < len(teams) else f"P{pi}"
            r["game"] = os.path.basename(f)
            rows.append(r)

    print(f"{'team':22s} {'money':>8s} {'herd':>5s} {'hire':>5s} {'care':>5s} "
          f"{'fert':>5s} {'feed':>5s} {'sell':>5s} {'plant':>5s} {'inflect':>7s}  top_sells")
    for r in rows:
        print(f"{r['team'][:22]:22s} {r['final_money']:8.0f} {r['peak_animals']:5d} "
              f"{r['hires']:5d} {r['cares']:5d} {r['collect_fert']:5d} {r['buy_feed']:5d} "
              f"{r['sells']:5d} {r['plants']:5d} {str(r['inflect_day']):>7s}  {r['top_sells']}")

    print("\n=== AGGREGATE (all top-player entries) ===")
    def avg(k): return statistics.mean(x[k] for x in rows if isinstance(x[k], (int, float)))
    print(f"  avg final money : {avg('final_money'):.0f}")
    print(f"  avg peak herd   : {avg('peak_animals'):.1f} animals")
    print(f"  avg HIRE orders : {avg('hires'):.0f}")
    print(f"  avg CARE        : {avg('cares'):.0f}")
    print(f"  avg collect fert: {avg('collect_fert'):.0f}")
    print(f"  avg buy feed    : {avg('buy_feed'):.0f}")
    all_sells = collections.Counter()
    for r in rows:
        for k, v in r["top_sells"].items():
            all_sells[k] += v
    print(f"  most-sold items : {dict(all_sells.most_common())}")


if __name__ == "__main__":
    main()
