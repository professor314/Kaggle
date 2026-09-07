"""Deep operational analysis of top-player replays — beyond aggregate counts.

Extracts the OPERATING PATTERNS we haven't mined yet:
  1. Bootstrap sequence: the ordered first ~40 market actions (what they buy, when).
  2. Land: if/when BUY_LAND fires, and herd size + money at that moment.
  3. Hire cadence: hires per day over the season (ramp curve), and WHEN in the day.
  4. Unit task mix: fraction of farmer/hand actions that are FEED/HARVEST/CARE/
     COLLECT/PLANT/PLACE/PICKUP/BUILD vs MOVE vs PASS (walk efficiency).
  5. Sell cadence: sells per day, avg qty per SELL order, per item.
  6. Composition over time: cows/sheep/geese by day (build order of the herd).
  7. What they ignore: crop planting counts, melon, buying fertilizer.
"""
import json, glob, os, collections

FOLDER = os.environ.get(
    "REPLAY_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "replays"),
)
ANIMALS = ("COW", "SHEEP", "GOOSE")
MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
TURNS_PER_DAY = 24


def herd_counts(farm):
    c = collections.Counter()
    for row in farm.get("tiles", []):
        for t in row:
            if isinstance(t, dict) and t.get("animal") in ANIMALS:
                c[t["animal"]] += 1
    return c


def analyze(steps, pi, label):
    first_market = []           # ordered (day.hour, op) for the opening
    land_events = []            # (day, herd, money) when BUY_LAND fires
    hires_by_day = collections.Counter()
    hire_hours = collections.Counter()
    sells_by_day = collections.Counter()
    sell_qty = collections.Counter()      # item -> total qty
    sell_orders = collections.Counter()   # item -> #orders
    task_mix = collections.Counter()
    comp_by_day = {}
    plant_ops = collections.Counter()
    buy_fert = 0

    for si, st in enumerate(steps):
        cell = st[pi]
        obs = cell.get("observation", {})
        act = cell.get("action")
        day = obs.get("day", si // TURNS_PER_DAY)
        hour = obs.get("hour", si % TURNS_PER_DAY)
        farms = obs.get("farms")
        if farms and pi < len(farms):
            comp_by_day[day] = herd_counts(farms[pi])
        if not isinstance(act, dict):
            continue
        # unit actions (farmer + hands)
        units = [act.get("farmer")] + list(act.get("hands") or [])
        for u in units:
            if not (isinstance(u, list) and u):
                task_mix["NONE"] += 1
                continue
            op = u[0]
            if op in MOVES:
                task_mix["MOVE"] += 1
            elif op == "PASS":
                task_mix["PASS"] += 1
            else:
                task_mix[op] += 1
                if op == "PLANT" and len(u) > 1:
                    plant_ops[u[1]] += 1
        # market actions
        for m in (act.get("market") or []):
            if not (isinstance(m, list) and m):
                continue
            op = m[0]
            if len(first_market) < 40 and op != "SELL":
                first_market.append((f"d{day}.{hour:02d}", " ".join(str(x) for x in m)))
            if op == "HIRE":
                hires_by_day[day] += 1
                hire_hours[hour] += 1
            elif op == "BUY_LAND":
                h = herd_counts(farms[pi]) if farms else collections.Counter()
                land_events.append((day, sum(h.values()),
                                    farms[pi]["money"] if farms else None))
            elif op == "SELL":
                sells_by_day[day] += 1
                if len(m) > 1:
                    sell_orders[m[1]] += 1
                    sell_qty[m[1]] += int(m[2]) if len(m) > 2 else 1
            elif op == "BUY_PRODUCT" and len(m) > 1 and m[1] == "FERTILIZER":
                buy_fert += 1

    tot_unit = sum(task_mix.values()) or 1
    print(f"\n########## {label} ##########")
    print("  -- opening (first non-sell market actions) --")
    for t, m in first_market[:18]:
        print(f"     {t}: {m}")
    print("  -- BUY_LAND events (day, herd, money) --")
    print("    ", land_events if land_events else "NONE — never bought land")
    print("  -- hires/day (ramp) --")
    days = sorted(hires_by_day)
    print("    ", {d: hires_by_day[d] for d in days})
    print(f"     hires by hour-of-day (top): "
          f"{dict(sorted(hire_hours.items())[:6])} ... total {sum(hire_hours.values())}")
    print("  -- unit task mix (% of all unit-turns) --")
    for op, n in task_mix.most_common():
        print(f"     {op:18s} {100*n/tot_unit:5.1f}%  ({n})")
    print("  -- sells: orders / total qty per item --")
    for it in sell_qty:
        print(f"     {it:11s} orders={sell_orders[it]:4d}  qty={sell_qty[it]:5d}  "
              f"avg/order={sell_qty[it]/max(1,sell_orders[it]):.1f}")
    print(f"  -- plant ops: {dict(plant_ops)}  |  BUY_PRODUCT FERTILIZER: {buy_fert}")
    print("  -- herd composition by day --")
    for d in sorted(comp_by_day):
        if d % 3 == 0 or d == max(comp_by_day):
            c = comp_by_day[d]
            print(f"     day {d:2d}: COW={c.get('COW',0)} SHEEP={c.get('SHEEP',0)} GOOSE={c.get('GOOSE',0)}")


def main():
    files = sorted(glob.glob(os.path.join(FOLDER, "*.json")))
    # analyze the single highest-scoring player entry per file to keep output focused
    for f in files:                           # all games
        try:
            data = json.load(open(f, encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  !! skipping unreadable replay {os.path.basename(f)}: {str(exc)[:60]}")
            continue
        steps = data.get("steps")
        teams = data.get("info", {}).get("TeamNames", [])
        for pi in range(len(steps[0])):
            label = f"{os.path.basename(f)} P{pi} ({teams[pi] if pi < len(teams) else '?'})"
            analyze(steps, pi, label)


if __name__ == "__main__":
    main()
