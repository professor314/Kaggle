# Kaggriculture agent — FARM OPERATOR (operational-competence strategy).
#
# Beats our previous GA champion 16/16 head-to-head (~$5,524 vs ~$3,549) and
# starter 16/16. The win is EXECUTION, not fancy economics: units MOVE to the
# nearest tile that needs work (water > harvest > dig weed > plant), so the farm
# stays full and watered instead of sitting idle and weeding over. High-velocity
# wheat loop; expand land only from clear surplus.
#
# LOADER RULE (kaggle-environments): the runner selects the LAST callable defined
# at module scope. Every helper is defined ABOVE; `agent()` is LAST.

CROP_FIRST_YIELD = {"WHEAT": 2, "CARROT": 4, "TOMATO": 5, "STRAWBERRY": 6, "MELON": 8}


def _move_toward(fx, fy, tx, ty):
    dx, dy = tx - fx, ty - fy
    if abs(dx) >= abs(dy):
        return ["EAST"] if dx > 0 else (["WEST"] if dx < 0 else None)
    return ["SOUTH"] if dy > 0 else (["NORTH"] if dy < 0 else None)


def _nearest(fx, fy, targets):
    if not targets:
        return None
    return min(targets, key=lambda p: abs(p[0] - fx) + abs(p[1] - fy))


def agent(obs):
    """Farm-operator agent. MUST be the last function defined (loader picks last callable)."""
    player = obs["player"]
    me = obs["farms"][player]
    priv = obs["private"]
    day = obs["day"]
    money = me["money"]
    tiles = me["tiles"]
    seeds = priv["seeds"]
    shed = priv["shed"]
    nq = len(me["unlocked_quadrants"])
    rows = len(tiles)
    cols = len(tiles[0]) if rows else 0

    harvestable, thirsty, empty, weeds = [], [], [], []
    for y in range(rows):
        for x in range(cols):
            t = tiles[y][x]
            if t == "LOCKED":
                continue
            if t is None:
                empty.append((x, y))
            elif isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT":
                    crop = t.get("crop", "WHEAT")
                    age = day - t.get("planted_day", day)
                    if age >= CROP_FIRST_YIELD.get(crop, 2) and t.get("yield_units", 0) > 0:
                        harvestable.append((x, y))
                    elif not t.get("watered_today", False):
                        thirsty.append((x, y))
                elif k == "WEED":
                    weeds.append((x, y))

    want_seed = min(len(empty) + 2, 12)
    if seeds.get("WHEAT", 0) < want_seed and money >= 100:
        market = [["BUY_SEED", "WHEAT", min(want_seed, int(money // 10))]]
    else:
        market = []
    for item, cnt in shed.items():
        if cnt > 0 and item != "FERTILIZER":
            market.append(["SELL", item, cnt])
        elif item == "FERTILIZER" and cnt > 0:
            market.append(["SELL", "FERTILIZER", cnt])
    if nq < 2 and money >= 4000 and len(empty) <= 2:
        market.append(["BUY_LAND"])
    market = market[:10]

    claimed = set()

    def assign(x, y):
        tile = tiles[y][x]
        if (x, y) in thirsty:
            return ["WATER"]
        if (x, y) in harvestable:
            return ["HARVEST"]
        if (x, y) in weeds:
            return ["DIG"]
        if tile is None and seeds.get("WHEAT", 0) > 0 and (x, y) not in claimed:
            claimed.add((x, y))
            return ["PLANT", "WHEAT"]
        for pool in (thirsty, harvestable, weeds, empty):
            tgt = _nearest(x, y, [p for p in pool if p not in claimed])
            if tgt:
                if pool is empty and seeds.get("WHEAT", 0) <= 0:
                    continue
                claimed.add(tgt)
                mv = _move_toward(x, y, tgt[0], tgt[1])
                if mv:
                    return mv
        return ["PASS"]

    fx, fy = me["farmer"]
    farmer = assign(fx, fy)
    hands = [assign(hx, hy) for hx, hy in me.get("hands", [])]
    return {"farmer": farmer, "hands": hands, "market": market}
