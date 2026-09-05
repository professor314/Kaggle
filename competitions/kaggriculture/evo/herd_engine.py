# Kaggriculture agent — HERD ENGINE v2.
#
# Reverse-engineered from the TOP players' replays (keiz, Dmytro Maliarenko,
# curiosity, Lunospital — all net $63k-$104k, ~15x our farm_operator's $5.5k).
# Universal winning strategy (STORY.md ch.12): a HERD of ~15 cows+sheep, ~10
# hires/day, constant CARE + COLLECT_FERTILIZER + BUY_PRODUCT(wheat feed) + SELL;
# #1 income is FERTILIZER, then MILK/WOOL. Near-broke to ~day 9, then compounds.
#
# THE MECHANIC EVERY PRIOR ANIMAL AGENT GOT WRONG (verified in the engine source):
#   * A bought animal lands in the SHED. To place it, a unit must be shed-adjacent,
#     PICKUP the animal into ITS OWN inventory, carry it to an empty matching
#     structure (pasture=cow/sheep, coop=goose), then PLACE.
#   * FEED consumes 1 WHEAT from the FEEDING UNIT'S inventory — NOT the shed. So a
#     unit must be CARRYING wheat to feed an animal. Units grab a wheat buffer at
#     the shed, then ferry it to animals.
#   * HARVEST / COLLECT_FERTILIZER add to the unit's inventory; it reaches the shed
#     at end-of-day auto-drop (then we SELL from the shed next day).
#
# So the core loop is a logistics problem: units carry wheat OUT to feed the herd
# and carry animals OUT to populate it. We grow the herd ONE animal at a time and
# only when the current herd is alive and fed, so we never starve a $400 asset.
#
# LOADER RULE: kaggle-environments picks the LAST callable at module scope, so
# `agent()` is LAST; every helper is above it.

CROP_FIRST_YIELD = {"WHEAT": 2, "CARROT": 4, "TOMATO": 5, "STRAWBERRY": 6, "MELON": 8}
BASE_PRICE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
              "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
PREMIUM = {"MELON", "STRAWBERRY", "MILK", "WOOL"}
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}


def _move_toward(fx, fy, tx, ty):
    dx, dy = tx - fx, ty - fy
    if abs(dx) >= abs(dy):
        return ["EAST"] if dx > 0 else (["WEST"] if dx < 0 else None)
    return ["SOUTH"] if dy > 0 else (["NORTH"] if dy < 0 else None)


def _nearest(fx, fy, targets):
    if not targets:
        return None
    return min(targets, key=lambda p: abs(p[0] - fx) + abs(p[1] - fy))


def _shed_tiles(rows, cols):
    hx, hy = cols // 2, rows // 2
    return [(hx - 1, hy - 1), (hx, hy - 1), (hx - 1, hy), (hx, hy)]


def _good_sell(item, price):
    b = BASE_PRICE.get(item, 1)
    if item in PREMIUM:
        return price >= 0.5 * b
    return True


def agent(obs):
    player = obs["player"]
    me = obs["farms"][player]
    priv = obs["private"]
    day = obs["day"]
    money = me["money"]
    tiles = me["tiles"]
    seeds = priv["seeds"]
    shed = priv["shed"]
    prices = obs["market"]["prices"]
    nq = len(me["unlocked_quadrants"])
    rows = len(tiles)
    cols = len(tiles[0]) if rows else 0
    invs = priv.get("inventories", [])
    n_units = 1 + len(me.get("hands", []))

    # --- classify every tile once ---
    harvestable, thirsty, empty, weeds = [], [], [], []
    feed_needed, care_needed, fert_ready, animal_harvest = [], [], [], []
    empty_coops, empty_pastures = [], []
    animals = 0
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
                elif k in ("COOP", "PASTURE"):
                    if t.get("animal"):
                        animals += 1
                        if not t.get("fed_today", False):
                            feed_needed.append((x, y))
                        if not t.get("cared_today", False):
                            care_needed.append((x, y))
                        if t.get("fertilizer_available"):
                            fert_ready.append((x, y))
                        if t.get("yield_units", 0) > 0:
                            animal_harvest.append((x, y))
                    elif k == "COOP":
                        empty_coops.append((x, y))
                    else:
                        empty_pastures.append((x, y))

    unplaced_cow = shed.get("COW", 0)
    unplaced_sheep = shed.get("SHEEP", 0)
    unplaced_goose = shed.get("GOOSE", 0)
    unplaced = unplaced_cow + unplaced_sheep + unplaced_goose
    wheat_shed = shed.get("WHEAT", 0)
    # wheat carried by units (available for FEED right now)
    wheat_carried = sum(iv.get("WHEAT", 0) for iv in invs if isinstance(iv, dict))

    # =====================================================================
    # MARKET ORDERS (cap 10)
    # =====================================================================
    market = []

    # 1) FEED SUPPLY — keep enough wheat (shed + carried) to feed the herd for
    #    ~2 days. Wheat is cheap (~$20-45); a starved animal is a $400+ loss.
    if animals > 0:
        want = animals * 2 + 4
        have = wheat_shed + wheat_carried
        if have < want and money >= 150:
            market.append(["BUY_PRODUCT", "WHEAT", min(want - have, 15)])

    # 2) SELL everything sellable. Fertilizer is the #1 income stream; milk/wool
    #    the premium payoff. Keep a wheat feed reserve.
    for item, cnt in shed.items():
        if cnt <= 0 or item in ("COW", "SHEEP", "GOOSE"):
            continue
        if item == "WHEAT":
            reserve = animals * 2 + 4
            surplus = cnt - reserve
            if surplus > 0:
                market.append(["SELL", "WHEAT", surplus])
            continue
        price = prices.get(item, 0)
        if item == "FERTILIZER":
            market.append(["SELL", "FERTILIZER", cnt])
        elif item in PREMIUM:
            if _good_sell(item, price):
                market.append(["SELL", item, cnt])
        else:
            market.append(["SELL", item, cnt])

    # 3) HIRE labor — scale hands with the amount of work (herd + crops). Cost is
    #    fib(hires_today): 1,1,2,3,5,8,... resets daily. Cheap early, harder later.
    hires_today = me.get("hires_today", 0)
    work = animals * 2 + len(thirsty) + len(harvestable) + unplaced * 2
    target_hands = min(9, max(1, work // 3))
    if day < 2:
        target_hands = 1
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    if hires_today < target_hands and hires_today < len(fib):
        if money >= fib[hires_today] + 40:
            market.append(["HIRE"])

    # 4) BUY ANIMALS — grow the herd ONE at a time, only when the current herd is
    #    fully fed (nothing starving) and no animal is still waiting to be placed.
    #    This prevents the buy-and-starve bleed that killed every prior attempt.
    HERD_TARGET = 15
    herd_ok = (len(feed_needed) == 0) and (unplaced == 0)
    if animals + unplaced < HERD_TARGET and herd_ok and day >= 1:
        # need a spare unit-action to place it this/next turn
        if money >= ANIMAL_COST["COW"] + 300:
            market.append(["BUY_ANIMAL", "COW", 1])
        elif money >= ANIMAL_COST["GOOSE"] + 200 and animals < 3:
            market.append(["BUY_ANIMAL", "GOOSE", 1])

    # 5) SEEDS — small wheat supply to farm bootstrap cash / early feed.
    want_seed = 4 if day < 6 else 2
    if seeds.get("WHEAT", 0) < want_seed and money >= 60:
        market.append(["BUY_SEED", "WHEAT", want_seed])

    # 6) LAND — expand once home quadrant is full and we have a real herd + cash.
    if nq < 2 and money >= 4000 and len(empty) <= 2 and animals >= 8:
        market.append(["BUY_LAND"])

    market = market[:10]

    # =====================================================================
    # UNIT ROUTING. Each unit does the highest-value thing it can, else moves
    # toward the nearest work. Logistics: carry animals out of the shed to place
    # them; carry a WHEAT buffer out to FEED the herd.
    # =====================================================================
    shed_tiles = _shed_tiles(rows, cols)
    claimed = set()
    picked_animal = {"COW": 0, "SHEEP": 0, "GOOSE": 0}
    picked_wheat = [0]  # mutable counter of wheat grabbed from shed this turn
    shed_pool = {"COW": unplaced_cow, "SHEEP": unplaced_sheep, "GOOSE": unplaced_goose}
    # how much wheat we still need units to be carrying to feed everyone unfed
    feed_deficit = [max(0, len(feed_needed) - wheat_carried)]

    def inv_of(uidx):
        return invs[uidx] if uidx < len(invs) and isinstance(invs[uidx], dict) else {}

    def carried_animal(uidx):
        iv = inv_of(uidx)
        for a in ("COW", "SHEEP", "GOOSE"):
            if iv.get(a, 0) > 0:
                return a
        return None

    def struct_for(a):
        return "PASTURE" if a in ("COW", "SHEEP") else "COOP"

    def assign(x, y, uidx):
        tile = tiles[y][x]
        here = (x, y)
        iv = inv_of(uidx)
        carrying = carried_animal(uidx)
        my_wheat = iv.get("WHEAT", 0)

        # ---- carrying an animal: place it on a structure (build one if needed) ----
        if carrying:
            need = struct_for(carrying)
            if isinstance(tile, dict) and tile.get("kind") == need and not tile.get("animal") and here not in claimed:
                claimed.add(here)
                return ["PLACE", carrying, 1]
            pool = empty_pastures if need == "PASTURE" else empty_coops
            tgt = _nearest(x, y, [p for p in pool if p not in claimed])
            if tgt:
                claimed.add(tgt)
                mv = _move_toward(x, y, tgt[0], tgt[1])
                if mv:
                    return mv
            if tile is None and here not in claimed:
                claimed.add(here)
                return ["BUILD_PASTURE"] if need == "PASTURE" else ["BUILD_COOP"]
            tgt = _nearest(x, y, [p for p in empty if p not in claimed])
            if tgt:
                claimed.add(tgt)
                mv = _move_toward(x, y, tgt[0], tgt[1])
                if mv:
                    return mv
            return ["PASS"]

        # ---- feed the animal we're standing on (if we carry wheat) ----
        if here in feed_needed and my_wheat > 0:
            return ["FEED"]
        # ---- other upkeep/production on the current tile ----
        if here in thirsty:
            return ["WATER"]
        if here in animal_harvest:
            return ["HARVEST"]
        if here in fert_ready:
            return ["COLLECT_FERTILIZER"]
        if here in care_needed:
            return ["CARE"]
        if here in harvestable:
            return ["HARVEST"]
        if here in weeds:
            return ["DIG"]

        # ---- ferry an animal out of the shed ----
        waiting = None
        for a in ("COW", "SHEEP", "GOOSE"):
            if shed_pool[a] - picked_animal[a] > 0:
                waiting = a
                break
        if waiting:
            if here in shed_tiles:
                picked_animal[waiting] += 1
                return ["PICKUP", waiting, 1]
            tgt = _nearest(x, y, shed_tiles)
            if tgt:
                mv = _move_toward(x, y, tgt[0], tgt[1])
                if mv:
                    return mv

        # ---- if animals need feeding and we have no wheat, grab a buffer ----
        if feed_deficit[0] > 0 and my_wheat == 0:
            avail = wheat_shed - picked_wheat[0]
            if avail > 0:
                if here in shed_tiles:
                    grab = min(avail, max(2, len(feed_needed)))
                    picked_wheat[0] += grab
                    feed_deficit[0] = max(0, feed_deficit[0] - grab)
                    return ["PICKUP", "WHEAT", grab]
                tgt = _nearest(x, y, shed_tiles)
                if tgt:
                    mv = _move_toward(x, y, tgt[0], tgt[1])
                    if mv:
                        return mv

        # ---- if we carry wheat and an animal needs feed, go to it ----
        if my_wheat > 0 and feed_needed:
            tgt = _nearest(x, y, feed_needed)
            if tgt:
                mv = _move_toward(x, y, tgt[0], tgt[1])
                if mv:
                    return mv

        # ---- move toward the nearest production/upkeep work ----
        for pool in (thirsty, animal_harvest, fert_ready, care_needed, harvestable, weeds):
            tgt = _nearest(x, y, [p for p in pool if p not in claimed])
            if tgt:
                mv = _move_toward(x, y, tgt[0], tgt[1])
                if mv:
                    return mv

        # ---- fill empty tiles with wheat (bootstrap cash + feed supply) ----
        if tile is None and seeds.get("WHEAT", 0) > 0 and here not in claimed:
            claimed.add(here)
            return ["PLANT", "WHEAT"]
        if seeds.get("WHEAT", 0) > 0:
            tgt = _nearest(x, y, [p for p in empty if p not in claimed])
            if tgt:
                claimed.add(tgt)
                mv = _move_toward(x, y, tgt[0], tgt[1])
                if mv:
                    return mv
        return ["PASS"]

    fx, fy = me["farmer"]
    farmer = assign(fx, fy, 0)
    hands = [assign(hx, hy, i + 1) for i, (hx, hy) in enumerate(me.get("hands", []))]
    return {"farmer": farmer, "hands": hands, "market": market}
