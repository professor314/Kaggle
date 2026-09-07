"""Parameterized FARM-OPERATOR agent (the GA substrate).

Keeps the winning routing logic (units MOVE to the nearest tile that needs work:
water > harvest > dig > plant) but exposes the real decision knobs as GENES so a
small-tweak GA can tune them AND layer in economics (melon, animals) on top of a
farm that actually operates.

Genes:
  seed_buffer        extra wheat seeds to keep beyond empty-tile count
  land_gate          money needed to buy the next land quadrant
  land_surplus       only expand when empty tiles <= this (don't outrun upkeep)
  melon_gate         money above which some tiles grow melon instead of wheat
  melon_frac         fraction of empty tiles to devote to melon (0..1)
  melon_sell_price   only sell melon at/above this price (avoid the $1 glut floor)
  animal_gate        money above which we buy ONE goose (perpetual egg income)
  hire_gate          money above which we hire a hand (more actions) when full
  sell_wheat_min     min wheat price to sell (else hold)
  harvest_first      >0.5 prioritise harvest over watering when both pending
"""
from __future__ import annotations

CROP_FIRST_YIELD = {"WHEAT": 2, "CARROT": 4, "TOMATO": 5, "STRAWBERRY": 6, "MELON": 8}

DEFAULT_OP_GENES = {
    "seed_buffer": 2.0,
    "land_gate": 4000.0,
    "land_surplus": 2.0,
    "melon_gate": 100000.0,   # default: never (pure wheat operator = our proven winner)
    "melon_frac": 0.0,
    "melon_sell_price": 120.0,
    "animal_gate": 100000.0,  # default: never
    "hire_gate": 100000.0,    # default: never
    "sell_wheat_min": 1.0,
    "harvest_first": 0.0,
}

OP_GENE_BOUNDS = {
    "seed_buffer": (0, 6),
    "land_gate": (1500, 9000),
    "land_surplus": (0, 8),
    "melon_gate": (300, 8000),
    "melon_frac": (0.0, 0.6),
    "melon_sell_price": (60, 240),
    "animal_gate": (700, 8000),
    "hire_gate": (1000, 12000),
    "sell_wheat_min": (1, 20),
    "harvest_first": (0, 1),
}


def _move_toward(fx, fy, tx, ty):
    dx, dy = tx - fx, ty - fy
    if abs(dx) >= abs(dy):
        return ["EAST"] if dx > 0 else (["WEST"] if dx < 0 else None)
    return ["SOUTH"] if dy > 0 else (["NORTH"] if dy < 0 else None)


def _nearest(fx, fy, targets):
    if not targets:
        return None
    return min(targets, key=lambda p: abs(p[0] - fx) + abs(p[1] - fy))


def make_operator(genes: dict):
    g = dict(DEFAULT_OP_GENES)
    g.update(genes or {})

    def agent(obs):
        me = obs["farms"][obs["player"]]
        priv = obs["private"]
        day = obs["day"]; money = me["money"]
        tiles = me["tiles"]; seeds = priv["seeds"]; shed = priv["shed"]
        prices = obs["market"]["prices"]; nq = len(me["unlocked_quadrants"])
        rows = len(tiles); cols = len(tiles[0]) if rows else 0

        harvestable, thirsty, empty, weeds = [], [], [], []
        coops_empty, pastures_empty = [], []
        n_animals = 0
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
                    elif k == "COOP":
                        if t.get("animal"):
                            n_animals += 1
                            if not t.get("fed_today", False):
                                thirsty.append((x, y))   # treat "feed" as upkeep work
                        else:
                            coops_empty.append((x, y))

        # --- market ---
        want_seed = min(len(empty) + int(g["seed_buffer"]), 12)
        market = []
        if seeds.get("WHEAT", 0) < want_seed and money >= 100:
            market.append(["BUY_SEED", "WHEAT", min(want_seed, int(money // 10))])
        if money >= g["melon_gate"] and g["melon_frac"] > 0 and seeds.get("MELON", 0) < 2:
            market.append(["BUY_SEED", "MELON", 2])
        # animals: buy one goose (needs a coop built + wheat feed)
        if money >= g["animal_gate"] and shed.get("GOOSE", 0) == 0 and n_animals == 0 and money >= 700:
            market.append(["BUY_ANIMAL", "GOOSE", 1])
        if n_animals and shed.get("WHEAT", 0) < 3 and money >= 150:
            market.append(["BUY_PRODUCT", "WHEAT", 3])
        # sells
        for item, cnt in shed.items():
            if cnt <= 0:
                continue
            if item == "FERTILIZER":
                market.append(["SELL", "FERTILIZER", cnt]); continue
            if item == "MELON":
                if prices.get("MELON", 0) >= g["melon_sell_price"]:
                    market.append(["SELL", "MELON", min(2, cnt)])
            elif item == "WHEAT":
                if prices.get("WHEAT", 0) >= g["sell_wheat_min"]:
                    market.append(["SELL", "WHEAT", cnt])
            else:
                market.append(["SELL", item, cnt])
        if nq < 2 and money >= g["land_gate"] and len(empty) <= g["land_surplus"]:
            market.append(["BUY_LAND"])
        if money >= g["hire_gate"] and me.get("hires_today", 0) < 2 and len(empty) <= g["land_surplus"]:
            market.append(["HIRE"])
        market = market[:10]

        # --- routing: units go to the nearest work ---
        claimed = set()
        want_melon = money >= g["melon_gate"] and g["melon_frac"] > 0 and seeds.get("MELON", 0) > 0

        def assign(x, y):
            tile = tiles[y][x]
            # standing on actionable work
            if (x, y) in thirsty:
                t = tiles[y][x]
                if isinstance(t, dict) and t.get("kind") == "COOP":
                    return ["FEED"]
                return ["WATER"]
            if (x, y) in harvestable:
                return ["HARVEST"]
            if (x, y) in weeds:
                return ["DIG"]
            if tile is None and (x, y) not in claimed:
                if want_melon and seeds.get("MELON", 0) > 0:
                    claimed.add((x, y)); return ["PLANT", "MELON"]
                if seeds.get("WHEAT", 0) > 0:
                    claimed.add((x, y)); return ["PLANT", "WHEAT"]
                if shed.get("GOOSE", 0):
                    return ["BUILD_COOP"]
            if isinstance(tile, dict) and tile.get("kind") == "COOP" and not tile.get("animal") and shed.get("GOOSE", 0):
                return ["PLACE", "GOOSE", 1]
            # move toward nearest work
            order = (harvestable, thirsty, weeds, empty) if g["harvest_first"] >= 0.5 \
                else (thirsty, harvestable, weeds, empty)
            for pool in order:
                tgt = _nearest(x, y, [p for p in pool if p not in claimed])
                if tgt:
                    if pool is empty and seeds.get("WHEAT", 0) <= 0 and not want_melon:
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

    return agent
