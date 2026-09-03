# Kaggriculture agent — GA-evolved genome (run 1, gen 10 best).
#
# Self-contained: the winning genome is hardcoded below, so this file needs no
# external data. Evolved by evo/ga.py over 6,624 self-play games vs `starter`;
# this genome scored ~$3,994 avg money at 100% winrate vs starter.
#
# LOADER RULE (kaggle-environments): the runner selects the LAST callable defined
# at module scope. So every helper is defined ABOVE, and `agent()` is LAST.

GENES = {
    "wheat_seed_target": 5.351399153338243,
    "carrot_gate": 1360.176406325152,
    "tomato_gate": 6000,
    "melon_gate": 5097.811470328456,
    "land2_gate": 8000,
    "land3_gate": 10951.337487587725,
    "land4_gate": 25000,
    "animal_gate": 8000,
    "cow_gate": 5746.118033054801,
    "sell_frac": 0.4366451997749106,
    "sell_min_price": 43.431414641947114,
    "hoard_days": 4.192730867116591,
    "fertilize": 0.0896318242520942,
    "care_animals": 0.4675343420700794,
    "hire_gate": 80875.01483255414,
    "plant_priority": 0.6985819173145595,
}

CROP_FIRST_YIELD = {"WHEAT": 2, "CARROT": 4, "TOMATO": 5, "STRAWBERRY": 6, "MELON": 8}
_CROP_VALUE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "MELON": 250}


def _which_crops(money, g):
    allowed = ["WHEAT"]
    if money >= g["carrot_gate"]:
        allowed.append("CARROT")
    if money >= g["tomato_gate"]:
        allowed.append("TOMATO")
    if money >= g["melon_gate"]:
        allowed.append("MELON")
    allowed.sort(key=lambda c: _CROP_VALUE[c], reverse=(g["plant_priority"] >= 0.5))
    return allowed


def _decide(tile, tiles, fx, fy, seeds, day, money, g):
    if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE"):
        if tile.get("animal"):
            if not tile.get("fed_today", False):
                return ["FEED"]
            if g["care_animals"] >= 0.5 and not tile.get("cared_today", False):
                return ["CARE"]
            if tile.get("fertilizer_available", 0):
                return ["COLLECT_FERTILIZER"]
            if tile.get("yield_units", 0) > 0:
                return ["HARVEST"]
        return ["PASS"]
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop = tile.get("crop", "WHEAT")
        age = day - tile.get("planted_day", day)
        if age >= CROP_FIRST_YIELD.get(crop, 2) and tile.get("yield_units", 0) > 0:
            return ["HARVEST"]
        if not tile.get("watered_today", False):
            return ["WATER"]
        if (g["fertilize"] >= 0.5 and tile.get("fertilized_until_day", -1) < day
                and crop in ("WHEAT", "CARROT", "MELON")):
            return ["FERTILIZE"]
        return ["PASS"]
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return ["DIG"]
    if tile is None:
        for crop in _which_crops(money, g):
            if seeds.get(crop, 0) > 0:
                return ["PLANT", crop]
    return ["PASS"]


def _place_or_build(tile, shed, fx, fy, tiles, seeds, day, money, g):
    owns_goose = shed.get("GOOSE", 0) > 0
    owns_cow = shed.get("COW", 0) > 0
    if tile is None and (owns_goose or owns_cow):
        return ["BUILD_COOP"] if owns_goose else ["BUILD_PASTURE"]
    if isinstance(tile, dict) and tile.get("kind") == "COOP" and not tile.get("animal") and owns_goose:
        return ["PLACE", "GOOSE", 1]
    if isinstance(tile, dict) and tile.get("kind") == "PASTURE" and not tile.get("animal") and owns_cow:
        return ["PLACE", "COW", 1]
    return _decide(tile, tiles, fx, fy, seeds, day, money, g)


def agent(obs):
    """GA-evolved Kaggriculture agent. MUST be the last function defined."""
    g = GENES
    player = obs["player"]
    me = obs["farms"][player]
    priv = obs["private"]
    day = obs["day"]
    fx, fy = me["farmer"]
    tiles = me["tiles"]
    money = me["money"]
    seeds = priv["seeds"]
    shed = priv["shed"]
    prices = obs["market"]["prices"]
    unlocked = me["unlocked_quadrants"]

    market = []
    # selling: throttle to avoid crashing prices; hold cheap/early
    if day >= g["hoard_days"]:
        for item, count in shed.items():
            if item == "FERTILIZER" or count <= 0:
                continue
            if prices.get(item, 0) < g["sell_min_price"]:
                continue
            market.append(["SELL", item, max(1, int(count * g["sell_frac"]))])
    if shed.get("FERTILIZER", 0) > 0:
        market.append(["SELL", "FERTILIZER", shed["FERTILIZER"]])

    # buying seeds
    if seeds.get("WHEAT", 0) < g["wheat_seed_target"] and money >= 100:
        market.append(["BUY_SEED", "WHEAT", min(10, int(money // 10))])
    if money >= g["carrot_gate"] and seeds.get("CARROT", 0) < 3:
        market.append(["BUY_SEED", "CARROT", 3])
    if money >= g["tomato_gate"] and seeds.get("TOMATO", 0) < 2:
        market.append(["BUY_SEED", "TOMATO", 2])
    if money >= g["melon_gate"] and seeds.get("MELON", 0) < 2:
        market.append(["BUY_SEED", "MELON", 2])

    # animals
    if money >= g["animal_gate"] and shed.get("GOOSE", 0) == 0 and money >= 320:
        market.append(["BUY_ANIMAL", "GOOSE", 1])
    if money >= g["cow_gate"] and shed.get("COW", 0) == 0 and money >= 420:
        market.append(["BUY_ANIMAL", "COW", 1])

    # land
    nq = len(unlocked)
    if nq < 2 and money >= g["land2_gate"]:
        market.append(["BUY_LAND"])
    elif nq < 3 and money >= g["land3_gate"]:
        market.append(["BUY_LAND"])
    elif nq < 4 and money >= g["land4_gate"]:
        market.append(["BUY_LAND"])

    # hire
    if money >= g["hire_gate"] and me.get("hires_today", 0) < 2:
        market.append(["HIRE"])

    market = market[:10]

    farmer_action = _place_or_build(tiles[fy][fx], shed, fx, fy, tiles, seeds, day, money, g)
    hands = []
    for (hx, hy) in me.get("hands", []):
        hands.append(_decide(tiles[hy][hx], tiles, hx, hy, seeds, day, money, g))

    return {"farmer": farmer_action, "hands": hands, "market": market}
