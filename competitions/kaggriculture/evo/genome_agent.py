"""Parameterized ("genome-driven") Kaggriculture agent for the GA.

The agent's behavior is controlled by a dict of numeric GENES. The GA evolves
these genes; this module turns a genome into an agent callable. Kept dependency-
free so a winning genome can be baked into a standalone main.py for submission.

Genes (all floats unless noted), with sane defaults:
  wheat_seed_target   how many wheat seeds to keep on hand
  carrot_gate         money above which we also grow carrots (big -> never)
  tomato_gate         money above which we also grow tomatoes
  melon_gate          money above which we also grow melons
  land2_gate          money to buy the 2nd quadrant
  land3_gate          money to buy the 3rd
  land4_gate          money to buy the 4th
  animal_gate         money above which we invest in a goose (coop + bird)
  cow_gate            money above which we invest in a cow (pasture + cow)
  sell_frac           fraction of a stack to sell per turn (throttle gluts)
  sell_min_price      don't sell a unit below this price (hold instead)
  hoard_days          early days to withhold selling (let prices/inventory build)
  fertilize           >0.5 -> fertilize crops in their bonus window
  care_animals        >0.5 -> CARE for animals when fed
  hire_gate           money above which we hire a farm hand each day
  plant_priority      0..1 blends toward higher-value crops when multiple gated in
"""
from __future__ import annotations

CROP_FIRST_YIELD = {"WHEAT": 2, "CARROT": 4, "TOMATO": 5, "STRAWBERRY": 6, "MELON": 8}
CROP_SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}

DEFAULT_GENES = {
    "wheat_seed_target": 5.0,
    "carrot_gate": 600.0,
    "tomato_gate": 1500.0,
    "melon_gate": 1200.0,
    "land2_gate": 2500.0,
    "land3_gate": 6000.0,
    "land4_gate": 12000.0,
    "animal_gate": 1500.0,
    "cow_gate": 4000.0,
    "sell_frac": 0.5,
    "sell_min_price": 5.0,
    "hoard_days": 1.0,
    "fertilize": 0.0,
    "care_animals": 0.0,
    "hire_gate": 100000.0,
    "plant_priority": 0.5,
}

# Search bounds for each gene (min, max) — used by the GA for init + mutation.
GENE_BOUNDS = {
    "wheat_seed_target": (1, 15),
    "carrot_gate": (0, 5000),
    "tomato_gate": (0, 6000),
    "melon_gate": (0, 6000),
    "land2_gate": (1000, 8000),
    "land3_gate": (3000, 15000),
    "land4_gate": (7000, 25000),
    "animal_gate": (400, 8000),
    "cow_gate": (900, 15000),
    "sell_frac": (0.1, 1.0),
    "sell_min_price": (1, 60),
    "hoard_days": (0, 8),
    "fertilize": (0, 1),
    "care_animals": (0, 1),
    "hire_gate": (500, 100000),
    "plant_priority": (0, 1),
}


def make_agent(genes: dict):
    """Return an `agent(obs)` callable driven by `genes` (missing keys -> defaults)."""
    g = dict(DEFAULT_GENES)
    g.update(genes or {})

    def _which_crops(money):
        """Ordered list of crops we're willing to plant at this money level."""
        allowed = ["WHEAT"]
        if money >= g["carrot_gate"]:
            allowed.append("CARROT")
        if money >= g["tomato_gate"]:
            allowed.append("TOMATO")
        if money >= g["melon_gate"]:
            allowed.append("MELON")
        # plant_priority high -> prefer higher-value crops first
        value = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "MELON": 250}
        allowed.sort(key=lambda c: value[c], reverse=(g["plant_priority"] >= 0.5))
        return allowed

    def _decide(tile, tiles, fx, fy, seeds, day, money):
        # animal structures
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
        # plants
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
            for crop in _which_crops(money):
                if seeds.get(crop, 0) > 0:
                    return ["PLANT", crop]
        return ["PASS"]

    def agent(obs):
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
        # --- selling: throttle to avoid crashing prices; hold cheap/early ---
        if day >= g["hoard_days"]:
            for item, count in shed.items():
                if item == "FERTILIZER" or count <= 0:
                    continue
                price = prices.get(item, 0)
                if price < g["sell_min_price"]:
                    continue
                qty = max(1, int(count * g["sell_frac"]))
                market.append(["SELL", item, qty])
        # fertilizer: sell freely (it's a byproduct)
        if shed.get("FERTILIZER", 0) > 0:
            market.append(["SELL", "FERTILIZER", shed["FERTILIZER"]])

        # --- buying seeds ---
        if seeds.get("WHEAT", 0) < g["wheat_seed_target"] and money >= 100:
            market.append(["BUY_SEED", "WHEAT", min(10, int(money // 10))])
        if money >= g["carrot_gate"] and seeds.get("CARROT", 0) < 3:
            market.append(["BUY_SEED", "CARROT", 3])
        if money >= g["tomato_gate"] and seeds.get("TOMATO", 0) < 2:
            market.append(["BUY_SEED", "TOMATO", 2])
        if money >= g["melon_gate"] and seeds.get("MELON", 0) < 2:
            market.append(["BUY_SEED", "MELON", 2])

        # --- animals: buy wheat to feed, coop/pasture + animal ---
        want_goose = money >= g["animal_gate"]
        want_cow = money >= g["cow_gate"]
        if want_goose and shed.get("GOOSE", 0) == 0 and money >= 320:
            market.append(["BUY_ANIMAL", "GOOSE", 1])
        if want_cow and shed.get("COW", 0) == 0 and money >= 420:
            market.append(["BUY_ANIMAL", "COW", 1])
        # keep some wheat as feed if we own animals
        owns_animal = shed.get("GOOSE", 0) or shed.get("COW", 0)

        # --- land ---
        nq = len(unlocked)
        if nq < 2 and money >= g["land2_gate"]:
            market.append(["BUY_LAND"])
        elif nq < 3 and money >= g["land3_gate"]:
            market.append(["BUY_LAND"])
        elif nq < 4 and money >= g["land4_gate"]:
            market.append(["BUY_LAND"])

        # --- hire ---
        if money >= g["hire_gate"] and me.get("hires_today", 0) < 2:
            market.append(["HIRE"])

        market = market[:10]

        # --- unit actions: farmer + hands ---
        # place a bought animal if standing on a matching empty structure
        farmer_tile = tiles[fy][fx]
        farmer_action = _place_or_build(farmer_tile, shed, priv, fx, fy, tiles, seeds, day, money, _decide, g)

        hands = []
        for (hx, hy) in me.get("hands", []):
            hands.append(_decide(tiles[hy][hx], tiles, hx, hy, seeds, day, money))

        return {"farmer": farmer_action, "hands": hands, "market": market}

    return agent


def _place_or_build(tile, shed, priv, fx, fy, tiles, seeds, day, money, decide, g):
    # If we own an animal but no structure here, build one on an empty tile.
    owns_goose = shed.get("GOOSE", 0) > 0
    owns_cow = shed.get("COW", 0) > 0
    if tile is None and (owns_goose or owns_cow):
        return ["BUILD_COOP"] if owns_goose else ["BUILD_PASTURE"]
    if isinstance(tile, dict) and tile.get("kind") == "COOP" and not tile.get("animal") and owns_goose:
        return ["PLACE", "GOOSE", 1]
    if isinstance(tile, dict) and tile.get("kind") == "PASTURE" and not tile.get("animal") and owns_cow:
        return ["PLACE", "COW", 1]
    return decide(tile, tiles, fx, fy, seeds, day, money)
