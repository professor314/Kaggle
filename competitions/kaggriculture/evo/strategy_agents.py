"""Theory-driven hand-built Kaggriculture agents.

These are NOT genome-knob tunings — they encode real strategy logic derived from
the game's economics (see STORY.md Chapter 8):

- The market is a shared, exhaustible commons: prices fall as you sell, premium
  goods (melon/strawberry/milk/wool) crash to the $1 floor on gluts. So METER
  premium sales and sell into demand, don't dump.
- Animals are a PERPETUAL income engine: one-time cost -> milk/wool/eggs every
  interval for the rest of the season. The compounding asset.
- Actions/turn is the binding constraint: 720 turns, 1 action/unit. Hiring hands
  buys actions; buying land buys tiles. money -> land+hands -> production -> money.

Each agent is a plain function agent(obs) with the standard action dict.
Shared helpers do the tile bookkeeping so each strategy only expresses its policy.
"""
from __future__ import annotations

CROP_FIRST_YIELD = {"WHEAT": 2, "CARROT": 4, "TOMATO": 5, "STRAWBERRY": 6, "MELON": 8}
# base prices, for "is this a good price to sell?" decisions
BASE_PRICE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
              "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
PREMIUM = {"MELON", "STRAWBERRY", "MILK", "WOOL"}   # crash hard on gluts


def _me(obs):
    p = obs["player"]
    return obs["farms"][p], obs["private"]


def _empty_tiles(tiles, unlocked_only=True):
    out = []
    for y, row in enumerate(tiles):
        for x, t in enumerate(row):
            if t is None:
                out.append((x, y))
    return out


def _tend_current_tile(tile, day):
    """Universal 'keep what I have alive/harvested' policy for the standing unit."""
    if isinstance(tile, dict):
        k = tile.get("kind")
        if k in ("COOP", "PASTURE"):
            if tile.get("animal"):
                if not tile.get("fed_today", False):
                    return ["FEED"]
                if not tile.get("cared_today", False):
                    return ["CARE"]                       # bank the yield bonus
                if tile.get("fertilizer_available", 0):
                    return ["COLLECT_FERTILIZER"]
                if tile.get("yield_units", 0) > 0:
                    return ["HARVEST"]
            return ["PASS"]
        if k == "PLANT":
            crop = tile.get("crop", "WHEAT")
            age = day - tile.get("planted_day", day)
            if age >= CROP_FIRST_YIELD.get(crop, 2) and tile.get("yield_units", 0) > 0:
                return ["HARVEST"]
            if not tile.get("watered_today", False):
                return ["WATER"]
            return ["PASS"]
        if k == "WEED":
            return ["DIG"]
    return None


def _good_sell_price(item, price):
    """Sell premium goods only near/above base (avoid the $1-floor glut); staples freely."""
    b = BASE_PRICE.get(item, 1)
    if item in PREMIUM:
        return price >= 0.6 * b
    return price >= 0.4 * b


def _metered_sells(shed, prices, frac_premium=0.25, frac_staple=0.6):
    """Sell a trickle of premium goods, more of staples, only at decent prices."""
    orders = []
    for item, cnt in shed.items():
        if cnt <= 0:
            continue
        price = prices.get(item, 0)
        if item == "FERTILIZER":
            orders.append(["SELL", "FERTILIZER", cnt])       # byproduct, always sell
            continue
        if not _good_sell_price(item, price):
            continue
        frac = frac_premium if item in PREMIUM else frac_staple
        qty = max(1, int(cnt * frac))
        orders.append(["SELL", item, qty])
    return orders


# ============================================================================
# 1) ANIMAL ENGINE — rush land + livestock, hire hands, meter premium sales
# ============================================================================
def animal_engine(obs):
    me, priv = _me(obs)
    day = obs["day"]; money = me["money"]
    tiles = me["tiles"]; seeds = priv["seeds"]; shed = priv["shed"]
    prices = obs["market"]["prices"]; nq = len(me["unlocked_quadrants"])
    fx, fy = me["farmer"]

    market = []
    # keep wheat flowing: it's cash early AND animal feed forever
    if seeds.get("WHEAT", 0) < 6 and money >= 60:
        market.append(["BUY_SEED", "WHEAT", min(8, int(money // 10))])
    # aggressive reinvestment ladder: land, then animals
    if nq < 2 and money >= 2000:
        market.append(["BUY_LAND"])
    elif nq < 3 and money >= 5000:
        market.append(["BUY_LAND"])
    # animals = perpetual engine; buy as soon as we can sustainably feed them
    if money >= 900 and shed.get("COW", 0) == 0 and (shed.get("WHEAT", 0) + seeds.get("WHEAT", 0)) >= 2:
        market.append(["BUY_ANIMAL", "COW", 1])
    if money >= 1400 and shed.get("SHEEP", 0) == 0:
        market.append(["BUY_ANIMAL", "SHEEP", 1])
    if money >= 700 and shed.get("GOOSE", 0) == 0:
        market.append(["BUY_ANIMAL", "GOOSE", 1])
    # buy extra wheat as feed if we own animals and are short
    owns = shed.get("COW", 0) or shed.get("SHEEP", 0) or shed.get("GOOSE", 0)
    if owns and shed.get("WHEAT", 0) < 4 and money >= 200:
        market.append(["BUY_PRODUCT", "WHEAT", 3])
    # hire a hand once we have enough tiles/animals to keep it busy
    if money >= 1200 and me.get("hires_today", 0) < 2 and (nq >= 2 or owns):
        market.append(["HIRE"])
    market += _metered_sells(shed, prices)
    market = market[:10]

    # unit policy: place/build animals, else tend, else fill empty tiles with wheat
    def unit(x, y):
        tile = tiles[y][x]
        og, oc, osh = shed.get("GOOSE", 0), shed.get("COW", 0), shed.get("SHEEP", 0)
        if tile is None:
            if oc or osh:
                return ["BUILD_PASTURE"]
            if og:
                return ["BUILD_COOP"]
            if seeds.get("WHEAT", 0) > 0:
                return ["PLANT", "WHEAT"]
            return ["PASS"]
        if isinstance(tile, dict):
            if tile.get("kind") == "PASTURE" and not tile.get("animal"):
                if oc:
                    return ["PLACE", "COW", 1]
                if osh:
                    return ["PLACE", "SHEEP", 1]
            if tile.get("kind") == "COOP" and not tile.get("animal") and og:
                return ["PLACE", "GOOSE", 1]
        act = _tend_current_tile(tile, day)
        return act if act else ["PASS"]

    farmer = unit(fx, fy)
    hands = [unit(hx, hy) for hx, hy in me.get("hands", [])]
    return {"farmer": farmer, "hands": hands, "market": market}


# ============================================================================
# 2) MELON BARON — highest-value crop, sold in a trickle to keep price high
# ============================================================================
def melon_baron(obs):
    me, priv = _me(obs)
    day = obs["day"]; money = me["money"]
    tiles = me["tiles"]; seeds = priv["seeds"]; shed = priv["shed"]
    prices = obs["market"]["prices"]; nq = len(me["unlocked_quadrants"])
    fx, fy = me["farmer"]

    market = []
    if seeds.get("WHEAT", 0) < 3 and money >= 40:
        market.append(["BUY_SEED", "WHEAT", 3])
    if money >= 300 and seeds.get("MELON", 0) < 4:
        market.append(["BUY_SEED", "MELON", 3])
    if nq < 2 and money >= 3000:
        market.append(["BUY_LAND"])
    # melon crashes on gluts: sell at most 1-2 per turn, only at a strong price
    mprice = prices.get("MELON", 0)
    if shed.get("MELON", 0) > 0 and mprice >= 120:
        market.append(["SELL", "MELON", min(2, shed["MELON"])])
    # staples freely
    for item in ("WHEAT", "CARROT", "TOMATO"):
        if shed.get(item, 0) > 0 and _good_sell_price(item, prices.get(item, 0)):
            market.append(["SELL", item, shed[item]])
    if shed.get("FERTILIZER", 0) > 0:
        market.append(["SELL", "FERTILIZER", shed["FERTILIZER"]])
    market = market[:10]

    def unit(x, y):
        tile = tiles[y][x]
        if tile is None:
            if seeds.get("MELON", 0) > 0:
                return ["PLANT", "MELON"]
            if seeds.get("WHEAT", 0) > 0:
                return ["PLANT", "WHEAT"]
            return ["PASS"]
        # fertilize melon in its bonus window (doubles the watered-day bonus)
        if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "MELON":
            if tile.get("watered_today") and tile.get("fertilized_until_day", -1) < day:
                return ["FERTILIZE"]
        act = _tend_current_tile(tile, day)
        return act if act else ["PASS"]

    farmer = unit(fx, fy)
    hands = [unit(hx, hy) for hx, hy in me.get("hands", [])]
    return {"farmer": farmer, "hands": hands, "market": market}


# ============================================================================
# 3) MARKET MAKER — sell into town demand / scarcity, hold when cheap, diversified
# ============================================================================
def market_maker(obs):
    me, priv = _me(obs)
    day = obs["day"]; money = me["money"]
    tiles = me["tiles"]; seeds = priv["seeds"]; shed = priv["shed"]
    prices = obs["market"]["prices"]; inv = obs["market"]["inventory"]
    shops = obs.get("town", {}).get("unlocked_shops", [])
    nq = len(me["unlocked_quadrants"]); fx, fy = me["farmer"]

    # what does the town demand? sell those preferentially (their price holds up)
    demand = set()
    shop_demand = {
        "BAKERY": ["EGG", "WHEAT"], "PIZZA SHOP": ["MILK", "TOMATO", "WHEAT"],
        "BRUNCH SPOT": ["EGG", "WHEAT", "STRAWBERRY"], "YARN STORE": ["WOOL"],
        "ICE CREAM SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET CAFE": ["CARROT"],
        "SMOOTHIE SHOP": ["STRAWBERRY", "MILK"],
        "FARMERS MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
    }
    for s in shops:
        for d in shop_demand.get(str(s).upper(), []):
            demand.add(d)

    market = []
    if seeds.get("WHEAT", 0) < 4 and money >= 60:
        market.append(["BUY_SEED", "WHEAT", 4])
    if money >= 400 and seeds.get("CARROT", 0) < 3:
        market.append(["BUY_SEED", "CARROT", 3])
    if money >= 1000 and seeds.get("TOMATO", 0) < 2:
        market.append(["BUY_SEED", "TOMATO", 2])
    if nq < 2 and money >= 2500:
        market.append(["BUY_LAND"])
    if money >= 2000 and shed.get("GOOSE", 0) == 0:
        market.append(["BUY_ANIMAL", "GOOSE", 1])
    # sell: town-demanded goods first (price holds), then any good-priced staple;
    # hold premium unless price is strong
    for item, cnt in shed.items():
        if cnt <= 0 or item == "FERTILIZER":
            continue
        price = prices.get(item, 0)
        wanted = item in demand
        if wanted and _good_sell_price(item, price):
            market.append(["SELL", item, cnt])              # dump into demand
        elif _good_sell_price(item, price) and item not in PREMIUM:
            market.append(["SELL", item, max(1, cnt // 2)])
        elif item in PREMIUM and price >= 0.8 * BASE_PRICE.get(item, 1):
            market.append(["SELL", item, max(1, cnt // 3)])
    if shed.get("FERTILIZER", 0) > 0:
        market.append(["SELL", "FERTILIZER", shed["FERTILIZER"]])
    market = market[:10]

    def unit(x, y):
        tile = tiles[y][x]
        if tile is None:
            og = shed.get("GOOSE", 0)
            if og:
                return ["BUILD_COOP"]
            for crop in ("TOMATO", "CARROT", "WHEAT"):
                if seeds.get(crop, 0) > 0:
                    return ["PLANT", crop]
            return ["PASS"]
        if isinstance(tile, dict) and tile.get("kind") == "COOP" and not tile.get("animal") and shed.get("GOOSE", 0):
            return ["PLACE", "GOOSE", 1]
        act = _tend_current_tile(tile, day)
        return act if act else ["PASS"]

    farmer = unit(fx, fy)
    hands = [unit(hx, hy) for hx, hy in me.get("hands", [])]
    return {"farmer": farmer, "hands": hands, "market": market}


# ============================================================================
# 4) DISCIPLINED ENGINE — the corrected theory agent.
#    Core rule: MAINTENANCE BEFORE EXPANSION. Never own more than the current
#    action budget (farmer + hands) can feed/water each day. Hire BEFORE expanding.
#    A starved animal / weeded crop is a total loss, so upkeep always wins.
# ============================================================================
def _count_upkeep(tiles):
    """How many tiles need a daily action (unwatered plants + unfed animals)."""
    need = 0
    plants = animals = 0
    for row in tiles:
        for t in row:
            if isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT":
                    plants += 1
                    if not t.get("watered_today", False):
                        need += 1
                elif k in ("COOP", "PASTURE") and t.get("animal"):
                    animals += 1
                    if not t.get("fed_today", False):
                        need += 1
    return need, plants, animals


def disciplined_engine(obs):
    me, priv = _me(obs)
    day = obs["day"]; money = me["money"]
    tiles = me["tiles"]; seeds = priv["seeds"]; shed = priv["shed"]
    prices = obs["market"]["prices"]; nq = len(me["unlocked_quadrants"])
    fx, fy = me["farmer"]
    n_units = 1 + len(me.get("hands", []))
    need, plants, animals = _count_upkeep(tiles)
    owns_animal = animals > 0 or any(shed.get(a, 0) for a in ("COW", "SHEEP", "GOOSE"))

    # capacity headroom: can our units maintain everything AND do one more thing?
    # each unit does ~1 upkeep action/turn; we want slack before taking on more.
    at_capacity = (plants + animals) >= n_units * 6   # ~6 maintainable tiles/unit heuristic

    market = []
    # 1) FEED SUPPLY FIRST: if we own animals, guarantee wheat on hand to feed them
    if owns_animal and shed.get("WHEAT", 0) < 4 and money >= 150:
        market.append(["BUY_PRODUCT", "WHEAT", 4])
    # 2) seeds to keep planting (only staples we can reliably water)
    if seeds.get("WHEAT", 0) < 4 and money >= 60:
        market.append(["BUY_SEED", "WHEAT", 4])
    # 3) HIRE before expanding: more actions = more we can maintain
    if at_capacity and money >= 1500 and me.get("hires_today", 0) < 2:
        market.append(["HIRE"])
    # 4) EXPAND only when we have action headroom (not at capacity)
    if not at_capacity:
        if nq < 2 and money >= 2500:
            market.append(["BUY_LAND"])
        elif nq < 3 and money >= 6000:
            market.append(["BUY_LAND"])
        # one animal at a time, only if we can feed it and have a free unit-action
        if money >= 900 and shed.get("COW", 0) == 0 and shed.get("WHEAT", 0) >= 3 and animals < n_units:
            market.append(["BUY_ANIMAL", "COW", 1])
        elif money >= 700 and shed.get("GOOSE", 0) == 0 and shed.get("WHEAT", 0) >= 3 and animals < n_units:
            market.append(["BUY_ANIMAL", "GOOSE", 1])
    # 5) metered sells (keep prices up)
    market += _metered_sells(shed, prices)
    market = market[:10]

    def unit(x, y):
        tile = tiles[y][x]
        # upkeep of the current tile always comes first
        act = _tend_current_tile(tile, day)
        if act and act != ["PASS"]:
            return act
        # place a bought animal on a matching empty structure
        if isinstance(tile, dict):
            if tile.get("kind") == "PASTURE" and not tile.get("animal") and shed.get("COW", 0):
                return ["PLACE", "COW", 1]
            if tile.get("kind") == "COOP" and not tile.get("animal") and shed.get("GOOSE", 0):
                return ["PLACE", "GOOSE", 1]
        # empty tile: build a structure for a bought animal, else plant wheat
        if tile is None:
            if shed.get("COW", 0):
                return ["BUILD_PASTURE"]
            if shed.get("GOOSE", 0):
                return ["BUILD_COOP"]
            if seeds.get("WHEAT", 0) > 0 and not at_capacity:
                return ["PLANT", "WHEAT"]
        return ["PASS"]

    farmer = unit(fx, fy)
    hands = [unit(hx, hy) for hx, hy in me.get("hands", [])]
    return {"farmer": farmer, "hands": hands, "market": market}


# ============================================================================
# 5) FARM OPERATOR — operational competence first. Units MOVE to the nearest
#    tile that needs work (water > harvest > plant), so the farm stays full and
#    watered instead of sitting idle and weeding over. Pure high-velocity wheat
#    loop; expand land only from surplus. This targets the 20-43 idle tiles the
#    trace exposed as pure upside.
# ============================================================================
def _is_unlocked(tile):
    return tile != "LOCKED"


def _move_toward(fx, fy, tx, ty):
    dx, dy = tx - fx, ty - fy
    if abs(dx) >= abs(dy):
        return ["EAST"] if dx > 0 else (["WEST"] if dx < 0 else None)
    return ["SOUTH"] if dy > 0 else (["NORTH"] if dy < 0 else None)


def _nearest(fx, fy, targets):
    if not targets:
        return None
    return min(targets, key=lambda p: abs(p[0] - fx) + abs(p[1] - fy))


def farm_operator(obs):
    me, priv = _me(obs)
    day = obs["day"]; money = me["money"]
    tiles = me["tiles"]; seeds = priv["seeds"]; shed = priv["shed"]
    prices = obs["market"]["prices"]; nq = len(me["unlocked_quadrants"])
    rows = len(tiles); cols = len(tiles[0]) if rows else 0

    # classify tiles once
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

    # market: keep enough wheat seed to fill every empty tile, sell harvest freely
    want_seed = min(len(empty) + 2, 12)
    if seeds.get("WHEAT", 0) < want_seed and money >= 100:
        market = [["BUY_SEED", "WHEAT", min(want_seed, int(money // 10))]]
    else:
        market = []
    for item, cnt in shed.items():
        if cnt > 0 and item != "FERTILIZER":
            market.append(["SELL", item, cnt])            # wheat sells fine near base
        elif item == "FERTILIZER" and cnt > 0:
            market.append(["SELL", "FERTILIZER", cnt])
    # expand land only from clear surplus (don't starve the loop)
    if nq < 2 and money >= 4000 and len(empty) <= 2:
        market.append(["BUY_LAND"])
    market = market[:10]

    # assign each unit to the nearest useful action. Priority: WATER (save a crop)
    # > HARVEST (bank money) > DIG weed > PLANT empty. Units MOVE if not already there.
    claimed = set()

    def assign(x, y):
        tile = tiles[y][x]
        # already standing on actionable tile? do it.
        if (x, y) in thirsty:
            return ["WATER"]
        if (x, y) in harvestable:
            return ["HARVEST"]
        if (x, y) in weeds:
            return ["DIG"]
        if tile is None and seeds.get("WHEAT", 0) > 0 and (x, y) not in claimed:
            claimed.add((x, y))
            return ["PLANT", "WHEAT"]
        # otherwise move toward the nearest work (water first, then harvest, plant)
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


STRATEGIES = {
    "animal_engine": animal_engine,
    "melon_baron": melon_baron,
    "market_maker": market_maker,
    "disciplined_engine": disciplined_engine,
    "farm_operator": farm_operator,
}
