"""
Kaggriculture Agent — Greedy Farming Baseline
==============================================
A simple strategy agent that follows a greedy wheat-farming loop:
1. Buy wheat seeds when low
2. Move farmer to available tiles
3. Plant wheat on empty tiles
4. Water planted crops daily
5. Harvest when ready
6. Sell all produce immediately
7. Scale up to better crops as money grows

The agent prioritizes actions based on what's immediately available
on the farmer's current tile, with market orders handled separately.
"""


def agent(obs, cfg=None):
    """Main agent entry point for Kaggriculture."""
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    day = obs["day"]
    hour = obs["hour"]
    market_state = obs["market"]

    fx, fy = me["farmer"]
    tiles = me["tiles"]
    tile = tiles[fy][fx]
    money = me["money"]
    seeds = private["seeds"]
    shed = private["shed"]
    unlocked = me["unlocked_quadrants"]

    # --- Market orders (independent of farmer position) ---
    market = []

    # Sell everything in the shed first
    for item, count in shed.items():
        if count > 0 and item != "FERTILIZER":
            market.append(["SELL", item, count])

    # Buy seeds strategy: wheat is cheapest and fastest
    wheat_seeds = seeds.get("WHEAT", 0)
    if wheat_seeds < 5 and money >= 100:
        buy_count = min(10, money // 10)  # Wheat seeds cost $10 each
        if buy_count > 0:
            market.append(["BUY_SEED", "WHEAT", buy_count])

    # As money grows, diversify to better crops
    if money >= 500:
        carrot_seeds = seeds.get("CARROT", 0)
        if carrot_seeds < 3:
            market.append(["BUY_SEED", "CARROT", 3])

    if money >= 1500:
        tomato_seeds = seeds.get("TOMATO", 0)
        if tomato_seeds < 2:
            market.append(["BUY_SEED", "TOMATO", 2])

    # Buy land when we can afford it and have unlocked fewer quadrants
    if money >= 1500 and len(unlocked) < 4:
        market.append(["BUY_LAND"])

    # Limit market orders to avoid exceeding cap
    market = market[:10]

    # --- Farmer action ---
    farmer_action = decide_farmer_action(tile, tiles, fx, fy, seeds, day, unlocked)

    # --- Farm hands ---
    hands_actions = []
    for i, hand_pos in enumerate(me.get("hands", [])):
        hx, hy = hand_pos
        hand_tile = tiles[hy][hx]
        hand_action = decide_farmer_action(hand_tile, tiles, hx, hy, seeds, day, unlocked)
        hands_actions.append(hand_action)

    return {
        "farmer": farmer_action,
        "hands": hands_actions,
        "market": market,
    }


def decide_farmer_action(tile, tiles, fx, fy, seeds, day, unlocked):
    """Decide what a farmer/hand should do on their current tile."""

    # If standing on a plant
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop_age = day - tile.get("planted_day", day)

        # Harvest if ready (wheat first_yield_day = 2, carrot = 4, tomato = 5)
        crop = tile.get("crop", "WHEAT")
        first_yield = get_first_yield_day(crop)

        if crop_age >= first_yield and tile.get("yield_units", 0) > 0:
            return ["HARVEST"]

        # Water if not watered today
        if not tile.get("watered_today", False):
            return ["WATER"]

        # Already watered and not ready to harvest — move to find work
        target = find_nearest_work(tiles, fx, fy, seeds, day, unlocked)
        if target:
            return move_toward(fx, fy, target[0], target[1])
        return ["PASS"]

    # If standing on a weed, dig it
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return ["DIG"]

    # If standing on an animal structure
    if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE"):
        # Feed if not fed today
        if tile.get("animal") and not tile.get("fed_today", False):
            return ["FEED"]
        # Collect fertilizer if available
        if tile.get("fertilizer_available", 0) > 0:
            return ["COLLECT_FERTILIZER"]
        # Move elsewhere
        target = find_nearest_work(tiles, fx, fy, seeds, day, unlocked)
        if target:
            return move_toward(fx, fy, target[0], target[1])
        return ["PASS"]

    # If on an empty tile, plant if we have seeds
    if tile is None:
        # Prefer better seeds if available
        for crop in ["TOMATO", "CARROT", "WHEAT"]:
            if seeds.get(crop, 0) > 0:
                return ["PLANT", crop]

    # No action possible here — find somewhere useful to go
    target = find_nearest_work(tiles, fx, fy, seeds, day, unlocked)
    if target:
        return move_toward(fx, fy, target[0], target[1])

    return ["PASS"]


def get_first_yield_day(crop):
    """Return the first day a crop can be harvested."""
    yield_days = {
        "WHEAT": 2,
        "CARROT": 4,
        "TOMATO": 5,
        "STRAWBERRY": 6,
        "MELON": 8,
    }
    return yield_days.get(crop, 2)


def find_nearest_work(tiles, fx, fy, seeds, day, unlocked):
    """Find the nearest tile that needs attention."""
    best_target = None
    best_dist = float('inf')

    rows = len(tiles)
    cols = len(tiles[0]) if rows > 0 else 0

    has_seeds = any(seeds.get(crop, 0) > 0 for crop in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"])

    for y in range(rows):
        for x in range(cols):
            tile = tiles[y][x]

            # Skip locked tiles for work (but can pass through)
            if tile == "LOCKED":
                continue

            dist = abs(x - fx) + abs(y - fy)
            if dist == 0:
                continue  # Already here

            priority = dist  # Lower is better

            # Unwatered plants are high priority
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                if not tile.get("watered_today", False):
                    priority = dist - 100  # Very high priority
                crop_age = day - tile.get("planted_day", day)
                crop = tile.get("crop", "WHEAT")
                if crop_age >= get_first_yield_day(crop) and tile.get("yield_units", 0) > 0:
                    priority = dist - 200  # Harvest is highest priority

            # Empty tiles where we can plant
            elif tile is None and has_seeds:
                priority = dist - 50

            # Weeds should be cleared
            elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                priority = dist - 20

            # Animals that need feeding
            elif isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE"):
                if tile.get("animal") and not tile.get("fed_today", False):
                    priority = dist - 80

            if priority < best_dist:
                best_dist = priority
                best_target = (x, y)

    return best_target


def move_toward(fx, fy, tx, ty):
    """Return a movement action toward the target."""
    dx = tx - fx
    dy = ty - fy

    # Prefer moving along the longer axis first
    if abs(dx) >= abs(dy):
        if dx > 0:
            return ["EAST"]
        elif dx < 0:
            return ["WEST"]

    if dy > 0:
        return ["SOUTH"]
    elif dy < 0:
        return ["NORTH"]

    # Already at target
    return ["PASS"]
