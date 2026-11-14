def agent(obs):
    """Kaggriculture greedy farming baseline agent."""
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    day = obs["day"]
    hour = obs["hour"]

    fx, fy = me["farmer"]
    tiles = me["tiles"]
    tile = tiles[fy][fx]
    money = me["money"]
    seeds = private["seeds"]
    shed = private["shed"]
    unlocked = me["unlocked_quadrants"]

    # --- Market orders ---
    market = []

    # Sell everything in the shed
    for item, count in shed.items():
        if count > 0 and item != "FERTILIZER":
            market.append(["SELL", item, count])

    # Buy wheat seeds
    wheat_seeds = seeds.get("WHEAT", 0)
    if wheat_seeds < 5 and money >= 100:
        buy_count = min(10, money // 10)
        if buy_count > 0:
            market.append(["BUY_SEED", "WHEAT", buy_count])

    # Diversify as money grows
    if money >= 500:
        if seeds.get("CARROT", 0) < 3:
            market.append(["BUY_SEED", "CARROT", 3])

    if money >= 1500:
        if seeds.get("TOMATO", 0) < 2:
            market.append(["BUY_SEED", "TOMATO", 2])

    # Buy land
    if money >= 1500 and len(unlocked) < 4:
        market.append(["BUY_LAND"])

    market = market[:10]

    # --- Farmer action ---
    farmer_action = _decide_action(tile, tiles, fx, fy, seeds, day, unlocked)

    # --- Hands ---
    hands_actions = []
    for hand_pos in me.get("hands", []):
        hx, hy = hand_pos
        hand_tile = tiles[hy][hx]
        hands_actions.append(_decide_action(hand_tile, tiles, hx, hy, seeds, day, unlocked))

    return {
        "farmer": farmer_action,
        "hands": hands_actions,
        "market": market,
    }


def _decide_action(tile, tiles, fx, fy, seeds, day, unlocked):
    """Decide action for a unit on a given tile."""
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop = tile.get("crop", "WHEAT")
        crop_age = day - tile.get("planted_day", day)
        first_yield = _first_yield_day(crop)

        if crop_age >= first_yield and tile.get("yield_units", 0) > 0:
            return ["HARVEST"]
        if not tile.get("watered_today", False):
            return ["WATER"]

        target = _find_work(tiles, fx, fy, seeds, day, unlocked)
        if target:
            return _move_toward(fx, fy, target[0], target[1])
        return ["PASS"]

    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return ["DIG"]

    if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE"):
        if tile.get("animal") and not tile.get("fed_today", False):
            return ["FEED"]
        if tile.get("fertilizer_available", 0) > 0:
            return ["COLLECT_FERTILIZER"]
        target = _find_work(tiles, fx, fy, seeds, day, unlocked)
        if target:
            return _move_toward(fx, fy, target[0], target[1])
        return ["PASS"]

    if tile is None:
        for crop in ["TOMATO", "CARROT", "WHEAT"]:
            if seeds.get(crop, 0) > 0:
                return ["PLANT", crop]

    target = _find_work(tiles, fx, fy, seeds, day, unlocked)
    if target:
        return _move_toward(fx, fy, target[0], target[1])
    return ["PASS"]


def _first_yield_day(crop):
    return {"WHEAT": 2, "CARROT": 4, "TOMATO": 5, "STRAWBERRY": 6, "MELON": 8}.get(crop, 2)


def _find_work(tiles, fx, fy, seeds, day, unlocked):
    """Find nearest tile needing attention."""
    best = None
    best_pri = float('inf')
    rows = len(tiles)
    cols = len(tiles[0]) if rows > 0 else 0
    has_seeds = any(seeds.get(c, 0) > 0 for c in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"])

    for y in range(rows):
        for x in range(cols):
            t = tiles[y][x]
            if t == "LOCKED":
                continue
            dist = abs(x - fx) + abs(y - fy)
            if dist == 0:
                continue
            pri = dist

            if isinstance(t, dict) and t.get("kind") == "PLANT":
                if not t.get("watered_today", False):
                    pri = dist - 100
                crop_age = day - t.get("planted_day", day)
                crop = t.get("crop", "WHEAT")
                if crop_age >= _first_yield_day(crop) and t.get("yield_units", 0) > 0:
                    pri = dist - 200
            elif t is None and has_seeds:
                pri = dist - 50
            elif isinstance(t, dict) and t.get("kind") == "WEED":
                pri = dist - 20
            elif isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE"):
                if t.get("animal") and not t.get("fed_today", False):
                    pri = dist - 80

            if pri < best_pri:
                best_pri = pri
                best = (x, y)
    return best


def _move_toward(fx, fy, tx, ty):
    dx = tx - fx
    dy = ty - fy
    if abs(dx) >= abs(dy):
        if dx > 0:
            return ["EAST"]
        elif dx < 0:
            return ["WEST"]
    if dy > 0:
        return ["SOUTH"]
    elif dy < 0:
        return ["NORTH"]
    return ["PASS"]
