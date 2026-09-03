"""Opponent pool for co-evolution (Tier 1.1).

These are the "exam" — the agents our evolving population must beat. They are
NOT evolved. They are: the built-in `starter` (by name), a couple of sharp
hand-written genome presets, and (added at runtime by the GA) our own past
champions (Hall of Fame).

Each preset is expressed as a genome so it reuses make_agent(). Two distinct,
plausible strategies so a candidate can't win by exploiting one weakness.
"""
from __future__ import annotations

from genome_agent import DEFAULT_GENES, make_agent


def _preset(**overrides):
    g = dict(DEFAULT_GENES)
    g.update(overrides)
    return g


# Aggressive expander: buy land + hire early, plant high-value crops, sell freely.
AGGRESSIVE_EXPAND = _preset(
    wheat_seed_target=8,
    carrot_gate=300, tomato_gate=900, melon_gate=1000,
    land2_gate=1200, land3_gate=3500, land4_gate=8000,
    animal_gate=900, cow_gate=2500,
    sell_frac=0.7, sell_min_price=3, hoard_days=0,
    fertilize=1.0, care_animals=1.0, hire_gate=1500, plant_priority=1.0,
)

# Market timer: patient, holds stock, sells only at good prices, staple-heavy.
MARKET_TIMER = _preset(
    wheat_seed_target=6,
    carrot_gate=1500, tomato_gate=3000, melon_gate=2500,
    land2_gate=3000, land3_gate=8000, land4_gate=16000,
    animal_gate=2500, cow_gate=6000,
    sell_frac=0.35, sell_min_price=30, hoard_days=6,
    fertilize=0.0, care_animals=0.0, hire_gate=100000, plant_priority=0.5,
)


def base_opponents():
    """The fixed part of the opponent pool.

    Returns a list of (name, spec) where spec is either the string "starter"
    (a built-in agent, passed by name to env.run) or a genome dict (turned into
    a callable inside the worker).
    """
    return [
        ("starter", "starter"),
        ("aggressive_expand", AGGRESSIVE_EXPAND),
        ("market_timer", MARKET_TIMER),
    ]


def resolve(spec):
    """Turn an opponent spec into something env.run accepts: a name or a callable."""
    if isinstance(spec, str):
        return spec                    # built-in agent name
    return make_agent(spec)            # genome dict -> callable
