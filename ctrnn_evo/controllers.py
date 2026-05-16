from __future__ import annotations

import jax
import jax.numpy as jnp

from .world import WorldConfig, WorldState


def random_walk(
    key: jax.Array,
    sensors: jnp.ndarray,
    state: WorldState,
    wcfg: WorldConfig,
) -> jnp.ndarray:
    """
    Uniform random velocity each step.

    Baseline controller that should reliably starve — the expected energy
    gain from random movement is less than the metabolism + movement cost
    at the default WorldConfig parameters.
    """
    return jax.random.uniform(key, (2,), minval=-1.0, maxval=1.0)


def nearest_hotspot(
    key: jax.Array,
    sensors: jnp.ndarray,
    state: WorldState,
    wcfg: WorldConfig,
) -> jnp.ndarray:
    """
    Move directly toward the nearest food hotspot at full speed.

    Validation tool only — reads exact hotspot coordinates from WorldState,
    which evolved agents cannot access.  Sensors are ignored entirely.
    Its sole purpose is to confirm the world is survivable in principle
    before asking evolution to solve it.
    """
    diff      = state.hotspot_pos - state.agent_pos[None, :]   # [n_food, 2]
    distances = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))          # [n_food]
    nearest   = jnp.argmin(distances)
    direction = diff[nearest] / (distances[nearest] + 1e-8)    # unit vector [2]
    return direction
