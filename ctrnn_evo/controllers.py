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


def gradient_follower(
    key: jax.Array,
    sensors: jnp.ndarray,
    state: WorldState,
    wcfg: WorldConfig,
) -> jnp.ndarray:
    """
    Move directly toward the nearest food hotspot at full speed.

    This controller has access to full WorldState (including exact hotspot
    positions) and is a validation tool only — not restricted to the sensors
    an evolved agent receives.

    Uses direct Euclidean distance rather than the food-field gradient to
    avoid float32 underflow at large distances from narrow hotspots.
    """
    diff      = state.hotspot_pos - state.agent_pos[None, :]   # [n_food, 2]
    distances = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))          # [n_food]
    nearest   = jnp.argmin(distances)
    direction = diff[nearest] / (distances[nearest] + 1e-8)    # unit vector [2]
    return direction
