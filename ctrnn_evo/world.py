from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp


# ── WorldConfig ───────────────────────────────────────────────────────────────

@dataclass
class WorldConfig:
    # Arena
    arena_size:    float = 100.0

    # Food hotspots
    n_food:        int   = 3
    hotspot_sigma: float = 5.0    # Gaussian spread (units)
    hotspot_drift: float = 0.3    # std of per-step random walk of hotspot centres

    # Energy economics
    init_energy:   float = 0.5
    metabolism:    float = 0.005  # passive drain per world step
    move_cost:     float = 0.001  # additional drain per unit of speed
    eat_rate:      float = 0.08   # energy gain = eat_rate * clipped_food_density
    max_energy:    float = 1.0

    # Agent physics
    max_speed:     float = 3.0    # units per world step

    # Episode
    episode_steps: int   = 2000


# ── WorldState ────────────────────────────────────────────────────────────────

@dataclass
class WorldState:
    """
    All mutable world state for one episode step.

    Registered as a JAX pytree so jit / lax.scan can look inside.
    """
    agent_pos:    jnp.ndarray  # [2]           float32 — position in [0, arena_size]²
    agent_energy: jnp.ndarray  # []            float32 — in [0, max_energy]
    hotspot_pos:  jnp.ndarray  # [n_food, 2]   float32 — hotspot centres
    step:         jnp.ndarray  # []            int32
    rng_key:      jax.Array    # PRNG key for stochastic drift


jax.tree_util.register_pytree_node(
    WorldState,
    lambda s: (
        [s.agent_pos, s.agent_energy, s.hotspot_pos, s.step, s.rng_key],
        None,
    ),
    lambda _, children: WorldState(*children),
)


# ── Food field ────────────────────────────────────────────────────────────────

def food_at(
    pos: jnp.ndarray,
    hotspot_pos: jnp.ndarray,
    wcfg: WorldConfig,
) -> jnp.ndarray:
    """
    Raw food density at pos: sum of Gaussians centred at each hotspot.

    Returns values in [0, n_food].  Clip to [0, 1] for sensor / energy use
    (one hotspot at its own centre contributes exactly 1.0).
    """
    diff    = pos[None, :] - hotspot_pos                             # [n_food, 2]
    sq_dist = jnp.sum(diff ** 2, axis=-1)                            # [n_food]
    return jnp.sum(jnp.exp(-sq_dist / (2.0 * wcfg.hotspot_sigma ** 2)))


# ── Sensor readout ────────────────────────────────────────────────────────────

def sensor_readout(state: WorldState, wcfg: WorldConfig) -> jnp.ndarray:
    """
    Returns [food_density, energy_level] both normalised to [0, 1].

    food_density: clipped raw food value — saturates at 1 near any hotspot.
    energy_level: agent_energy / max_energy.
    """
    food_norm  = jnp.clip(food_at(state.agent_pos, state.hotspot_pos, wcfg), 0.0, 1.0)
    energy_norm = state.agent_energy / wcfg.max_energy
    return jnp.array([food_norm, energy_norm])


# ── World step ────────────────────────────────────────────────────────────────

def step_world(
    state: WorldState,
    action: jnp.ndarray,
    wcfg: WorldConfig,
) -> WorldState:
    """
    Advance the world by one step.

    action: [v_x, v_y] in [-1, 1]; scaled by max_speed internally.
    Boundary: reflective clamp (agent cannot leave the arena).
    """
    # --- Physics ---
    v       = jnp.clip(action, -1.0, 1.0) * wcfg.max_speed
    speed   = jnp.sqrt(jnp.sum(v ** 2))
    new_pos = jnp.clip(state.agent_pos + v, 0.0, wcfg.arena_size)

    # --- Energy ---
    food_norm  = jnp.clip(food_at(new_pos, state.hotspot_pos, wcfg), 0.0, 1.0)
    gained     = wcfg.eat_rate * food_norm
    lost       = wcfg.metabolism + wcfg.move_cost * speed
    new_energy = jnp.clip(state.agent_energy + gained - lost, 0.0, wcfg.max_energy)

    # --- Hotspot drift ---
    key, subkey  = jax.random.split(state.rng_key)
    noise        = jax.random.normal(subkey, state.hotspot_pos.shape) * wcfg.hotspot_drift
    new_hotspots = jnp.clip(state.hotspot_pos + noise, 0.0, wcfg.arena_size)

    return WorldState(
        agent_pos=new_pos,
        agent_energy=new_energy,
        hotspot_pos=new_hotspots,
        step=state.step + 1,
        rng_key=key,
    )


# ── Episode initialisation ────────────────────────────────────────────────────

def reset_world(key: jax.Array, wcfg: WorldConfig) -> WorldState:
    """Initialise a fresh episode with random agent position and hotspot layout."""
    k1, k2, k3 = jax.random.split(key, 3)
    return WorldState(
        agent_pos=jax.random.uniform(k1, (2,)) * wcfg.arena_size,
        agent_energy=jnp.array(wcfg.init_energy, dtype=jnp.float32),
        hotspot_pos=jax.random.uniform(k2, (wcfg.n_food, 2)) * wcfg.arena_size,
        step=jnp.array(0, dtype=jnp.int32),
        rng_key=k3,
    )


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    key: jax.Array,
    controller_fn,
    wcfg: WorldConfig,
) -> tuple[WorldState, jnp.ndarray]:
    """
    Run a full episode with controller_fn for wcfg.episode_steps steps.

    controller_fn(key, sensors, state, wcfg) -> action [2] in [-1, 1]

    Returns (final_state, steps_survived) where steps_survived counts the
    number of steps on which the agent had energy > 0 after the step.
    Surviving the full episode gives steps_survived == episode_steps.
    """
    k1, k2 = jax.random.split(key)
    state   = reset_world(k1, wcfg)

    def body(carry, _):
        state, k = carry
        k, ctrl_key = jax.random.split(k)
        sensors   = sensor_readout(state, wcfg)
        action    = controller_fn(ctrl_key, sensors, state, wcfg)
        new_state = step_world(state, action, wcfg)
        alive     = new_state.agent_energy > 0.0
        return (new_state, k), alive

    (final_state, _), alive_mask = jax.lax.scan(
        body, (state, k2), None, length=wcfg.episode_steps
    )
    steps_survived = jnp.sum(alive_mask.astype(jnp.int32))
    return final_state, steps_survived
