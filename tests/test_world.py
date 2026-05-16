"""
Milestone 3 — World simulator tests.

Written before the implementation (TDD).  These tests define the required
interface and all validation gates from the milestone spec:

  - Gradient-following controller survives the full episode
  - Random-walk controller reliably starves
  - Energy economics balance (metabolism, eating, movement cost)
  - Food hotspots drift at the intended rate
  - Boundary reflection keeps the agent inside the arena
  - Difficulty band: clear fitness gap between good and bad controllers

Expected interface in ctrnn_evo.world:

    WorldConfig   dataclass
    WorldState    dataclass (JAX pytree)

    reset_world(key, wcfg)                     -> WorldState
    step_world(state, action, wcfg)            -> WorldState
    sensor_readout(state, wcfg)                -> jnp.ndarray [2]  (food, energy)
    food_at(pos, hotspot_pos, wcfg)            -> float
    run_episode(key, controller_fn, wcfg)      -> (WorldState, int)
        controller_fn(sensors, state, wcfg) -> action [2] in [-1, 1]

Expected interface in ctrnn_evo.controllers:

    random_walk(key, sensors, state, wcfg)     -> action [2]
    nearest_hotspot(key, sensors, state, wcfg) -> action [2]
        (nearest_hotspot may use full WorldState — it is a validation tool,
         not an evolved agent, and intentionally has more information than sensors)
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ctrnn_evo.world import (
    WorldConfig,
    WorldState,
    reset_world,
    step_world,
    sensor_readout,
    food_at,
    run_episode,
)
from ctrnn_evo.controllers import random_walk, nearest_hotspot


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def wcfg():
    return WorldConfig()


@pytest.fixture
def state(wcfg):
    return reset_world(jax.random.PRNGKey(0), wcfg)


# ── WorldConfig ───────────────────────────────────────────────────────────────

class TestWorldConfig:
    def test_has_required_fields(self, wcfg):
        assert hasattr(wcfg, "arena_size")
        assert hasattr(wcfg, "n_food")
        assert hasattr(wcfg, "hotspot_sigma")
        assert hasattr(wcfg, "hotspot_drift")
        assert hasattr(wcfg, "init_energy")
        assert hasattr(wcfg, "metabolism")
        assert hasattr(wcfg, "move_cost")
        assert hasattr(wcfg, "eat_rate")
        assert hasattr(wcfg, "max_energy")
        assert hasattr(wcfg, "max_speed")
        assert hasattr(wcfg, "episode_steps")

    def test_positive_values(self, wcfg):
        assert wcfg.arena_size > 0
        assert wcfg.n_food > 0
        assert wcfg.hotspot_sigma > 0
        assert wcfg.metabolism > 0
        assert wcfg.eat_rate > 0
        assert wcfg.max_speed > 0
        assert wcfg.episode_steps > 0


# ── reset_world ───────────────────────────────────────────────────────────────

class TestResetWorld:
    def test_returns_world_state(self, wcfg):
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        assert isinstance(state, WorldState)

    def test_agent_pos_in_arena(self, wcfg):
        state = reset_world(jax.random.PRNGKey(1), wcfg)
        assert jnp.all(state.agent_pos >= 0)
        assert jnp.all(state.agent_pos <= wcfg.arena_size)

    def test_energy_initialised_correctly(self, wcfg):
        state = reset_world(jax.random.PRNGKey(2), wcfg)
        assert float(state.agent_energy) == pytest.approx(wcfg.init_energy)

    def test_hotspot_positions_in_arena(self, wcfg):
        state = reset_world(jax.random.PRNGKey(3), wcfg)
        assert state.hotspot_pos.shape == (wcfg.n_food, 2)
        assert jnp.all(state.hotspot_pos >= 0)
        assert jnp.all(state.hotspot_pos <= wcfg.arena_size)

    def test_step_counter_zero(self, wcfg):
        state = reset_world(jax.random.PRNGKey(4), wcfg)
        assert int(state.step) == 0

    def test_different_keys_give_different_states(self, wcfg):
        s1 = reset_world(jax.random.PRNGKey(5), wcfg)
        s2 = reset_world(jax.random.PRNGKey(6), wcfg)
        assert not jnp.array_equal(s1.agent_pos, s2.agent_pos)

    def test_is_jax_pytree(self, wcfg):
        state = reset_world(jax.random.PRNGKey(7), wcfg)
        leaves, treedef = jax.tree_util.tree_flatten(state)
        state2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.array_equal(state.agent_pos, state2.agent_pos)


# ── food_at ───────────────────────────────────────────────────────────────────

class TestFoodAt:
    def test_peak_at_hotspot_centre(self, wcfg):
        hotspot_pos = jnp.array([[50.0, 50.0]])
        wcfg_single = WorldConfig(n_food=1)
        density = food_at(jnp.array([50.0, 50.0]), hotspot_pos, wcfg_single)
        assert float(density) == pytest.approx(1.0, abs=1e-4)

    def test_decays_with_distance(self, wcfg):
        hotspot_pos = jnp.array([[50.0, 50.0]])
        wcfg_single = WorldConfig(n_food=1)
        near  = food_at(jnp.array([51.0, 50.0]), hotspot_pos, wcfg_single)
        far   = food_at(jnp.array([70.0, 50.0]), hotspot_pos, wcfg_single)
        assert float(near) > float(far)

    def test_nonnegative(self, state, wcfg):
        density = food_at(state.agent_pos, state.hotspot_pos, wcfg)
        assert float(density) >= 0.0

    def test_multiple_hotspots_add(self):
        wcfg2 = WorldConfig(n_food=2)
        # Two hotspots at same point — density should be ~2x single
        hotspot_pos = jnp.array([[50.0, 50.0], [50.0, 50.0]])
        wcfg1 = WorldConfig(n_food=1)
        hotspot_pos1 = jnp.array([[50.0, 50.0]])
        d2 = food_at(jnp.array([50.0, 50.0]), hotspot_pos,  wcfg2)
        d1 = food_at(jnp.array([50.0, 50.0]), hotspot_pos1, wcfg1)
        assert float(d2) == pytest.approx(float(d1) * 2, rel=1e-4)


# ── sensor_readout ────────────────────────────────────────────────────────────

class TestSensorReadout:
    def test_shape(self, state, wcfg):
        sensors = sensor_readout(state, wcfg)
        assert sensors.shape == (2,)

    def test_food_sensor_in_range(self, state, wcfg):
        sensors = sensor_readout(state, wcfg)
        assert 0.0 <= float(sensors[0]) <= 1.0

    def test_energy_sensor_matches_state(self, state, wcfg):
        sensors = sensor_readout(state, wcfg)
        assert float(sensors[1]) == pytest.approx(float(state.agent_energy), rel=1e-4)

    def test_food_sensor_high_at_hotspot(self, wcfg):
        # Place agent exactly at a hotspot centre
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        centre = state.hotspot_pos[0]
        at_hotspot = WorldState(
            agent_pos=centre,
            agent_energy=state.agent_energy,
            hotspot_pos=state.hotspot_pos,
            step=state.step,
            rng_key=state.rng_key,
        )
        sensors = sensor_readout(at_hotspot, wcfg)
        assert float(sensors[0]) > 0.5, "Food sensor should be high at hotspot centre"

    def test_food_sensor_low_far_from_hotspots(self, wcfg):
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        # Move agent far from all hotspots (corner, hotspots initialised near centre)
        corner = WorldState(
            agent_pos=jnp.array([0.0, 0.0]),
            agent_energy=state.agent_energy,
            hotspot_pos=jnp.full((wcfg.n_food, 2), wcfg.arena_size * 0.75),
            step=state.step,
            rng_key=state.rng_key,
        )
        sensors = sensor_readout(corner, wcfg)
        assert float(sensors[0]) < 0.1, "Food sensor should be low far from hotspots"


# ── step_world ────────────────────────────────────────────────────────────────

class TestStepWorld:
    def test_step_counter_increments(self, state, wcfg):
        action = jnp.zeros(2)
        s2 = step_world(state, action, wcfg)
        assert int(s2.step) == int(state.step) + 1

    def test_stationary_agent_loses_energy(self, wcfg):
        # Place agent far from all food, zero velocity
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        barren = WorldState(
            agent_pos=jnp.array([0.0, 0.0]),
            agent_energy=jnp.array(0.5),
            hotspot_pos=jnp.full((wcfg.n_food, 2), wcfg.arena_size),
            step=state.step,
            rng_key=state.rng_key,
        )
        action = jnp.zeros(2)  # stationary
        s2 = step_world(barren, action, wcfg)
        assert float(s2.agent_energy) < 0.5, "Stationary agent in barren area must lose energy"

    def test_eating_restores_energy(self, wcfg):
        # Place agent at hotspot centre with low energy
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        centre = state.hotspot_pos[0]
        at_hotspot = WorldState(
            agent_pos=centre,
            agent_energy=jnp.array(0.1),
            hotspot_pos=state.hotspot_pos,
            step=state.step,
            rng_key=state.rng_key,
        )
        action = jnp.zeros(2)
        s2 = step_world(at_hotspot, action, wcfg)
        assert float(s2.agent_energy) > 0.1, "Agent at hotspot centre must gain energy"

    def test_energy_capped_at_max(self, wcfg):
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        full_energy = WorldState(
            agent_pos=state.hotspot_pos[0],      # at hotspot
            agent_energy=jnp.array(wcfg.max_energy),
            hotspot_pos=state.hotspot_pos,
            step=state.step,
            rng_key=state.rng_key,
        )
        action = jnp.zeros(2)
        for _ in range(10):
            full_energy = step_world(full_energy, action, wcfg)
        assert float(full_energy.agent_energy) <= wcfg.max_energy + 1e-5

    def test_energy_never_negative(self, wcfg):
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        empty = WorldState(
            agent_pos=jnp.array([0.0, 0.0]),
            agent_energy=jnp.array(0.0),
            hotspot_pos=jnp.full((wcfg.n_food, 2), wcfg.arena_size),
            step=state.step,
            rng_key=state.rng_key,
        )
        action = jnp.zeros(2)
        s2 = step_world(empty, action, wcfg)
        assert float(s2.agent_energy) >= 0.0

    def test_movement_costs_energy(self, wcfg):
        # Two agents at same barren position: one moving, one stationary
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        barren = WorldState(
            agent_pos=jnp.array([0.0, 0.0]),
            agent_energy=jnp.array(0.5),
            hotspot_pos=jnp.full((wcfg.n_food, 2), wcfg.arena_size),
            step=state.step,
            rng_key=state.rng_key,
        )
        stationary = step_world(barren, jnp.zeros(2), wcfg)
        moving     = step_world(barren, jnp.ones(2), wcfg)   # max speed
        assert float(moving.agent_energy) < float(stationary.agent_energy), \
            "Moving agent should spend more energy than stationary agent"

    def test_boundary_reflection(self, wcfg):
        # Agent at corner, moving out of bounds
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        at_edge = WorldState(
            agent_pos=jnp.array([0.0, 0.0]),
            agent_energy=jnp.array(0.5),
            hotspot_pos=state.hotspot_pos,
            step=state.step,
            rng_key=state.rng_key,
        )
        action = jnp.array([-1.0, -1.0])  # max speed toward negative corner
        for _ in range(10):
            at_edge = step_world(at_edge, action, wcfg)
        assert jnp.all(at_edge.agent_pos >= 0.0), "Agent escaped arena lower bound"
        assert jnp.all(at_edge.agent_pos <= wcfg.arena_size), "Agent escaped arena upper bound"

    def test_deterministic_given_same_key(self, state, wcfg):
        action = jnp.array([0.3, -0.5])
        s2a = step_world(state, action, wcfg)
        s2b = step_world(state, action, wcfg)
        assert jnp.array_equal(s2a.agent_pos,    s2b.agent_pos)
        assert jnp.array_equal(s2a.agent_energy, s2b.agent_energy)
        assert jnp.array_equal(s2a.hotspot_pos,  s2b.hotspot_pos)

    def test_hotspot_drift_over_time(self, wcfg):
        """
        After T steps, mean squared displacement of hotspot centres should
        be approximately T * hotspot_drift^2 per coordinate (random walk variance).
        """
        T = 500
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        initial_pos = state.hotspot_pos.copy()
        action = jnp.zeros(2)
        for _ in range(T):
            state = step_world(state, action, wcfg)
        displacement = state.hotspot_pos - initial_pos
        msd = float(jnp.mean(displacement ** 2))
        expected_msd = T * wcfg.hotspot_drift ** 2
        # Allow 3x tolerance: boundary reflections and clamping reduce actual drift
        assert msd > 0.0, "Hotspots did not drift at all"
        assert msd < expected_msd * 3, f"Hotspots drifted far more than expected (msd={msd:.3f})"


# ── Energy economics unit check ───────────────────────────────────────────────

class TestEnergyEconomics:
    def test_metabolism_rate_matches_config(self, wcfg):
        """
        A stationary agent in a food-free environment loses exactly
        metabolism energy per step.
        """
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        barren = WorldState(
            agent_pos=jnp.array([0.0, 0.0]),
            agent_energy=jnp.array(0.5),
            hotspot_pos=jnp.full((wcfg.n_food, 2), wcfg.arena_size * 10),  # far away
            step=state.step,
            rng_key=state.rng_key,
        )
        action = jnp.zeros(2)
        s2 = step_world(barren, action, wcfg)
        expected = 0.5 - wcfg.metabolism
        assert float(s2.agent_energy) == pytest.approx(expected, abs=1e-4)


# ── Controllers ───────────────────────────────────────────────────────────────

class TestControllers:
    def test_random_walk_action_shape(self, state, wcfg):
        sensors = sensor_readout(state, wcfg)
        action  = random_walk(jax.random.PRNGKey(0), sensors, state, wcfg)
        assert action.shape == (2,)

    def test_random_walk_action_in_range(self, state, wcfg):
        sensors = sensor_readout(state, wcfg)
        for i in range(10):
            action = random_walk(jax.random.PRNGKey(i), sensors, state, wcfg)
            assert jnp.all(action >= -1.0) and jnp.all(action <= 1.0), \
                "random_walk must return actions in [-1, 1]"

    def test_nearest_hotspot_action_shape(self, state, wcfg):
        sensors = sensor_readout(state, wcfg)
        action  = nearest_hotspot(jax.random.PRNGKey(0), sensors, state, wcfg)
        assert action.shape == (2,)

    def test_nearest_hotspot_action_in_range(self, state, wcfg):
        sensors = sensor_readout(state, wcfg)
        action  = nearest_hotspot(jax.random.PRNGKey(0), sensors, state, wcfg)
        assert jnp.all(action >= -1.0) and jnp.all(action <= 1.0)

    def test_nearest_hotspot_moves_toward_food(self, wcfg):
        """
        Starting from a known offset from a single hotspot, nearest_hotspot
        should produce an action that moves the agent closer to the hotspot.
        """
        wcfg_single = WorldConfig(n_food=1, hotspot_drift=0.0)
        state = reset_world(jax.random.PRNGKey(0), wcfg_single)
        hotspot = state.hotspot_pos[0]
        # Place agent 20 units to the left of the hotspot
        offset_pos = hotspot + jnp.array([-20.0, 0.0])
        offset_pos = jnp.clip(offset_pos, 0.0, wcfg_single.arena_size)
        at_offset = WorldState(
            agent_pos=offset_pos,
            agent_energy=jnp.array(0.5),
            hotspot_pos=state.hotspot_pos,
            step=state.step,
            rng_key=state.rng_key,
        )
        sensors = sensor_readout(at_offset, wcfg_single)
        action  = nearest_hotspot(jax.random.PRNGKey(0), sensors, at_offset, wcfg_single)
        # Action x-component should be positive (toward hotspot at +x direction)
        assert float(action[0]) > 0.0, "Gradient follower should move toward hotspot"


# ── Validation gates ──────────────────────────────────────────────────────────

class TestValidationGates:
    def test_nearest_hotspot_survives_full_episode(self):
        """
        Key M3 gate: a gradient-following controller must survive the entire
        episode without starving.
        """
        wcfg = WorldConfig(episode_steps=500)
        _, steps = run_episode(jax.random.PRNGKey(0), nearest_hotspot, wcfg)
        assert steps == wcfg.episode_steps, (
            f"Gradient follower starved at step {steps}/{wcfg.episode_steps}"
        )

    def test_random_walk_dies_before_episode_end(self):
        """
        Key M3 gate: a random-walk controller must reliably starve well before
        the episode ends.  We run several seeds and require all of them to die.
        """
        wcfg  = WorldConfig(episode_steps=2000)
        seeds = [10, 11, 12, 13, 14]
        steps_list = []
        for seed in seeds:
            _, steps = run_episode(jax.random.PRNGKey(seed), random_walk, wcfg)
            steps_list.append(steps)

        max_steps = max(steps_list)
        assert max_steps < wcfg.episode_steps, (
            f"Random walker survived the full episode (seed produced {max_steps} steps). "
            "World may be too easy — check eat_rate / metabolism ratio."
        )

    def test_difficulty_band(self):
        """
        Quantitative gap: gradient follower fitness must be substantially
        higher than random walker fitness across multiple seeds.
        """
        wcfg  = WorldConfig(episode_steps=1000)
        seeds = [20, 21, 22]

        gf_steps = [run_episode(jax.random.PRNGKey(s), nearest_hotspot, wcfg)[1] for s in seeds]
        rw_steps = [run_episode(jax.random.PRNGKey(s), random_walk,        wcfg)[1] for s in seeds]

        mean_gf = np.mean(gf_steps)
        mean_rw = np.mean(rw_steps)

        assert mean_gf > mean_rw * 2, (
            f"Difficulty band too narrow: nearest_hotspot mean={mean_gf:.0f}, "
            f"random_walk mean={mean_rw:.0f}.  Gap should be at least 2x."
        )
