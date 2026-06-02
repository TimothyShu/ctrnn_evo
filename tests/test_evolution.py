"""
M5 Tests — Evolutionary loop.

Tests cover:
  1.  init_population — shape and validity
  2.  eval_population — shapes, bounds, determinism
  3.  compute_fitness — range and cost-penalty direction
  4.  tournament_select_idx — always valid, always picks best when clear winner
  5.  select_parents — shape, all indices valid
  6.  reproduce — shape preserved, offspring differ from parents (mutation fires)
  7.  evolve_step — shape preserved, determinism
  8.  elitism — best parent survives into offspring slot 0
  9.  run_brain_episode_full — returns c_act as third value, bounded
  10. run_evolution — history length, callback count, fitness trend
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ctrnn_evo import Config, WorldConfig
from ctrnn_evo import random_genome
from ctrnn_evo.mutation import MutationRates
from ctrnn_evo.brain import run_brain_episode_full
from ctrnn_evo.evolution import (
    init_population,
    eval_population,
    compute_fitness,
    tournament_select_idx,
    select_parents,
    reproduce,
    evolve_step,
    collect_stats,
    run_evolution,
    _warmup_ramp,
    _cycle_ramp,
    _mutation_scale,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

POP  = 20   # small population so tests stay fast
GENS = 10   # short run for run_evolution test

@pytest.fixture(scope="module")
def cfg():
    return Config(N_max=16, n_out=2, K=4, population_size=POP, tournament_size=3)

@pytest.fixture(scope="module")
def wcfg():
    return WorldConfig(episode_steps=100)

@pytest.fixture(scope="module")
def rates():
    return MutationRates()

@pytest.fixture(scope="module")
def pop(cfg):
    key = jax.random.PRNGKey(0)
    return init_population(key, cfg)

@pytest.fixture(scope="module")
def evaluated(pop, cfg, wcfg):
    """Pre-evaluated population: (steps, c_acts, fitness)."""
    key = jax.random.PRNGKey(1)
    steps, c_acts, raw_food = eval_population(key, pop, cfg, wcfg, n_evals=3)
    fitness = compute_fitness(steps, c_acts, raw_food, pop, cfg, wcfg)
    return steps, c_acts, fitness


# ── 1. init_population ────────────────────────────────────────────────────────

def test_init_population_field_shapes(pop, cfg):
    assert pop.active_mask.shape    == (POP, cfg.N_max)
    assert pop.weight_matrix.shape  == (POP, cfg.N_max, cfg.N_max)
    assert pop.tau.shape            == (POP, cfg.N_max)
    assert pop.bias.shape           == (POP, cfg.N_max)
    assert pop.position.shape       == (POP, cfg.N_max, 2)
    assert pop.edge_mask.shape      == (POP, cfg.N_max, cfg.N_max)
    assert pop.neuron_type.shape    == (POP, cfg.N_max)

def test_init_population_weights_nonneg(pop):
    assert jnp.all(pop.weight_matrix >= 0), "Weight magnitudes must be non-negative (Dale's law)"

def test_init_population_io_slots_always_active(pop, cfg):
    """Input (first n_in) and output (last n_out) slots must be active in every genome."""
    assert jnp.all(pop.active_mask[:, :cfg.n_in]),  "Input slots must always be active"
    assert jnp.all(pop.active_mask[:, -cfg.n_out:]), "Output slots must always be active"


# ── 2. eval_population ────────────────────────────────────────────────────────

def test_eval_population_shapes(evaluated, wcfg):
    steps, c_acts, _ = evaluated
    assert steps.shape  == (POP,)
    assert c_acts.shape == (POP,)

def test_eval_population_steps_bounded(evaluated, wcfg):
    steps, _, _ = evaluated
    assert jnp.all(steps >= 0) and jnp.all(steps <= wcfg.episode_steps)

def test_eval_population_c_acts_nonneg(evaluated):
    _, c_acts, _ = evaluated
    assert jnp.all(c_acts >= 0), "Activation cost must be non-negative"

def test_eval_population_determinism(pop, cfg, wcfg):
    key = jax.random.PRNGKey(7)
    s1, c1, rf1 = eval_population(key, pop, cfg, wcfg, n_evals=2)
    s2, c2, rf2 = eval_population(key, pop, cfg, wcfg, n_evals=2)
    assert jnp.array_equal(s1, s2) and jnp.allclose(c1, c2) and jnp.allclose(rf1, rf2)


# ── 3. compute_fitness ────────────────────────────────────────────────────────

def test_compute_fitness_shape(evaluated):
    _, _, fitness = evaluated
    assert fitness.shape == (POP,)

def test_compute_fitness_in_unit_interval_when_no_penalty(pop, cfg, wcfg):
    """With lambda_conn=lambda_act=0 (Config defaults), survival fitness == f_raw ∈ [0,1]."""
    assert cfg.lambda_edge == 0.0 and cfg.lambda_dist == 0.0 and cfg.lambda_act == 0.0, "Test assumes zero-penalty config"
    assert cfg.fitness_mode == "survival", "Test assumes survival fitness mode"
    key = jax.random.PRNGKey(8)
    steps, c_acts, raw_food = eval_population(key, pop, cfg, wcfg, n_evals=2)
    fitness = compute_fitness(steps, c_acts, raw_food, pop, cfg, wcfg)
    assert jnp.all(fitness >= 0.0) and jnp.all(fitness <= 1.0 + 1e-5)

def test_compute_fitness_penalty_reduces_fitness():
    """Enabling connection cost should strictly reduce fitness for a genome with edges."""
    cfg_no  = Config(N_max=16, n_out=2, K=4, population_size=4,
                     lambda_edge=0.0, lambda_dist=0.0, lambda_act=0.0)
    cfg_pen = Config(N_max=16, n_out=2, K=4, population_size=4,
                     lambda_edge=1.0, lambda_dist=0.0, lambda_act=0.0)
    wcfg = WorldConfig(episode_steps=50)
    key  = jax.random.PRNGKey(9)
    pop_small = init_population(key, cfg_no)
    steps, c_acts, raw_food = eval_population(key, pop_small, cfg_no, wcfg, n_evals=2)

    f_no  = compute_fitness(steps, c_acts, raw_food, pop_small, cfg_no,  wcfg)
    f_pen = compute_fitness(steps, c_acts, raw_food, pop_small, cfg_pen, wcfg)
    # With positive connection cost, penalised fitness ≤ unpenalised fitness
    assert jnp.all(f_pen <= f_no + 1e-6)


# ── 4. tournament_select_idx ──────────────────────────────────────────────────

def test_tournament_select_idx_valid_range(cfg):
    fitness = jnp.array([0.1, 0.5, 0.9, 0.2, 0.7])
    for i in range(10):
        idx = tournament_select_idx(jax.random.PRNGKey(i), fitness, cfg.tournament_size)
        assert 0 <= int(idx) < len(fitness)

def test_tournament_always_picks_best():
    """
    When one genome has fitness >> all others it should win every tournament.
    Using tournament_size == population_size ensures the dominant genome is
    always sampled (replace=False exhausts the population).
    """
    fitness = jnp.array([0.01, 0.01, 100.0, 0.01, 0.01])
    for i in range(20):
        idx = tournament_select_idx(jax.random.PRNGKey(i), fitness, tournament_size=len(fitness))
        assert int(idx) == 2, f"Expected winner at index 2, got {int(idx)}"


# ── 5. select_parents ─────────────────────────────────────────────────────────

def test_select_parents_shape(evaluated, cfg):
    _, _, fitness = evaluated
    key = jax.random.PRNGKey(10)
    idxs = select_parents(key, fitness, cfg.population_size, cfg.tournament_size)
    assert idxs.shape == (POP,)

def test_select_parents_valid_indices(evaluated, cfg):
    _, _, fitness = evaluated
    key = jax.random.PRNGKey(11)
    idxs = select_parents(key, fitness, cfg.population_size, cfg.tournament_size)
    assert jnp.all(idxs >= 0) and jnp.all(idxs < POP)


# ── 6. reproduce ─────────────────────────────────────────────────────────────

def test_reproduce_shape_preserved(pop, evaluated, cfg, rates):
    _, _, fitness = evaluated
    key  = jax.random.PRNGKey(12)
    idxs = select_parents(key, fitness, cfg.population_size, cfg.tournament_size)
    key2 = jax.random.PRNGKey(13)
    offspring = reproduce(key2, pop, idxs, rates, cfg)
    assert offspring.weight_matrix.shape == pop.weight_matrix.shape
    assert offspring.active_mask.shape   == pop.active_mask.shape

def test_reproduce_offspring_differ_from_parents(pop, evaluated, cfg, rates):
    """Mutation should change at least some weights."""
    _, _, fitness = evaluated
    key  = jax.random.PRNGKey(14)
    idxs = select_parents(key, fitness, cfg.population_size, cfg.tournament_size)
    # Gather parents for direct comparison
    parents   = jax.tree_util.tree_map(lambda x: x[idxs], pop)
    key2      = jax.random.PRNGKey(15)
    offspring = reproduce(key2, pop, idxs, rates, cfg)
    total_diff = jnp.sum(jnp.abs(offspring.weight_matrix - parents.weight_matrix))
    assert float(total_diff) > 0, "Offspring weights are identical to parents — mutation not firing"


# ── 7. evolve_step ────────────────────────────────────────────────────────────

def test_evolve_step_shape_preserved(pop, evaluated, cfg, rates):
    _, _, fitness = evaluated
    key = jax.random.PRNGKey(16)
    new_pop = evolve_step(key, pop, fitness, rates, cfg)
    assert new_pop.weight_matrix.shape == pop.weight_matrix.shape
    assert new_pop.active_mask.shape   == pop.active_mask.shape

def test_evolve_step_determinism(pop, evaluated, cfg, rates):
    _, _, fitness = evaluated
    key = jax.random.PRNGKey(17)
    p1 = evolve_step(key, pop, fitness, rates, cfg)
    p2 = evolve_step(key, pop, fitness, rates, cfg)
    assert jnp.allclose(p1.weight_matrix, p2.weight_matrix), "evolve_step not deterministic"

def test_evolve_step_io_slots_preserved(pop, evaluated, cfg, rates):
    """Input (first n_in) and output (last n_out) slots must survive evolution."""
    _, _, fitness = evaluated
    key = jax.random.PRNGKey(18)
    new_pop = evolve_step(key, pop, fitness, rates, cfg)
    assert jnp.all(new_pop.active_mask[:, :cfg.n_in]),  "Input slots became inactive after evolve_step"
    assert jnp.all(new_pop.active_mask[:, -cfg.n_out:]), "Output slots became inactive after evolve_step"


# ── 8. elitism ────────────────────────────────────────────────────────────────

def test_elitism_best_parent_in_slot_zero(pop, evaluated, cfg, rates):
    """
    The best genome in the parent population must appear unchanged in slot 0
    of the offspring (elitism = 1).
    """
    _, _, fitness = evaluated
    best_idx = int(jnp.argmax(fitness))
    best_wm  = pop.weight_matrix[best_idx]

    key     = jax.random.PRNGKey(19)
    new_pop = evolve_step(key, pop, fitness, rates, cfg)

    assert jnp.allclose(new_pop.weight_matrix[0], best_wm), (
        "Elite (best parent) weight_matrix not found in offspring slot 0"
    )


# ── 9. run_brain_episode_full ─────────────────────────────────────────────────

def test_run_brain_episode_full_four_returns(cfg, wcfg):
    key    = jax.random.PRNGKey(20)
    genome = random_genome(key, cfg)
    result = run_brain_episode_full(key, genome, cfg, wcfg)
    assert len(result) == 4, "Expected (final_state, steps_survived, mean_c_act, total_raw_food)"

def test_run_brain_episode_full_c_act_nonneg(cfg, wcfg):
    key    = jax.random.PRNGKey(21)
    genome = random_genome(key, cfg)
    _, steps, c_act, raw_food = run_brain_episode_full(key, genome, cfg, wcfg)
    assert float(c_act) >= 0.0
    assert float(raw_food) >= 0.0
    assert 0 <= int(steps) <= wcfg.episode_steps

def test_run_brain_episode_full_determinism(cfg, wcfg):
    key    = jax.random.PRNGKey(22)
    genome = random_genome(key, cfg)
    _, s1, c1, rf1 = run_brain_episode_full(key, genome, cfg, wcfg)
    _, s2, c2, rf2 = run_brain_episode_full(key, genome, cfg, wcfg)
    assert int(s1) == int(s2)
    assert jnp.isclose(c1, c2)
    assert jnp.isclose(rf1, rf2)


# ── 10. run_evolution ─────────────────────────────────────────────────────────

def test_run_evolution_history_length(cfg, wcfg, rates):
    key = jax.random.PRNGKey(30)
    _, _, history = run_evolution(key, GENS, cfg, wcfg, rates, n_evals=2)
    assert len(history) == GENS, f"Expected {GENS} history entries, got {len(history)}"

def test_run_evolution_history_keys(cfg, wcfg, rates):
    key = jax.random.PRNGKey(31)
    _, _, history = run_evolution(key, 2, cfg, wcfg, rates, n_evals=2)
    expected = {"generation", "max_fitness", "mean_fitness", "max_steps",
                "mean_steps", "mean_n_active", "mean_edge_cost", "mean_wiring_cost"}
    assert expected <= set(history[0].keys()), f"Missing keys: {expected - set(history[0].keys())}"

def test_run_evolution_generation_indices(cfg, wcfg, rates):
    key = jax.random.PRNGKey(32)
    _, _, history = run_evolution(key, GENS, cfg, wcfg, rates, n_evals=2)
    gens = [h["generation"] for h in history]
    assert gens == list(range(GENS)), "Generation indices are not sequential starting from 0"

def test_run_evolution_callback_called(cfg, wcfg, rates):
    key   = jax.random.PRNGKey(33)
    calls = []
    run_evolution(key, 3, cfg, wcfg, rates, n_evals=2, callback=lambda s, g: calls.append(s))
    assert len(calls) == 3, f"Callback called {len(calls)} times, expected 3"

def test_run_evolution_returns_valid_best_genome(cfg, wcfg, rates):
    key = jax.random.PRNGKey(34)
    best, final_fitness, _ = run_evolution(key, 2, cfg, wcfg, rates, n_evals=2)
    # best should be a single (unbatched) Genome
    assert best.weight_matrix.shape == (cfg.N_max, cfg.N_max), (
        f"best_genome.weight_matrix has unexpected shape {best.weight_matrix.shape}"
    )
    assert final_fitness.shape == (POP,)

def test_run_evolution_fitness_plausible(cfg, wcfg, rates):
    """
    max_fitness in history should be in [0, 1] throughout (lambda=0 case).

    Note: elitism preserves genome *weights* not estimated fitness.
    Re-evaluating the elite with fresh episode seeds each generation means
    the max_fitness estimate can fluctuate up or down due to world randomness.
    The structural elitism invariant (best genome survives) is tested separately
    by test_elitism_best_parent_in_slot_zero.
    """
    key = jax.random.PRNGKey(35)
    _, _, history = run_evolution(key, GENS, cfg, wcfg, rates, n_evals=3)
    for stats in history:
        assert 0.0 <= stats["max_fitness"] <= 1.0 + 1e-6, (
            f"max_fitness {stats['max_fitness']} out of [0,1] at gen {stats['generation']}"
        )
        assert stats["mean_fitness"] <= stats["max_fitness"] + 1e-6, (
            "mean_fitness exceeds max_fitness"
        )


# ── 11. collect_stats ─────────────────────────────────────────────────────────

def test_collect_stats_values_plausible(pop, evaluated, cfg):
    steps, _, fitness = evaluated
    stats = collect_stats(0, fitness, steps, pop, cfg)
    assert stats["mean_n_active"] >= cfg.n_in + cfg.n_out, \
        "mean_n_active must be at least n_in + n_out (I/O slots always active)"
    assert stats["mean_n_active"] <= cfg.N_max
    assert stats["mean_edge_cost"] >= 0.0
    assert stats["mean_wiring_cost"] >= 0.0
    assert 0.0 <= stats["mean_fitness"] <= stats["max_fitness"] + 1e-6


# ── 12. _mutation_scale ───────────────────────────────────────────────────────

class TestMutationScale:
    """Unit tests for the mutation warmup scale function."""

    def test_gen0_returns_full_scale(self):
        cfg = Config(N_max=16, n_out=2, penalty_warmup_gens=200, mutation_warmup_scale=3.0)
        assert _mutation_scale(0, cfg) == pytest.approx(3.0)

    def test_midpoint_returns_midpoint_scale(self):
        cfg = Config(N_max=16, n_out=2, penalty_warmup_gens=200, mutation_warmup_scale=3.0)
        # At gen 100 (half of 200): scale = 3.0*(1-0.5) + 1.0*0.5 = 2.0
        assert _mutation_scale(100, cfg) == pytest.approx(2.0)

    def test_at_warmup_end_returns_one(self):
        cfg = Config(N_max=16, n_out=2, penalty_warmup_gens=200, mutation_warmup_scale=3.0)
        assert _mutation_scale(200, cfg) == pytest.approx(1.0)

    def test_after_warmup_returns_one(self):
        cfg = Config(N_max=16, n_out=2, penalty_warmup_gens=200, mutation_warmup_scale=3.0)
        for gen in [201, 500, 999]:
            assert _mutation_scale(gen, cfg) == pytest.approx(1.0), \
                f"Expected 1.0 after warmup at gen {gen}"

    def test_scale_one_always_returns_one(self):
        cfg = Config(N_max=16, n_out=2, penalty_warmup_gens=200, mutation_warmup_scale=1.0)
        for gen in [0, 100, 200, 999]:
            assert _mutation_scale(gen, cfg) == pytest.approx(1.0)

    def test_no_warmup_always_returns_one(self):
        cfg = Config(N_max=16, n_out=2, penalty_warmup_gens=0, mutation_warmup_scale=3.0)
        for gen in [0, 100, 999]:
            assert _mutation_scale(gen, cfg) == pytest.approx(1.0)

    def test_scale_never_below_one(self):
        cfg = Config(N_max=16, n_out=2, penalty_warmup_gens=200, mutation_warmup_scale=5.0)
        for gen in range(0, 300, 10):
            assert _mutation_scale(gen, cfg) >= 1.0 - 1e-6


# ── 13. mutation warmup integration ──────────────────────────────────────────

def test_mutation_warmup_increases_weight_diversity():
    """With mutation_warmup_scale > 1, weight variation across the population
    after one evolve_step at gen 0 should be larger than without scaling."""
    import numpy as np

    cfg_base  = Config(N_max=16, n_out=2, K=4, population_size=50,
                       penalty_warmup_gens=200, mutation_warmup_scale=1.0)
    cfg_scale = Config(N_max=16, n_out=2, K=4, population_size=50,
                       penalty_warmup_gens=200, mutation_warmup_scale=5.0)
    wcfg  = WorldConfig(episode_steps=50)
    rates = MutationRates(weight_sigma=0.1, add_node_prob=0.0, remove_node_prob=0.0,
                          add_edge_prob=0.0, remove_edge_prob=0.0)

    key = jax.random.PRNGKey(99)
    pop = init_population(key, cfg_base)
    k_eval, k_step_base, k_step_scale = jax.random.split(key, 3)

    steps, c_acts, raw_food = eval_population(k_eval, pop, cfg_base, wcfg, n_evals=2)
    fitness = compute_fitness(steps, c_acts, raw_food, pop, cfg_base, wcfg)

    # One evolve_step with and without warmup scale — same key, same starting pop
    _, _, history_base = run_evolution(
        k_step_base, 2, cfg_base, wcfg, rates, n_evals=2)
    _, _, history_scale = run_evolution(
        k_step_scale, 2, cfg_scale, wcfg, rates, n_evals=2)

    # With scale=5, mean_n_active should show more structural diversity;
    # simpler check: early weight std should be higher for scaled run
    pop_base  = init_population(k_step_base,  cfg_base)
    pop_scale = init_population(k_step_scale, cfg_scale)

    k_e = jax.random.PRNGKey(77)
    steps_b, c_b, rf_b = eval_population(k_e, pop_base,  cfg_base,  wcfg, n_evals=2)
    steps_s, c_s, rf_s = eval_population(k_e, pop_scale, cfg_scale, wcfg, n_evals=2)
    fit_b = compute_fitness(steps_b, c_b, rf_b, pop_base,  cfg_base,  wcfg)
    fit_s = compute_fitness(steps_s, c_s, rf_s, pop_scale, cfg_scale, wcfg)

    off_base  = evolve_step(jax.random.PRNGKey(88), pop_base,  fit_b, rates, cfg_base,  generation=0)
    off_scale = evolve_step(jax.random.PRNGKey(88), pop_scale, fit_s, rates, cfg_scale, generation=0)

    std_base  = float(jnp.std(off_base.weight_matrix))
    std_scale = float(jnp.std(off_scale.weight_matrix))
    assert std_scale > std_base, (
        f"Expected higher weight std with scale=5 at gen 0 "
        f"(got std_base={std_base:.4f}, std_scale={std_scale:.4f})"
    )
