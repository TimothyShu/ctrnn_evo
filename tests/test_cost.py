import dataclasses
import jax
import jax.numpy as jnp
import pytest
from ctrnn_evo import Config, random_genome, edge_count_cost, dist_cost, adjusted_fitness


@pytest.fixture
def cfg():
    return Config(N_max=8, n_out=1)


def test_edge_count_cost_zero_no_edges(cfg):
    key = jax.random.PRNGKey(0)
    g   = random_genome(key, cfg)
    g_no_edges = g.__class__(**{**vars(g), "edge_mask": jnp.zeros_like(g.edge_mask)})
    assert float(edge_count_cost(g_no_edges)) == pytest.approx(0.0)


def test_edge_count_cost_nonnegative(cfg):
    key = jax.random.PRNGKey(1)
    g   = random_genome(key, cfg)
    assert float(edge_count_cost(g)) >= 0.0


def test_edge_count_cost_invariant_to_position(cfg):
    """Edge count cost must not change when neuron positions change."""
    key = jax.random.PRNGKey(2)
    g   = random_genome(key, cfg)
    g_moved = g.__class__(**{**vars(g), "position": jnp.zeros((cfg.N_max, 2))})
    assert float(edge_count_cost(g)) == pytest.approx(float(edge_count_cost(g_moved)))


def test_dist_cost_zero_no_edges(cfg):
    key = jax.random.PRNGKey(0)
    g   = random_genome(key, cfg)
    g_no_edges = g.__class__(**{**vars(g), "edge_mask": jnp.zeros_like(g.edge_mask)})
    assert float(dist_cost(g_no_edges)) == pytest.approx(0.0)


def test_dist_cost_nonnegative(cfg):
    key = jax.random.PRNGKey(1)
    g   = random_genome(key, cfg)
    assert float(dist_cost(g)) >= 0.0


def test_dist_cost_scales_with_distance(cfg):
    """Packed neurons (all at origin) should have lower dist_cost than spread ones."""
    key = jax.random.PRNGKey(2)
    g   = random_genome(key, cfg)
    g_packed = g.__class__(**{**vars(g), "position": jnp.zeros((cfg.N_max, 2))})
    assert float(dist_cost(g)) >= float(dist_cost(g_packed))


def test_adjusted_fitness_lambda_dist(cfg):
    """lambda_dist > 0 reduces fitness when edges exist."""
    key   = jax.random.PRNGKey(3)
    g     = random_genome(key, cfg)
    f_raw = 1.0
    c_act = 0.0
    cfg_no   = dataclasses.replace(cfg, lambda_dist=0.0)
    cfg_with = dataclasses.replace(cfg, lambda_dist=1.0)
    assert float(adjusted_fitness(f_raw, g, c_act, cfg_no))   == pytest.approx(f_raw)
    assert float(adjusted_fitness(f_raw, g, c_act, cfg_with)) <  f_raw


def test_adjusted_fitness_lambda_edge(cfg):
    """lambda_edge > 0 reduces fitness when edges exist."""
    key   = jax.random.PRNGKey(3)
    g     = random_genome(key, cfg)
    f_raw = 1.0
    c_act = 0.0
    cfg_no   = dataclasses.replace(cfg, lambda_edge=0.0)
    cfg_with = dataclasses.replace(cfg, lambda_edge=1.0)
    assert float(adjusted_fitness(f_raw, g, c_act, cfg_no))   == pytest.approx(f_raw)
    assert float(adjusted_fitness(f_raw, g, c_act, cfg_with)) <  f_raw


def test_activation_cost_penalty(cfg):
    """lambda_act > 0 and nonzero c_act reduces fitness."""
    key   = jax.random.PRNGKey(4)
    g     = random_genome(key, cfg)
    f_raw = 1.0
    c_act = 5.0
    cfg_no   = dataclasses.replace(cfg, lambda_act=0.0)
    cfg_with = dataclasses.replace(cfg, lambda_act=0.1)
    assert float(adjusted_fitness(f_raw, g, c_act, cfg_no))   == pytest.approx(f_raw)
    assert float(adjusted_fitness(f_raw, g, c_act, cfg_with)) <  f_raw
