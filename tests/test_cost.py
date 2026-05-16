import jax
import jax.numpy as jnp
import pytest
from ctrnn_evo import Config, random_genome, connection_cost, adjusted_fitness


@pytest.fixture
def cfg():
    return Config(N_max=8, n_in=1, n_out=1)


def test_connection_cost_zero_no_edges(cfg):
    """A genome with no active edges has zero connection cost."""
    key = jax.random.PRNGKey(0)
    g   = random_genome(key, cfg)
    # Remove all edges
    g_no_edges = g.__class__(
        **{**vars(g), "edge_mask": jnp.zeros_like(g.edge_mask)}
    )
    assert float(connection_cost(g_no_edges)) == pytest.approx(0.0)


def test_connection_cost_nonnegative(cfg):
    key = jax.random.PRNGKey(1)
    g   = random_genome(key, cfg)
    assert float(connection_cost(g)) >= 0.0


def test_connection_cost_scales_with_distance(cfg):
    """
    Two organisms identical except one has neurons packed at the same point
    and one has them spread across the unit square.  The spread one should
    have higher connection cost when the same edges are active.
    """
    key  = jax.random.PRNGKey(2)
    g    = random_genome(key, cfg)
    N    = cfg.N_max

    # Packed: all neurons at the origin
    g_packed = g.__class__(
        **{**vars(g), "position": jnp.zeros((N, 2))}
    )
    # Spread: neurons at corners / spread positions (already random from g)
    cost_packed = float(connection_cost(g_packed))
    cost_spread = float(connection_cost(g))

    assert cost_spread >= cost_packed


def test_adjusted_fitness_decreases_with_lambda(cfg):
    """With lambda_conn > 0 and active edges, adjusted fitness < raw fitness."""
    key    = jax.random.PRNGKey(3)
    g      = random_genome(key, cfg)
    f_raw  = 1.0
    c_act  = 0.0

    cfg_no_cost   = Config(**{**vars(cfg), "lambda_conn": 0.0})
    cfg_with_cost = Config(**{**vars(cfg), "lambda_conn": 1.0})

    f_no_cost   = float(adjusted_fitness(f_raw, g, c_act, cfg_no_cost))
    f_with_cost = float(adjusted_fitness(f_raw, g, c_act, cfg_with_cost))

    assert f_no_cost == pytest.approx(f_raw)
    assert f_with_cost < f_raw


def test_activation_cost_penalty(cfg):
    """With lambda_act > 0 and nonzero c_act, adjusted fitness is reduced."""
    key   = jax.random.PRNGKey(4)
    g     = random_genome(key, cfg)
    f_raw = 1.0
    c_act = 5.0  # nonzero activity

    cfg_no_act   = Config(**{**vars(cfg), "lambda_act": 0.0})
    cfg_with_act = Config(**{**vars(cfg), "lambda_act": 0.1})

    f_no_act   = float(adjusted_fitness(f_raw, g, c_act, cfg_no_act))
    f_with_act = float(adjusted_fitness(f_raw, g, c_act, cfg_with_act))

    assert f_no_act   == pytest.approx(f_raw)
    assert f_with_act  < f_raw
