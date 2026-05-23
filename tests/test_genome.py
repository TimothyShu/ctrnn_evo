import jax
import jax.numpy as jnp
import pytest
from ctrnn_evo import Config, Genome, random_genome, validate_genome, E, FSI, SII


@pytest.fixture
def cfg():
    return Config(N_max=16, n_out=2)


def test_random_genome_is_valid(cfg):
    key = jax.random.PRNGKey(0)
    g = random_genome(key, cfg)
    validate_genome(g, cfg)


def test_io_slots_always_active(cfg):
    key = jax.random.PRNGKey(1)
    g = random_genome(key, cfg)
    assert jnp.all(g.active_mask[:cfg.n_in]),  "Input slots must be active"
    assert jnp.all(g.active_mask[-cfg.n_out:]), "Output slots must be active"


def test_io_slots_are_excitatory(cfg):
    key = jax.random.PRNGKey(2)
    g = random_genome(key, cfg)
    assert jnp.all(g.neuron_type[:cfg.n_in]  == E)
    assert jnp.all(g.neuron_type[-cfg.n_out:] == E)


def test_weight_magnitudes_nonnegative(cfg):
    key = jax.random.PRNGKey(3)
    g = random_genome(key, cfg)
    assert jnp.all(g.weight_matrix >= 0)


def test_tau_within_type_ranges(cfg):
    key = jax.random.PRNGKey(4)
    g = random_genome(key, cfg)
    tau_lo = jnp.array([cfg.tau_e_range[0], cfg.tau_fsi_range[0], cfg.tau_sii_range[0]])
    tau_hi = jnp.array([cfg.tau_e_range[1], cfg.tau_fsi_range[1], cfg.tau_sii_range[1]])
    lo = tau_lo[g.neuron_type]
    hi = tau_hi[g.neuron_type]
    active = g.active_mask
    assert jnp.all(jnp.where(active, g.tau >= lo - 1e-4, True))
    assert jnp.all(jnp.where(active, g.tau <= hi + 1e-4, True))


def test_positions_in_unit_square(cfg):
    key = jax.random.PRNGKey(5)
    g = random_genome(key, cfg)
    assert jnp.all(g.position >= 0) and jnp.all(g.position <= 1)


def test_genome_is_jax_pytree(cfg):
    key = jax.random.PRNGKey(6)
    g = random_genome(key, cfg)
    leaves, treedef = jax.tree_util.tree_flatten(g)
    g2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jnp.array_equal(g.tau, g2.tau)


def test_batch_genome_vmappable(cfg):
    """random_genome vmapped over a batch of keys produces a valid batched Genome."""
    keys = jax.random.split(jax.random.PRNGKey(7), 8)
    batch = jax.vmap(random_genome, in_axes=(0, None))(keys, cfg)
    assert batch.active_mask.shape == (8, cfg.N_max)
    assert batch.weight_matrix.shape == (8, cfg.N_max, cfg.N_max)
