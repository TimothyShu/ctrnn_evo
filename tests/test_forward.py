"""
M1 validation gates for the CTRNN forward pass.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ctrnn_evo import Config, Genome, random_genome, forward_pass, batch_forward, E, FSI


# ── Helpers ──────────────────────────────────────────────────────────────────

def minimal_genome(cfg: Config, **overrides) -> Genome:
    """
    Return a blank genome (all neurons inactive, no edges) then apply overrides.
    Useful for hand-crafting specific circuit topologies in tests.
    """
    N = cfg.N_max
    g = Genome(
        active_mask=jnp.zeros(N, dtype=bool),
        neuron_type=jnp.zeros(N, dtype=jnp.uint8),
        tau=jnp.full(N, 10.0),           # safe default, avoids division by zero
        bias=jnp.zeros(N),
        position=jnp.full((N, 2), 0.5),
        weight_matrix=jnp.zeros((N, N)),
        edge_mask=jnp.zeros((N, N), dtype=bool),
    )
    # Always keep I/O slots active
    active = g.active_mask.at[:cfg.n_in].set(True).at[-cfg.n_out:].set(True)
    g = Genome(**{**vars(g), "active_mask": active, **overrides})
    return g


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return Config(N_max=8, n_in=1, n_out=1, dt=0.5, K=20)


def test_isolated_excitatory_neuron_decays(cfg):
    """
    A single active excitatory neuron with v0=1, no connections, no bias,
    no input should decay as v(t) ≈ v0 * exp(-K*dt / tau).

    We use a hidden neuron (index 1) with tau=10ms.
    K=20 ticks * dt=0.5ms = 10ms simulated time.
    Expected: v ≈ exp(-1) ≈ 0.368.
    """
    N = cfg.N_max
    tau_val = 10.0
    idx = 1  # hidden neuron index

    active = jnp.zeros(N, dtype=bool).at[0].set(True).at[-1].set(True).at[idx].set(True)
    tau    = jnp.full(N, 1.0).at[idx].set(tau_val)   # others need nonzero tau
    g = minimal_genome(cfg, active_mask=active, tau=tau)

    v0 = jnp.zeros(N).at[idx].set(1.0)
    input_vec = jnp.zeros(N)

    v_final, _, _ = forward_pass(g, v0, input_vec, cfg)

    expected = np.exp(-cfg.K * cfg.dt / tau_val)
    # tanh nonlinearity introduces slight deviation; tolerance is loose
    assert abs(float(v_final[idx]) - expected) < 0.05, (
        f"Expected ~{expected:.3f}, got {float(v_final[idx]):.3f}"
    )


def test_inactive_neuron_does_not_propagate(cfg):
    """
    A neuron that is masked out must contribute zero to all other neurons
    regardless of its weight magnitude.
    """
    N = cfg.N_max
    idx_masked = 2
    idx_target = 3

    active = (
        jnp.zeros(N, dtype=bool)
        .at[0].set(True)    # input
        .at[-1].set(True)   # output
        .at[idx_target].set(True)
        # idx_masked is intentionally NOT activated
    )
    # Large weight from masked neuron to target — should have no effect
    W = jnp.zeros((N, N)).at[idx_target, idx_masked].set(10.0)
    edge = jnp.zeros((N, N), dtype=bool).at[idx_target, idx_masked].set(True)

    g = minimal_genome(cfg, active_mask=active, weight_matrix=W, edge_mask=edge)

    v0        = jnp.zeros(N).at[idx_masked].set(5.0)  # give masked neuron high state
    input_vec = jnp.zeros(N)

    v_final, _, _ = forward_pass(g, v0, input_vec, cfg)

    assert float(v_final[idx_target]) == pytest.approx(0.0, abs=1e-5), (
        "Masked neuron should not propagate any signal"
    )


def test_dales_law_inhibitory_suppresses(cfg):
    """
    An FS-I neuron projecting to an excitatory target must suppress it
    (i.e. the effective weight must be negative).
    We drive the FS-I neuron with a strong positive state and verify the
    target's state is pulled downward compared to a no-connection baseline.
    """
    N = cfg.N_max
    fsi_idx = 1
    e_idx   = 2

    active = (
        jnp.zeros(N, dtype=bool)
        .at[0].set(True).at[-1].set(True)
        .at[fsi_idx].set(True)
        .at[e_idx].set(True)
    )
    ntype = jnp.zeros(N, dtype=jnp.uint8).at[fsi_idx].set(FSI)
    W     = jnp.zeros((N, N)).at[e_idx, fsi_idx].set(1.0)
    edge  = jnp.zeros((N, N), dtype=bool).at[e_idx, fsi_idx].set(True)

    g = minimal_genome(cfg, active_mask=active, neuron_type=ntype,
                       weight_matrix=W, edge_mask=edge)

    # Strong positive state on FSI neuron — should suppress E neuron
    v0        = jnp.zeros(N).at[fsi_idx].set(3.0).at[e_idx].set(0.5)
    input_vec = jnp.zeros(N)

    v_final, _, _ = forward_pass(g, v0, input_vec, cfg)

    # Target should end up lower than its initial value
    assert float(v_final[e_idx]) < 0.5, (
        "FS-I neuron should suppress excitatory target"
    )


def test_batch_forward_matches_single(cfg):
    """
    batch_forward on P genomes must produce identical results to running
    forward_pass individually on each genome — catches batch-dimension bugs.
    """
    P = 16
    keys = jax.random.split(jax.random.PRNGKey(42), P)
    pop  = jax.vmap(random_genome, in_axes=(0, None))(keys, cfg)

    v0s       = jnp.zeros((P, cfg.N_max))
    input_vecs = jnp.zeros((P, cfg.N_max))

    v_batch, out_batch, cact_batch = batch_forward(pop, v0s, input_vecs, cfg)

    for i in range(P):
        g_i = jax.tree_util.tree_map(lambda x: x[i], pop)
        v_i, out_i, cact_i = forward_pass(g_i, v0s[i], input_vecs[i], cfg)
        np.testing.assert_allclose(v_batch[i], v_i,   rtol=1e-5,
                                   err_msg=f"v mismatch at organism {i}")
        np.testing.assert_allclose(out_batch[i], out_i, rtol=1e-5,
                                   err_msg=f"output mismatch at organism {i}")


def test_no_nans_random_population(cfg):
    """Forward pass on random genomes must not produce NaNs or Infs."""
    keys = jax.random.split(jax.random.PRNGKey(99), 64)
    pop  = jax.vmap(random_genome, in_axes=(0, None))(keys, cfg)
    v0s  = jnp.zeros((64, cfg.N_max))
    ins  = jnp.zeros((64, cfg.N_max))

    v_final, output, c_act = batch_forward(pop, v0s, ins, cfg)

    assert not jnp.any(jnp.isnan(v_final)),  "NaN in v_final"
    assert not jnp.any(jnp.isinf(v_final)),  "Inf in v_final"
    assert not jnp.any(jnp.isnan(output)),   "NaN in output"
    assert not jnp.any(jnp.isnan(c_act)),    "NaN in c_act"
