"""
M7 Tests — Analysis layer.

Tests cover:
  1.  modularity_q — valid range, degenerate cases, two-cluster > fully-connected
  2.  network_stats — keys, values consistent with genome fields
  3.  analyse_genome — keys, internal consistency
  4.  analyse_population — shapes, all Q values valid
  5.  summarise_run — keys, list lengths, scalar ranges
"""

from __future__ import annotations

import sys
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ctrnn_evo import Config, WorldConfig, Genome, random_genome, E, FSI
from ctrnn_evo.evolution import init_population, eval_population, compute_fitness
from ctrnn_evo.analysis import (
    modularity_q,
    network_stats,
    analyse_genome,
    analyse_population,
    summarise_run,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg():
    return Config(N_max=16, n_in=2, n_out=2, K=4, population_size=8, tournament_size=3)


@pytest.fixture(scope="module")
def wcfg():
    return WorldConfig(episode_steps=50)


@pytest.fixture(scope="module")
def genome(cfg):
    return random_genome(jax.random.PRNGKey(0), cfg)


@pytest.fixture(scope="module")
def pop(cfg):
    return init_population(jax.random.PRNGKey(1), cfg)


@pytest.fixture(scope="module")
def history():
    """Minimal fake history list (5 generations)."""
    return [
        {
            "generation": i,
            "max_fitness": 0.5 + i * 0.05,
            "mean_fitness": 0.3 + i * 0.04,
            "max_steps": 50 + i * 2,
            "mean_steps": 30.0 + i * 1.5,
            "mean_n_active": 7.0,
            "mean_conn_cost": 1.2,
        }
        for i in range(5)
    ]


def _make_two_cluster_genome(cfg: Config) -> Genome:
    """
    Construct a genome with a clear two-module structure:
    - Neurons 0..n_in-1  : inputs  (always active, excitatory)
    - Neurons n_in..5    : cluster A (active, strong intra-cluster weights)
    - Neurons 6..9       : cluster B (active, strong intra-cluster weights)
    - Cross-cluster edges: absent
    - Outputs (last n_out): active, connected to each cluster

    Weight matrix has large values within each cluster and zeros across.
    """
    N = cfg.N_max
    n_in  = cfg.n_in
    n_out = cfg.n_out

    # Active mask: inputs + neurons 2..9 + outputs
    active_mask = jnp.zeros(N, dtype=bool)
    cluster_A = list(range(n_in, 6))       # indices 2,3,4,5
    cluster_B = list(range(6, 10))          # indices 6,7,8,9
    output_idxs = list(range(N - n_out, N))
    active_idxs = list(range(n_in)) + cluster_A + cluster_B + output_idxs
    active_mask = active_mask.at[jnp.array(active_idxs)].set(True)

    # Neuron types: all excitatory
    neuron_type = jnp.zeros(N, dtype=jnp.uint8)

    # Tau: excitatory range midpoints
    tau_mid = (cfg.tau_e_range[0] + cfg.tau_e_range[1]) / 2.0
    tau = jnp.full(N, tau_mid)

    # Weights: 1.0 within each cluster, 0 elsewhere
    wm = np.zeros((N, N), dtype=np.float32)
    for i in cluster_A:
        for j in cluster_A:
            wm[i, j] = 1.0
    for i in cluster_B:
        for j in cluster_B:
            wm[i, j] = 1.0
    weight_matrix = jnp.array(wm)

    # Edge mask: mirrors weight structure (only within clusters)
    em = np.zeros((N, N), dtype=bool)
    for i in cluster_A:
        for j in cluster_A:
            if i != j:
                em[i, j] = True
    for i in cluster_B:
        for j in cluster_B:
            if i != j:
                em[i, j] = True
    edge_mask = jnp.array(em)

    # Positions: spread out in unit square
    pos = np.random.default_rng(42).uniform(0, 1, (N, 2)).astype(np.float32)
    position = jnp.array(pos)
    bias = jnp.zeros(N)

    return Genome(
        active_mask=active_mask,
        neuron_type=neuron_type,
        tau=tau,
        bias=bias,
        position=position,
        weight_matrix=weight_matrix,
        edge_mask=edge_mask,
    )


def _make_fully_connected_genome(cfg: Config) -> Genome:
    """
    Genome with all active neurons fully connected — single dense module, low Q.
    Uses the same 10 active neurons as the two-cluster genome.
    """
    N = cfg.N_max
    n_in  = cfg.n_in
    n_out = cfg.n_out

    active_idxs = list(range(n_in)) + list(range(n_in, 10)) + list(range(N - n_out, N))
    active_mask = jnp.zeros(N, dtype=bool).at[jnp.array(active_idxs)].set(True)
    neuron_type = jnp.zeros(N, dtype=jnp.uint8)
    tau_mid = (cfg.tau_e_range[0] + cfg.tau_e_range[1]) / 2.0
    tau = jnp.full(N, tau_mid)

    # All-ones weight matrix between active neurons
    am = np.array(active_mask)
    wm = np.outer(am.astype(float), am.astype(float)).astype(np.float32)
    np.fill_diagonal(wm, 0)
    weight_matrix = jnp.array(wm)
    edge_mask = jnp.array(wm > 0)

    pos = np.random.default_rng(7).uniform(0, 1, (N, 2)).astype(np.float32)
    position = jnp.array(pos)
    bias = jnp.zeros(N)

    return Genome(
        active_mask=active_mask,
        neuron_type=neuron_type,
        tau=tau,
        bias=bias,
        position=position,
        weight_matrix=weight_matrix,
        edge_mask=edge_mask,
    )


# ── 1. modularity_q ───────────────────────────────────────────────────────────

def test_modularity_q_returns_float(genome, cfg):
    q = modularity_q(genome, cfg)
    assert isinstance(q, float)

def test_modularity_q_in_valid_range(cfg):
    """Q must be in (-0.5, 1.0] for any random genome."""
    for seed in range(10):
        g = random_genome(jax.random.PRNGKey(seed), cfg)
        q = modularity_q(g, cfg)
        assert -0.5 - 1e-6 <= q <= 1.0 + 1e-6, f"Q={q:.4f} out of range for seed {seed}"

def test_modularity_q_no_edges_returns_zero(cfg):
    """A genome with no active edges should return Q=0."""
    g = random_genome(jax.random.PRNGKey(10), cfg)
    g = Genome(
        active_mask=g.active_mask,
        neuron_type=g.neuron_type,
        tau=g.tau,
        bias=g.bias,
        position=g.position,
        weight_matrix=g.weight_matrix,
        edge_mask=jnp.zeros_like(g.edge_mask),
    )
    assert modularity_q(g, cfg) == 0.0

def test_modularity_q_single_active_returns_zero(cfg):
    """Fewer than 2 active neurons is degenerate — should return 0."""
    g = random_genome(jax.random.PRNGKey(11), cfg)
    # Force only input slot 0 active (keep I/O constraints by having only 1 neuron)
    # Easier: zero out edge mask and set active to only n_in+n_out slots
    tiny_mask = jnp.zeros(cfg.N_max, dtype=bool)
    tiny_mask = tiny_mask.at[0].set(True)   # single neuron
    g = Genome(
        active_mask=tiny_mask,
        neuron_type=g.neuron_type,
        tau=g.tau,
        bias=g.bias,
        position=g.position,
        weight_matrix=g.weight_matrix,
        edge_mask=jnp.zeros_like(g.edge_mask),
    )
    assert modularity_q(g, cfg) == 0.0

def test_modularity_q_two_clusters_higher_than_fully_connected(cfg):
    """
    A clear two-module block-diagonal network should have higher Q than
    a fully-connected network with the same number of active neurons.
    """
    q_modular  = modularity_q(_make_two_cluster_genome(cfg), cfg)
    q_dense    = modularity_q(_make_fully_connected_genome(cfg), cfg)
    assert q_modular > q_dense, (
        f"Two-cluster Q={q_modular:.4f} should exceed fully-connected Q={q_dense:.4f}"
    )

def test_modularity_q_two_clusters_positive(cfg):
    """A block-diagonal network should have Q > 0."""
    q = modularity_q(_make_two_cluster_genome(cfg), cfg)
    assert q > 0.0, f"Expected Q > 0 for two-cluster network, got {q:.4f}"

def test_modularity_q_deterministic(genome, cfg):
    """Same genome always returns the same Q."""
    q1 = modularity_q(genome, cfg)
    q2 = modularity_q(genome, cfg)
    assert q1 == q2


# ── 2. network_stats ──────────────────────────────────────────────────────────

def test_network_stats_returns_dict(genome, cfg):
    stats = network_stats(genome, cfg)
    assert isinstance(stats, dict)

def test_network_stats_keys(genome, cfg):
    stats = network_stats(genome, cfg)
    expected = {"n_active", "n_edges", "density", "mean_weight", "connection_cost"}
    assert expected <= set(stats.keys()), f"Missing keys: {expected - set(stats.keys())}"

def test_network_stats_n_active_matches_mask(genome, cfg):
    stats = network_stats(genome, cfg)
    expected_n_active = int(jnp.sum(genome.active_mask))
    assert stats["n_active"] == expected_n_active

def test_network_stats_n_edges_matches_mask(genome, cfg):
    stats = network_stats(genome, cfg)
    # Edges = active edge_mask entries between two active neurons
    expected = int(jnp.sum(
        genome.edge_mask
        & genome.active_mask[:, None]
        & genome.active_mask[None, :]
    ))
    assert stats["n_edges"] == expected

def test_network_stats_density_range(cfg):
    for seed in range(5):
        g = random_genome(jax.random.PRNGKey(seed + 20), cfg)
        stats = network_stats(g, cfg)
        assert 0.0 <= stats["density"] <= 1.0 + 1e-6, \
            f"density {stats['density']:.4f} out of [0, 1]"

def test_network_stats_connection_cost_nonneg(genome, cfg):
    stats = network_stats(genome, cfg)
    assert stats["connection_cost"] >= 0.0

def test_network_stats_mean_weight_nonneg(genome, cfg):
    stats = network_stats(genome, cfg)
    assert stats["mean_weight"] >= 0.0

def test_network_stats_no_edges_mean_weight_zero(cfg):
    g = random_genome(jax.random.PRNGKey(30), cfg)
    g = Genome(
        active_mask=g.active_mask, neuron_type=g.neuron_type,
        tau=g.tau, bias=g.bias, position=g.position,
        weight_matrix=g.weight_matrix,
        edge_mask=jnp.zeros_like(g.edge_mask),
    )
    stats = network_stats(g, cfg)
    assert stats["mean_weight"] == 0.0
    assert stats["n_edges"] == 0
    assert stats["density"] == 0.0


# ── 3. analyse_genome ─────────────────────────────────────────────────────────

def test_analyse_genome_keys(genome, cfg):
    result = analyse_genome(genome, cfg)
    expected = {"q", "n_active", "n_edges", "density", "mean_weight", "connection_cost"}
    assert expected <= set(result.keys()), f"Missing keys: {expected - set(result.keys())}"

def test_analyse_genome_all_scalars(genome, cfg):
    result = analyse_genome(genome, cfg)
    for k, v in result.items():
        assert isinstance(v, (int, float)), f"Value for '{k}' is {type(v)}, expected scalar"

def test_analyse_genome_q_consistent(genome, cfg):
    """q from analyse_genome must equal standalone modularity_q."""
    q_standalone = modularity_q(genome, cfg)
    q_combined   = analyse_genome(genome, cfg)["q"]
    assert abs(q_combined - q_standalone) < 1e-9

def test_analyse_genome_n_active_consistent(genome, cfg):
    """n_active from analyse_genome must equal standalone network_stats."""
    stats = network_stats(genome, cfg)
    combined = analyse_genome(genome, cfg)
    assert combined["n_active"] == stats["n_active"]


# ── 4. analyse_population ─────────────────────────────────────────────────────

def test_analyse_population_returns_dict(pop, cfg):
    result = analyse_population(pop, cfg)
    assert isinstance(result, dict)

def test_analyse_population_keys(pop, cfg):
    result = analyse_population(pop, cfg)
    expected = {"q", "n_active", "n_edges", "density", "mean_weight", "connection_cost"}
    assert expected <= set(result.keys())

def test_analyse_population_list_lengths(pop, cfg):
    result = analyse_population(pop, cfg)
    for k, v in result.items():
        assert len(v) == cfg.population_size, \
            f"Key '{k}' has length {len(v)}, expected {cfg.population_size}"

def test_analyse_population_q_values_valid(pop, cfg):
    result = analyse_population(pop, cfg)
    for i, q in enumerate(result["q"]):
        assert -0.5 - 1e-6 <= q <= 1.0 + 1e-6, \
            f"Q={q:.4f} out of range for genome {i}"

def test_analyse_population_n_active_positive(pop, cfg):
    result = analyse_population(pop, cfg)
    for n in result["n_active"]:
        assert n >= cfg.n_in + cfg.n_out, \
            f"n_active={n} less than minimum I/O neurons"


# ── 5. summarise_run ─────────────────────────────────────────────────────────

def test_summarise_run_returns_dict(history, pop, genome, cfg):
    result = summarise_run(history, pop, genome, cfg)
    assert isinstance(result, dict)

def test_summarise_run_expected_keys(history, pop, genome, cfg):
    result = summarise_run(history, pop, genome, cfg)
    expected = {
        "fitness_max", "fitness_mean", "steps_mean",
        "final_q_mean", "final_q_max",
        "final_n_active_mean", "final_conn_cost_mean",
        "best_q", "best_n_active",
    }
    assert expected <= set(result.keys()), \
        f"Missing keys: {expected - set(result.keys())}"

def test_summarise_run_fitness_list_length(history, pop, genome, cfg):
    result = summarise_run(history, pop, genome, cfg)
    n = len(history)
    assert len(result["fitness_max"])  == n
    assert len(result["fitness_mean"]) == n
    assert len(result["steps_mean"])   == n

def test_summarise_run_fitness_values_from_history(history, pop, genome, cfg):
    result = summarise_run(history, pop, genome, cfg)
    for i, h in enumerate(history):
        assert abs(result["fitness_max"][i]  - h["max_fitness"])  < 1e-9
        assert abs(result["fitness_mean"][i] - h["mean_fitness"]) < 1e-9
        assert abs(result["steps_mean"][i]   - h["mean_steps"])   < 1e-9

def test_summarise_run_best_q_valid(history, pop, genome, cfg):
    result = summarise_run(history, pop, genome, cfg)
    assert -0.5 - 1e-6 <= result["best_q"] <= 1.0 + 1e-6

def test_summarise_run_best_n_active_valid(history, pop, genome, cfg):
    result = summarise_run(history, pop, genome, cfg)
    assert result["best_n_active"] >= cfg.n_in + cfg.n_out
    assert result["best_n_active"] <= cfg.N_max

def test_summarise_run_final_q_mean_valid(history, pop, genome, cfg):
    result = summarise_run(history, pop, genome, cfg)
    assert -0.5 - 1e-6 <= result["final_q_mean"] <= 1.0 + 1e-6

def test_summarise_run_final_q_max_ge_mean(history, pop, genome, cfg):
    result = summarise_run(history, pop, genome, cfg)
    assert result["final_q_max"] >= result["final_q_mean"] - 1e-6
