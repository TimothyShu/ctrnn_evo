"""
Milestone 2 — Mutation operator tests.

Written before the implementation (TDD). These tests define the required
interface and all validation gates from the milestone spec:

  - Every operator produces a genome that passes validate_genome()
  - Repeated mutation does not drift fields that should not change
  - Every operator is vmappable over a population batch
  - Structural statistics drift in sensible directions under repeated mutation
  - I/O neuron slots are never structurally mutated

Expected mutation interface (all in ctrnn_evo.mutation):

    perturb_weights(key, genome, cfg, *, sigma=0.1)   -> Genome
    perturb_tau    (key, genome, cfg, *, sigma=0.1)   -> Genome
    perturb_bias   (key, genome, cfg, *, sigma=0.1)   -> Genome
    perturb_position(key, genome, cfg, *, sigma=0.05) -> Genome
    type_flip      (key, genome, cfg, *, flip_prob=0.05) -> Genome
    add_node       (key, genome, cfg)                 -> Genome
    remove_node    (key, genome, cfg)                 -> Genome
    add_edge       (key, genome, cfg)                 -> Genome
    remove_edge    (key, genome, cfg)                 -> Genome
    mutate         (key, genome, cfg, rates)          -> Genome
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ctrnn_evo import Config, random_genome, validate_genome, E, FSI, SII
from dataclasses import replace

from ctrnn_evo.mutation import (
    perturb_weights,
    perturb_tau,
    perturb_bias,
    perturb_position,
    type_flip,
    add_node,
    remove_node,
    add_edge,
    remove_edge,
    mutate,
    prune_isolated,
    MutationRates,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    # Small network so tests run fast; large enough for structural ops to have room
    return Config(N_max=16, n_out=2)


@pytest.fixture
def genome(cfg):
    return random_genome(jax.random.PRNGKey(0), cfg)


@pytest.fixture
def genome_sparse(cfg):
    """Genome with very few active nodes and edges — for add_* edge cases."""
    g = random_genome(jax.random.PRNGKey(1), cfg, n_active=cfg.n_in + cfg.n_out)
    # Clear all edges
    return g.__class__(**{**vars(g), "edge_mask": jnp.zeros_like(g.edge_mask)})


@pytest.fixture
def genome_full(cfg):
    """Genome with all nodes active — for add_node boundary case."""
    g = random_genome(jax.random.PRNGKey(2), cfg, n_active=cfg.N_max)
    return g


# ── Helpers ───────────────────────────────────────────────────────────────────

def active_count(g):
    return int(jnp.sum(g.active_mask))


def edge_count(g):
    return int(jnp.sum(g.edge_mask & g.active_mask[:, None] & g.active_mask[None, :]))


def fields_equal(g1, g2, *field_names):
    """Assert that listed fields are identical between two genomes."""
    for name in field_names:
        a, b = getattr(g1, name), getattr(g2, name)
        if not jnp.array_equal(a, b):
            raise AssertionError(f"Field '{name}' changed unexpectedly")


# ─────────────────────────────────────────────────────────────────────────────
# perturb_weights
# ─────────────────────────────────────────────────────────────────────────────

class TestPerturbWeights:
    def test_valid_after(self, genome, cfg):
        g2 = perturb_weights(jax.random.PRNGKey(10), genome, cfg)
        validate_genome(g2, cfg)

    def test_weights_remain_nonnegative(self, genome, cfg):
        g2 = perturb_weights(jax.random.PRNGKey(11), genome, cfg, sigma=10.0)
        assert jnp.all(g2.weight_matrix >= 0), "Weights went negative after perturbation"

    def test_only_weight_matrix_changes(self, genome, cfg):
        g2 = perturb_weights(jax.random.PRNGKey(12), genome, cfg)
        fields_equal(genome, g2, "active_mask", "neuron_type", "tau",
                     "bias", "position", "edge_mask")

    def test_weights_actually_change(self, genome, cfg):
        g2 = perturb_weights(jax.random.PRNGKey(13), genome, cfg, sigma=1.0)
        assert not jnp.array_equal(genome.weight_matrix, g2.weight_matrix)

    def test_vmappable(self, cfg):
        keys = jax.random.split(jax.random.PRNGKey(14), 32)
        pop  = jax.vmap(random_genome, in_axes=(0, None))(keys, cfg)
        keys2 = jax.random.split(jax.random.PRNGKey(15), 32)
        pop2 = jax.vmap(perturb_weights, in_axes=(0, 0, None))(keys2, pop, cfg)
        assert pop2.weight_matrix.shape == (32, cfg.N_max, cfg.N_max)


# ─────────────────────────────────────────────────────────────────────────────
# perturb_tau
# ─────────────────────────────────────────────────────────────────────────────

class TestPerturbTau:
    def test_valid_after(self, genome, cfg):
        g2 = perturb_tau(jax.random.PRNGKey(20), genome, cfg)
        validate_genome(g2, cfg)

    def test_tau_clamped_to_type_range(self, genome, cfg):
        # Large sigma to stress-test clamping
        g2 = perturb_tau(jax.random.PRNGKey(21), genome, cfg, sigma=500.0)
        tau_lo = jnp.array([cfg.tau_e_range[0], cfg.tau_fsi_range[0], cfg.tau_sii_range[0]])
        tau_hi = jnp.array([cfg.tau_e_range[1], cfg.tau_fsi_range[1], cfg.tau_sii_range[1]])
        lo = tau_lo[g2.neuron_type]
        hi = tau_hi[g2.neuron_type]
        assert jnp.all(g2.tau >= lo - 1e-4)
        assert jnp.all(g2.tau <= hi + 1e-4)

    def test_only_tau_changes(self, genome, cfg):
        g2 = perturb_tau(jax.random.PRNGKey(22), genome, cfg)
        fields_equal(genome, g2, "active_mask", "neuron_type", "bias",
                     "position", "weight_matrix", "edge_mask")

    def test_vmappable(self, cfg):
        keys  = jax.random.split(jax.random.PRNGKey(23), 32)
        pop   = jax.vmap(random_genome, in_axes=(0, None))(keys, cfg)
        keys2 = jax.random.split(jax.random.PRNGKey(24), 32)
        pop2  = jax.vmap(perturb_tau, in_axes=(0, 0, None))(keys2, pop, cfg)
        assert pop2.tau.shape == (32, cfg.N_max)


# ─────────────────────────────────────────────────────────────────────────────
# perturb_bias
# ─────────────────────────────────────────────────────────────────────────────

class TestPerturbBias:
    def test_valid_after(self, genome, cfg):
        g2 = perturb_bias(jax.random.PRNGKey(30), genome, cfg)
        validate_genome(g2, cfg)

    def test_only_bias_changes(self, genome, cfg):
        g2 = perturb_bias(jax.random.PRNGKey(31), genome, cfg)
        fields_equal(genome, g2, "active_mask", "neuron_type", "tau",
                     "position", "weight_matrix", "edge_mask")

    def test_bias_actually_changes(self, genome, cfg):
        g2 = perturb_bias(jax.random.PRNGKey(32), genome, cfg, sigma=1.0)
        assert not jnp.array_equal(genome.bias, g2.bias)

    def test_vmappable(self, cfg):
        keys  = jax.random.split(jax.random.PRNGKey(33), 32)
        pop   = jax.vmap(random_genome, in_axes=(0, None))(keys, cfg)
        keys2 = jax.random.split(jax.random.PRNGKey(34), 32)
        pop2  = jax.vmap(perturb_bias, in_axes=(0, 0, None))(keys2, pop, cfg)
        assert pop2.bias.shape == (32, cfg.N_max)


# ─────────────────────────────────────────────────────────────────────────────
# perturb_position
# ─────────────────────────────────────────────────────────────────────────────

class TestPerturbPosition:
    def test_valid_after(self, genome, cfg):
        g2 = perturb_position(jax.random.PRNGKey(40), genome, cfg)
        validate_genome(g2, cfg)

    def test_positions_stay_in_unit_square(self, genome, cfg):
        g2 = perturb_position(jax.random.PRNGKey(41), genome, cfg, sigma=10.0)
        assert jnp.all(g2.position >= 0.0)
        assert jnp.all(g2.position <= 1.0)

    def test_only_position_changes(self, genome, cfg):
        g2 = perturb_position(jax.random.PRNGKey(42), genome, cfg)
        fields_equal(genome, g2, "active_mask", "neuron_type", "tau",
                     "bias", "weight_matrix", "edge_mask")

    def test_vmappable(self, cfg):
        keys  = jax.random.split(jax.random.PRNGKey(43), 32)
        pop   = jax.vmap(random_genome, in_axes=(0, None))(keys, cfg)
        keys2 = jax.random.split(jax.random.PRNGKey(44), 32)
        pop2  = jax.vmap(perturb_position, in_axes=(0, 0, None))(keys2, pop, cfg)
        assert pop2.position.shape == (32, cfg.N_max, 2)


# ─────────────────────────────────────────────────────────────────────────────
# type_flip
# ─────────────────────────────────────────────────────────────────────────────

class TestTypeFlip:
    def test_valid_after(self, genome, cfg):
        g2 = type_flip(jax.random.PRNGKey(50), genome, cfg, flip_prob=0.5)
        validate_genome(g2, cfg)

    def test_tau_reclamped_to_new_type(self, genome, cfg):
        g2 = type_flip(jax.random.PRNGKey(51), genome, cfg, flip_prob=1.0)
        tau_lo = jnp.array([cfg.tau_e_range[0], cfg.tau_fsi_range[0], cfg.tau_sii_range[0]])
        tau_hi = jnp.array([cfg.tau_e_range[1], cfg.tau_fsi_range[1], cfg.tau_sii_range[1]])
        lo = tau_lo[g2.neuron_type]
        hi = tau_hi[g2.neuron_type]
        assert jnp.all(jnp.where(g2.active_mask, g2.tau >= lo - 1e-4, True))
        assert jnp.all(jnp.where(g2.active_mask, g2.tau <= hi + 1e-4, True))

    def test_io_slots_never_flip(self, genome, cfg):
        # Run many flips; I/O neuron types must never change
        _flip = jax.jit(lambda key, g: type_flip(key, g, cfg, flip_prob=1.0))
        g = genome
        for i in range(20):
            g = _flip(jax.random.PRNGKey(52 + i), g)
        assert jnp.all(g.neuron_type[:cfg.n_in]  == E), "Input neurons changed type"
        assert jnp.all(g.neuron_type[-cfg.n_out:] == E), "Output neurons changed type"

    def test_types_change_at_high_prob(self, genome, cfg):
        g2 = type_flip(jax.random.PRNGKey(53), genome, cfg, flip_prob=1.0)
        # At least some hidden neurons should have changed type
        n_hidden = cfg.N_max - cfg.n_in - cfg.n_out
        if n_hidden > 0:
            changed = ~jnp.array_equal(
                genome.neuron_type[cfg.n_in:-cfg.n_out],
                g2.neuron_type[cfg.n_in:-cfg.n_out],
            )
            assert changed, "No neuron changed type at flip_prob=1.0"

    def test_structural_fields_unchanged(self, genome, cfg):
        g2 = type_flip(jax.random.PRNGKey(54), genome, cfg)
        fields_equal(genome, g2, "active_mask", "bias", "position",
                     "weight_matrix", "edge_mask")

    def test_vmappable(self, cfg):
        keys  = jax.random.split(jax.random.PRNGKey(55), 32)
        pop   = jax.vmap(random_genome, in_axes=(0, None))(keys, cfg)
        keys2 = jax.random.split(jax.random.PRNGKey(56), 32)
        pop2  = jax.vmap(type_flip, in_axes=(0, 0, None))(keys2, pop, cfg)
        assert pop2.neuron_type.shape == (32, cfg.N_max)


# ─────────────────────────────────────────────────────────────────────────────
# add_node
# ─────────────────────────────────────────────────────────────────────────────

class TestAddNode:
    def test_valid_after(self, genome, cfg):
        g2 = add_node(jax.random.PRNGKey(60), genome, cfg)
        validate_genome(g2, cfg)

    def test_node_count_increases_when_room(self, genome_sparse, cfg):
        n_before = active_count(genome_sparse)
        g2 = add_node(jax.random.PRNGKey(61), genome_sparse, cfg)
        n_after = active_count(g2)
        assert n_after == n_before + 1, (
            f"Expected {n_before + 1} active nodes, got {n_after}"
        )

    def test_no_change_when_full(self, genome_full, cfg):
        n_before = active_count(genome_full)
        g2 = add_node(jax.random.PRNGKey(62), genome_full, cfg)
        assert active_count(g2) == n_before, "add_node on full genome should be a no-op"
        validate_genome(g2, cfg)

    def test_new_node_has_valid_tau(self, genome_sparse, cfg):
        g2 = add_node(jax.random.PRNGKey(63), genome_sparse, cfg)
        # Find the newly activated slot
        new_slots = g2.active_mask & ~genome_sparse.active_mask
        tau_lo = jnp.array([cfg.tau_e_range[0], cfg.tau_fsi_range[0], cfg.tau_sii_range[0]])
        tau_hi = jnp.array([cfg.tau_e_range[1], cfg.tau_fsi_range[1], cfg.tau_sii_range[1]])
        lo = tau_lo[g2.neuron_type]
        hi = tau_hi[g2.neuron_type]
        assert jnp.all(jnp.where(new_slots, g2.tau >= lo - 1e-4, True))
        assert jnp.all(jnp.where(new_slots, g2.tau <= hi + 1e-4, True))

    def test_io_slots_unaffected(self, genome_sparse, cfg):
        g2 = add_node(jax.random.PRNGKey(64), genome_sparse, cfg)
        assert jnp.array_equal(g2.active_mask[:cfg.n_in],  genome_sparse.active_mask[:cfg.n_in])
        assert jnp.array_equal(g2.active_mask[-cfg.n_out:], genome_sparse.active_mask[-cfg.n_out:])

    def test_new_node_has_one_incoming_and_one_outgoing_edge(self, genome_sparse, cfg):
        """New node must have ≥1 incoming and ≥1 outgoing edge among active neurons."""
        g2 = add_node(jax.random.PRNGKey(67), genome_sparse, cfg)
        new_slots = np.where(np.array(g2.active_mask) & ~np.array(genome_sparse.active_mask))[0]
        assert len(new_slots) == 1, "Expected exactly one new node"
        slot = int(new_slots[0])
        active = np.array(g2.active_mask)
        has_in  = bool(np.any(np.array(g2.edge_mask)[:, slot] & active))
        has_out = bool(np.any(np.array(g2.edge_mask)[slot, :] & active))
        assert has_in,  "New node has no incoming edges"
        assert has_out, "New node has no outgoing edges"

    def test_vmappable(self, cfg):
        keys  = jax.random.split(jax.random.PRNGKey(65), 32)
        pop   = jax.vmap(random_genome, in_axes=(0, None))(keys, cfg)
        keys2 = jax.random.split(jax.random.PRNGKey(66), 32)
        pop2  = jax.vmap(add_node, in_axes=(0, 0, None))(keys2, pop, cfg)
        assert pop2.active_mask.shape == (32, cfg.N_max)


# ─────────────────────────────────────────────────────────────────────────────
# remove_node
# ─────────────────────────────────────────────────────────────────────────────

class TestRemoveNode:
    def test_valid_after(self, genome, cfg):
        g2 = remove_node(jax.random.PRNGKey(70), genome, cfg)
        validate_genome(g2, cfg)

    def test_node_count_decreases_when_hidden_exist(self, genome, cfg):
        n_before = active_count(genome)
        n_hidden_active = n_before - cfg.n_in - cfg.n_out
        g2 = remove_node(jax.random.PRNGKey(71), genome, cfg)
        n_after = active_count(g2)
        if n_hidden_active > 0:
            assert n_after == n_before - 1
        else:
            assert n_after == n_before  # no-op when nothing to remove

    def test_io_slots_never_removed(self, genome, cfg):
        _remove = jax.jit(lambda key, g: remove_node(key, g, cfg))
        g = genome
        for i in range(20):
            g = _remove(jax.random.PRNGKey(72 + i), g)
        assert jnp.all(g.active_mask[:cfg.n_in]),  "Input neurons were removed"
        assert jnp.all(g.active_mask[-cfg.n_out:]), "Output neurons were removed"

    def test_no_change_when_no_hidden(self, genome_sparse, cfg):
        # genome_sparse has only I/O neurons active
        n_before = active_count(genome_sparse)
        g2 = remove_node(jax.random.PRNGKey(73), genome_sparse, cfg)
        assert active_count(g2) == n_before, "remove_node with no hidden should be a no-op"
        validate_genome(g2, cfg)

    def test_removed_node_edges_also_cleared(self, genome, cfg):
        g2 = remove_node(jax.random.PRNGKey(74), genome, cfg)
        removed = genome.active_mask & ~g2.active_mask
        # Edges to/from the removed neuron must be cleared
        if jnp.any(removed):
            assert jnp.all(~g2.edge_mask[removed, :]), "Outgoing edges of removed node not cleared"
            assert jnp.all(~g2.edge_mask[:, removed]), "Incoming edges of removed node not cleared"

    def test_vmappable(self, cfg):
        keys  = jax.random.split(jax.random.PRNGKey(75), 32)
        pop   = jax.vmap(random_genome, in_axes=(0, None))(keys, cfg)
        keys2 = jax.random.split(jax.random.PRNGKey(76), 32)
        pop2  = jax.vmap(remove_node, in_axes=(0, 0, None))(keys2, pop, cfg)
        assert pop2.active_mask.shape == (32, cfg.N_max)


# ─────────────────────────────────────────────────────────────────────────────
# add_edge
# ─────────────────────────────────────────────────────────────────────────────

class TestAddEdge:
    def test_valid_after(self, genome, cfg):
        g2 = add_edge(jax.random.PRNGKey(80), genome, cfg)
        validate_genome(g2, cfg)

    def test_edge_count_increases_when_room(self, genome_sparse, cfg):
        # Add two active hidden neurons so there's room for edges
        g = add_node(jax.random.PRNGKey(80), genome_sparse, cfg)
        g = add_node(jax.random.PRNGKey(81), g, cfg)
        n_before = edge_count(g)
        g2 = add_edge(jax.random.PRNGKey(82), g, cfg)
        assert edge_count(g2) == n_before + 1

    def test_new_edge_has_positive_weight(self, genome_sparse, cfg):
        g = add_node(jax.random.PRNGKey(83), genome_sparse, cfg)
        g = add_node(jax.random.PRNGKey(84), g, cfg)
        g2 = add_edge(jax.random.PRNGKey(85), g, cfg)
        new_edges = g2.edge_mask & ~g.edge_mask
        assert jnp.all(jnp.where(new_edges, g2.weight_matrix > 0, True)), \
            "Newly added edge should have a positive weight magnitude"

    def test_only_active_neuron_pairs_get_edges(self, genome, cfg):
        g2 = add_edge(jax.random.PRNGKey(86), genome, cfg)
        # No edge may exist between two inactive neurons
        inactive = ~g2.active_mask
        assert jnp.all(~g2.edge_mask[inactive[:, None] | inactive[None, :]])

    def test_vmappable(self, cfg):
        keys  = jax.random.split(jax.random.PRNGKey(87), 32)
        pop   = jax.vmap(random_genome, in_axes=(0, None))(keys, cfg)
        keys2 = jax.random.split(jax.random.PRNGKey(88), 32)
        pop2  = jax.vmap(add_edge, in_axes=(0, 0, None))(keys2, pop, cfg)
        assert pop2.edge_mask.shape == (32, cfg.N_max, cfg.N_max)


# ─────────────────────────────────────────────────────────────────────────────
# remove_edge
# ─────────────────────────────────────────────────────────────────────────────

class TestRemoveEdge:
    def test_valid_after(self, genome, cfg):
        g2 = remove_edge(jax.random.PRNGKey(90), genome, cfg)
        validate_genome(g2, cfg)

    def test_edge_count_decreases_when_edges_exist(self, genome, cfg):
        n_before = edge_count(genome)
        g2 = remove_edge(jax.random.PRNGKey(91), genome, cfg)
        n_after = edge_count(g2)
        if n_before > 0:
            assert n_after == n_before - 1
        else:
            assert n_after == 0  # no-op

    def test_no_change_when_no_edges(self, genome_sparse, cfg):
        n_before = edge_count(genome_sparse)
        assert n_before == 0
        g2 = remove_edge(jax.random.PRNGKey(92), genome_sparse, cfg)
        assert edge_count(g2) == 0
        validate_genome(g2, cfg)

    def test_vmappable(self, cfg):
        keys  = jax.random.split(jax.random.PRNGKey(93), 32)
        pop   = jax.vmap(random_genome, in_axes=(0, None))(keys, cfg)
        keys2 = jax.random.split(jax.random.PRNGKey(94), 32)
        pop2  = jax.vmap(remove_edge, in_axes=(0, 0, None))(keys2, pop, cfg)
        assert pop2.edge_mask.shape == (32, cfg.N_max, cfg.N_max)


# ─────────────────────────────────────────────────────────────────────────────
# prune_isolated
# ─────────────────────────────────────────────────────────────────────────────

class TestPruneIsolated:
    def test_isolated_hidden_neuron_deactivated(self, genome, cfg):
        """A hidden neuron with no active edges should be deactivated."""
        g = add_node(jax.random.PRNGKey(200), genome, cfg)
        new_slot = int(np.argmax(np.array(g.active_mask) & ~np.array(genome.active_mask)))
        # Manually strip all edges for that slot
        new_edges = g.edge_mask.at[new_slot, :].set(False).at[:, new_slot].set(False)
        g_isolated = replace(g, edge_mask=new_edges)
        g_pruned = prune_isolated(g_isolated, cfg)
        assert not bool(g_pruned.active_mask[new_slot]), "Isolated hidden node was not pruned"

    def test_connected_neurons_preserved(self, genome, cfg):
        """Neurons that have at least one active edge must not be deactivated."""
        active_pairs = genome.active_mask[:, None] & genome.active_mask[None, :]
        has_edge = (
            jnp.any(genome.edge_mask & active_pairs, axis=0) |
            jnp.any(genome.edge_mask & active_pairs, axis=1)
        )
        connected_active = genome.active_mask & has_edge
        g2 = prune_isolated(genome, cfg)
        assert jnp.all(g2.active_mask[connected_active]), "Connected neuron was wrongly pruned"

    def test_io_neurons_never_pruned(self, cfg):
        """I/O neurons must not be deactivated even when they have no edges at all."""
        g = random_genome(jax.random.PRNGKey(201), cfg)
        g_no_edges = replace(g, edge_mask=jnp.zeros_like(g.edge_mask))
        g_pruned = prune_isolated(g_no_edges, cfg)
        assert jnp.all(g_pruned.active_mask[:cfg.n_in]),   "Input neurons were pruned"
        assert jnp.all(g_pruned.active_mask[-cfg.n_out:]), "Output neurons were pruned"

    def test_vmappable(self, cfg):
        keys = jax.random.split(jax.random.PRNGKey(202), 32)
        pop  = jax.vmap(random_genome, in_axes=(0, None))(keys, cfg)
        pop2 = jax.vmap(prune_isolated, in_axes=(0, None))(pop, cfg)
        assert pop2.active_mask.shape == (32, cfg.N_max)

    def test_valid_after(self, genome, cfg):
        g2 = prune_isolated(genome, cfg)
        validate_genome(g2, cfg)


# ─────────────────────────────────────────────────────────────────────────────
# mutate (combined operator)
# ─────────────────────────────────────────────────────────────────────────────

class TestMutate:
    def test_valid_after(self, genome, cfg):
        rates = MutationRates()
        g2 = mutate(jax.random.PRNGKey(100), genome, cfg, rates)
        validate_genome(g2, cfg)

    def test_vmappable(self, cfg):
        keys  = jax.random.split(jax.random.PRNGKey(101), 64)
        pop   = jax.vmap(random_genome, in_axes=(0, None))(keys, cfg)
        keys2 = jax.random.split(jax.random.PRNGKey(102), 64)
        rates = MutationRates()
        pop2  = jax.vmap(mutate, in_axes=(0, 0, None, None))(keys2, pop, cfg, rates)
        assert pop2.active_mask.shape == (64, cfg.N_max)

    def test_no_isolated_hidden_neurons_after_mutate(self, cfg):
        """After mutate(), no active hidden neuron should have zero active edges."""
        rates = MutationRates(add_node_prob=1.0, remove_edge_prob=0.5)
        P     = 64
        keys  = jax.random.split(jax.random.PRNGKey(110), P)
        pop   = jax.vmap(random_genome, in_axes=(0, None))(keys, cfg)
        keys2 = jax.random.split(jax.random.PRNGKey(111), P)
        pop2  = jax.vmap(mutate, in_axes=(0, 0, None, None))(keys2, pop, cfg, rates)

        active_pairs = pop2.active_mask[:, :, None] * pop2.active_mask[:, None, :]
        active_edges = pop2.edge_mask * active_pairs
        has_edge     = jnp.any(active_edges, axis=1) | jnp.any(active_edges, axis=2)  # [P, N]
        hidden       = jnp.zeros(cfg.N_max, bool).at[cfg.n_in: cfg.N_max - cfg.n_out].set(True)
        isolated     = pop2.active_mask & ~has_edge & hidden[None, :]
        assert not jnp.any(isolated), "Isolated hidden neurons found after mutate()"

    def test_zero_rates_produces_identical_genome(self, genome, cfg):
        """With all structural rates at 0, structural fields must not change."""
        rates = MutationRates(
            weight_sigma=0.0, tau_sigma=0.0, bias_sigma=0.0, position_sigma=0.0,
            type_flip_prob=0.0, add_node_prob=0.0, remove_node_prob=0.0,
            add_edge_prob=0.0, remove_edge_prob=0.0,
        )
        g2 = mutate(jax.random.PRNGKey(103), genome, cfg, rates)
        fields_equal(genome, g2,
                     "active_mask", "neuron_type", "tau", "bias",
                     "position", "weight_matrix", "edge_mask")


# ─────────────────────────────────────────────────────────────────────────────
# Milestone validation gates: repeated mutation & statistical drift
# ─────────────────────────────────────────────────────────────────────────────

class TestMilestoneGates:
    def test_repeated_param_mutation_no_structural_drift(self, genome, cfg):
        """
        100 rounds of parameter-only mutation must not change structural fields
        (active_mask, edge_mask) or neuron types.
        """
        rates = MutationRates(
            weight_sigma=0.5, tau_sigma=0.5, bias_sigma=0.5, position_sigma=0.1,
            type_flip_prob=0.0, add_node_prob=0.0, remove_node_prob=0.0,
            add_edge_prob=0.0, remove_edge_prob=0.0,
        )
        _mutate = jax.jit(lambda key, g: mutate(key, g, cfg, rates))
        g = genome
        for i in range(100):
            g = _mutate(jax.random.PRNGKey(200 + i), g)

        fields_equal(genome, g, "active_mask", "edge_mask", "neuron_type")
        validate_genome(g, cfg)

    def test_add_node_increases_population_mean_node_count(self, cfg):
        """
        Repeated add_node on a population should increase mean active node count.
        """
        P    = 64
        keys = jax.random.split(jax.random.PRNGKey(300), P)
        pop  = jax.vmap(random_genome, in_axes=(0, None, None))(keys, cfg, cfg.n_in + cfg.n_out)

        mean_before = float(jnp.mean(jnp.sum(pop.active_mask, axis=1)))

        _add_node = jax.jit(jax.vmap(lambda k, g: add_node(k, g, cfg), in_axes=(0, 0)))
        for i in range(10):
            op_keys = jax.random.split(jax.random.PRNGKey(301 + i), P)
            pop = _add_node(op_keys, pop)

        mean_after = float(jnp.mean(jnp.sum(pop.active_mask, axis=1)))
        assert mean_after > mean_before, (
            f"Mean node count should increase after repeated add_node "
            f"(before={mean_before:.2f}, after={mean_after:.2f})"
        )

    def test_remove_node_decreases_population_mean_node_count(self, cfg):
        """
        Repeated remove_node on a population with hidden nodes should decrease
        mean active node count.
        """
        P    = 64
        keys = jax.random.split(jax.random.PRNGKey(400), P)
        pop  = jax.vmap(random_genome, in_axes=(0, None))(keys, cfg)  # n_active = N_max//2

        mean_before = float(jnp.mean(jnp.sum(pop.active_mask, axis=1)))

        _remove_node = jax.jit(jax.vmap(lambda k, g: remove_node(k, g, cfg), in_axes=(0, 0)))
        for i in range(10):
            op_keys = jax.random.split(jax.random.PRNGKey(401 + i), P)
            pop = _remove_node(op_keys, pop)

        mean_after = float(jnp.mean(jnp.sum(pop.active_mask, axis=1)))
        assert mean_after < mean_before, (
            f"Mean node count should decrease after repeated remove_node "
            f"(before={mean_before:.2f}, after={mean_after:.2f})"
        )

    def test_add_edge_increases_population_mean_edge_count(self, cfg):
        """Repeated add_edge should increase mean edge count in the population."""
        P    = 64
        keys = jax.random.split(jax.random.PRNGKey(500), P)
        pop  = jax.vmap(random_genome, in_axes=(0, None))(keys, cfg)

        def count_edges(pop):
            return jnp.sum(
                pop.edge_mask
                * pop.active_mask[:, :, None]
                * pop.active_mask[:, None, :],
                axis=(1, 2),
            )

        mean_before = float(jnp.mean(count_edges(pop)))

        _add_edge = jax.jit(jax.vmap(lambda k, g: add_edge(k, g, cfg), in_axes=(0, 0)))
        for i in range(10):
            op_keys = jax.random.split(jax.random.PRNGKey(501 + i), P)
            pop = _add_edge(op_keys, pop)

        mean_after = float(jnp.mean(count_edges(pop)))
        assert mean_after > mean_before

    def test_all_operators_produce_valid_genomes_large_batch(self, cfg):
        """Smoke test: mutate a large batch and validate every genome."""
        P     = 256
        keys  = jax.random.split(jax.random.PRNGKey(600), P)
        pop   = jax.vmap(random_genome, in_axes=(0, None))(keys, cfg)
        rates = MutationRates()
        keys2 = jax.random.split(jax.random.PRNGKey(601), P)
        pop2  = jax.vmap(mutate, in_axes=(0, 0, None, None))(keys2, pop, cfg, rates)

        # Validate a sample (full validation requires Python loop; check invariants directly)
        assert jnp.all(pop2.weight_matrix >= 0),    "Negative weights in batch"
        assert jnp.all(pop2.tau > 0),               "Non-positive tau in batch"
        assert jnp.all(pop2.neuron_type <= 2),      "Invalid type in batch"
        assert jnp.all(pop2.active_mask[:, :cfg.n_in]),   "Input neurons deactivated"
        assert jnp.all(pop2.active_mask[:, -cfg.n_out:]), "Output neurons deactivated"
        assert jnp.all(pop2.position >= 0) and jnp.all(pop2.position <= 1), \
            "Positions out of unit square"
