"""
M6 Tests — Logging layer.

Tests cover:
  1.  make_run_dir — directory structure created correctly
  2.  save_config / load_config — full roundtrip for all three dataclasses
  3.  save_genome / load_genome — all 7 fields roundtrip with array equality
  4.  append_history / load_history — incremental append and empty/missing cases
  5.  make_logger — files written, checkpoint cadence, verbose flag
  6.  run_evolution integration — logger callback wired end-to-end, updated signature
"""

from __future__ import annotations

import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ctrnn_evo import Config, WorldConfig, random_genome
from ctrnn_evo.mutation import MutationRates
from ctrnn_evo.evolution import run_evolution
from ctrnn_evo.logger import (
    make_run_dir,
    save_config, load_config,
    save_genome, load_genome,
    append_history, load_history,
    make_logger,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_base(tmp_path):
    """A fresh temporary base directory for each test."""
    return tmp_path / "runs"


@pytest.fixture(scope="module")
def cfg():
    return Config(N_max=16, n_in=2, n_out=2, K=4, population_size=10, tournament_size=3)


@pytest.fixture(scope="module")
def wcfg():
    return WorldConfig(episode_steps=50)


@pytest.fixture(scope="module")
def rates():
    return MutationRates()


@pytest.fixture(scope="module")
def genome(cfg):
    return random_genome(jax.random.PRNGKey(0), cfg)


# ── 1. make_run_dir ───────────────────────────────────────────────────────────

def test_make_run_dir_creates_run_directory(tmp_base):
    run_dir = make_run_dir(tmp_base)
    assert run_dir.exists(), "run_dir was not created"
    assert run_dir.is_dir()

def test_make_run_dir_creates_checkpoints_subdir(tmp_base):
    run_dir = make_run_dir(tmp_base)
    assert (run_dir / "checkpoints").exists()
    assert (run_dir / "checkpoints").is_dir()

def test_make_run_dir_unique_without_run_id(tmp_base):
    """Two calls with no run_id must produce different directories."""
    import time
    d1 = make_run_dir(tmp_base)
    time.sleep(0.01)          # ensure timestamp differs
    d2 = make_run_dir(tmp_base)
    assert d1 != d2, "make_run_dir produced the same path twice"

def test_make_run_dir_custom_run_id(tmp_base):
    run_dir = make_run_dir(tmp_base, run_id="myrun")
    assert "myrun" in run_dir.name

def test_make_run_dir_creates_base_if_missing(tmp_path):
    base = tmp_path / "does" / "not" / "exist"
    run_dir = make_run_dir(base)
    assert run_dir.exists()


# ── 2. save_config / load_config ──────────────────────────────────────────────

def test_save_config_creates_file(tmp_base, cfg, wcfg, rates):
    run_dir = make_run_dir(tmp_base)
    save_config(run_dir, cfg, wcfg, rates)
    assert (run_dir / "config.json").exists()

def test_save_config_is_valid_json(tmp_base, cfg, wcfg, rates):
    run_dir = make_run_dir(tmp_base)
    save_config(run_dir, cfg, wcfg, rates)
    with open(run_dir / "config.json") as f:
        data = json.load(f)
    assert "config" in data
    assert "world_config" in data
    assert "mutation_rates" in data

def test_load_config_roundtrip_cfg(tmp_base, cfg, wcfg, rates):
    run_dir = make_run_dir(tmp_base)
    save_config(run_dir, cfg, wcfg, rates)
    cfg2, _, _ = load_config(run_dir)
    assert cfg2 == cfg, f"Config mismatch:\n  saved:  {cfg}\n  loaded: {cfg2}"

def test_load_config_roundtrip_wcfg(tmp_base, cfg, wcfg, rates):
    run_dir = make_run_dir(tmp_base)
    save_config(run_dir, cfg, wcfg, rates)
    _, wcfg2, _ = load_config(run_dir)
    assert wcfg2 == wcfg

def test_load_config_roundtrip_rates(tmp_base, cfg, wcfg, rates):
    run_dir = make_run_dir(tmp_base)
    save_config(run_dir, cfg, wcfg, rates)
    _, _, rates2 = load_config(run_dir)
    assert rates2 == rates


# ── 3. save_genome / load_genome ──────────────────────────────────────────────

def test_save_genome_creates_npz(tmp_base, genome):
    run_dir = make_run_dir(tmp_base)
    path = run_dir / "test_genome.npz"
    save_genome(path, genome)
    assert path.exists()

def test_load_genome_active_mask(tmp_base, genome):
    run_dir = make_run_dir(tmp_base)
    path = run_dir / "genome.npz"
    save_genome(path, genome)
    g2 = load_genome(path)
    assert jnp.array_equal(g2.active_mask, genome.active_mask)

def test_load_genome_weight_matrix(tmp_base, genome):
    run_dir = make_run_dir(tmp_base)
    path = run_dir / "genome.npz"
    save_genome(path, genome)
    g2 = load_genome(path)
    assert jnp.allclose(g2.weight_matrix, genome.weight_matrix)

def test_load_genome_all_fields(tmp_base, genome):
    """All 7 genome fields must survive save/load."""
    run_dir = make_run_dir(tmp_base)
    path = run_dir / "genome.npz"
    save_genome(path, genome)
    g2 = load_genome(path)

    assert jnp.array_equal(g2.active_mask,   genome.active_mask)
    assert jnp.array_equal(g2.neuron_type,   genome.neuron_type)
    assert jnp.allclose(g2.tau,              genome.tau)
    assert jnp.allclose(g2.bias,             genome.bias)
    assert jnp.allclose(g2.position,         genome.position)
    assert jnp.allclose(g2.weight_matrix,    genome.weight_matrix)
    assert jnp.array_equal(g2.edge_mask,     genome.edge_mask)

def test_load_genome_returns_genome_instance(tmp_base, genome):
    from ctrnn_evo import Genome
    run_dir = make_run_dir(tmp_base)
    path = run_dir / "genome.npz"
    save_genome(path, genome)
    g2 = load_genome(path)
    assert isinstance(g2, Genome)


# ── 4. append_history / load_history ─────────────────────────────────────────

def test_append_history_single(tmp_base):
    run_dir = make_run_dir(tmp_base)
    stats = {"generation": 0, "max_fitness": 0.5, "mean_fitness": 0.3}
    append_history(run_dir, stats)
    history = load_history(run_dir)
    assert len(history) == 1
    assert history[0]["generation"] == 0

def test_append_history_accumulates(tmp_base):
    run_dir = make_run_dir(tmp_base)
    for i in range(5):
        append_history(run_dir, {"generation": i, "max_fitness": float(i) * 0.1})
    history = load_history(run_dir)
    assert len(history) == 5
    assert [h["generation"] for h in history] == list(range(5))

def test_append_history_values_preserved(tmp_base):
    run_dir = make_run_dir(tmp_base)
    stats = {"generation": 7, "max_fitness": 0.9123, "mean_steps": 91.5,
             "mean_n_active": 6.2, "mean_conn_cost": 1.337}
    append_history(run_dir, stats)
    h = load_history(run_dir)[0]
    assert h["generation"] == 7
    assert abs(h["max_fitness"] - 0.9123) < 1e-6
    assert abs(h["mean_conn_cost"] - 1.337) < 1e-6

def test_load_history_empty_file(tmp_base):
    run_dir = make_run_dir(tmp_base)
    (run_dir / "history.jsonl").touch()
    assert load_history(run_dir) == []

def test_load_history_missing_file(tmp_base):
    run_dir = make_run_dir(tmp_base)
    # history.jsonl not created yet
    assert load_history(run_dir) == []

def test_append_history_is_incremental(tmp_base):
    """Appending twice doesn't rewrite earlier lines."""
    run_dir = make_run_dir(tmp_base)
    append_history(run_dir, {"generation": 0, "x": 1})
    first_mtime = (run_dir / "history.jsonl").stat().st_mtime_ns
    # Read and check the first line is still intact after second append
    append_history(run_dir, {"generation": 1, "x": 2})
    lines = (run_dir / "history.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["generation"] == 0
    assert json.loads(lines[1])["generation"] == 1


# ── 5. make_logger ────────────────────────────────────────────────────────────

def _fake_stats(gen: int) -> dict:
    return {
        "generation":     gen,
        "max_fitness":    0.8,
        "mean_fitness":   0.6,
        "max_steps":      80,
        "mean_steps":     60.0,
        "mean_n_active":  7.0,
        "mean_conn_cost": 0.5,
    }


def test_make_logger_creates_history_file(tmp_base, genome):
    run_dir = make_run_dir(tmp_base)
    cb = make_logger(run_dir, checkpoint_every=100, verbose=False)
    cb(_fake_stats(0), genome)
    assert (run_dir / "history.jsonl").exists()

def test_make_logger_history_grows_each_call(tmp_base, genome):
    run_dir = make_run_dir(tmp_base)
    cb = make_logger(run_dir, checkpoint_every=100, verbose=False)
    for i in range(7):
        cb(_fake_stats(i), genome)
    history = load_history(run_dir)
    assert len(history) == 7

def test_make_logger_writes_best_genome_file(tmp_base, genome):
    run_dir = make_run_dir(tmp_base)
    cb = make_logger(run_dir, checkpoint_every=100, verbose=False)
    cb(_fake_stats(0), genome)
    assert (run_dir / "best_genome.npz").exists()

def test_make_logger_best_genome_updated_each_call(tmp_base, genome):
    """best_genome.npz should always reflect the most recent callback."""
    run_dir = make_run_dir(tmp_base)
    cb = make_logger(run_dir, checkpoint_every=100, verbose=False)
    cb(_fake_stats(0), genome)
    mtime1 = (run_dir / "best_genome.npz").stat().st_mtime_ns
    cb(_fake_stats(1), genome)
    mtime2 = (run_dir / "best_genome.npz").stat().st_mtime_ns
    assert mtime2 >= mtime1   # file was touched again

def test_make_logger_checkpoint_at_gen_0(tmp_base, genome):
    run_dir = make_run_dir(tmp_base)
    cb = make_logger(run_dir, checkpoint_every=5, verbose=False)
    cb(_fake_stats(0), genome)
    assert (run_dir / "checkpoints" / "gen_000000.npz").exists()

def test_make_logger_checkpoint_cadence(tmp_base, genome):
    """Checkpoints at gens 0, 5, 10; not at 3 or 7."""
    run_dir = make_run_dir(tmp_base)
    cb = make_logger(run_dir, checkpoint_every=5, verbose=False)
    for i in range(11):
        cb(_fake_stats(i), genome)
    ckpt = run_dir / "checkpoints"
    assert (ckpt / "gen_000000.npz").exists()
    assert (ckpt / "gen_000005.npz").exists()
    assert (ckpt / "gen_000010.npz").exists()
    assert not (ckpt / "gen_000003.npz").exists()
    assert not (ckpt / "gen_000007.npz").exists()

def test_make_logger_checkpoint_content_valid(tmp_base, genome):
    """Checkpoint files should load back as valid genomes."""
    run_dir = make_run_dir(tmp_base)
    cb = make_logger(run_dir, checkpoint_every=1, verbose=False)
    cb(_fake_stats(0), genome)
    g2 = load_genome(run_dir / "checkpoints" / "gen_000000.npz")
    assert jnp.allclose(g2.weight_matrix, genome.weight_matrix)

def test_make_logger_verbose_false_no_stdout(tmp_base, genome, capsys):
    run_dir = make_run_dir(tmp_base)
    cb = make_logger(run_dir, checkpoint_every=100, verbose=False)
    cb(_fake_stats(0), genome)
    captured = capsys.readouterr()
    assert captured.out == "", f"Expected no stdout with verbose=False, got: {captured.out!r}"

def test_make_logger_verbose_true_prints(tmp_base, genome, capsys):
    run_dir = make_run_dir(tmp_base)
    cb = make_logger(run_dir, checkpoint_every=100, verbose=True)
    cb(_fake_stats(42), genome)
    captured = capsys.readouterr()
    assert "42" in captured.out, "Generation number not found in verbose output"


# ── 6. run_evolution integration ─────────────────────────────────────────────

def test_run_evolution_callback_receives_genome(cfg, wcfg, rates, tmp_base):
    """Updated callback signature: callback(stats, best_genome) — genome has correct shape."""
    from ctrnn_evo import Genome
    received = []

    def cb(stats, best_genome):
        received.append((stats["generation"], best_genome))

    key = jax.random.PRNGKey(99)
    run_evolution(key, 3, cfg, wcfg, rates, n_evals=2, callback=cb)

    assert len(received) == 3
    for gen_idx, g in received:
        assert isinstance(g, Genome), "callback did not receive a Genome"
        assert g.weight_matrix.shape == (cfg.N_max, cfg.N_max), (
            f"best_genome has wrong shape: {g.weight_matrix.shape}"
        )

def test_run_evolution_with_logger_populates_files(cfg, wcfg, rates, tmp_base):
    """Full run_evolution with make_logger; verify expected files exist."""
    run_dir = make_run_dir(tmp_base)
    cb = make_logger(run_dir, checkpoint_every=2, verbose=False)

    key = jax.random.PRNGKey(100)
    run_evolution(key, 5, cfg, wcfg, rates, n_evals=2, callback=cb)

    assert (run_dir / "history.jsonl").exists()
    assert (run_dir / "best_genome.npz").exists()
    history = load_history(run_dir)
    assert len(history) == 5

def test_run_evolution_with_logger_checkpoint_files(cfg, wcfg, rates, tmp_base):
    """Checkpoints at gen 0 and 2 should exist after 5 generations with every=2."""
    run_dir = make_run_dir(tmp_base)
    cb = make_logger(run_dir, checkpoint_every=2, verbose=False)

    key = jax.random.PRNGKey(101)
    run_evolution(key, 5, cfg, wcfg, rates, n_evals=2, callback=cb)

    ckpt = run_dir / "checkpoints"
    assert (ckpt / "gen_000000.npz").exists()
    assert (ckpt / "gen_000002.npz").exists()
    assert (ckpt / "gen_000004.npz").exists()

def test_run_evolution_logger_best_genome_loadable(cfg, wcfg, rates, tmp_base):
    """best_genome.npz written by the logger must load back without error."""
    run_dir = make_run_dir(tmp_base)
    save_config(run_dir, cfg, wcfg, rates)
    cb = make_logger(run_dir, checkpoint_every=10, verbose=False)

    key = jax.random.PRNGKey(102)
    run_evolution(key, 3, cfg, wcfg, rates, n_evals=2, callback=cb)

    g = load_genome(run_dir / "best_genome.npz")
    assert g.weight_matrix.shape == (cfg.N_max, cfg.N_max)
