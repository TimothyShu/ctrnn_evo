"""
logger.py — Logging and checkpointing for evolutionary runs.

Public API
----------
make_run_dir(base_dir, run_id) -> Path
    Create a timestamped run directory with a checkpoints/ subdirectory.

save_config(run_dir, cfg, wcfg, rates) -> None
    Serialise all three hyperparameter dataclasses to config.json.

load_config(run_dir) -> (Config, WorldConfig, MutationRates)
    Reconstruct dataclasses from config.json.

save_genome(path, genome) -> None
    Persist all 7 genome fields to a .npz archive.

load_genome(path) -> Genome
    Reconstruct a Genome from a .npz archive.

append_history(run_dir, stats) -> None
    Append one stats dict as a line to history.jsonl (safe for partial runs).

load_history(run_dir) -> list[dict]
    Read all lines from history.jsonl; returns [] if file is absent or empty.

make_logger(run_dir, checkpoint_every, verbose) -> callback
    Return callback(stats, best_genome) that logs, saves, and checkpoints.
"""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import numpy as np

from .config import Config
from .genome import Genome
from .world import WorldConfig
from .mutation import MutationRates


# Genome field order must match the pytree registration in genome.py:
# lambda g: [g.active_mask, g.neuron_type, g.tau, g.bias,
#            g.position, g.weight_matrix, g.edge_mask]
_GENOME_FIELDS = [
    "active_mask",
    "neuron_type",
    "tau",
    "bias",
    "position",
    "weight_matrix",
    "edge_mask",
]


# ── Directory management ──────────────────────────────────────────────────────

def make_run_dir(base_dir: str | Path = "runs", run_id: str | None = None) -> Path:
    """
    Create a fresh run directory under base_dir.

    Name format: run_{YYYYMMDD_HHMMSS_ffffff}_{run_id}
    If run_id is None, a 4-char random hex suffix is generated so that
    two calls in the same second still produce distinct paths.

    Creates base_dir and the checkpoints/ subdirectory if they don't exist.
    Returns the Path to the new run directory.
    """
    base_dir = Path(base_dir)
    if run_id is None:
        run_id = os.urandom(2).hex()
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = base_dir / f"run_{ts}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


# ── Config serialisation ──────────────────────────────────────────────────────

def save_config(
    run_dir: Path,
    cfg: Config,
    wcfg: WorldConfig,
    rates: MutationRates,
) -> None:
    """Write all three hyperparameter dataclasses to config.json."""
    data = {
        "config":         dataclasses.asdict(cfg),
        "world_config":   dataclasses.asdict(wcfg),
        "mutation_rates": dataclasses.asdict(rates),
    }
    with open(Path(run_dir) / "config.json", "w") as f:
        json.dump(data, f, indent=2)


def load_config(run_dir: Path) -> tuple[Config, WorldConfig, MutationRates]:
    """Reconstruct Config, WorldConfig, MutationRates from config.json.

    JSON encodes tuples as lists; we convert any list field values back to
    tuples so that the roundtrip is exact (Config stores tau ranges as tuples).
    """
    def _fix_tuples(d: dict) -> dict:
        return {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}

    with open(Path(run_dir) / "config.json") as f:
        data = json.load(f)
    cfg   = Config(**_fix_tuples(data["config"]))
    wcfg  = WorldConfig(**data["world_config"])
    rates = MutationRates(**data["mutation_rates"])
    return cfg, wcfg, rates


# ── Genome serialisation ──────────────────────────────────────────────────────

def save_genome(path: str | Path, genome: Genome) -> None:
    """
    Save all 7 genome fields to a .npz archive.

    Field order in the archive matches _GENOME_FIELDS, which is the same
    order as the Genome pytree registration so load_genome can reconstruct
    via Genome(*children) without a dict lookup.
    """
    arrays = {field: np.array(getattr(genome, field)) for field in _GENOME_FIELDS}
    np.savez(str(path), **arrays)


def load_genome(path: str | Path) -> Genome:
    """Reconstruct a Genome from a .npz archive saved by save_genome."""
    archive  = np.load(str(path))
    children = [jnp.array(archive[field]) for field in _GENOME_FIELDS]
    return Genome(*children)


# ── History (newline-delimited JSON) ──────────────────────────────────────────

def append_history(run_dir: Path, stats: dict) -> None:
    """
    Append one stats dict as a single JSON line to history.jsonl.

    Uses append mode — never rewrites existing content, safe for partial runs.
    """
    with open(Path(run_dir) / "history.jsonl", "a") as f:
        f.write(json.dumps(stats) + "\n")


def load_history(run_dir: Path) -> list[dict]:
    """
    Read all lines from history.jsonl and parse each as JSON.

    Returns an empty list if the file is missing or empty.
    """
    path = Path(run_dir) / "history.jsonl"
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ── Logger factory ────────────────────────────────────────────────────────────

def make_logger(
    run_dir: Path,
    checkpoint_every: int = 100,
    verbose: bool = True,
) -> Callable[[dict, Genome], None]:
    """
    Return a callback(stats, best_genome) for use with run_evolution.

    Each call:
      1. Appends stats to history.jsonl.
      2. Overwrites best_genome.npz with the current best.
      3. Saves checkpoints/gen_{N:06d}.npz every checkpoint_every generations.
      4. Prints a one-line summary if verbose=True.

    Parameters
    ----------
    run_dir          : directory created by make_run_dir
    checkpoint_every : save a named checkpoint every this many generations
    verbose          : print progress to stdout each generation
    """
    run_dir = Path(run_dir)

    def callback(stats: dict, best_genome: Genome) -> None:
        gen = stats["generation"]

        # 1. Append to history
        append_history(run_dir, stats)

        # 2. Overwrite current best
        save_genome(run_dir / "best_genome.npz", best_genome)

        # 3. Named checkpoint
        if gen % checkpoint_every == 0:
            ckpt_path = run_dir / "checkpoints" / f"gen_{gen:06d}.npz"
            save_genome(ckpt_path, best_genome)

        # 4. Progress line
        if verbose:
            print(
                f"gen {gen:04d} | "
                f"fit {stats['max_fitness']:.4f} "
                f"(mean {stats['mean_fitness']:.4f}) | "
                f"steps {stats['mean_steps']:.1f} | "
                f"nodes {stats['mean_n_active']:.1f}"
            )

    return callback
