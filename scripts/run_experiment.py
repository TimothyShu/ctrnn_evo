#!/usr/bin/env python
"""
scripts/run_experiment.py — M8 Clune 2013 modularity replication.

Runs two conditions in sequence:
  baseline  — lambda_conn=0.0   (no modularity pressure)
  modular   — lambda_conn=LAMBDA (connection cost penalty)

Each condition is repeated for --n-replicates independent evolutionary runs.
Results are saved under --output-dir/{condition}/run_*/  using the standard
logger format (history.jsonl, best_genome.npz, checkpoints/).

A summary table is printed at the end showing mean/std Q and fitness per condition.

Usage examples
--------------
# Quick smoke test (2 replicates, 5 generations):
python scripts/run_experiment.py --smoke-test

# Default experiment (10 replicates, 500 generations):
python scripts/run_experiment.py

# Full Clune replication (20 replicates, 1000 generations):
python scripts/run_experiment.py --n-replicates 20 --n-generations 1000

# Custom connection cost strength:
python scripts/run_experiment.py --lambda-conn 0.002

# Verbose per-generation output:
python scripts/run_experiment.py --verbose
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp

from ctrnn_evo import Config, WorldConfig
from ctrnn_evo.mutation import MutationRates
from ctrnn_evo.evolution import (
    init_population, eval_population, compute_fitness, run_evolution,
)
from ctrnn_evo.analysis import analyse_genome
from ctrnn_evo.logger import make_run_dir, save_config, make_logger


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="M8 Clune 2013 modularity replication experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-replicates",  type=int,   default=10,     help="independent runs per condition")
    p.add_argument("--n-generations", type=int,   default=500,    help="generations per run")
    p.add_argument("--n-evals",       type=int,   default=5,      help="episodes per fitness estimate")
    p.add_argument("--pop-size",      type=int,   default=1000,   help="population size")
    p.add_argument("--lambda-conn",   type=float, default=0.001,  help="connection cost coefficient for modular condition")
    p.add_argument("--output-dir",    type=str,   default="runs/m8", help="root directory for all run output")
    p.add_argument("--seed",          type=int,   default=0,      help="base random seed")
    p.add_argument("--verbose",       action="store_true",        help="print per-generation progress")
    p.add_argument("--smoke-test",    action="store_true",        help="quick run: 2 replicates × 5 generations × 1 eval")
    return p.parse_args()


# ── XLA warm-up ───────────────────────────────────────────────────────────────

def warmup(cfg: Config, wcfg: WorldConfig, key: jax.Array) -> None:
    """
    Compile the eval_population kernel before the timed experiment starts.

    The first call to eval_population triggers XLA compilation (30-90 s on
    first run with a new set of parameters).  Subsequent calls at the same
    shapes are instantaneous cache hits.
    """
    print("Warming up XLA compilation — this takes 30-90 s on first run...")
    pop = init_population(key, cfg)
    _, k = jax.random.split(key)
    steps, _ = eval_population(k, pop, cfg, wcfg, n_evals=1)
    steps.block_until_ready()
    print("Compilation done.\n")


# ── Single condition runner ───────────────────────────────────────────────────

def run_condition(
    condition_name: str,
    cfg: Config,
    wcfg: WorldConfig,
    rates: MutationRates,
    n_replicates: int,
    n_generations: int,
    n_evals: int,
    output_dir: Path,
    rep_keys: list[jax.Array],
    verbose: bool,
) -> list[dict]:
    """
    Run all replicates for one condition.  Returns a list of result dicts.
    """
    condition_dir = output_dir / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for rep, rep_key in enumerate(rep_keys):
        print(f"  [{condition_name}] replicate {rep + 1}/{n_replicates}")

        run_dir = make_run_dir(condition_dir, run_id=f"rep{rep:02d}")
        save_config(run_dir, cfg, wcfg, rates)
        cb = make_logger(run_dir, checkpoint_every=100, verbose=verbose)

        t0 = time.perf_counter()
        best_genome, final_fitness, history = run_evolution(
            rep_key, n_generations, cfg, wcfg, rates,
            n_evals=n_evals, callback=cb,
        )
        elapsed = time.perf_counter() - t0

        metrics = analyse_genome(best_genome, cfg)

        result = {
            "condition":           condition_name,
            "replicate":           rep,
            "final_max_fitness":   history[-1]["max_fitness"],
            "final_mean_fitness":  history[-1]["mean_fitness"],
            "mean_final_fitness":  float(jnp.mean(final_fitness)),
            "best_q":              metrics["q"],
            "best_n_active":       metrics["n_active"],
            "best_conn_cost":      metrics["connection_cost"],
            "elapsed_s":           elapsed,
            "run_dir":             str(run_dir),
        }
        results.append(result)

        print(
            f"    → Q={result['best_q']:.3f}  "
            f"fit={result['final_max_fitness']:.3f}  "
            f"n_active={result['best_n_active']}  "
            f"({elapsed / 60:.1f} min)"
        )

    return results


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(all_results: list[dict]) -> None:
    conditions = sorted(set(r["condition"] for r in all_results))

    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print(f"{'Condition':<14} {'Q mean':>8} {'Q std':>7} "
          f"{'Fitness':>9} {'Fit std':>8} {'Nodes':>7}")
    print("-" * 65)

    for cond in conditions:
        reps     = [r for r in all_results if r["condition"] == cond]
        qs       = [r["best_q"]             for r in reps]
        fits     = [r["final_max_fitness"]   for r in reps]
        nodes    = [r["best_n_active"]       for r in reps]
        print(
            f"{cond:<14} "
            f"{np.mean(qs):>8.3f} {np.std(qs):>7.3f} "
            f"{np.mean(fits):>9.3f} {np.std(fits):>8.3f} "
            f"{np.mean(nodes):>7.1f}"
        )

    print("=" * 65)

    # Interpretation hint
    reps_b = [r for r in all_results if r["condition"] == "baseline"]
    reps_m = [r for r in all_results if r["condition"] == "modular"]
    if reps_b and reps_m:
        dq   = np.mean([r["best_q"] for r in reps_m]) - np.mean([r["best_q"] for r in reps_b])
        dfit = np.mean([r["final_max_fitness"] for r in reps_m]) - \
               np.mean([r["final_max_fitness"] for r in reps_b])
        print(f"\nModularity gain (modular − baseline):  ΔQ = {dq:+.3f}")
        print(f"Fitness cost   (modular − baseline): ΔFit = {dfit:+.3f}")
        if dq > 0.05:
            print("✓ Connection cost pressure increased modularity.")
        else:
            print("△ Modularity difference is small — consider increasing --lambda-conn.")
        if abs(dfit) < 0.05:
            print("✓ Fitness was not significantly degraded.")
        else:
            print("△ Fitness dropped noticeably — consider reducing --lambda-conn.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Smoke-test overrides
    if args.smoke_test:
        args.n_replicates  = 2
        args.n_generations = 5
        args.n_evals       = 1
        args.pop_size      = 20
        print("Smoke-test mode: 2 replicates × 5 generations × pop=20\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Configs ───────────────────────────────────────────────────────────────
    base_cfg = Config(population_size=args.pop_size)
    wcfg     = WorldConfig()
    rates    = MutationRates()

    cfg_baseline = Config(
        population_size=args.pop_size,
        lambda_conn=0.0,
        lambda_act=0.0,
    )
    cfg_modular = Config(
        population_size=args.pop_size,
        lambda_conn=args.lambda_conn,
        lambda_act=0.0,
    )

    # ── Key schedule ──────────────────────────────────────────────────────────
    # Each condition gets its own independent stream of keys so that
    # replicates are not correlated across conditions.
    base_key = jax.random.PRNGKey(args.seed)
    k_warmup, k_baseline, k_modular = jax.random.split(base_key, 3)
    baseline_keys = list(jax.random.split(k_baseline, args.n_replicates))
    modular_keys  = list(jax.random.split(k_modular,  args.n_replicates))

    # ── XLA warm-up ───────────────────────────────────────────────────────────
    warmup(base_cfg, wcfg, k_warmup)

    # ── Print experiment header ───────────────────────────────────────────────
    print(f"Experiment: {args.n_replicates} replicates × {args.n_generations} generations")
    print(f"  population_size = {args.pop_size}")
    print(f"  n_evals         = {args.n_evals}")
    print(f"  lambda_conn     = 0.0  (baseline)  vs  {args.lambda_conn}  (modular)")
    print(f"  output_dir      = {output_dir.resolve()}\n")

    t_total = time.perf_counter()
    all_results: list[dict] = []

    # ── Baseline condition ────────────────────────────────────────────────────
    print("── Condition: baseline (lambda_conn=0.0) ──────────────────────")
    all_results += run_condition(
        condition_name="baseline",
        cfg=cfg_baseline,
        wcfg=wcfg,
        rates=rates,
        n_replicates=args.n_replicates,
        n_generations=args.n_generations,
        n_evals=args.n_evals,
        output_dir=output_dir,
        rep_keys=baseline_keys,
        verbose=args.verbose,
    )

    # ── Modular condition ─────────────────────────────────────────────────────
    print(f"\n── Condition: modular (lambda_conn={args.lambda_conn}) ────────────────")
    all_results += run_condition(
        condition_name="modular",
        cfg=cfg_modular,
        wcfg=wcfg,
        rates=rates,
        n_replicates=args.n_replicates,
        n_generations=args.n_generations,
        n_evals=args.n_evals,
        output_dir=output_dir,
        rep_keys=modular_keys,
        verbose=args.verbose,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_total
    print(f"\nTotal wall time: {total_elapsed / 3600:.2f} hrs  ({total_elapsed / 60:.1f} min)")
    print_summary(all_results)
    print(f"\nAll results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
