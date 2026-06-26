#!/usr/bin/env python
"""
scripts/run_experiment.py — M8 Clune 2013 modularity replication.

Runs two conditions in sequence:
  baseline  — no cost   (no modularity pressure)
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
    fitness_threshold, convergence_stop,
)
from ctrnn_evo.analysis import analyse_genome
from ctrnn_evo.logger import make_run_dir, save_config, make_logger, latest_state_checkpoint


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
    p.add_argument("--lambda-edge",   type=float, default=0.0,    help="edge-count cost coefficient — absolute mode (penalises every edge equally)")
    p.add_argument("--lambda-dist",   type=float, default=0.001,  help="distance-weighted wiring cost coefficient — absolute mode (penalises long edges more)")
    p.add_argument("--lambda-act",    type=float, default=0.0,    help="activation cost coefficient — absolute mode (penalises mean firing per tick)")
    p.add_argument("--dist-frac",     type=float, default=0.0,
                   help="wiring penalty as a fraction of f_raw (proportional mode, overrides --lambda-dist when >0). "
                        "E.g. 0.05 = 5%% of fitness deducted when wiring is at reference density. "
                        "Self-calibrates across fitness regimes — no recalibration needed.")
    p.add_argument("--act-frac",      type=float, default=0.0,
                   help="activation penalty as a fraction of f_raw (proportional mode, overrides --lambda-act when >0).")
    p.add_argument("--edge-frac",     type=float, default=0.0,
                   help="edge-count penalty as a fraction of f_raw (proportional mode, overrides --lambda-edge when >0).")
    p.add_argument("--n-food-types",  type=int,   default=1,      help="number of distinct food types (each with its own energy resource and sensor channel)")
    p.add_argument("--hotspot-drift",  type=float, default=0.6,   help="std-dev of per-step hotspot Gaussian drift (default 0.6; use ~0.2 with strip placement to keep types separated)")
    p.add_argument("--hotspot-sigma",  type=float, default=5.0,   help="Gaussian radius of food reward patch (default 5.0; smaller = harder to find food)")
    p.add_argument("--metabolism",     type=float, default=0.01,  help="energy drain per step from baseline metabolism (default 0.01)")
    p.add_argument("--output-dir",    type=str,   default="runs/m8", help="root directory for all run output")
    p.add_argument("--seed",          type=int,   default=0,      help="base random seed")
    p.add_argument("--fitness-threshold",  type=float, default=None,
                   help="stop a replicate early when max_fitness reaches this value (e.g. 0.95)")
    p.add_argument("--convergence-window", type=int,   default=None,
                   help="stop when max_fitness hasn't improved by --convergence-tol over this many generations")
    p.add_argument("--convergence-tol",    type=float, default=1e-3,
                   help="minimum improvement required within --convergence-window (default: 0.001)")
    p.add_argument("--save-state-every", type=int,   default=100,
                   help="save full training state every N generations for resume support (0 = disabled)")
    p.add_argument("--resume-from",      type=str,   default=None,
                   help="path to a state_gen_*.npz checkpoint; resumes that single replicate "
                        "and skips the second condition.  For per-replicate auto-resume within "
                        "a full experiment use --resume-run-dir instead.")
    p.add_argument("--resume-run-dir",   type=str,   default=None,
                   help="path to an interrupted run directory; the script will auto-detect the "
                        "latest state checkpoint and resume that single replicate.")
    p.add_argument("--condition",      choices=["both", "baseline", "modular"], default="both",
                   help="which condition(s) to run (default: both)")
    p.add_argument("--lambda-sweep",   type=float, nargs="+", default=None, metavar="L",
                   help="test multiple lambda values in one run, each saved to modular_<L>/ subdir "
                        "(overrides --lambda-conn and --condition)")
    p.add_argument("--fitness-mode",   type=str,   default="survival",
                   choices=["survival", "food"],
                   help="fitness metric: 'survival'=steps_survived/T ∈ [0,1] (default); "
                        "'food'=cumulative raw food score / (T * n_food_types), can exceed 1.0 "
                        "for agents that actively forage near hotspot centres")
    p.add_argument("--position-sensors", action="store_true", default=False,
                   help="give the agent normalised (x, y) position as two extra input sensors "
                        "(appended after food/energy sensors). Adds 2 to n_in. Without this, "
                        "agents starting far from food have zero gradient and run open-loop.")
    p.add_argument("--penalty-warmup-gens", type=int, default=0,
                   help="linearly ramp all λ penalties from 0 to their full values over this many "
                        "generations (0 = disabled, penalties are constant from gen 0). "
                        "Prevents early-generation over-pruning when food signal is weak.")
    p.add_argument("--penalty-cycle-gens", type=int, default=0,
                   help="cyclic loosening cycle length after warmup (0 = disabled). "
                        "Every penalty_cycle_gens generations, the last penalty_cycle_free_gens "
                        "have all penalties set to zero, allowing cold reps to escape local optima.")
    p.add_argument("--penalty-cycle-free-gens", type=int, default=0,
                   help="free (penalty=0) window at the end of each cycle (0 = disabled).")
    p.add_argument("--mutation-warmup-scale", type=float, default=1.0,
                   help="scale factor for continuous mutation sigmas at gen 0, decaying linearly "
                        "to 1.0 by penalty_warmup_gens. Mirrors the penalty ramp: high exploration "
                        "when penalty=0, normal rates when penalty=full. 1.0 = disabled.")
    p.add_argument("--verbose",        action="store_true",       help="print per-generation progress")
    p.add_argument("--smoke-test",     action="store_true",       help="quick run: 2 replicates × 5 generations × 1 eval")
    p.add_argument("--quick-test",     action="store_true",       help="lambda sweep validation: 3 replicates × 150 generations × pop=500")
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
    steps, _, _ = eval_population(k, pop, cfg, wcfg, n_evals=1)
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
    early_stop_fn=None,
    save_state_every: int = 100,
    resume_from: "str | None" = None,
) -> list[dict]:
    """
    Run all replicates for one condition.  Returns a list of result dicts.

    Parameters
    ----------
    save_state_every : save full training state every N generations (0 = off).
                       State files land in run_dir/checkpoints/state_gen_*.npz.
    resume_from      : path to a state_gen_*.npz checkpoint.  When set, only
                       one replicate is run (the resumed one) and the rep_keys
                       list is not used.
    """
    condition_dir = output_dir / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for rep, rep_key in enumerate(rep_keys):
        # When resuming a specific state file we only run one replicate
        if resume_from is not None and rep > 0:
            break

        print(f"  [{condition_name}] replicate {rep + 1}/{n_replicates}")

        run_dir = make_run_dir(condition_dir, run_id=f"rep{rep:02d}")
        save_config(run_dir, cfg, wcfg, rates)
        cb = make_logger(run_dir, checkpoint_every=100, verbose=verbose)

        state_ckpt_dir = (run_dir / "checkpoints") if save_state_every > 0 else None

        t0 = time.perf_counter()
        best_genome, final_fitness, history = run_evolution(
            rep_key, n_generations, cfg, wcfg, rates,
            n_evals=n_evals, callback=cb,
            # early_stop_fn is a factory so each replicate gets its own
            # independent state (convergence_stop tracks its own window)
            early_stop_fn=early_stop_fn() if callable(early_stop_fn) else early_stop_fn,
            resume_from=resume_from,
            state_checkpoint_dir=state_ckpt_dir,
            state_checkpoint_every=save_state_every if save_state_every > 0 else 100,
        )
        elapsed = time.perf_counter() - t0

        metrics = analyse_genome(best_genome, cfg)

        generations_run = len(history)
        result = {
            "condition":           condition_name,
            "replicate":           rep,
            "generations_run":     generations_run,
            "final_max_fitness":   history[-1]["max_fitness"],
            "final_mean_fitness":  history[-1]["mean_fitness"],
            "mean_final_fitness":  float(jnp.mean(final_fitness)),
            "best_q":              metrics["q"],
            "best_n_active":       metrics["n_active"],
            "best_conn_cost":      metrics["wiring_cost"],
            "elapsed_s":           elapsed,
            "run_dir":             str(run_dir),
        }
        results.append(result)

        stopped_early = generations_run < n_generations
        print(
            f"    → Q={result['best_q']:.3f}  "
            f"fit={result['final_max_fitness']:.3f}  "
            f"n_active={result['best_n_active']}  "
            f"({elapsed / 60:.1f} min)"
            + (f"  [stopped at gen {generations_run}]" if stopped_early else "")
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

    # Quick-test overrides (lambda sweep validation)
    if args.quick_test:
        args.n_replicates  = 3
        args.n_generations = 150
        args.n_evals       = 3
        args.pop_size      = 500
        print("Quick-test mode: 3 replicates × 150 generations × pop=500\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Configs ───────────────────────────────────────────────────────────────
    base_cfg = Config(population_size=args.pop_size, n_food_types=args.n_food_types,
                      fitness_mode=args.fitness_mode, position_sensors=args.position_sensors)
    wcfg     = WorldConfig(n_food_types=args.n_food_types, hotspot_drift=args.hotspot_drift,
                           hotspot_sigma=args.hotspot_sigma, metabolism=args.metabolism,
                           position_sensors=args.position_sensors)
    rates    = MutationRates()

    cfg_baseline = Config(
        population_size=args.pop_size,
        n_food_types=args.n_food_types,
        fitness_mode=args.fitness_mode,
        position_sensors=args.position_sensors,
        penalty_warmup_gens=args.penalty_warmup_gens,
        penalty_cycle_gens=args.penalty_cycle_gens,
        penalty_cycle_free_gens=args.penalty_cycle_free_gens,
        mutation_warmup_scale=args.mutation_warmup_scale,
    )
    cfg_modular = Config(
        population_size=args.pop_size,
        n_food_types=args.n_food_types,
        lambda_edge=args.lambda_edge,
        lambda_dist=args.lambda_dist,
        lambda_act=args.lambda_act,
        dist_frac=args.dist_frac,
        act_frac=args.act_frac,
        edge_frac=args.edge_frac,
        fitness_mode=args.fitness_mode,
        position_sensors=args.position_sensors,
        penalty_warmup_gens=args.penalty_warmup_gens,
        penalty_cycle_gens=args.penalty_cycle_gens,
        penalty_cycle_free_gens=args.penalty_cycle_free_gens,
        mutation_warmup_scale=args.mutation_warmup_scale,
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
    print(f"  fitness_mode    = {args.fitness_mode}")
    print(f"  position_sensors= {args.position_sensors}")
    print(f"  lambda_edge     = 0.0  (baseline)  vs  {args.lambda_edge}  (modular)")
    print(f"  lambda_dist     = 0.0  (baseline)  vs  {args.lambda_dist}  (modular)")
    print(f"  lambda_act      = 0.0  (baseline)  vs  {args.lambda_act}  (modular)")
    if args.dist_frac or args.act_frac or args.edge_frac:
        print(f"  dist_frac       = 0.0  (baseline)  vs  {args.dist_frac}  (modular)  [proportional]")
        print(f"  act_frac        = 0.0  (baseline)  vs  {args.act_frac}   (modular)  [proportional]")
        print(f"  edge_frac       = 0.0  (baseline)  vs  {args.edge_frac}  (modular)  [proportional]")
    print(f"  penalty_warmup  = {args.penalty_warmup_gens} gens")
    print(f"  output_dir      = {output_dir.resolve()}\n")

    # ── Early stop function ───────────────────────────────────────────────────
    # Build a factory (called once per replicate) so each replicate gets
    # independent state (convergence_stop tracks its own sliding window).
    if args.fitness_threshold is not None:
        _threshold = args.fitness_threshold
        early_stop_factory = lambda: fitness_threshold(_threshold)
        print(f"  early_stop      = fitness_threshold({args.fitness_threshold})\n")
    elif args.convergence_window is not None:
        _window, _tol = args.convergence_window, args.convergence_tol
        early_stop_factory = lambda: convergence_stop(_window, _tol)
        print(f"  early_stop      = convergence_stop(window={_window}, tol={_tol})\n")
    else:
        early_stop_factory = None

    # ── Resolve resume path ───────────────────────────────────────────────────
    resume_from = None
    if args.resume_from is not None:
        resume_from = args.resume_from
        print(f"  resume_from     = {resume_from}\n")
    elif args.resume_run_dir is not None:
        resume_from = latest_state_checkpoint(args.resume_run_dir)
        if resume_from is None:
            print(f"WARNING: no state_gen_*.npz found in {args.resume_run_dir}/checkpoints — "
                  "starting fresh.\n")
        else:
            print(f"  resume_from     = {resume_from}  (auto-detected)\n")

    save_state_every = args.save_state_every

    t_total = time.perf_counter()
    all_results: list[dict] = []

    # ── Lambda sweep mode ─────────────────────────────────────────────────────
    if args.lambda_sweep is not None:
        # Run baseline once, then each lambda as its own named subdir
        print("── Condition: baseline (no cost) ──────────────────────")
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
            early_stop_fn=early_stop_factory,
            save_state_every=save_state_every,
            resume_from=resume_from,
        )
        for lam in args.lambda_sweep:
            cfg_lam = Config(
                population_size=args.pop_size,
                lambda_dist=lam,
                lambda_act=args.lambda_act,
                dist_frac=args.dist_frac,
                act_frac=args.act_frac,
                edge_frac=args.edge_frac,
                fitness_mode=args.fitness_mode,
                position_sensors=args.position_sensors,
                penalty_warmup_gens=args.penalty_warmup_gens,
            )
            # Derive a deterministic key stream for this lambda from the base seed
            lam_key = jax.random.fold_in(jax.random.PRNGKey(args.seed), int(lam * 1_000_000))
            lam_keys = list(jax.random.split(lam_key, args.n_replicates))
            subdir_name = f"modular_{lam:.4f}".rstrip("0").rstrip(".")
            print(f"\n── Sweep: {subdir_name} (lambda_dist={lam}) ────────────────")
            all_results += run_condition(
                condition_name=subdir_name,
                cfg=cfg_lam,
                wcfg=wcfg,
                rates=rates,
                n_replicates=args.n_replicates,
                n_generations=args.n_generations,
                n_evals=args.n_evals,
                output_dir=output_dir,
                rep_keys=lam_keys,
                verbose=args.verbose,
                early_stop_fn=early_stop_factory,
                save_state_every=save_state_every,
            )

    else:
        # ── Baseline condition ────────────────────────────────────────────────
        if args.condition in ("both", "baseline"):
            print("── Condition: baseline (no cost) ──────────────────────")
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
                early_stop_fn=early_stop_factory,
                save_state_every=save_state_every,
                resume_from=resume_from,
            )

        # ── Modular condition ─────────────────────────────────────────────────
        if args.condition in ("both", "modular") and resume_from is None:
            print(f"\n── Condition: modular (edge={args.lambda_edge} dist={args.lambda_dist} act={args.lambda_act}) ──")
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
                early_stop_fn=early_stop_factory,
                save_state_every=save_state_every,
            )
        elif resume_from is not None:
            print("\n(Skipping modular condition — single-replicate resume mode.)")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_total
    print(f"\nTotal wall time: {total_elapsed / 3600:.2f} hrs  ({total_elapsed / 60:.1f} min)")
    print_summary(all_results)

    # Persist per-run results so they survive beyond the printed summary
    # (consumed by analysis tooling and the forge-queue contract emitter).
    import json as _json
    def _jsonable(o):
        return o.item() if hasattr(o, "item") else float(o)
    (output_dir / "results.json").write_text(
        _json.dumps(all_results, indent=2, default=_jsonable)
    )
    print(f"\nAll results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
