#!/usr/bin/env python3
"""
Post-hoc analysis of completed M8 experiment runs.

Loads all valid runs (500 generations) from runs/m8/baseline and
runs/m8/modular, groups modular runs by lambda_conn (read from config.json),
and prints a per-replicate table, condition summary, Mann-Whitney tests,
fitness trajectories, and the Clune 2013 replication verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path
from scipy import stats as scipy_stats
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ctrnn_evo.logger import load_history, load_genome, load_config
from ctrnn_evo.analysis import analyse_genome

RUNS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "runs" / "m8"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_n_gen_full() -> int:
    """Return the modal (most common) history length across all run dirs."""
    from collections import Counter
    lengths = []
    for cond_dir in RUNS_DIR.iterdir():
        if not cond_dir.is_dir():
            continue
        for run_dir in cond_dir.iterdir():
            if not run_dir.is_dir():
                continue
            h = load_history(run_dir)
            if h:
                lengths.append(len(h))
    if not lengths:
        return 500
    return Counter(lengths).most_common(1)[0][0]


def load_condition(condition: str, n_gen_full: int) -> list[dict]:
    """Load all valid replicates for a condition, grouped by lambda_conn."""
    cond_dir = RUNS_DIR / condition
    if not cond_dir.exists():
        print(f"  [warn] {cond_dir} not found")
        return []

    reps = []
    for run_dir in sorted(cond_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        history = load_history(run_dir)
        if len(history) < n_gen_full:
            print(f"  [skip] {run_dir.name}  ({len(history)} gens - incomplete)")
            continue

        cfg, wcfg, rates = load_config(run_dir)
        genome  = load_genome(run_dir / "best_genome.npz")
        metrics = analyse_genome(genome, cfg)
        final   = history[-1]

        reps.append({
            "run_dir":     run_dir.name,
            "condition":   condition,
            "lambda_edge": cfg.lambda_edge,
            "lambda_dist": cfg.lambda_dist,
            "lambda_act":  cfg.lambda_act,
            "q":           metrics["q"],
            "n_active":    metrics["n_active"],
            "n_edges":     metrics["n_edges"],
            "density":     metrics["density"],
            "conn_cost":   metrics["wiring_cost"],
            "fit_max":     final["max_fitness"],
            "fit_mean":    final["mean_fitness"],
            "history":     history,
        })

    return reps


def _condition_key(r: dict) -> tuple:
    return (r["lambda_edge"], r["lambda_dist"], r["lambda_act"])


def _condition_label(key: tuple) -> str:
    e, d, a = key
    parts = []
    if e: parts.append(f"edge={e}")
    if d: parts.append(f"dist={d}")
    if a: parts.append(f"act={a}")
    return ", ".join(parts) if parts else "baseline"


def group_by_lambda(reps: list[dict]) -> dict[tuple, list[dict]]:
    """Group replicate dicts by their (lambda_edge, lambda_dist, lambda_act) tuple."""
    groups: dict[tuple, list[dict]] = {}
    for r in reps:
        groups.setdefault(_condition_key(r), []).append(r)
    return dict(sorted(groups.items()))


def summarise(reps: list[dict]) -> dict:
    return {
        "q":        [r["q"]        for r in reps],
        "fit":      [r["fit_max"]  for r in reps],
        "nodes":    [r["n_active"] for r in reps],
        "edges":    [r["n_edges"]  for r in reps],
        "density":  [r["density"]  for r in reps],
        "conn_cost":[r["conn_cost"] for r in reps],
    }


def trajectory(reps: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.array([[h[key] for h in r["history"]] for r in reps])
    return arr.mean(axis=0), arr.std(axis=0)


def mw(a, b):
    u, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    return u, p, sig


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading results...")
    n_gen_full = _detect_n_gen_full()
    print(f"  Detected run length: {n_gen_full} generations\n")

    baseline_reps = load_condition("baseline", n_gen_full)

    # Collect all modular* subdirs (handles "modular", "modular_0.0005", etc.)
    modular_reps: list[dict] = []
    if RUNS_DIR.exists():
        for subdir in sorted(RUNS_DIR.iterdir()):
            if subdir.is_dir() and subdir.name.startswith("modular"):
                modular_reps += load_condition(subdir.name, n_gen_full)

    modular_groups = group_by_lambda(modular_reps)
    n_baseline     = len(baseline_reps)
    n_mod_total    = len(modular_reps)
    print(f"\nValid replicates - baseline: {n_baseline}, modular: {n_mod_total} "
          f"({', '.join(f'{_condition_label(k)}: {len(r)}' for k, r in modular_groups.items())})\n")

    # ── Per-replicate table ───────────────────────────────────────────────────
    print("=" * 82)
    print("PER-REPLICATE SUMMARY")
    print("=" * 82)
    print(f"{'Run':<45} {'Q':>6} {'Fit':>6} {'Nodes':>6} {'Edges':>6} {'Density':>8}")
    print("-" * 82)

    print("  BASELINE")
    for r in baseline_reps:
        print(f"  {r['run_dir']:<43} {r['q']:6.3f} "
              f"{r['fit_max']:6.3f} {r['n_active']:6d} {r['n_edges']:6d} {r['density']:8.4f}")
    print()

    for key, reps in modular_groups.items():
        print(f"  MODULAR  ({_condition_label(key)})")
        for r in reps:
            print(f"  {r['run_dir']:<43} {r['q']:6.3f} "
                  f"{r['fit_max']:6.3f} {r['n_active']:6d} {r['n_edges']:6d} {r['density']:8.4f}")
        print()

    # ── Condition summary ─────────────────────────────────────────────────────
    print("=" * 82)
    print("CONDITION SUMMARY (mean ± std)")
    print("=" * 82)
    fmt = "{:<18} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}"
    print(fmt.format("Condition", "Q", "Fitness", "Nodes", "Edges", "Density", "ConnCost"))
    print("-" * 82)

    all_summaries: dict[str, dict] = {}

    b_sum = summarise(baseline_reps)
    all_summaries["baseline"] = b_sum
    print(fmt.format(
        "baseline",
        f"{np.mean(b_sum['q']):.3f}±{np.std(b_sum['q']):.3f}",
        f"{np.mean(b_sum['fit']):.3f}±{np.std(b_sum['fit']):.3f}",
        f"{np.mean(b_sum['nodes']):.1f}±{np.std(b_sum['nodes']):.1f}",
        f"{np.mean(b_sum['edges']):.1f}±{np.std(b_sum['edges']):.1f}",
        f"{np.mean(b_sum['density']):.4f}±{np.std(b_sum['density']):.4f}",
        f"{np.mean(b_sum['conn_cost']):.2f}±{np.std(b_sum['conn_cost']):.2f}",
    ))

    for key, reps in modular_groups.items():
        label = _condition_label(key)
        s     = summarise(reps)
        all_summaries[label] = s
        print(fmt.format(
            label[:18],
            f"{np.mean(s['q']):.3f}±{np.std(s['q']):.3f}",
            f"{np.mean(s['fit']):.3f}±{np.std(s['fit']):.3f}",
            f"{np.mean(s['nodes']):.1f}±{np.std(s['nodes']):.1f}",
            f"{np.mean(s['edges']):.1f}±{np.std(s['edges']):.1f}",
            f"{np.mean(s['density']):.4f}±{np.std(s['density']):.4f}",
            f"{np.mean(s['conn_cost']):.2f}±{np.std(s['conn_cost']):.2f}",
        ))

    # ── Statistical tests vs baseline ─────────────────────────────────────────
    print("\n" + "=" * 82)
    print("STATISTICAL TESTS vs BASELINE  (Mann-Whitney U, two-sided)")
    print("=" * 82)
    metrics_labels = [("q","Q"), ("fit","Fitness"), ("nodes","Nodes"),
                      ("edges","Edges"), ("density","Density"), ("conn_cost","ConnCost")]

    for key, reps in modular_groups.items():
        print(f"\n  {_condition_label(key)}:")
        s = summarise(reps)
        for metric, label in metrics_labels:
            u, p, sig = mw(b_sum[metric], s[metric])
            delta = np.mean(s[metric]) - np.mean(b_sum[metric])
            print(f"    {label:<10}  D={delta:+.4f}   U={u:.0f}  p={p:.4f}  {sig}")

    # ── Fitness trajectories ──────────────────────────────────────────────────
    print("\n" + "=" * 82)
    print("FITNESS TRAJECTORY - max fitness (mean ± std, every 50 gens)")
    print("=" * 82)

    col_w = 20
    header_parts = [f"{'Gen':>5}"] + [f"{'baseline':>{col_w}}"] + \
                   [f"{_condition_label(k):>{col_w}}" for k in modular_groups]
    print(" | ".join(header_parts))
    print("-" * (7 + (col_w + 3) * (1 + len(modular_groups))))

    b_max, b_max_std = trajectory(baseline_reps, "max_fitness")
    mod_traj = {k: trajectory(r, "max_fitness") for k, r in modular_groups.items()}

    step = max(1, n_gen_full // 10)
    for g in list(range(0, n_gen_full, step)) + [n_gen_full - 1]:
        row = [f"{g:5d}"]
        row.append(f"{b_max[g]:.3f}±{b_max_std[g]:.3f}".rjust(col_w))
        for k, (m, s) in mod_traj.items():
            row.append(f"{m[g]:.3f}±{s[g]:.3f}".rjust(col_w))
        print(" | ".join(row))

    # ── Clune verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 82)
    print("CLUNE 2013 REPLICATION VERDICT")
    print("=" * 82)
    for key, reps in modular_groups.items():
        s    = summarise(reps)
        dq   = np.mean(s["q"])   - np.mean(b_sum["q"])
        dfit = np.mean(s["fit"]) - np.mean(b_sum["fit"])
        _, pq, _ = mw(b_sum["q"], s["q"])
        print(f"\n  {_condition_label(key)}:")
        print(f"    DQ   = {dq:+.4f}  (p={pq:.4f})")
        print(f"    DFit = {dfit:+.4f}")
        if dq > 0.05 and pq < 0.05:
            print("    [YES] Modularity increased significantly.")
        elif dq > 0 and pq >= 0.05:
            print("    [~] Q trend positive but not significant - more reps/gens needed.")
        else:
            print("    [NO] Modularity did not increase.")
        if abs(dfit) < 0.05:
            print("    [YES] Fitness preserved (|DFit| < 0.05).")
        else:
            print(f"    [^] Fitness changed noticeably: DFit={dfit:+.4f}.")


if __name__ == "__main__":
    main()
