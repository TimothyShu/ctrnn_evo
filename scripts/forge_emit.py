#!/usr/bin/env python3
"""forge_emit.py — ctrnn_evo → forge-queue result-contract adapter.

Reads the experiment's results.json + per-run history.jsonl from a run dir and
writes the standard forge-queue contract files into the same dir:

  records.csv   one row per condition x replicate
  metrics.json  aggregate scalar summary (mean/std Q + fitness per condition)
  series.csv    per-generation curves, aggregated across reps per condition
  manifest.json tags every best_genome.npz as a `network` artifact

This is an adapter living in the tenant: it keeps run_experiment.py
forge-agnostic. Usage: forge_emit.py <run_dir>  (run_dir == FORGE_RUN_DIR).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

SERIES_KEYS = ["max_fitness", "mean_n_active"]   # per-generation curves to emit


def _read_results(run_dir: Path):
    p = run_dir / "results.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return []


def write_records(run_dir: Path, results: list) -> None:
    cols = ["condition", "replicate", "best_q", "best_n_active",
            "best_conn_cost", "final_max_fitness", "generations_run", "elapsed_s"]
    with (run_dir / "records.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([r.get(c, "") for c in cols])


def write_metrics(run_dir: Path, results: list) -> None:
    metrics = {"n_runs": len(results)}
    conditions = sorted({r["condition"] for r in results})
    for cond in conditions:
        reps = [r for r in results if r["condition"] == cond]
        qs = [float(r["best_q"]) for r in reps]
        fits = [float(r["final_max_fitness"]) for r in reps]
        nodes = [float(r["best_n_active"]) for r in reps]
        metrics[f"{cond}_n"] = len(reps)
        metrics[f"{cond}_mean_q"] = round(mean(qs), 4)
        metrics[f"{cond}_std_q"] = round(pstdev(qs), 4) if len(qs) > 1 else 0.0
        metrics[f"{cond}_mean_fitness"] = round(mean(fits), 4)
        metrics[f"{cond}_mean_n_active"] = round(mean(nodes), 2)
    if "baseline" in conditions and "modular" in conditions:
        metrics["q_gain_modular"] = round(
            metrics["modular_mean_q"] - metrics["baseline_mean_q"], 4)
        metrics["fitness_cost_modular"] = round(
            metrics["modular_mean_fitness"] - metrics["baseline_mean_fitness"], 4)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def write_series(run_dir: Path, results: list) -> None:
    conditions = sorted({r["condition"] for r in results})
    # agg: cond -> generation -> key -> [values across reps]
    agg: dict = {}
    for r in results:
        hist = Path(r["run_dir"]) / "history.jsonl"
        if not hist.exists():
            continue
        for line in hist.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gen = row.get("generation")
            if gen is None:
                continue
            slot = agg.setdefault(r["condition"], {}).setdefault(gen, {})
            for k in SERIES_KEYS:
                if k in row:
                    slot.setdefault(k, []).append(float(row[k]))
    if not agg:
        return
    header = ["generation"] + [f"{c}_{k}" for c in conditions for k in SERIES_KEYS]
    all_gens = sorted({g for cond in agg for g in agg[cond]})
    with (run_dir / "series.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for g in all_gens:
            out = [g]
            for cond in conditions:
                for k in SERIES_KEYS:
                    vals = agg.get(cond, {}).get(g, {}).get(k)
                    out.append(round(mean(vals), 4) if vals else "")
            w.writerow(out)


def write_manifest(run_dir: Path) -> None:
    artifacts = {npz.relative_to(run_dir).as_posix(): {"type": "network"}
                 for npz in run_dir.rglob("best_genome.npz")}
    if artifacts:
        (run_dir / "manifest.json").write_text(
            json.dumps({"artifacts": artifacts}, indent=2))


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: forge_emit.py <run_dir>", file=sys.stderr)
        return 2
    run_dir = Path(args[0])
    results = _read_results(run_dir)
    write_manifest(run_dir)   # tag genomes even if results.json is missing
    if not results:
        print(f"[forge_emit] no results.json in {run_dir} — emitted manifest only")
        return 0
    write_records(run_dir, results)
    write_metrics(run_dir, results)
    write_series(run_dir, results)
    print(f"[forge_emit] wrote records.csv / metrics.json / series.csv / manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
