# ctrnn-evo

Evolutionary optimisation of Continuous-Time Recurrent Neural Networks (CTRNNs) using JAX.
Reproduces the modularity experiment of Clune, Mouret & Lipson (2013) in a 2D foraging world.

## Installation

**Requirements:** Python 3.10+, CUDA 12.x (GPU only)

### CPU (default)

```bash
pip install -e .
```

### GPU — CUDA (recommended for full experiments)

```bash
pip install -e ".[cuda]"
```

This installs `jax[cuda12]`, which bundles a CUDA-compatible XLA and cuDNN automatically.
No separate cuDNN install is required.

### Development

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## GPU Setup (NVIDIA)

### 1. Verify CUDA 12.x is installed

```bash
nvidia-smi          # driver version ≥ 525
nvcc --version      # CUDA 12.x
```

If CUDA is not installed, download it from
[developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

### 2. Install with the CUDA extra

```bash
pip install -e ".[cuda]"
```

### 3. Verify JAX sees the GPU

```python
import jax
print(jax.devices())   # expected: [CudaDevice(id=0)]
```

If `CpuDevice` is shown instead, JAX did not find CUDA.
Check that `nvidia-smi` works and that `jax[cuda12]` (not plain `jax`) is installed.

### 4. XLA warm-up

The **first generation is slow** (30–90 s) because XLA compiles the vmapped and scanned
computation graph into CUDA kernels. Every subsequent generation runs at full speed.
Add a warm-up call before starting your experiment timer:

```python
import jax
from ctrnn_evo import Config, WorldConfig
from ctrnn_evo.evolution import init_population, eval_population

cfg  = Config()
wcfg = WorldConfig()
key  = jax.random.PRNGKey(0)

pop = init_population(key, cfg)

print("Warming up XLA compilation...")
key, k = jax.random.split(key)
steps, _ = eval_population(k, pop, cfg, wcfg, n_evals=1)
steps.block_until_ready()
print("Ready.")
```

## Estimated training time

Benchmarked at production parameters:
`population_size=1000, episode_steps=2000, K=20, n_evals=5`

| Hardware | Per generation | 10 rep × 500 gen | 20 rep × 1000 gen |
|---|---|---|---|
| Apple M4 Max (CPU) | ~19 s | ~53 hrs | ~211 hrs |
| RTX 4080 Super | ~2 s | ~6 hrs | ~22 hrs |
| A100 | ~0.65 s | ~1.8 hrs | ~7 hrs |

The M8 validation experiment (Clune replication) requires two conditions
(with / without connection cost penalty). The minimum meaningful replication is
10 replicates × 500 generations per condition.

## Running the M8 experiment (Clune 2013 replication)

The validation experiment evolves two populations in parallel conditions:

| Condition | `lambda_conn` | Expected result |
|---|---|---|
| `baseline` | 0.0 | task fitness, low modularity |
| `modular`  | 0.001 | comparable fitness, higher modularity Q |

### Quick smoke test (< 1 minute)

Verifies the full pipeline works before committing to a long run:

```bash
python scripts/run_experiment.py --smoke-test
```

### Default experiment (10 replicates × 500 generations)

```bash
python scripts/run_experiment.py
```

### Full Clune replication (20 replicates × 1000 generations)

```bash
python scripts/run_experiment.py --n-replicates 20 --n-generations 1000
```

### All options

```
--n-replicates       INT    independent runs per condition        (default: 10)
--n-generations      INT    generations per run                   (default: 500)
--n-evals            INT    episodes per fitness estimate         (default: 5)
--pop-size           INT    population size                       (default: 1000)
--lambda-conn        FLOAT  connection cost strength, modular     (default: 0.001)
--output-dir         PATH   root directory for saved runs         (default: runs/m8)
--seed               INT    base random seed                      (default: 0)
--fitness-threshold  FLOAT  stop a replicate early at this fitness
--convergence-window INT    stop when fitness plateaus for N gens
--convergence-tol    FLOAT  minimum improvement to count as progress (default: 0.001)
--save-state-every   INT    full training state snapshot cadence  (default: 100, 0=off)
--resume-from        PATH   resume a single replicate from a state_gen_*.npz file
--resume-run-dir     PATH   auto-detect latest checkpoint in a run directory and resume
--verbose                   print per-generation progress
--smoke-test                override to 2 rep × 5 gen × pop=20
```

### Output structure

```
runs/m8/
  baseline/
    run_20260517_143022_rep00/
      config.json               ← hyperparameters
      history.jsonl             ← one stats line per generation
      best_genome.npz           ← best genome at end of run
      checkpoints/
        gen_000000.npz          ← best-genome snapshots (every 100 gens)
        gen_000100.npz
        ...
        state_gen_000100.npz    ← full training state for resume (every 100 gens)
        state_gen_000200.npz
        ...
    run_20260517_143022_rep01/
    ...
  modular/
    run_20260517_143022_rep00/
    ...
```

### Resuming an interrupted run

Every 100 generations (configurable with `--save-state-every`) the script saves a
`state_gen_*.npz` snapshot containing the full population, fitness array, steps array,
and RNG key — enough to resume exactly where training stopped.

**Auto-detect the latest checkpoint from a run directory:**

```bash
python scripts/run_experiment.py \
  --resume-run-dir runs/m8/baseline/run_20260517_143022_rep00
```

**Point at a specific snapshot:**

```bash
python scripts/run_experiment.py \
  --resume-from runs/m8/baseline/run_20260517_143022_rep00/checkpoints/state_gen_000300.npz
```

**Programmatically (e.g. in a custom script):**

```python
from ctrnn_evo.logger import latest_state_checkpoint, load_training_state
from ctrnn_evo.evolution import run_evolution

run_dir    = "runs/m8/baseline/run_20260517_143022_rep00"
state_path = latest_state_checkpoint(run_dir)          # finds state_gen_000300.npz

best, fitness, new_history = run_evolution(
    key=jax.random.PRNGKey(0),    # ignored when resume_from is set
    n_generations=500,
    cfg=cfg, wcfg=wcfg, rates=rates,
    resume_from=state_path,
    state_checkpoint_dir=f"{run_dir}/checkpoints",
)
```

The resumed `run_evolution` restores the saved RNG key so the random sequence is
identical to what it would have been without the interruption. History from the
resumed run covers only the new generations; use `load_history(run_dir)` for the
full per-generation record.

> **Note for full experiments:** `--resume-from` / `--resume-run-dir` runs a single
> replicate and skips the second condition. For resuming a multi-replicate experiment
> interrupted mid-run, re-launch without those flags — already-completed replicates
> will produce new run directories alongside the existing ones. Merge results from both
> sessions when analysing.

### Reading results

```python
from ctrnn_evo.logger import load_history, load_genome, load_config
from ctrnn_evo.analysis import analyse_genome
from pathlib import Path

run_dir = Path("runs/m8/modular/run_<timestamp>_rep00")
cfg, wcfg, rates = load_config(run_dir)
history = load_history(run_dir)          # list of dicts, one per generation
genome  = load_genome(run_dir / "best_genome.npz")
metrics = analyse_genome(genome, cfg)
print(f"Q = {metrics['q']:.3f}")
```

### Interpreting the summary table

```
=================================================================
RESULTS SUMMARY
=================================================================
Condition        Q mean   Q std   Fitness  Fit std   Nodes
-----------------------------------------------------------------
baseline          0.142   0.031     0.847    0.062    38.1
modular           0.381   0.058     0.821    0.071    31.4
=================================================================
Modularity gain (modular − baseline):  ΔQ = +0.239
Fitness cost   (modular − baseline): ΔFit = -0.026
✓ Connection cost pressure increased modularity.
✓ Fitness was not significantly degraded.
```

The key result from Clune 2013 is that `ΔQ > 0` with minimal `ΔFit`.
If `ΔFit` is large and negative, decrease `--lambda-conn`.
If `ΔQ` is small, increase `--lambda-conn`.

### Tuning `--lambda-conn`

The connection cost penalty must be large enough to drive modularity without
collapsing task performance. A useful heuristic:

- `lambda_conn=0.001` — recommended starting point
- `lambda_conn=0.0005` — lighter pressure, safer if fitness collapses
- `lambda_conn=0.002` — stronger modularity pressure, may reduce task performance

---

## What to expect from your first run

This section describes the expected results for the default 10 rep × 500 gen
experiment run with `lambda_act=0.0` (both conditions) — the configuration as
of Phase 0. The activation cost term is implemented but intentionally disabled
at this stage; only `lambda_conn` differs between conditions.

### What this run is measuring

The experiment has one structural lever: **connection cost**, which penalises
the total wire length of the evolved network. Wire length is the sum of
Euclidean distances between every connected pair of active neurons, where
positions evolve in the unit square [0,1]².

```
f = f_raw - lambda_conn × Σ( ||pos_i - pos_j|| × edge_mask[i,j] )
```

With positions in [0,1]² and a moderately connected network (~35 active neurons,
15–25% density), a typical `C_conn` is 80–200, so `lambda_conn=0.001` applies a
penalty of roughly **0.08–0.20** against a maximum raw fitness of 1.0. This is
a meaningful but not crushing penalty — strong enough to drive structural change,
small enough to leave room for task performance.

### Expected numbers

These are honest ballpark estimates, not guarantees. Replicate variance is high
and 500 generations may not fully converge.

| Metric | Baseline (λ=0) | Modular (λ=0.001) | Target signal |
|---|---|---|---|
| Q mean | 0.10–0.22 | 0.20–0.40 | ΔQ > 0.05 |
| Q std  | 0.04–0.08 | 0.05–0.12 | — |
| Adj. fitness mean | 0.65–0.90 | 0.50–0.80 | ΔFit > −0.15 |
| Active nodes mean | 35–50 | 20–38 | modular < baseline |

**The primary signal is ΔQ.** The `Fitness` column in the summary table shows
the *adjusted* fitness (raw performance minus the cost penalty), so some of the
modular condition's fitness gap is the penalty itself rather than genuine task
degradation. The more useful question is whether modular networks achieve similar
raw performance with fewer, shorter connections — that is what ΔQ captures.

### What a successful first run looks like

```
Modularity gain (modular − baseline):  ΔQ = +0.10 to +0.25
Fitness cost   (modular − baseline): ΔFit = −0.05 to −0.20
✓ Connection cost pressure increased modularity.
```

Even a modest ΔQ of +0.05 with ΔFit around −0.10 is a meaningful result for
a first run — it confirms the cost penalty is driving structural reorganisation.
The Clune 2013 paper shows much larger ΔQ (+0.2 to +0.3) but used a simpler
one-type network, a different world, and 1000+ generations. Our network has three
neuron types (E, FSI, SII), Dale's law, and spatially embedded wiring — more
biological realism means slower, noisier convergence.

**Watch for these secondary signals too:**

- `n_active` (Nodes column) should be lower in the modular condition — the cost
  pressure prunes neurons that are not earning their connection budget.
- `best_conn_cost` in each replicate's result dict should be markedly lower for
  modular runs.
- Within the modular condition, replicates with higher Q should also show lower
  connection cost (they're on the same Pareto front — fewer/shorter connections
  and modular structure go together).

### What would call for tuning

| Observation | Diagnosis | Fix |
|---|---|---|
| ΔFit < −0.20 | penalty too aggressive, dominates selection | lower `--lambda-conn` to 0.0005 |
| ΔQ < 0.03 and fitness looks fine | too little structural pressure | raise `--lambda-conn` to 0.002 |
| ΔQ ≈ 0 and modular fitness is very low | penalty crushing all networks | lower `--lambda-conn` to 0.0005 |
| High variance, some replicates Q ≈ 0 | 500 gens not enough for all seeds | rerun with `--n-generations 1000` |

### What this run is NOT testing (yet)

- **Activation cost (`lambda_act`)** — currently `0.0` for both conditions. This
  penalty will later reward sparse, energy-efficient firing patterns. Without it,
  neurons are free to fire persistently, which can blur module boundaries. Adding
  this in a future run is expected to sharpen Q further.
- **LTC neurons and disinhibitory types** — the three current types (E, FSI, SII)
  are all standard CTRNN units. Future phases add Liquid Time-Constant dynamics and
  a disinhibitory cell class.
- **Full 1000-generation convergence** — 500 gens gives a useful early signal but
  Q typically keeps rising slowly through generation 800–1000 in our world.
- **Modularity at the functional level** — Q measures structural modularity (graph
  community structure). Functional modularity (whether sub-circuits independently
  process the food vs. energy signals) requires additional analysis not yet
  implemented.

### Runtime

On an RTX 4080 Super at the default settings:

```
XLA warm-up:          1–2 min
Baseline (10 × 500):  ~2.8 hrs
Modular  (10 × 500):  ~2.8 hrs
─────────────────────────────
Total:                ~6 hrs
```

State snapshots are saved every 100 generations automatically so an overnight
interruption loses at most 200 seconds of work (≈ 100 generations per replicate).

---

## Project structure

```
scripts/
  run_experiment.py   — M8 Clune replication (two-condition evolutionary run)
ctrnn_evo/
  config.py       — Config and WorldConfig dataclasses
  genome.py       — Genome pytree, random_genome, effective_weights
  forward.py      — CTRNN forward pass (jax.lax.scan over K ticks)
  cost.py         — connection_cost, adjusted_fitness
  world.py        — WorldState, step_world, run_episode
  controllers.py  — random_walk, nearest_hotspot (baseline controllers)
  brain.py        — run_brain_episode, batch_run_brain_episode
  mutation.py     — MutationRates, all mutation operators
  evolution.py    — init_population, eval_population, evolve_step, run_evolution
  logger.py       — make_run_dir, save/load genome+config+history+training state
  analysis.py     — modularity_q, network_stats, analyse_genome, summarise_run
tests/
  test_genome.py
  test_forward.py
  test_cost.py
  test_mutation.py
  test_world.py
  test_brain.py
  test_evolution.py
  test_logger.py
  test_analysis.py
```
