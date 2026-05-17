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
--n-replicates    INT    independent runs per condition        (default: 10)
--n-generations   INT    generations per run                   (default: 500)
--n-evals         INT    episodes per fitness estimate         (default: 5)
--pop-size        INT    population size                       (default: 1000)
--lambda-conn     FLOAT  connection cost strength, modular     (default: 0.001)
--output-dir      PATH   root directory for saved runs         (default: runs/m8)
--seed            INT    base random seed                      (default: 0)
--verbose                print per-generation progress
--smoke-test             override to 2 rep × 5 gen × pop=20
```

### Output structure

```
runs/m8/
  baseline/
    run_20260517_143022_rep00/
      config.json         ← hyperparameters
      history.jsonl       ← one stats line per generation
      best_genome.npz     ← best genome at end of run
      checkpoints/
        gen_000000.npz
        gen_000100.npz
        ...
    run_20260517_143022_rep01/
    ...
  modular/
    run_20260517_143022_rep00/
    ...
```

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

- `lambda_conn=0.001` — recommended starting point (penalty ≈ 8% of fitness range)
- `lambda_conn=0.0005` — lighter pressure, safer for harder worlds
- `lambda_conn=0.002` — stronger pressure, may reduce fitness

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
  logger.py       — make_run_dir, save/load genome+config+history, make_logger
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
