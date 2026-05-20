# Scripts & Job Queue — Bootstrap Guide

## Overview

```
queue/pending/    drop a .yaml job spec here to submit
queue/queued/     moved here atomically by watcher before tsp submission
queue/processing/ moved here by run_job.sh when a job starts
queue/done/       completed specs land here
queue/failed/     failed specs land here
queued-jobs/      historical archive committed to git
configs/          base config YAMLs and example job specs
```

Job flow: `pending/ → queued/ → processing/ → done/ | failed/ + queued-jobs/`

---

## Bootstrap on a fresh WSL2 machine

### 1. Install system dependencies

```bash
sudo apt update
sudo apt install -y task-spooler curl
```

### 2. Clone and install Python dependencies

```bash
git clone <repo-url>
cd ctrnn_evo
pip install -e ".[cuda]"   # installs jax[cuda12], numpy, pyyaml, etc.
```

### 3. Configure ntfy

```bash
cp .env.example .env
# Edit .env and set your ntfy topic (keep it non-obvious)
nano .env
```

Install ntfy on your phone, subscribe to the same topic, and you will get
push notifications when jobs start and finish.

### 4. Configure tsp for single-GPU use

```bash
# tsp runs one job at a time by default — confirm:
tsp -S       # should print 1
# If not: tsp -S 1
```

### 5. Start the queue watcher

```bash
bash scripts/queue_watcher.sh &
# Or to keep it running across WSL sessions, add to ~/.bashrc:
# nohup bash /path/to/scripts/queue_watcher.sh >> ~/queue_watcher.log 2>&1 &
```

---

## Submitting a job

1. Copy or write a job spec YAML (see `configs/example_job.yaml` for the schema).
2. Drop it in `queue/pending/`:

```bash
cp configs/example_job.yaml queue/pending/phase1a_edge_0007.yaml
```

The watcher picks it up within 10 seconds and hands it to `tsp`.

### Job spec schema

```yaml
config: configs/base_m8.yaml   # optional base defaults file
overrides:                      # merged on top of config; keys = CLI flag names
  lambda-edge: 0.0007
  output-dir:  runs/my_run
notes: "Human-readable description shown in ntfy notification"
```

Keys in `overrides:` map directly to `run_experiment.py` CLI flags (without `--`).
YAML handles types natively: `0.0007` is float, `10` is int, `true` is a bare flag.

### All available overrides

| Key | Type | Default |
|-----|------|---------|
| `n-replicates` | int | 10 |
| `n-generations` | int | 500 |
| `n-evals` | int | 5 |
| `pop-size` | int | 1000 |
| `lambda-edge` | float | 0.0 |
| `lambda-dist` | float | 0.0 |
| `lambda-act` | float | 0.0 |
| `condition` | str | `both` |
| `output-dir` | str | `runs/default` |
| `seed` | int | 0 |
| `verbose` | bool | true |
| `fitness-threshold` | float | — |
| `convergence-window` | int | — |

Typical run (both conditions, 10 reps x 500 gens): ~5 hours on RTX 4080 Super.

---

## Monitoring

```bash
tsp                  # show queue status
tsp -t <id>          # tail output of a running job
tsp -c <id>          # show full output of a finished job
tsp -r <id>          # remove a queued job before it starts
tsp -k <id>          # kill a running job
tsp -C               # clear finished jobs from the list
tail -f queue/processing/*.yaml   # see which spec is active
ls queued-jobs/       # historical log of completed specs
```

---

## Script reference

| Script | Purpose |
|--------|---------|
| `queue_watcher.sh` | Polls `queue/pending/` every 10s; submits to tsp |
| `run_job.sh <spec>` | Runs one job: git pull, build args, train, notify |
| `notify.sh <title> <body> [priority]` | Send ntfy push notification |

---

## Historical log

Every completed spec (both done and failed) is copied to `queued-jobs/` with
the git commit SHA and status appended to the filename, then committed to git.
This gives a permanent record of exactly what configuration produced each run.
