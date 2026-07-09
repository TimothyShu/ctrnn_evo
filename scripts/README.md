# Scripts

Experiment code and the forge-queue integration for ctrnn_evo.

## Running experiments

Jobs run on the **forge** rig via [forge-queue](https://github.com/TimothyShu/forge-queue),
a generic GPU job queue. This repo integrates with it purely through the
`run.sh` contract at the repo root — forge-queue pins a GPU, activates the venv
(`~/jax-env`), and calls `run.sh`, which loads the config, applies params, and
runs `run_experiment.py`. Results are archived to `forge:/mnt/archive/ctrnn_evo/`.

> The old in-repo scheduler (`queue_watcher.sh`, `run_job.sh`, `notify.sh`, and
> the `queue/` directories) has been removed — forge-queue replaces it. Its
> historical run log is frozen in [`../queued-jobs/`](../queued-jobs/).

### Submitting a job

Write a spec in the forge-queue schema (see
[`../configs/example_job.yaml`](../configs/example_job.yaml)) — experiment
settings go under `params:`, **not** at the top level:

```yaml
project: ctrnn_evo
venv: ~/jax-env
notes: "Human-readable description shown in the ntfy notification"
params:
  config: configs/base_m8.yaml   # optional base defaults; keys = run_experiment.py flags
  lambda-edge: 0.0007            # any override is a sibling param (no leading --)
```

Then either paste it into the dashboard's YAML box
(https://forge.tail9b71a9.ts.net) or `cp` it into `~/queue/pending/` on forge.
The queue forces `--output-dir` to the archived run dir, so don't set it. The
job name (and archive dir) comes from the spec filename.

`run_experiment.py --help` lists every available flag.

## Script reference

| Script | Purpose |
|--------|---------|
| `run_experiment.py` | The experiment: evolves CTRNNs across conditions/replicates |
| `../run.sh` | forge-queue entrypoint — config+params → `run_experiment.py` |
| `forge_emit.py` | Writes the forge-queue result contract (metrics/series/records/manifest) |
| `analyse_results.py` | Post-hoc analysis of archived runs |
| `visualise_network.py` | Render an evolved network to PNG |
| `visualise_agent.py` | Render/animate an agent in the world |
