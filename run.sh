#!/usr/bin/env bash
# run.sh — forge-queue entrypoint for ctrnn_evo.
#
# Called by the forge-queue watcher, which has already: pinned a GPU via
# CUDA_VISIBLE_DEVICES, activated the venv (~/jax-env), and cd'd here. See
# ~/forge-queue/docs/conventions.md for the full contract.
#
# Job-spec params (under params:) map to scripts/run_experiment.py flags:
#   config: <path>    optional base-config YAML; its keys become defaults
#   <flag>: <value>   any run_experiment.py flag without the leading -- ;
#                     underscores and dashes are interchangeable.
#                     booleans: true -> bare flag, false -> omitted.
# Example: params: {config: configs/base_m8.yaml, n-generations: 500}
#       -> python3 scripts/run_experiment.py --n-generations 500 <config keys...>
set -euo pipefail

cd "${FORGE_PROJECT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

# Best-effort pull of latest code; keep running on failure (e.g. no creds).
git pull --ff-only 2>&1 || echo "[run.sh] WARN: git pull failed, using current code"
echo "[run.sh] commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown) gpu=${FORGE_GPU:-?}"

# Build the CLI args: load optional base config, overlay the FORGE_PARAM_*
# values (minus the special 'config' key) on top. Emitting one token per line
# keeps values with spaces intact.
readarray -t ARGS < <(python3 - <<'PY'
import os, sys, yaml, pathlib

params = {}
for k, v in os.environ.items():
    if k.startswith("FORGE_PARAM_"):
        flag = k[len("FORGE_PARAM_"):].lower().replace("_", "-")
        params[flag] = v

base = {}
cfg = params.pop("config", "")
if cfg:
    p = pathlib.Path(cfg)
    if p.exists():
        base = yaml.safe_load(p.read_text()) or {}
    else:
        print(f"# WARN: config {cfg!r} not found, using params only", file=sys.stderr)
base = {str(k).replace("_", "-"): v for k, v in base.items()}

merged = {**base, **params}   # params override base config
for k, v in merged.items():
    s = str(v).strip().lower()
    if s in ("true", "false"):          # boolean flag
        if s == "true":
            print(f"--{k}")
    elif v not in (None, ""):
        print(f"--{k}")
        print(str(v))
PY
)

# Force all experiment output into the forge-queue run dir so it gets archived.
# Appended last so it overrides any output-dir from the config/params.
if [ -n "${FORGE_RUN_DIR:-}" ]; then
    ARGS+=(--output-dir "$FORGE_RUN_DIR")
fi

echo "[run.sh] scripts/run_experiment.py ${ARGS[*]:-(defaults)}"
set +e
python3 -u scripts/run_experiment.py "${ARGS[@]}"
rc=$?
set -e

# Emit the forge-queue result contract (records/metrics/series/manifest) into
# the run dir so the archive + dashboard can read it. Advisory — never fail the
# job over emission, and still attempt it on a failed run (partial results).
if [ -n "${FORGE_RUN_DIR:-}" ]; then
    python3 scripts/forge_emit.py "$FORGE_RUN_DIR" || echo "[run.sh] WARN: forge_emit failed"
fi

exit "$rc"
