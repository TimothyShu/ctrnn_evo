#!/usr/bin/env bash
# Usage: run_job.sh <path-to-job-spec.yaml>
#
# Called by tsp via queue_watcher.sh. Does git pull, builds CLI args from the
# job spec YAML, runs run_experiment.py, archives the spec, and sends ntfy notifications.
#
# Job spec schema:
#   config:    path/to/base-config.yaml   # optional defaults file
#   overrides:                            # merged on top of config
#     key: value
#   notes: "human-readable description"
#
# Keys in config/overrides match run_experiment.py flag names (without --).
# Boolean true  → bare flag (e.g. verbose: true → --verbose).
# Boolean false → flag is omitted entirely.

set -euo pipefail

SPEC_ARG="$1"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# ── Env (ntfy topic, etc.) ────────────────────────────────────────────────────
[ -f .env ] && source .env

# ── Move spec to processing/ (signals job is active) ─────────────────────────
SPEC_NAME="$(basename "$SPEC_ARG")"
mkdir -p queue/processing
SPEC="queue/processing/$SPEC_NAME"
mv "$SPEC_ARG" "$SPEC"

# ── Pull latest code ──────────────────────────────────────────────────────────
echo "[run_job] git pull..."
git pull --ff-only origin main 2>&1 || echo "[run_job] WARNING: git pull failed, continuing with current code"
COMMIT="$(git rev-parse --short HEAD)"

# ── Parse spec: extract notes and build CLI arg array ────────────────────────
NOTES="$(python3 -c "
import yaml, sys
s = yaml.safe_load(open(sys.argv[1]))
print(s.get('notes') or '')
" "$SPEC")"

# Python writes one token per line; readarray builds a bash array.
# This handles values that contain spaces without word-splitting.
readarray -t ARGS < <(python3 - "$SPEC" <<'PYEOF'
import sys, yaml, pathlib

spec      = yaml.safe_load(open(sys.argv[1]))
cfg_path  = spec.get('config') or ''
overrides = spec.get('overrides') or {}

base = {}
if cfg_path:
    p = pathlib.Path(cfg_path)
    if p.exists():
        base = yaml.safe_load(p.read_text()) or {}
    else:
        print(f'# WARNING: config {cfg_path!r} not found, using overrides only', file=__import__('sys').stderr)

merged = {**base, **overrides}

for k, v in merged.items():
    if isinstance(v, bool):
        if v:
            print(f'--{k}')          # bare flag for true; omit for false
    elif v is not None:
        print(f'--{k}')
        print(str(v))
PYEOF
)

echo "[run_job] spec=$SPEC_NAME  commit=$COMMIT"
[ ${#ARGS[@]} -gt 0 ] && echo "[run_job] args: ${ARGS[*]}" || echo "[run_job] args: (none — using script defaults)"

bash scripts/notify.sh \
  "GPU job started" \
  "spec=$SPEC_NAME commit=$COMMIT${NOTES:+$'\n'$NOTES}"

# ── Run experiment ────────────────────────────────────────────────────────────
START=$(date +%s)
EXIT_CODE=0
python3 -u scripts/run_experiment.py "${ARGS[@]}" 2>&1 || EXIT_CODE=$?
ELAPSED=$(( ($(date +%s) - START) / 60 ))

# ── Archive spec to done/ or failed/ ─────────────────────────────────────────
if [ "$EXIT_CODE" -eq 0 ]; then
    STATUS=done;   PRIORITY=default
else
    STATUS=failed; PRIORITY=high
fi

DEST_NAME="${SPEC_NAME%.yaml}_${COMMIT}_${STATUS}.yaml"
mkdir -p "queue/$STATUS" queued-jobs
mv "$SPEC" "queue/$STATUS/$DEST_NAME"
cp "queue/$STATUS/$DEST_NAME" "queued-jobs/$DEST_NAME"

echo "[run_job] $STATUS in ${ELAPSED}m  (exit=$EXIT_CODE)  ->  queue/$STATUS/$DEST_NAME"

bash scripts/notify.sh \
  "GPU job $STATUS (${ELAPSED}m)" \
  "spec=$SPEC_NAME commit=$COMMIT exit=$EXIT_CODE${NOTES:+$'\n'$NOTES}" \
  "$PRIORITY"

exit "$EXIT_CODE"
