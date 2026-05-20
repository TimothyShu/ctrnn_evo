#!/usr/bin/env bash
# Queue watcher — submit pending job specs to tsp one at a time.
#
# Start once on WSL startup (add to ~/.bashrc or run manually):
#   bash scripts/queue_watcher.sh &
#
# To check the tsp queue:   tsp
# To cancel a queued job:   tsp -r <job-id>
# To clear finished jobs:   tsp -C

# Don't use set -e here: a transient tsp hiccup or failed mv must not kill
# the watcher — it needs to keep looping and picking up future jobs.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PENDING="$REPO_DIR/queue/pending"
QUEUED="$REPO_DIR/queue/queued"
mkdir -p "$PENDING" "$QUEUED"

# Enforce single-job execution — one GPU, no parallelism.
tsp -S 1

echo "[watcher] started, polling $PENDING every 10s"
echo "[watcher] tsp parallelism: $(tsp -S) slot(s)"

while true; do
    for spec in "$PENDING"/*.yaml "$PENDING"/*.yml; do
        # glob expands to literal string when no files match
        [ -f "$spec" ] || continue

        SPEC_NAME="$(basename "$spec")"

        # Atomic move to queued/ prevents double-submission if the watcher
        # loops again before tsp has started the job.
        DEST="$QUEUED/$SPEC_NAME"
        if ! mv "$spec" "$DEST" 2>/dev/null; then
            echo "[watcher] WARNING: could not move $SPEC_NAME (already claimed?), skipping"
            continue
        fi

        echo "[watcher] submitting: $SPEC_NAME"
        tsp bash "$REPO_DIR/scripts/run_job.sh" "$DEST" || \
            echo "[watcher] WARNING: tsp submission failed for $SPEC_NAME"
    done

    sleep 10
done
