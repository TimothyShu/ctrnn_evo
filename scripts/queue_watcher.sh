#!/usr/bin/env bash
# Queue watcher — submit pending job specs to tsp one at a time.
#
# Start once on WSL startup (add to ~/.bashrc or run manually):
#   bash scripts/queue_watcher.sh &
#
# To check the tsp queue:   tsp
# To cancel a queued job:   tsp -r <job-id>
# To clear finished jobs:   tsp -C

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PENDING="$REPO_DIR/queue/pending"
QUEUED="$REPO_DIR/queue/queued"
mkdir -p "$PENDING" "$QUEUED"

echo "[watcher] started, polling $PENDING every 10s"
echo "[watcher] tsp queue: $(tsp -l 2>/dev/null | wc -l) jobs"

while true; do
    for spec in "$PENDING"/*.yaml "$PENDING"/*.yml; do
        # glob expands to literal string when no files match
        [ -f "$spec" ] || continue

        SPEC_NAME="$(basename "$spec")"

        # Atomic move to queued/ prevents the watcher from double-submitting
        # a spec that tsp hasn't started yet (tsp queues asynchronously).
        DEST="$QUEUED/$SPEC_NAME"
        mv "$spec" "$DEST"

        echo "[watcher] submitting: $SPEC_NAME"
        tsp bash "$REPO_DIR/scripts/run_job.sh" "$DEST"
    done

    sleep 10
done
