#!/usr/bin/env bash
# Run the Phase 1a connection-cost experiment under the harder world parameters.
#
# World changes vs m8 baseline:
#   metabolism:    0.005 -> 0.010  (passive energy drain doubled)
#   move_cost:     0.001 -> 0.003  (movement cost tripled)
#   hotspot_drift: 0.3   -> 0.6    (food moves faster)
#
# Results saved to runs/m8_hard/{baseline,modular}/
#
# Run from WSL2 inside the project root:
#   bash scripts/run_m8_hard.sh

set -euo pipefail
cd "$(dirname "$0")/.."

python3 -u scripts/run_experiment.py \
  --lambda-conn 0.0007 \
  --output-dir  runs/m8_hard \
  --n-replicates  10 \
  --n-generations 500 \
  --verbose \
  "$@"
