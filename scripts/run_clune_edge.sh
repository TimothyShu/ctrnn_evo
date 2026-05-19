#!/usr/bin/env bash
# Clune 2013 replication using pure edge-count cost (lambda_edge).
#
# Clune et al. 2013 penalised the *number* of connections regardless of length.
# Previous runs used lambda_dist (distance-weighted wiring), which is different.
# This script uses lambda_edge to match the original paper's cost structure.
#
# Results saved to runs/m8_edge/{baseline,modular}/
#
# Run from WSL2 inside the project root:
#   bash scripts/run_clune_edge.sh

set -euo pipefail
cd "$(dirname "$0")/.."

python3 -u scripts/run_experiment.py \
  --lambda-edge  0.0007 \
  --lambda-dist  0.0 \
  --output-dir   runs/m8_edge \
  --n-replicates  10 \
  --n-generations 500 \
  --verbose \
  "$@"
