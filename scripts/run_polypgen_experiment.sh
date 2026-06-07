#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${ROOT}/scripts/experiments/run_experiment.py" \
  --config "${ROOT}/experiments/polypgen_medsam2_yolo_sam2.json" \
  "$@"
