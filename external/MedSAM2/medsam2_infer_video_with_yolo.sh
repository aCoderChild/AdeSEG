#!/usr/bin/env bash
set -euo pipefail

export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Always run from repo root so relative paths resolve consistently.
cd "${ROOT}"

if [ -z "${PYTHON_BIN}" ] || [ "${PYTHON_BIN}" = "python3" ]; then
    if [ -x "${ROOT}/.venv/bin/python" ]; then
        PYTHON_BIN="${ROOT}/.venv/bin/python"
    fi
fi

export PYTHONPATH="${ROOT}:${ROOT}/external/MedSAM2${PYTHONPATH:+:${PYTHONPATH}}"

SCRIPT="${ROOT}/external/MedSAM2/medsam2_infer_video_with_yolo.py"

if [ ! -f "${SCRIPT}" ]; then
    echo "Error: target script not found: ${SCRIPT}" >&2
    exit 1
fi

for i in {1..23}; do
    # "${PYTHON_BIN}" "${SCRIPT}" --seq_num "${i}"
    "${PYTHON_BIN}" "${SCRIPT}" --seq_num "${i}" --eval_mask_dir external/MedSAM2/data/polypgen_vid_seq/seq"${i}"/masks # --yolo_first_frame_only
done

#  bash external/MedSAM2/medsam2_infer_video_with_yolo.sh