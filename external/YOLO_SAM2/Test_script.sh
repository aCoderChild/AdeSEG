set -euo pipefail

export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${ROOT}:${ROOT}/external/YOLO_SAM2${PYTHONPATH:+:${PYTHONPATH}}"

for i in {1..23}; do 
    "${PYTHON_BIN}" "${ROOT}/external/YOLO_SAM2/Test.py" --seq_num ${i}
done

# bash external/YOLO_SAM2/Test_script.sh