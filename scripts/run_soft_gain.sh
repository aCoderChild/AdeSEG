#!/bin/zsh
# Soft score-scaled Kalman gain: dev calibration -> dev selection -> C6 test once.
#
# Pre-registered selection rule (same shape the EMA ablation used):
#   The chosen (alpha, beta) must beat the plain Kalman run on BOTH
#     dev lost-polyp-frame fraction, and
#     dev high-motion polyp-frame Dice (motion > 0.117, dev tercile cut),
#   otherwise do not run C6.
#
# Expects the untrained Kalman checkpoint under ~/Library/Caches/adseg_work/untrained.pt.

set -e
ROOT="/Users/maianhpham/Documents/AdeSEG"
OUT="/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/Kalman"
W="${HOME}/Library/Caches/adseg_work"
PYTHON="${ROOT}/.venv/bin/python"
CKPT="${W}/untrained.pt"
CALIBRATION="${OUT}/26_soft_gain_calibration.json"

cd "${ROOT}"

"${PYTHON}" scripts/soft_gain_calibrate.py --out "${CALIBRATION}"
ALL_ALPHA=$(${PYTHON} -c "import json; print(json.load(open('${CALIBRATION}'))['all_frames']['alpha'])")
ALL_BETA=$(${PYTHON} -c "import json; print(json.load(open('${CALIBRATION}'))['all_frames']['beta'])")
PRESENT_ALPHA=$(${PYTHON} -c "import json; print(json.load(open('${CALIBRATION}'))['present_only']['alpha'])")
PRESENT_BETA=$(${PYTHON} -c "import json; print(json.load(open('${CALIBRATION}'))['present_only']['beta'])")

echo "all-frames fit  : alpha=${ALL_ALPHA}  beta=${ALL_BETA}"
echo "present-only fit: alpha=${PRESENT_ALPHA}  beta=${PRESENT_BETA}"

run () {
  local tag="$1"; local alpha="$2"; local beta="$3"; local split="$4"
  local dir="${OUT}/${tag}" ; [[ "${split}" == "test" ]] && dir="${OUT}/${tag}"
  mkdir -p "${dir}"
  caffeinate -i "${PYTHON}" scripts/infer.py \
    --sam2_cfg configs/sam2.1_hiera_t512.yaml \
    --sam2_checkpoint checkpoints/MedSAM2_latest.pt \
    --manifest data/polypgen_sequence.jsonl --split "${split}" --label_ids 1 \
    --memory_backend kalman --kalman_checkpoint "${CKPT}" \
    --score_scale "${alpha}" "${beta}" \
    --device mps --output_dir "${dir}" 2>&1 | tee "${dir}/run.log"
  "${PYTHON}" evaluation/temporal.py --output_mask_dir "${dir}/masks" \
    --sequences seq1 seq2 seq3 seq4 seq5 seq6 seq7 seq8 seq9 seq10 seq11 seq12 seq13 seq14 seq15 2>&1 | tail -5
}

# Dev calibration pass: try both fits.
run "26_soft_gain_all"     "${ALL_ALPHA}"     "${ALL_BETA}"     dev
run "26_soft_gain_present" "${PRESENT_ALPHA}" "${PRESENT_BETA}" dev

echo "Compare on dev vs 10_kalman_memory via scripts/analyze_reappearance.py / illumination."
echo "Then, if the pre-registered rule is met, re-run with --split test to produce C6 numbers."
