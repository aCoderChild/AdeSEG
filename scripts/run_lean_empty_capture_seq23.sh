#!/usr/bin/env bash
set -euo pipefail
D='/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment';O="$D/lean_encoder_empty_controls_v1"
for policy in recent spread; do
 extra=()
 if [ "$policy" = recent ]; then W="$D/lean_token_window_candidate_v1";else W="$D/lean_temporal_spread_window_v1";extra=(--spread-crop-verification "$W/seq23_gate_verification_attempt01");fi
 .venv/bin/python scripts/capture_lean_encoder_and_read.py --policy "$policy" --entry crop --capture-reference "$W/seq23_attempt01" --gate-verification "$O/capture_gate_verification_attempt01" --mode candidate \
  --checkpoint "$D/author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt" --detector "$PWD/checkpoints/polypgen_yolov8n.pt" \
  --crop-gate-a "$D/lean_seq23_no_inset_v1/gate_attempt03" --crop-gate-b "$D/lean_seq23_no_inset_v1/gate_attempt04" \
  --baseline "$D/lean_seq23_no_inset_v1/full_attempt02" --verification "$W/gate_verification_attempt01" \
  --window-crop-gate-a "$W/seq23_gate_attempt01" --window-crop-gate-b "$W/seq23_gate_attempt02" "${extra[@]}" \
  --output "$O/${policy}_seq23_capture_attempt01" > "$O/${policy}_seq23_capture_attempt01.log" 2>&1
done
