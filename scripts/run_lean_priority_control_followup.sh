#!/usr/bin/env bash
# Run remaining frozen matched-size controls; all outputs go to Drive.
set -euo pipefail
task_drive='/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment'
checkpoint="$task_drive/author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt"
.venv/bin/python - "$task_drive/lean_raw_two_window_control_v1/seq8_attempt01/raw_two_control_receipt.json" <<'PY'
import json,sys
if not json.load(open(sys.argv[1]))['passed']:raise ValueError('First raw-two run is incomplete')
PY
run_control() {
  local kind="$1" sequence="$2" part="$3" root dispatcher
  if [ "$kind" = raw ]; then
    root="$task_drive/lean_raw_two_window_control_v1"
    dispatcher=scripts/run_lean_raw_two_control.py
  else
    root="$task_drive/lean_fused_token_window_control_v1"
    dispatcher=scripts/run_lean_fused_window_control.py
  fi
  .venv/bin/python "$dispatcher" --entry token --variant window --mode candidate --sequence "$sequence" \
    --checkpoint "$checkpoint" \
    --merged "$task_drive/lean_auto_multiscale_v1/cohort_combined_attempt01" \
    --reference "$task_drive/lean_cpu_rope_repeat_fix_v1/part_${part}_attempt01/$sequence" \
    --indexed-reference "$task_drive/lean_spatial_rope_diagnostic_v1/part_${part}_attempt01/$sequence" \
    --cpu-verification "$task_drive/lean_cpu_rope_repeat_fix_v1/engineering_verification.json" \
    --observed-reference "$task_drive/lean_spatial_rope_diagnostic_v1/seq3_probe_attempt01" \
    --indexed-gate-a "$task_drive/lean_spatial_rope_diagnostic_v1/indexed_gate_attempt01" \
    --indexed-gate-b "$task_drive/lean_spatial_rope_diagnostic_v1/indexed_gate_attempt02" \
    --noop-gate-a "$root/noop_gate_attempt01" --noop-gate-b "$root/noop_gate_attempt02" \
    --window-gate-a "$root/window_gate_attempt01" --window-gate-b "$root/window_gate_attempt02" \
    --output "$root/${sequence}_attempt01" > "$root/${sequence}_attempt01.log" 2>&1
}
run_control fused seq8 a
run_control raw seq20 b
run_control fused seq20 b
