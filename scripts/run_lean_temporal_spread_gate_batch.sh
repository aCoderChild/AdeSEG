#!/usr/bin/env bash
set -euo pipefail
D='/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment'
O="$D/lean_temporal_spread_window_v1"
mkdir -p "$O"
for variant in noop window; do
 for n in 01 02; do
  extra=()
  if [ "$variant" = window ]; then extra=(--noop-gate-a "$O/noop_gate_attempt01" --noop-gate-b "$O/noop_gate_attempt02"); fi
  .venv/bin/python scripts/run_lean_temporal_spread_window.py --variant "$variant" --mode gate --sequence seq3 \
   --checkpoint "$D/author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt" \
   --merged "$D/lean_auto_multiscale_v1/cohort_combined_attempt01" \
   --reference "$D/lean_cpu_rope_repeat_fix_v1/part_a_attempt01/seq3" \
   --indexed-reference "$D/lean_spatial_rope_diagnostic_v1/indexed_gate_attempt01" \
   --cpu-verification "$D/lean_cpu_rope_repeat_fix_v1/engineering_verification.json" \
   --observed-reference "$D/lean_spatial_rope_diagnostic_v1/seq3_probe_attempt01" \
   --indexed-gate-a "$D/lean_spatial_rope_diagnostic_v1/indexed_gate_attempt01" \
   --indexed-gate-b "$D/lean_spatial_rope_diagnostic_v1/indexed_gate_attempt02" \
   "${extra[@]}" --output "$O/${variant}_gate_attempt$n" > "$O/${variant}_gate_attempt$n.log" 2>&1
 done
done
.venv/bin/python scripts/verify_lean_temporal_spread_gates.py --noop-a "$O/noop_gate_attempt01" --noop-b "$O/noop_gate_attempt02" --window-a "$O/window_gate_attempt01" --window-b "$O/window_gate_attempt02" --reference "$D/lean_spatial_rope_diagnostic_v1/indexed_gate_attempt01" --output "$O/gate_verification_attempt01" > "$O/gate_verification_attempt01.log" 2>&1
