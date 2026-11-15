#!/usr/bin/env bash
set -euo pipefail
D='/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment';O="$D/lean_encoder_empty_controls_v1";mkdir -p "$O"
for policy in recent spread; do
 if [ "$policy" = recent ]; then W="$D/lean_token_window_candidate_v1";ref="$W/window_gate_attempt03";na="$W/noop_gate_attempt05";nb="$W/noop_gate_attempt06";else W="$D/lean_temporal_spread_window_v1";ref="$W/window_gate_attempt01";na="$W/noop_gate_attempt01";nb="$W/noop_gate_attempt02";fi
 for n in 01 02; do
  .venv/bin/python scripts/capture_lean_encoder_and_read.py --policy "$policy" --entry gate --capture-reference "$ref" --gate-verification "$O/capture_gate_verification_attempt01" --variant window --mode gate --sequence seq3 \
   --checkpoint "$D/author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt" --merged "$D/lean_auto_multiscale_v1/cohort_combined_attempt01" \
   --reference "$D/lean_cpu_rope_repeat_fix_v1/part_a_attempt01/seq3" --indexed-reference "$D/lean_spatial_rope_diagnostic_v1/indexed_gate_attempt01" \
   --cpu-verification "$D/lean_cpu_rope_repeat_fix_v1/engineering_verification.json" --observed-reference "$D/lean_spatial_rope_diagnostic_v1/seq3_probe_attempt01" \
   --indexed-gate-a "$D/lean_spatial_rope_diagnostic_v1/indexed_gate_attempt01" --indexed-gate-b "$D/lean_spatial_rope_diagnostic_v1/indexed_gate_attempt02" \
   --noop-gate-a "$na" --noop-gate-b "$nb" --output "$O/${policy}_capture_gate_attempt$n" > "$O/${policy}_capture_gate_attempt$n.log" 2>&1
 done
done
P="$D/lean_newest_memory_component_v1";E="$D/lean_recent_spread_exchange_v1";S="$D/lean_temporal_slot_rank_v1"
.venv/bin/python scripts/verify_lean_combined_capture_gates.py --recent-first "$O/recent_capture_gate_attempt01" --recent-second "$O/recent_capture_gate_attempt02" --recent-encoder-reference "$P/recent_capture_gate_attempt01" --recent-read-reference "$E/capture_gate_attempt01" --spread-first "$O/spread_capture_gate_attempt01" --spread-second "$O/spread_capture_gate_attempt02" --spread-encoder-reference "$P/spread_capture_gate_attempt01" --spread-read-reference "$S/capture_gate_attempt01" --model-verification "$D/lean_token_window_candidate_v1/gate_verification_attempt01" --checkpoint "$D/author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt" --output "$O/capture_gate_verification_attempt01" > "$O/capture_gate_verification_attempt01.log" 2>&1
