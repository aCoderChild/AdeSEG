#!/bin/zsh
# Serial, offline SAM2Long comparator; start only after other MPS jobs finish.
set -eu
if [[ $# -ne 1 ]]; then
  print -u2 'Usage: run_sam2long_automatic_priority_full.sh <Drive outputs/experiment>'
  exit 2
fi
BASE=$1
ROOT=${0:A:h:h}
cd "$ROOT"
for SEQ in seq20 seq23 seq11; do
  PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 .venv/bin/python scripts/run_sam2long_automatic_priority.py \
    --sequence "$SEQ" --device mps \
    --checkpoint "$BASE/author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt" \
    --preflight "$BASE/lean_channel_attention_training_pilot_v1/split_preflight_attempt03" \
    --cropped-reference "$BASE/lean_token_window_candidate_v1/seq23_attempt01" \
    --cropped-backup "$BASE/lean_seq23_no_inset_v1/full_attempt02" \
    --gate "$BASE/sam2long_automatic_priority_v1/${SEQ}_gate_verified_attempt03/gate.json" \
    --output "$BASE/sam2long_automatic_priority_v1/${SEQ}_full_mps_attempt01" \
    > "$BASE/sam2long_automatic_priority_v1/${SEQ}_full_mps_attempt01.log" 2>&1
  print "$SEQ full inference complete"
done
# All three full videos must be frozen before any scoring opens labels.
for SEQ in seq20 seq23 seq11; do
  .venv/bin/python scripts/score_sam2long_automatic_priority.py --sequence "$SEQ" \
    --lean "$BASE/contrast_refinement_v1/lean_mps_baseline_${SEQ}_full_attempt01" \
    --sam2long "$BASE/sam2long_automatic_priority_v1/${SEQ}_full_mps_attempt01" \
    --lean-gate "$BASE/contrast_refinement_v1/lean_mps_baseline_${SEQ}_gate_attempt02/gate.json" \
    --long-gate "$BASE/sam2long_automatic_priority_v1/${SEQ}_gate_verified_attempt03/gate.json" \
    --preflight "$BASE/lean_channel_attention_training_pilot_v1/split_preflight_attempt03" \
    --cropped-reference "$BASE/lean_token_window_candidate_v1/seq23_attempt01" \
    --cropped-backup "$BASE/lean_seq23_no_inset_v1/full_attempt02" \
    --output "$BASE/sam2long_automatic_priority_v1/${SEQ}_score_attempt01" \
    > "$BASE/sam2long_automatic_priority_v1/${SEQ}_score_attempt01.log" 2>&1
  print "$SEQ independent scoring complete"
done
