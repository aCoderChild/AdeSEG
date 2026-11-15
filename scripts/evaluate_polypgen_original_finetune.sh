#!/usr/bin/env bash
# Evaluation uses frozen detector checkpoints and writes only under Drive.
set -euo pipefail
D='/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment'
O="$D/lean_original_detector_adaptation_v1"
DATA="$D/lean_one_shot_detector_pilot_v1/dataset_attempt02"
DV="$D/lean_one_shot_detector_pilot_v1/dataset_verification_attempt02"
PF="$D/lean_channel_attention_training_pilot_v1/split_preflight_attempt03"
CROP="$D/lean_seq23_no_inset_v1/full_attempt02"
SAM="$D/author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt"
ORIG="$PWD/checkpoints/polypgen_yolov8n.pt"
BEST="$O/training_attempt01/train/weights/best.pt"
py="$PWD/.venv/bin/python"
runlog() { local log="$1"; shift; "$py" "$@" > "$O/$log.log" 2>&1; }
runlog training_verification_attempt01 scripts/verify_polypgen_original_detector_finetune.py --training "$O/training_attempt01" --dataset "$DATA" --dataset-verification "$DV" --smoke-verification "$O/smoke_verification_attempt01" --cache-verification "$O/label_cache_verification_attempt01" --pretrained "$ORIG" --output "$O/training_verification_attempt01"
for n in 01 02; do
 runlog candidate_attempt$n scripts/predict_polypgen_original_finetune.py --mode candidate --checkpoint "$BEST" --preflight "$PF" --crop-reference "$CROP" --gate-verification "$O/gate_verification_attempt01" --training-verification "$O/training_verification_attempt01" --output "$O/candidate_attempt$n"
done
runlog candidate_verification_attempt01 scripts/verify_polypgen_original_finetune_predictions.py --first "$O/candidate_attempt01" --second "$O/candidate_attempt02" --checkpoint "$BEST" --preflight "$PF" --crop-reference "$CROP" --training-verification "$O/training_verification_attempt01" --gate-verification "$O/gate_verification_attempt01" --output "$O/candidate_verification_attempt01"
runlog c6_inputs_attempt01 scripts/prepare_polypgen_finetune_c6_inputs.py --dataset "$DATA" --dataset-verification "$DV" --training-verification "$O/training_verification_attempt01" --output "$O/c6_inputs_attempt01"
for n in 01 02; do
 runlog c6_gate_attempt$n scripts/predict_polypgen_finetune_c6.py --mode gate --inputs "$O/c6_inputs_attempt01" --original "$ORIG" --candidate "$BEST" --training-verification "$O/training_verification_attempt01" --original-gate "$O/gate_verification_attempt01" --candidate-verification "$O/candidate_verification_attempt01" --gate-verification "$O/c6_gate_verification_attempt01" --output "$O/c6_gate_attempt$n"
done
runlog c6_gate_verification_attempt01 scripts/verify_polypgen_finetune_c6_gate.py --first "$O/c6_gate_attempt01" --second "$O/c6_gate_attempt02" --inputs "$O/c6_inputs_attempt01" --original "$ORIG" --candidate "$BEST" --output "$O/c6_gate_verification_attempt01"
for n in 01 02; do
 runlog c6_predictions_attempt$n scripts/predict_polypgen_finetune_c6.py --mode candidate --inputs "$O/c6_inputs_attempt01" --original "$ORIG" --candidate "$BEST" --training-verification "$O/training_verification_attempt01" --original-gate "$O/gate_verification_attempt01" --candidate-verification "$O/candidate_verification_attempt01" --gate-verification "$O/c6_gate_verification_attempt01" --output "$O/c6_predictions_attempt$n"
done
runlog c6_prediction_verification_attempt01 scripts/verify_polypgen_finetune_c6_predictions.py --first "$O/c6_predictions_attempt01" --second "$O/c6_predictions_attempt02" --inputs "$O/c6_inputs_attempt01" --gate "$O/c6_gate_verification_attempt01" --gate-reference "$O/c6_gate_attempt01" --original "$ORIG" --candidate "$BEST" --output "$O/c6_prediction_verification_attempt01"
for n in 01 02; do
 runlog masks_attempt$n scripts/predict_polypgen_original_finetune_masks.py --mode candidate --checkpoint "$SAM" --checkpoint-verification "$D/lean_raw_two_window_control_v1/gate_verification_attempt01" --fixed-box-gate "$D/lean_fixed_box_oracle_diagnostic_v1/gate_verification_attempt01" --fixed-box-reference "$D/lean_fixed_box_oracle_diagnostic_v1/gate_attempt01" --predictions "$O/candidate_attempt01" --prediction-verification "$O/candidate_verification_attempt01" --mask-gate "$O/mask_gate_verification_attempt01" --output "$O/masks_attempt$n"
done
runlog masks_verification_attempt01 scripts/verify_polypgen_original_finetune_masks.py --first "$O/masks_attempt01" --second "$O/masks_attempt02" --predictions "$O/candidate_attempt01" --prediction-verification "$O/candidate_verification_attempt01" --mask-gate "$O/mask_gate_verification_attempt01" --checkpoint "$SAM" --output "$O/masks_verification_attempt01"
runlog box_audit_attempt01 scripts/audit_polypgen_original_finetune_boxes.py --predictions "$O/candidate_attempt01" --verification "$O/candidate_verification_attempt01" --preflight "$PF" --crop-reference "$CROP" --output "$O/box_audit_attempt01"
runlog box_verification_attempt01 scripts/verify_polypgen_original_finetune_boxes.py --audit "$O/box_audit_attempt01" --predictions "$O/candidate_attempt01" --output "$O/box_verification_attempt01"
runlog c6_audit_attempt01 scripts/audit_polypgen_finetune_c6.py --predictions "$O/c6_predictions_attempt01" --prediction-verification "$O/c6_prediction_verification_attempt01" --inputs "$O/c6_inputs_attempt01" --dataset "$DATA" --dataset-verification "$DV" --output "$O/c6_audit_attempt01"
runlog c6_audit_verification_attempt01 scripts/verify_polypgen_finetune_c6_audit.py --audit "$O/c6_audit_attempt01" --predictions "$O/c6_predictions_attempt01" --inputs "$O/c6_inputs_attempt01" --dataset "$DATA" --output "$O/c6_audit_verification_attempt01"
for n in 01 02; do
 runlog oracle_attempt$n scripts/predict_polypgen_original_finetune_oracle.py --checkpoint "$SAM" --checkpoint-verification "$D/lean_raw_two_window_control_v1/gate_verification_attempt01" --predictions "$O/candidate_attempt01" --prediction-verification "$O/candidate_verification_attempt01" --masks "$O/masks_attempt01" --mask-verification "$O/masks_verification_attempt01" --preflight "$PF" --crop-reference "$CROP" --output "$O/oracle_attempt$n"
done
runlog oracle_verification_attempt01 scripts/verify_polypgen_original_finetune_oracle.py --first "$O/oracle_attempt01" --second "$O/oracle_attempt02" --predictions "$O/candidate_attempt01" --masks "$O/masks_attempt01" --mask-verification "$O/masks_verification_attempt01" --output "$O/oracle_verification_attempt01"
runlog mask_audit_attempt01 scripts/audit_polypgen_original_finetune_masks.py --predictions "$O/candidate_attempt01" --box-audit "$O/box_audit_attempt01" --box-verification "$O/box_verification_attempt01" --masks "$O/masks_attempt01" --mask-verification "$O/masks_verification_attempt01" --oracle "$O/oracle_attempt01" --oracle-verification "$O/oracle_verification_attempt01" --preflight "$PF" --crop-reference "$CROP" --priority-audit "$D/lean_priority_prompt_quality_v1/audit_attempt01" --priority-verification "$D/lean_priority_prompt_quality_v1/verification_attempt01" --output "$O/mask_audit_attempt01"
runlog mask_audit_verification_attempt01 scripts/verify_polypgen_original_finetune_mask_scores.py --audit "$O/mask_audit_attempt01" --masks "$O/masks_attempt01" --oracle "$O/oracle_attempt01" --preflight "$PF" --crop-reference "$CROP" --priority-audit "$D/lean_priority_prompt_quality_v1/audit_attempt01" --output "$O/mask_audit_verification_attempt01"
