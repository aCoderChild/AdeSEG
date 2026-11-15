#!/usr/bin/env bash
set -euo pipefail
D='/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment';O="$D/lean_encoder_empty_controls_v1";SAM="$D/author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt"
.venv/bin/python scripts/verify_lean_empty_encoder_captures.py --recent-encoder "$O/recent_seq23_capture_attempt01" --spread-encoder "$O/spread_seq23_capture_attempt01" --recent-read "$O/recent_seq23_capture_attempt01" --spread-read "$O/spread_seq23_capture_attempt01" --recent-reference "$D/lean_token_window_candidate_v1/seq23_attempt01" --spread-reference "$D/lean_temporal_spread_window_v1/seq23_attempt01" --capture-verification "$O/capture_gate_verification_attempt01" --model-verification "$D/lean_token_window_candidate_v1/gate_verification_attempt01" --checkpoint "$SAM" --output "$O/seq23_capture_verification_attempt01" > "$O/seq23_capture_verification_attempt01.log" 2>&1
.venv/bin/python - "$O" <<'PY'
import hashlib,json,sys
from pathlib import Path
O=Path(sys.argv[1]);h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();v=json.loads((O/'seq23_capture_verification_attempt01/verification.json').read_text())
if not v['passed'] or v['verifier_sha256']!=h('scripts/verify_lean_empty_encoder_captures.py'):raise ValueError('Independent seq23 capture verification changed')
for name,filename in (('recent','recent_capture_receipt.json'),('spread','spread_capture_receipt.json')):
 folder=O/f'{name}_seq23_capture_attempt01'
 if v['policies'][name]['encoder_receipt_sha256']!=h(folder/'encoder_capture_receipt.json') or v['policies'][name]['read_receipt_sha256']!=h(folder/filename):raise ValueError('Captured state changed')
PY
for n in 01 02; do
 .venv/bin/python scripts/replay_lean_empty_encoder_components.py --mode candidate --recent-encoder "$O/recent_seq23_capture_attempt01" --spread-encoder "$O/spread_seq23_capture_attempt01" --recent-read "$O/recent_seq23_capture_attempt01" --spread-read "$O/spread_seq23_capture_attempt01" --capture-verification "$O/capture_gate_verification_attempt01" --model-verification "$D/lean_token_window_candidate_v1/gate_verification_attempt01" --replay-verification "$O/replay_gate_verification_attempt01" --checkpoint "$SAM" --output "$O/seq23_replay_attempt$n" > "$O/seq23_replay_attempt$n.log" 2>&1
done
.venv/bin/python scripts/verify_lean_empty_component_gates.py --first "$O/seq23_replay_attempt01" --second "$O/seq23_replay_attempt02" --recent-encoder "$O/recent_seq23_capture_attempt01" --spread-encoder "$O/spread_seq23_capture_attempt01" --recent-read "$O/recent_seq23_capture_attempt01" --spread-read "$O/spread_seq23_capture_attempt01" --checkpoint "$SAM" --output "$O/seq23_replay_verification_attempt01" > "$O/seq23_replay_verification_attempt01.log" 2>&1
.venv/bin/python scripts/audit_lean_empty_encoder_components.py --replay "$O/seq23_replay_attempt01" --verification "$O/seq23_replay_verification_attempt01" --crop-reference "$D/lean_seq23_no_inset_v1/full_attempt02" --output "$O/seq23_audit_attempt01" > "$O/seq23_audit_attempt01.log" 2>&1
.venv/bin/python scripts/verify_lean_empty_component_audit.py --audit "$O/seq23_audit_attempt01" --replay "$O/seq23_replay_attempt01" --verification "$O/seq23_replay_verification_attempt01" --output "$O/seq23_audit_verification_attempt01" > "$O/seq23_audit_verification_attempt01.log" 2>&1
