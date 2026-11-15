#!/usr/bin/env bash
# Evaluate new policy only; reuse all original and recent-window references.
set -euo pipefail
D='/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment'
O="$D/lean_temporal_spread_window_v1"
.venv/bin/python - "$O" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
v=json.loads((root/'gate_verification_attempt01/verification.json').read_text())
if not v['passed'] or v['verifier_sha256']!=digest('scripts/verify_lean_temporal_spread_gates.py') or v['identity']['runner_sha256']!=digest('scripts/run_lean_temporal_spread_window.py') or v['identity']['protocol_sha256']!=digest('protocols/lean_temporal_spread_token_window_v1.json'):raise ValueError('Independent gate or implementation changed')
for variant in ('noop','window'):
 for n in ('01','02'):
  folder=root/f'{variant}_gate_attempt{n}';o=json.loads((folder/'token_window_observation.json').read_text())
  if not o['passed'] or o['identity']!=v['identity'] or digest(folder/'records.json')!=o['records_sha256'] or digest(folder/'token_window_events.json')!=o['events_sha256']:raise ValueError('Verified gate artifact changed')
PY
run_video() {
 local sequence="$1" part="$2"
 .venv/bin/python scripts/run_lean_temporal_spread_window.py --variant window --mode candidate --sequence "$sequence" \
  --checkpoint "$D/author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt" --merged "$D/lean_auto_multiscale_v1/cohort_combined_attempt01" \
  --reference "$D/lean_cpu_rope_repeat_fix_v1/part_${part}_attempt01/$sequence" --indexed-reference "$D/lean_spatial_rope_diagnostic_v1/part_${part}_attempt01/$sequence" \
  --cpu-verification "$D/lean_cpu_rope_repeat_fix_v1/engineering_verification.json" --observed-reference "$D/lean_spatial_rope_diagnostic_v1/seq3_probe_attempt01" \
  --indexed-gate-a "$D/lean_spatial_rope_diagnostic_v1/indexed_gate_attempt01" --indexed-gate-b "$D/lean_spatial_rope_diagnostic_v1/indexed_gate_attempt02" \
  --noop-gate-a "$O/noop_gate_attempt01" --noop-gate-b "$O/noop_gate_attempt02" --window-gate-a "$O/window_gate_attempt01" --window-gate-b "$O/window_gate_attempt02" \
  --output "$O/${sequence}_attempt01" > "$O/${sequence}_attempt01.log" 2>&1
}
run_video seq8 a
run_video seq20 b
run_video seq11 a
