# Verified-refiner runtime: development status

The live code is a research prototype, not yet a completed or independently
validated video study. The selection rules below were written after previous
development outcomes were inspected; they are not preregistration and do not
create an independent test set.

## Implemented safeguards

- `assess_refiner_candidate_promotion.py` independently reloads and reproduces
  the original 17/10 candidate and mask-only frozen MPS predictions before
  reading validation masks. It checks matched training metadata and equal
  model capacity, computes equal-per-video present Dice, and disallows
  degradation of false-alarm rate or mean false-positive area on any empty
  validation video. Both previously tested candidates fail this screen.
- `assess_region_conditioned_candidate.py` invokes the existing independent
  four-arm 19/10 auditor, then additionally compares region conditioning with
  baseline, global, shuffled and mask-only controls. The tested region
  checkpoint fails this exploratory screen too.
- `verified_mps_window_refiner` owns model creation, recomputes the appropriate
  screen from actual frozen outputs, verifies capture/detector/checkpoint
  provenance, compares CPU/MPS parameters and buffers, and replays the archived
  seq4 baseline twice on the *same predictor* before issuing an expiring
  predictor/refiner-bound lease. The lease checks active hooks and the 16,384
  token policy on each routed frame. This guards against accidental runtime
  mismatch; Python in-process code is not a security boundary.
- The priority runner supports seq8/11/20 and the fixed no-inset seq23 crop,
  plus seq4. It reuses the exact archived automatic mask helper from the
  SAMURAI comparator, records ROI and mask provenance, and requires a matching
  same-device arm A before B/C/D. It freezes masks without opening target
  sequence masks; the development screen separately reads validation labels.
  Arm B must preserve the full A compact state, memory events and positional
  and selection traces. The screen now supports both the original 17/10 and
  region-conditioned 19/10 schemas. No new inference threshold or redetection
  is added.

## Engineering checks performed

- A real token-window MPS baseline, identity A/B/C/D and forced-logit B/C/D
  were checked on seq4 frames 9–13; all eight freshly built predictors matched
  the actual 621 CPU reference tensor fingerprints, and identity/forced
  routes behaved as intended. `token_window_routing_mps_attempt03/` on Drive
  retains source snapshots and receipts. No learned candidate or label was used.
- A separate live-state check verified model parameters *and buffers* before
  and after a seq4 MPS replay; see `live_state_audit_attempt01/` on Drive.
- Direct archived automatic-mask injection was compared with historical
  box+points calls followed by the same archived mask: token-window MPS output
  and stored-state traces matched on seq4 frames 9–13 and cropped seq23 frames
  0–4 (`frozen_prompt_path_mps_attempt01/`,
  `frozen_prompt_seq23_mps_attempt01/`). These are short engineering checks.
- The new region-conditioned selection gate independently replayed the four
  models and rejected the current candidate. Its decision is in
  `region_window_runtime_gate_attempt02/` on Drive. No video candidate was run.

## Still required

The positive path of the owned runtime and factorial runner has **not** been
validated with a genuinely eligible trained refiner. No current checkpoint is
eligible, and no full candidate video has been run. An independently reviewed
positive end-to-end engineering check and source/provenance audit are required
when a new checkpoint actually passes the development criteria; do not bypass
the gate with test flags or synthetic metadata.

The seq23 archived prompt image was not bound in its original completion
receipt. The shared comparator helper checks a separate archived copy and
records both sources; this supports reproducibility but is not proof of the
historical generation path. The seq23 propagated frame-0 output is not the
initial prompt and differs from it.

Previously inspected PolypGen sequences remain development data. Patient and
center separation, external validation, matched comparator tracks, missing
ablations and training-seed variation remain unresolved. User-visible false
alarms and memory updates must be evaluated over entire videos before any
claim of a method improvement.
