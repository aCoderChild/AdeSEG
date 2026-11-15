# Asinh channel-value video development study

This is a proposed exploratory video test of the trained asinh head-input arm.
Its source evidence is the previously inspected 50-read seq9/15 development
evaluation; the observed fixed-read advantage is modest and does not establish
improved propagation. Seq10/11/20/23 were selected after their results and
failures were inspected. No independent or blinded efficacy claim is possible.

Use the exact archived automatic one-shot prompt and the established project
Lean 16,384-token window, with 3,072-token insurance reserve. Seq23 uses the
same retrospectively selected `[446,0,1280,720]` viewport and restores output
to the full canvas. Reuse the separately audited matched Lean A trajectories
from the source-bound V2 cross-fit index for these exact frames and prompts.
Do not replace YOLO with ground truth, redetect, change token pruning, add an
inference threshold or introduce a learned presence gate.

The residual refiner reads projected early and memory-conditioned deep features
and soft foreground/background contrast. In the asinh variant only the head's
native-logit input is `asinh(logit)`; the output residual anchor is the
unaltered native logit. The A/B/C/D routing is inherited from the earlier
factorial: A is the matched archived baseline; B changes only returned mask
logits; C changes only the mask passed to SAM2's memory encoder; D does both.
The conditioning prompt is unchanged. C/D do not replace native object score,
pointer, compact decoder mask, or all memory state. Future native predictions
may change through feedback. A refined mask can conflict with a native absent
score, and the native -1024 anchor may still prevent recovery. Telemetry must
record score and native/refined extrema and positive pixels, without using
those values as a gate. Timing includes staging, hashing, memory telemetry and
mask I/O, and is not model-only throughput.

Before any GPU run, a single source-bound certificate revalidates the 214
historical fixed-read captures, checkpoints, training plan, all frozen
validation predictions and the separate CPU metric audit without opening
labels. The certificate is a snapshot: child runs recheck its bytes, the
frozen checkpoint/manifest, source dependencies, environment and live model
and refiner fingerprints; they do not reread inactive historical training
tensors for every video. The short batch releases the GPU lock after its audit;
an explicit later resume reacquires the lock and rechecks the certificate,
source dependencies, smoke artifacts and every completed phase. An incomplete
next-phase folder and log are preserved in a separate in-batch partial-attempt
directory before retry; no prior predictions are deleted. The historical capture check remains
a snapshot at certificate creation, not a claim of continuous monitoring.
Per-video prompt, full image sequence, mask geometry,
finite logits, memory budget, source and state coverage remain checked.

Run only MPS smoke, with no CPU/MPS comparison or repeated-process gate:
one five-frame A/B/C/D factorial on nonzero-prompt seq20 plus one cropped
seq23 A run. Compare each A to the previously audited A prefix, enforce the
unchanged initial prompt and first-step routing, and independently audit the
smoke artifacts before full propagation. There is no claim that these checks
cover every video, random device variation, or long-term propagation.

After the smoke, freeze B/C/D full trajectories on seq10, 11, 20 and 23
(12 fresh video runs); A is reused from the exact matched, independently
audited prior trajectory. B must reproduce A's recorded compact masks,
memory features, pointers, object scores, memory events, positions and
token-selection traces across the whole video; unrecorded predictor state is
not covered by this comparison. C/D must preserve
the conditioning state and native decoder/pointer/presence at the first
post-prompt step; their later states may differ. Verify all 12 predictions and
matched A before opening any target label. Score the same post-initialization
frames, plus pre-prompt and full-frame outcomes, then independently audit
metrics using a separate CPU implementation with a shared structural verifier.
Record present Dice, boundary F, recall, zero overlap, empty alarms/area and
every per-sequence change. No result on these videos is a blinded test.

All outputs and temporary staging belong under the user's Drive experiment
folder. Check the existing 22-GiB free-space reserve before each phase and
stop an active child if that reserve is crossed. Record failed or partial
artifacts without replacing previous research results.
