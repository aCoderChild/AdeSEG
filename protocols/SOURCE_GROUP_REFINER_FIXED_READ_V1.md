# Source-group fixed-read refiner study

This is an exploratory preparation using previously inspected videos. It does
not establish patient independence, detector independence, or a working live
video refiner. The existing project Lean token window and automatic prompt are
the source of fixed reads; the study does not add a pruning rule, redetection,
ground-truth prompt, or inference-time gate.

The input is exactly the frozen source-group plan: seven training sequences
(seq2/3/4/5/12/13/14; 164 sampled reads) and two development validation
sequences (seq9/15; 50 reads). All nine full reference, capture and observer
audit artifacts must be complete. Every sequence is mapped explicitly to its
three artifact directories, so validated capture attempts can be combined
without moving files. Partial runs, aliases, changed inputs, incompatible
runtime identity and unsupported historical feature reuse are rejected.
The archived metadata contains previous label-derived scores; the bridge
projects label hashes and frame identities and does not open mask images.
Its `--capture-map` JSON must contain exactly the nine sequence keys; each
value must have `reference`, `capture` and `audit` directory paths. For
example, a seq2 entry can point to a different attempt directory from seq5.
Map creation must not use validation outcomes. The bridge binds the map's
bytes and checks all 27 resolved directories for reuse or symlink aliases.

The fixed-read arms all reuse `ForegroundConditionedRefiner`, with 48,005
parameters, identical initial weights, training samples, deterministic sample
orders, optimizer, loss and seed:

- `region`: early/deep features, predicted foreground/background channel
  conditioning and contrast map.
- `global`: the same features and head with the original global channel gate.
- `no_contrast`: the region gate with the contrast-map input zeroed.
- `mask_only`: the region gate with both early and deep inputs zeroed.
- The frozen native output is evaluated alongside these four fitted arms.

The pilot plan fixes one seed (20261115), ten epochs, Adam (learning rate
0.0003, weight decay 0.0001), and BCE-with-logits plus soft Dice. This is a
single exploratory seed, not a final statistical design. The hard-coded
training-only numerical parity reads are seq2 frame 4, seq4 frame 18 and
seq13 frame 8; seq4 was used in earlier investigations. Before training,
fixed-read CPU/MPS output, loss, gradient and resized-mask checks must pass.

`prepare_source_group_refiner_manifest_v1.py` rechecks the source-group
preflight, source freeze, short gate, full trajectory receipts, current frame
hashes, automatic prompts and saved feature/snapshot tensors. All output paths
and hashes are recorded. It blocks CPython opens of manifest mask images.
The derived feature pipeline identity describes the actual MPS capture and
separately preserves the historical CPU feature-pipeline reference.

`train_source_group_refiner_v1.py --mode plan` records model/source hashes and
the shared order before training. Parity and each arm use a guard that permits
only the seven training sequences' label masks. Each arm writes its checkpoint
and freezes all 50 validation predictions. No validation mask is opened by
the plan, parity, training or prediction-freezing process. The CPython audit
hook does not intercept arbitrary native-code file opens; source review is
still needed.

`verify_source_group_refiner_v1.py` requires every arm to be complete, checks
the source/plan/checkpoint hashes and initial conditions, and independently
replays every frozen native and refined validation mask under a guard that
blocks all manifest masks. Only the separate
`score_source_group_refiner_v1.py` process may then open validation labels,
after rechecking the replay receipt and all frozen output hashes. It reports
per-video and pooled present Dice, precision/recall, labeled-empty false alarms
and area; no empty frames means that false-alarm behavior is unmeasured.

The existing bounded-raw-candidate study cannot consume these reads directly:
they contain native post-suppression logits but not the three raw decoder
candidate outputs. That branch requires a separate label-free decoder replay
and audit. A continuous residual can in principle change a suppressed mask;
leaving the native presence score or object pointer unchanged does not prove
that final suppression or memory behavior is preserved. No live routing or
full-video rollout follows automatically from the fixed-read study.

The bridge and model training remain unrun while captures are incomplete.
Repeated per-read hashing and loading may total roughly 525 GiB of local reads
for four arms over ten epochs. Do not run them concurrently with the capture
batch; verify local free space and Drive synchronization/quota before launch.
