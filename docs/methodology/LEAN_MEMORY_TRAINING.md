# Lean-SAM2 early-FPN memory enrichment: reproducible training and Dice ablation

This is an **experimental, opt-in implementation on main**. It preserves the
native Lean-SAM2 memory feature geometry and uses native TAMP/TCIM/TARR.
The architecture is grounded in the complete
`backup/experiment-full-20261118` reference, but does **not** merge that
backup or replace the legacy RMFA mask-logit refiner.

## Executable stages

| Stage | Entry point | Status |
|---|---|---|
| Native / zero-residual / trained inference | `scripts/run_lean_memory_fusion_ablation.py` | Source implementation; requires real Lean checkout, checkpoint and videos |
| TRAIN-only native/oracle memory capture | `scripts/capture_lean_memory_train.py` | Source implementation; requires same resources and sealed groups |
| Offline **oracle-memory distillation** | `training/lean_memory_fusion.py` | CPU/MPS/CUDA code and synthetic tests |
| Independent paired Dice/IoU scoring | `scripts/score_lean_memory_ablations.py` | Operates only on frozen outputs |
| Live end-to-end gradient-through-future-video training | Not implemented | Research extension; do NOT claim it is trained this way |
| Real GPU/MPS propagation and benchmark results | Not executed here | Must be measured separately |

## Why this training approach

A fully differentiable video rollout through Lean-SAM2's temporal bank cannot
be obtained by simply backpropagating through `propagate_in_video`, which is
inference-mode decorated. Instead, this first experiment creates a **TRAIN-only
teacher memory** using the frozen Lean memory encoder with the corresponding
TRAIN ground-truth mask. The frozen predictor's ordinary (predicted-mask)
memory is the student input. Only the new fusion parameters are optimized.

Let `M_native` be native memory, `M_GT` the mask-conditioned oracle memory
generated from TRAIN ground truth, and `E_t` the early FPN maps.

```text
M_enriched = M_native + bounded_residual(E_t, M_native, pred_mask_logit, presence)
Loss = SmoothL1(M_enriched, M_GT) + 0.1 * SmoothL1(M_enriched, M_native)
```

This is **same-frame memory distillation**, not future-frame Dice training,
not ground-truth inference, and not evidence of a real video improvement.
The GT-conditioned teacher is NEVER placed into the predictor's temporal bank.
Video Dice is evaluated only after the trained adapter is frozen.

## Data and source-group preflight

Use the repository's standard video JSONL; each row provides `split`,
`video_id`, contiguous `frame_index`, JPEG `image`, and indexed `mask`.
Provide precomputed frozen **automatic** initialization-mask prompts for TRAIN
and a separate set for DEV/TEST. The prompt schema and SHA256 checks are in
`docs/methodology/LEAN_MEMORY_FUSION_ABLATIONS.md`.

Provide a group map JSON like
`{"seq2":"patient-A","seq3":"patient-B"}` covering every TRAIN video, and
a frozen group partition JSON like
`{"train":["patient-A","patient-B"],"dev":["patient-C"],"test":["patient-D"]}`.
The files prove declared group non-overlap, not genuine independent patient
identity; audit source provenance separately.

Do not use reviewed PolypGen DEV/TEST videos or labels to generate TRAIN
captures. Do not reuse the historical 0.742071 reference score as B2.

## Stage 1: capture actual frozen native and GT-conditioned memory

```bash
MODEL=/absolute/path/sam2.1_hiera_base_plus.pt
MODEL_SHA=$(shasum -a 256 "$MODEL" | cut -d ' ' -f1)

python -m scripts.capture_lean_memory_train \
  --manifest /absolute/path/train_video_manifest.jsonl \
  --prompt-manifest /absolute/path/train_frozen_auto_masks.jsonl \
  --split-file /absolute/path/group_partitions.json \
  --group-map /absolute/path/train_video_groups.json \
  --checkpoint "$MODEL" --backbone-sha256 "$MODEL_SHA" \
  --device mps \
  --output /absolute/path/train_memory_capture
```

This writes `train_memory_captures.json` and content-authenticated `.pt`
captures. The saved per-frame student inputs are original memory (64
channels), aligned early FPN (32/64 channels), pooled native prediction
probability encoded as logits, and native object-presence score.
The saved `teacher_memory` is produced with TRAIN GT mask conditioning,
not with predicted masks.

Keep captured tensors and GT outside public GitHub. Each capture includes
a source-group and a SHA256 of itself and its producer source.

## Stage 2: train ONLY the new residual adapter

```bash
python -m training.lean_memory_fusion \
  --manifest /absolute/path/train_memory_capture/train_memory_captures.json \
  --split-file /absolute/path/group_partitions.json \
  --backbone-sha256 "$MODEL_SHA" \
  --epochs 5 --lr 0.0001 --preservation-weight 0.1 \
  --seed 20261120 --device mps \
  --output /absolute/path/checkpoints/lean_memory_fusion.pt

FUSION=/absolute/path/checkpoints/lean_memory_fusion.pt
FUSION_SHA=$(shasum -a 256 "$FUSION" | cut -d ' ' -f1)
TRAIN_SHA=$(shasum -a 256 /absolute/path/train_memory_capture/train_memory_captures.json | cut -d ' ' -f1)
```

The checkpoint records exact implementation, training source, capture
producer, backbone, source groups, split manifest and train manifest hashes,
seed, optimizer settings, number of steps and the trained parameter values.
The inference entry point refuses a fake checkpoint lacking a matching
training protocol and producer fingerprints.

## Stage 3: infer B0/B1/B2 on SAME frozen DEV prompts

See `docs/methodology/LEAN_MEMORY_FUSION_ABLATIONS.md` for the complete
B0 and B1 commands. Run the learned B2 arm with the same
`--manifest`, `--prompt-manifest`, `--checkpoint`, `--checkpoint-sha256`
and `--device` arguments, plus:

```bash
python -m scripts.run_lean_memory_fusion_ablation \
  --arm fusion --manifest /absolute/path/dev_video_manifest.jsonl \
  --prompt-manifest /absolute/path/dev_frozen_auto_masks.jsonl \
  --checkpoint "$MODEL" --checkpoint-sha256 "$MODEL_SHA" \
  --fusion-checkpoint "$FUSION" --fusion-sha256 "$FUSION_SHA" \
  --train-manifest-sha256 "$TRAIN_SHA" \
  --device mps --split dev \
  --output /absolute/path/ablation/fusion
```

Each run emits raw binary prediction PNGs and a v2 receipt with source
and run-time hashes (code, model, bypass adapter, Lean configs, torch,
Python, device, precision), logit checksums, GT-file checksums for
**provenance only**, and observable resident-memory counters.

The immediate fusion output retains exactly the native memory
tensor geometry. Resident TAMP/TCIM counts are reported per arm. This is
not proof that peak VRAM, memory-attention tensor bytes or runtime are
the same: the fusion costs temporary computation.

## Stage 4: paired Dice with and without the module

```bash
python -m scripts.score_lean_memory_ablations \
  --manifest /absolute/path/dev_video_manifest.jsonl \
  --run native=/absolute/path/ablation/native \
  --run identity=/absolute/path/ablation/identity \
  --run fusion=/absolute/path/ablation/fusion \
  --baseline native --output /absolute/path/ablation/report
```

`report.md` reports equal-video macro GT-present Dice, frame-weighted
GT-present Dice, per-video deltas, and empty-frame false alarms.
The scorer requires exact protocol, source frame, software/hardware, source
image and annotation checksum matches. It checks native/identity prediction
**and logit** checksums and resident-memory parity; changed resident budgets
in other arms are separately flagged. Bootstrapping over sequences is
descriptive unless video/patient independence has been established.

A native/identity discrepancy blocks further claims. Optimize architecture
on group-disjoint DEV, not previously examined TEST. Only publish
held-out adenoid evidence after genuine adenoid evaluation.

## Required validation

```bash
python -m compileall -q modeling/lean_memory_feature_fusion.py \
    adeseg/inference/lean_memory_feature_fusion.py \
    scripts/capture_lean_memory_train.py training/lean_memory_fusion.py \
    scripts/run_lean_memory_fusion_ablation.py \
    scripts/score_lean_memory_ablations.py
python -m pytest -q tests/test_lean_memory_feature_fusion.py \
    tests/test_lean_memory_training.py tests/test_lean_memory_ablation.py
python -m pytest -q tests
```

Run the real DEV parity with the pinned Lean-SAM2 submodule before
trusting Dice. GitHub Actions has historically failed without starting job
steps; this is not a successful test result. The available environment has
no real Lean-SAM2 checkpoint or PolypGen video dataset, so no new Dice
score has been generated or verified.
