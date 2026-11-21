# Lean-SAM2 early-FPN memory enrichment: experiment preflight

**Scope:** research-stage PolypGen automatic video-polyp segmentation. This is
**not** adenoid-specific clinical validation. This protocol does **not**
modify native SAM2 image/memory encoders, decoder, TAMP, TCIM or TARR.

## Scientific question and integration point

At each non-conditioning frame `t`, the native predictor first segments
using memory read from earlier frames. It then creates a mask-conditioned memory
`M_t` from the unconditioned deepest *image-encoder* feature and the predicted
mask. The adapter aligns the earlier **image-encoder FPN levels** to the native
memory grid and writes `M_t + bounded_learned_residual`. Native positional
encoding, channel count, and spatial token count remain unchanged.

The enriched memory is read on subsequent frames. Native Lean-SAM2 TAMP
scores/prunes historical tokens *during later memory reads*, not during the
current-frame fusion. This reuses the **spatial multiscale motivation** of
CMSA-Net but does **not** implement CMSA-Net's cross-frame multiscale attention.

Sources:
- SAM2: https://arxiv.org/abs/2408.00714
- CMSA-Net: https://github.com/wangtong627/CMSA-Net
- Lean-SAM2: https://github.com/DeawhaleQwQ/Lean-SAM2

## What has and has not been established

- The module and scoped hook are implemented and retain native memory tensor
  geometry. Unit tests verify this interface in synthetic CPU predictors.
- **Training is same-frame oracle-memory distillation, not video BPTT.**
  TRAIN-only GT masks construct a target memory and are never written into
  the prediction memory bank. The actual inference fusion uses only predicted
  masks and image features.
- The distillation target may be **unreachable** under the bounded residual.
  Inspect `teacher_feasibility` and the fraction of unattainable elements.
- A lower training loss is **not evidence** of better future-frame segmentation.
  Only frozen, paired DEV/TEST video propagation can evaluate that.
- The anchor diagnostic measures fixed-anchor cosine shift between native and
  enriched memory on the **memory write**. It is NOT actual TAMP attention usage,
  retention-index overlap, or improved target discrimination.
- BF16 may eliminate tiny residual corrections. Check
  `memory_writes_quantized_changed` and
  `effective_memory_write_fraction`. No effective writes means the adapter
  is not a meaningful memory intervention, even if its FP32 tensor changed.
- No extra stored memory tokens does **not** imply zero additional compute.
- Conditioning-frame memory remains native in the default experiment. This
  deliberately keeps the initial target anchors fixed, but can create
  native/enriched feature-space mismatch. Inspect anchor cosine drift.
- These inference scripts start from **externally frozen automatic mask prompts**
  rather than running YOLO/FGPI inline. This isolates the memory contribution.

## Matched input-control arms

All controls use the **same adapter architecture** but must be **trained
separately**, with one immutable source partition and shared training setup.
The inference runner rejects a checkpoint trained for a different input mode.

| Mode | Early FPN 0 | Early FPN 1 | Explicit predicted mask | What it tests |
|---|---:|---:|---:|---|
| `full` | yes | yes | yes | Main multiscale hypothesis |
| `no_early` | no | no | yes | Memory-only + mask adapter control |
| `early0_only` | yes | no | yes | Contribution of higher-resolution FPN level |
| `early1_only` | no | yes | yes | Contribution of the other FPN level |
| `no_mask` | yes | yes | no | Whether explicit second mask input helps |

The native memory is mask-conditioned in **all** modes, including `no_mask`.
The mode `no_mask` only removes the **additional** mask input to the adapter.
A control trained with one mode must never be presented as evidence for another.

## Data and environment checks

1. Freeze the repository commit and Lean-SAM2 submodule revision. Use the
   exact same model weights, configs, runtime device and frame resolution.
2. Use leakage-free, independent **source/patient groups** for TRAIN/DEV/TEST
   wherever source provenance allows. Do not treat adjacent video frames as
   independent test samples.
3. Verify the PolypGen label-ID 1 masks, correct frame ordering, and complete
   source-frame identifiers.
4. Export automatic prompt masks once on images *without* reading test GT.
   Freeze the prompt manifest and SHA256 hashes. Exclude no-detection videos
   explicitly; report coverage.
5. Use an annotation-bearing TRAIN manifest only for capture; use
   **image-only** manifests for inference; open held-out labels **only** in
   `score_lean_memory_ablations.py`.
6. Run regression tests:
   ```bash
   python -m pytest -q \
     tests/test_lean_memory_feature_fusion.py \
     tests/test_lean_memory_training.py \
     tests/test_lean_memory_ablation.py
   ```
7. Use a short DEV video first; inspect memory-write telemetry before longer
   evaluations.

**Important:** Changes to the model/hook/trainer are SHA-bound. Earlier
fusion checkpoints may now fail provenance checks. Do not bypass these
checks or reuse an old checkpoint; recapture/retrain when required.

## Reproducible commands

All example paths below are placeholders; set them to your frozen, verified
artifacts. `shasum -a 256` works on macOS. Use `sha256sum` on Linux.

```bash
export TRAIN_MANIFEST=/path/to/train_with_masks.jsonl
export DEV_IMAGE_MANIFEST=/path/to/dev_images_only.jsonl
export DEV_EVAL_MANIFEST=/path/to/dev_with_masks.jsonl
export TRAIN_PROMPTS=/path/to/frozen_train_automatic_masks.jsonl
export DEV_PROMPTS=/path/to/frozen_dev_automatic_masks.jsonl
export SPLIT_FILE=/path/to/source_group_split.json
export GROUP_MAP=/path/to/train_video_to_source_group.json
export BACKBONE=/path/to/lean_hiera_bplus.pt
export OUT=/path/to/new_memory_fusion_experiment
export DEVICE=cuda

export BACKBONE_SHA=$(shasum -a 256 "$BACKBONE" | awk '{print $1}')
mkdir -p "$OUT"

# 1. Generate native / GT-conditioned memory capture pairs on TRAIN only.
python -m scripts.capture_lean_memory_train \
  --manifest "$TRAIN_MANIFEST" --prompt-manifest "$TRAIN_PROMPTS" \
  --group-map "$GROUP_MAP" --split-file "$SPLIT_FILE" \
  --checkpoint "$BACKBONE" --backbone-sha256 "$BACKBONE_SHA" \
  --device "$DEVICE" --output "$OUT/captures"

export CAPTURE_MANIFEST="$OUT/captures/train_memory_captures.json"
export CAPTURE_SHA=$(shasum -a 256 "$CAPTURE_MANIFEST" | awk '{print $1}')

# 2. Train matched input-mode controls on identical captures.
mkdir -p "$OUT/checkpoints"
for MODE in full no_early early0_only early1_only no_mask; do
  python -m training.lean_memory_fusion \
    --manifest "$CAPTURE_MANIFEST" --split-file "$SPLIT_FILE" \
    --output "$OUT/checkpoints/$MODE.pt" \
    --backbone-sha256 "$BACKBONE_SHA" \
    --input-mode "$MODE" --epochs 5 --lr 1e-4 \
    --preservation-weight 0.1 --seed 20261120 --device "$DEVICE"
done

# 3. Freeze native/identity evaluation using identical automatic prompts.
for ARM in native identity; do
  python -m scripts.run_lean_memory_fusion_ablation \
    --arm "$ARM" --manifest "$DEV_IMAGE_MANIFEST" \
    --prompt-manifest "$DEV_PROMPTS" \
    --checkpoint "$BACKBONE" --checkpoint-sha256 "$BACKBONE_SHA" \
    --split dev --device "$DEVICE" --output "$OUT/$ARM"
done

# 4. Evaluate each trained control with its matching mode.
for MODE in full no_early early0_only early1_only no_mask; do
  FUSION_SHA=$(shasum -a 256 "$OUT/checkpoints/$MODE.pt" | awk '{print $1}')
  python -m scripts.run_lean_memory_fusion_ablation \
    --arm fusion --fusion-input-mode "$MODE" \
    --fusion-checkpoint "$OUT/checkpoints/$MODE.pt" \
    --fusion-sha256 "$FUSION_SHA" \
    --train-manifest-sha256 "$CAPTURE_SHA" \
    --manifest "$DEV_IMAGE_MANIFEST" --prompt-manifest "$DEV_PROMPTS" \
    --checkpoint "$BACKBONE" --checkpoint-sha256 "$BACKBONE_SHA" \
    --split dev --device "$DEVICE" --output "$OUT/$MODE"
done

# 5. Independent paired DEV scoring; never feed DEV labels into inference.
python -m scripts.score_lean_memory_ablations \
  --run "native=$OUT/native" --run "identity=$OUT/identity" \
  --run "full=$OUT/full" --run "no_early=$OUT/no_early" \
  --run "early0_only=$OUT/early0_only" \
  --run "early1_only=$OUT/early1_only" \
  --run "no_mask=$OUT/no_mask" \
  --manifest "$DEV_EVAL_MANIFEST" --output "$OUT/scored_dev"
```

All runs compared by the scorer must be generated from the **same final
AdeSEG commit**. The scorer intentionally rejects mixed source revisions.
Train and DEV sequences may be different; inference and scoring DEV manifests
must reference the exact same source images/frame IDs.

## Decision gates before TEST

- The native and zero-initialized identity arms must produce **bit-identical
  mask and logit hashes** and identical temporal resident-memory snapshots.
- If the learned model has no effective BF16 memory writes, report the failure
  before interpreting differences.
- Confirm the mode-matched checkpoints, train split receipts, GT separation,
  no-object behavior, and unchanged memory-writing geometry.
- Inspect whether full fusion improves **paired sequence-level** Dice/IoU vs
  native and vs `no_early`; inspect GT-empty false alarms and temporal errors.
- Compare `early0_only` and `early1_only` to full for scale complementarity;
  compare `no_mask` to full for redundant/error-amplifying mask guidance.
- Treat native-anchor cosine shifts as **representation drift**, not proof of
  more correct TAMP top-K. To claim better retention, add a dedicated
  experiment logging actual TAMP retained indices and target-region
  correctness with a fixed token budget.
- Measure extra adapter runtime and actual GPU peak separately. Resident token
  counts are not peak VRAM, and dynamic native TAMP behavior may change the
  number of retained tokens even when write shapes are identical.
- Freeze hyperparameters and choose one candidate using DEV. Evaluate TEST
  once, without subsequent tuning, and label previously inspected sequences
  as exploratory if applicable.

**Failure interpretation:** If `full` improves over native but not over
`no_early`, the evidence supports an adapter or mask effect, **not**
the multiscale-feature hypothesis. If `full` beats `no_early` but
not the best single-scale arm, the claim that *both* early scales are needed
is not supported. No positive outcome should be assumed in advance.
