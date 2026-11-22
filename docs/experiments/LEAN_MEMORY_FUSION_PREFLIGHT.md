# Lean-SAM2 early-FPN memory enrichment: experiment preflight

**Scope:** research-stage PolypGen automatic video-polyp segmentation. This is
**not** adenoid-specific clinical validation. This protocol does **not**
modify native SAM2 image/memory encoders, decoder, TAMP, TCIM or TARR.

See also [literature-guided architecture corrections](../methodology/LITERATURE_GUIDED_CORRECTIONS.md)
for the verified native box/FGPI prompt path, GT-consistent teacher
object-presence, and separate synchronized timing benchmark.

See also the [source-level architectural and mathematical audit](../methodology/LEAN_MEMORY_ENRICHMENT_AUDIT.md)
for the exact FPN feature identity, invariants, theorem assumptions and
limits of what can be proved without video experiments.

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
  `memory_writes_quantized_changed`,
  `effective_memory_writes_affecting_future`, and
  `effective_memory_write_fraction`. A change written only on the final
  frame cannot improve a subsequent evaluated mask. A trained run with no
  BF16-effective future-facing write fails closed when the cohort has
  at least one eligible multi-frame video.
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

### Separately trained no-object-gate ablation

The default memory adapter exactly preserves native memory when Lean-SAM2
predicts the object absent. This prevents modification of its pretrained
no-object representation but cannot correct a false-negative *presence
decision*. An explicitly named alternative is now available **only during
adapter training**:

```bash
python -m training.lean_memory_fusion \
  --manifest "$CAPTURE_MANIFEST" --split-file "$SPLIT_FILE" \
  --output "$OUT/checkpoints/full_allow_absent.pt" \
  --backbone-sha256 "$BACKBONE_SHA" \
  --input-mode full --allow-no-object-memory-fusion \
  --epochs 5 --lr 1e-4 --preservation-weight 0.1 \
  --seed 20261120 --device "$DEVICE"
```

Run this **after** the capture step and environment variables below have
been established. Evaluate it using the existing `--arm fusion` inference
runner with the matching new checkpoint SHA; the checkpoint's saved
`preserve_native_no_object=False` configuration is authenticated and
written to the run receipt. This is an **experimental risk ablation**, not
a recommended default. It may create false positives or temporal drift.
Compare GT-empty false alarms explicitly.

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
  --initialization-mode direct_prompt --teacher-presence-mode gt_mask \
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
    --arm "$ARM" --initialization-mode direct_prompt \
    --manifest "$DEV_IMAGE_MANIFEST" \
    --prompt-manifest "$DEV_PROMPTS" \
    --checkpoint "$BACKBONE" --checkpoint-sha256 "$BACKBONE_SHA" \
    --split dev --device "$DEVICE" --output "$OUT/$ARM"
done

# 4. Evaluate each trained control with its matching mode.
for MODE in full no_early early0_only early1_only no_mask; do
  FUSION_SHA=$(shasum -a 256 "$OUT/checkpoints/$MODE.pt" | awk '{print $1}')
  python -m scripts.run_lean_memory_fusion_ablation \
    --arm fusion --fusion-input-mode "$MODE" \
    --initialization-mode direct_prompt \
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

## Separate native TAMP selection diagnostic (not timing)

To investigate **actual** Lean-SAM2 token selections, rerun matched native
and learned `full` arms on the same DEV video subset, this time enabling
the author's native index dump. This is separate from the normal inference
runs above; detailed JSONL writes add overhead and should never be used for
fair runtime comparisons.

```bash
# Same env variables and authenticated checkpoint as above.
FUSION_SHA=$(shasum -a 256 "$OUT/checkpoints/full.pt" | awk '{print $1}')

python -m scripts.run_lean_memory_fusion_ablation \
  --arm native --initialization-mode direct_prompt --dump-tamp-indices \
  --manifest "$DEV_IMAGE_MANIFEST" --prompt-manifest "$DEV_PROMPTS" \
  --checkpoint "$BACKBONE" --checkpoint-sha256 "$BACKBONE_SHA" \
  --split dev --device "$DEVICE" --output "$OUT/native_tamp"

python -m scripts.run_lean_memory_fusion_ablation \
  --arm fusion --fusion-input-mode full --dump-tamp-indices \
  --initialization-mode direct_prompt \
  --require-target-anchors \
  --fusion-checkpoint "$OUT/checkpoints/full.pt" \
  --fusion-sha256 "$FUSION_SHA" \
  --train-manifest-sha256 "$CAPTURE_SHA" \
  --manifest "$DEV_IMAGE_MANIFEST" --prompt-manifest "$DEV_PROMPTS" \
  --checkpoint "$BACKBONE" --checkpoint-sha256 "$BACKBONE_SHA" \
  --split dev --device "$DEVICE" --output "$OUT/full_tamp"

python -m scripts.compare_native_tamp_indices \
  --reference "$OUT/native_tamp" --candidate "$OUT/full_tamp" \
  --output "$OUT/tamp_overlap"
```

The comparison file `tamp_selection_overlap.json` records matched event
counts, token-count mismatches as hard errors, missing events, and Jaccard
overlap of **actual retained indices**. An **empty or incomplete paired
event set fails closed** by default; `--allow-incomplete-events` is only
for explicitly exploratory partial-coverage diagnostics. A changed top-K is not evidence
that the selected tokens align better with the target; that still requires
grounded region-level evaluation. Compare paired masks and Dice separately.
The native Lean writer handles the dump without changing its TAMP policy;
its output is checksum-verified before use.

## Decision gates before TEST

- The native and zero-initialized identity arms must produce **bit-identical
  mask and logit hashes** and identical temporal resident-memory snapshots.
- If the learned model has no effective BF16 memory writes, report the failure
  before interpreting differences.
- Confirm the mode-matched checkpoints, train split receipts, GT separation,
  no-object behavior, and unchanged memory-writing geometry.
- Inspect whether full fusion improves **paired sequence-level** Dice/IoU vs
  native and vs `no_early`; inspect GT-empty false alarms and temporal errors.
  For a strict TAMP interaction experiment, add `--require-target-anchors`
  to the fusion inference invocation so missing/disabled target anchors cannot
  silently pass as an anchor-aware result.
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


## New safeguards for the full study (F06–F24)

### 0. Audit source groups and exact duplicate frames

The auditor enforces declared TRAIN/DEV/TEST source-group separation and
checks identical JPEG bytes, even when different file paths or video IDs
are used. **It cannot prove different patient identity or detect visual
near-duplicates**: review the dataset's original provenance independently.

```bash
export TRAIN_IMAGE_MANIFEST=/path/to/train_images_only.jsonl
export TEST_IMAGE_MANIFEST=/path/to/test_images_only.jsonl
export GROUP_MAP_ALL=/path/to/all_video_id_to_source_group.json
export TEST_PROMPTS=/path/to/test_frozen_prompts.jsonl
python -m scripts.audit_source_partitions \
  --train-manifest "$TRAIN_MANIFEST" \
  --dev-image-manifest "$DEV_IMAGE_MANIFEST" \
  --test-image-manifest "$TEST_IMAGE_MANIFEST" \
  --split-file "$SPLIT_FILE" --group-map "$GROUP_MAP_ALL" \
  --output "$OUT/source_partition_audit.json"
```

The original capture producer uses `$GROUP_MAP` covering TRAIN videos;
`$GROUP_MAP_ALL` must cover **every** TRAIN/DEV/TEST video. Neither file
is allowed to assign the same real patient to different declared groups.

### 1. Same-backbone YOLO → FGPI / box-only prompt producer

The formerly missing connection to the active Lean-SAM2 Hiera-B+ backbone
is now implemented as an independently authenticated stage. It **does not**
embed YOLO inference into the memory-propagation loop; it exports frozen
automatic first-frame masks for downstream identical-memory comparisons.

```bash
export DETECTOR=/path/to/polyp_yolo.pt
export DETECTOR_SHA=$(shasum -a 256 "$DETECTOR" | awk '{print $1}')

# Export from image-only frames; no mask/annotation field is accepted.
for SPLIT in train dev test; do
  case "$SPLIT" in
    train) IMAGES="$TRAIN_IMAGE_MANIFEST";;
    dev) IMAGES="$DEV_IMAGE_MANIFEST";;
    test) IMAGES="$TEST_IMAGE_MANIFEST";;
  esac
  python -m scripts.export_yolo_detections \
    --manifest "$IMAGES" --checkpoint "$DETECTOR" \
    --target-class 0 --image-only-manifest --split "$SPLIT" \
    --output "$OUT/yolo_$SPLIT.jsonl"
  python -m scripts.export_lean_fgpi_prompts \
    --manifest "$IMAGES" \
    --detections "$OUT/yolo_$SPLIT.jsonl" \
    --detector-checkpoint "$DETECTOR" --detector-sha256 "$DETECTOR_SHA" \
    --checkpoint "$BACKBONE" --checkpoint-sha256 "$BACKBONE_SHA" \
    --mode fgpi --split "$SPLIT" --device "$DEVICE" \
    --output "$OUT/lean_prompts_$SPLIT"
done

export TRAIN_PROMPTS="$OUT/lean_prompts_train/frozen_prompts.jsonl"
export DEV_PROMPTS="$OUT/lean_prompts_dev/frozen_prompts.jsonl"
export TEST_PROMPTS="$OUT/lean_prompts_test/frozen_prompts.jsonl"
```

**Important:** `--target-class 0` above is an **example only**. Use the
actual detector class ID for polyps, verified from its training schema.
The exporter verifies original image coordinates, first eligible YOLO box,
native 32/64/256 projected FPN levels, joint box+points or box-only prompt
behavior, native initialization-logit geometry and a nonempty binary mask.
No-detection and empty-initialization videos are explicitly counted in the
prompt-producer receipt.

For the RQ1 comparison, run a **separate** `--mode box_only` export with
the same YOLO detections and compare initial-mask coverage and accuracy.
Do **not** substitute its prompts within a fusion-vs-native ablation;
those must use the exact same frozen prompt manifest.

For publication-oriented DEV runs, add
`--require-verified-prompt-producer` to memory-fusion inference.
TEST runs **always** require a producer receipt matching the current
Lean checkpoint, images, source revision, prompt bytes and mode.
Older external masks without that receipt are explicitly **unverified**;
their SHA alone cannot certify non-oracle generation.

### 2. Feasible teacher targets and independent controls

The default trainer now uses `--teacher-target-policy feasible_clipped`:

```math
T_t^{\mathrm{feasible}} =
M_t +
\operatorname{clip}\left(
 M_t^{GT}-M_t,
 -0.95\alpha\sigma(M_t),
 +0.95\alpha\sigma(M_t)
\right).
```

This is the elementwise Euclidean projection of the teacher difference
into the representable bounded-residual region (a 0.95 margin keeps the
tanh target away from its asymptotic endpoints). It removes the
*known impossible-target optimization request*, **not** the fundamental
limitation that same-frame oracle memory is an imperfect surrogate for
future-frame segmentation accuracy.

Use `--teacher-target-policy raw_oracle` only for a separately
named, independently trained optimization control. The saved checkpoint
binds the policy. Previous v1/v2 memory-fusion checkpoints and teacher captures are not
compatible with the current v3 GT-presence/provenance protocol and must
be regenerated. Separate `predicted` teacher-presence controls are
supported, but their trained checkpoint must record that policy.

### 3. Real DEV video/GPU smoke test (mandatory before full cohort)

```bash
FUSION_SHA=$(shasum -a 256 "$OUT/checkpoints/full.pt" | awk '{print $1}')
python -m scripts.smoke_lean_memory_fusion \
  --initialization-mode direct_prompt \
  --manifest "$DEV_IMAGE_MANIFEST" --prompt-manifest "$DEV_PROMPTS" \
  --checkpoint "$BACKBONE" --checkpoint-sha256 "$BACKBONE_SHA" \
  --fusion-checkpoint "$OUT/checkpoints/full.pt" \
  --fusion-sha256 "$FUSION_SHA" \
  --train-manifest-sha256 "$CAPTURE_SHA" \
  --require-target-anchors --device "$DEVICE" \
  --output "$OUT/smoke_dev"
```

This checks actual video propagation on the pinned author code, including
bitwise native/identity logits, per-frame memory parity, effective
BF16 changes **before** a future scored frame, anchor observability,
and CUDA peak allocation when available. The report has **no Dice or
GT**. If it fails, do not use full-run metrics or report a causal
benefit. It still does not prove that enhanced memory improved accuracy.

A separate, predeclared
`--max-anchor-cosine-shift 0.1` (example threshold, **not** a
literature-supported optimum) may be added to trained fusion inference.
An exceeding run fails closed; this monitors drift rather than
certifying correct TAMP retention. Native TAMP top-K selection events
must be examined separately with the optional dump.

### 4. Freeze one DEV-selected model before held-out TEST

After scoring and selecting a model on DEV, generate the TEST protocol
lock. Never fit, alter prompts, or choose a model based on TEST labels.

```bash
python -m scripts.freeze_lean_test_protocol \
  --dev-score "$OUT/scored_dev/results.json" \
  --dev-candidate-receipt "$OUT/full/receipt.json" \
  --test-image-manifest "$TEST_IMAGE_MANIFEST" \
  --test-prompt-manifest "$TEST_PROMPTS" \
  --backbone-checkpoint "$BACKBONE" \
  --fusion-checkpoint "$OUT/checkpoints/full.pt" \
  --selected-run-name full \
  --output "$OUT/test_lock.json"

# Re-use the same --test-lock for native, identity and selected fusion.
python -m scripts.run_lean_memory_fusion_ablation \
  --arm fusion --fusion-input-mode full \
  --initialization-mode direct_prompt \
  --fusion-checkpoint "$OUT/checkpoints/full.pt" \
  --fusion-sha256 "$FUSION_SHA" \
  --train-manifest-sha256 "$CAPTURE_SHA" \
  --manifest "$TEST_IMAGE_MANIFEST" \
  --prompt-manifest "$TEST_PROMPTS" \
  --checkpoint "$BACKBONE" --checkpoint-sha256 "$BACKBONE_SHA" \
  --split test --test-lock "$OUT/test_lock.json" \
  --require-verified-prompt-producer \
  --device "$DEVICE" --output "$OUT/final_test_fusion"
```

If DEV selects a mode other than `full`, substitute that mode and its
**separately trained** checkpoint and DEV receipt. The scorer requires
identical lock hashes across all TEST comparison arms. This creates
accountable evidence, not an unbreakable barrier against inspecting or
retesting held-out results.

### 5. No-detection/empty-GT and efficiency reporting

The independent scorer now reports:

- all source videos versus successfully initialized videos, with explicit
  unpromptable GT-present and GT-empty frame counts;
- frames before and including initialization separately from post-prompt
  scored frames (conditional Dice is **not** full-video accuracy);
- BF16-effective writes available to future frames and target-anchor
  diagnostics;
- observed CUDA peak **allocated** bytes and per-video propagation
  duration; these are not controlled latency/speedup benchmarks.

Do not silently drop no-detection videos, assert that matching write-time
token shapes implies matching resident memory budgets, or claim that a
different TAMP top-K means better target retention.

### Outstanding research questions that no assert can solve

The existing adapter area-pools early features to the coarse memory grid;
this preserves their **scale provenance**, not all original fine boundary
detail (F10). Native target anchors and enriched non-conditioning memory
may occupy different feature distributions (F11); their cosine drift and
actual native TAMP retained indices are measured, not guaranteed to improve.
Same-frame oracle memory distillation remains **non-temporal** (F08).
End-to-end accuracy, frame boundary quality, temporal propagation
benefits, and actual patient-independent generalization all require real
DEV/TEST data and GPU experiments. Do not change the architectural
hypothesis after examining TEST.
