# Matched Dice ablations: Lean-SAM2 memory feature fusion

The proposed memory-enrichment model is on `main`, **opt-in**:
`modeling/lean_memory_feature_fusion.py` and
`adeseg/inference/lean_memory_feature_fusion.py`.

These scripts add a *real inference/score pathway* for comparing native
Lean-SAM2 to a zero-residual identity control and a **trained** early-feature
memory-enrichment model on the SAME videos and frozen initialization masks.

- `scripts/run_lean_memory_fusion_ablation.py` performs prediction without opening
  GT masks. It uses the pinned Lean author builder, a fresh predictor per
  sequence, the same prompts, and the original temporal pruning machinery.
- `scripts/score_lean_memory_ablations.py` independently reads frozen binary
  predictions and GT masks to compute exact per-sequence and per-frame Dice.
- `tests/test_lean_memory_ablation.py` verifies scores on synthetic masks,
  identical prompt rules, binary masks, source hashes and missing-frame rejection.

## Ablation table to be filled by real video runs

| Arm | Early-FPN memory enrichment | Memory selection | Runnable now? |
| --- | --- | --- | --- |
| **B0 native** | OFF | Lean TAMP | Yes, with data and author checkpoint |
| **B1 identity** | Zero-initialized no-op | Lean TAMP | Yes, with data and author checkpoint |
| **B2 fusion** | **Trained early-feature memory fusion** | Lean TAMP | Needs authenticated trained fusion checkpoint |
| B3 scoring-only | OFF | Multiscale token scoring | **Not** an integrated video arm; shadow selection only |
| B4 fusion+scoring | Trained fusion | Multiscale scoring | **Not** implemented as integrated video arm |
| R0 RMFA output | Output-only logit refiner | Lean TAMP | Historical separate controlled runner |
| R1 RMFA memory-mask | Mask feedback to native encoder | Lean TAMP | Historical separate controlled runner |

Do **not** enter the historical 0.742071 Lean result as B2, nor mix the
four-video RMFA numbers and eight-video pooled-score numbers in the same
matched-run effect size. Those used different samples/prompts and were
exploratory.

## Required input manifest

**Two separate manifests are required.** Video inference accepts an
**image-only** JSONL manifest with `split, video_id, frame_index, image`
and deliberately rejects any `mask` field. Example:

~~~json
{"split":"dev","video_id":"seq2","frame_index":3,"image":"frames/seq2/000003.jpg"}
~~~

TRAIN capture generation and independent Dice scoring use the existing
labeled JSONL schema (`split, video_id, frame_index, image, mask`).
These manifests need not be byte-identical: the scorer matches frame IDs,
source-image bytes, frozen prompt digests, backbone/config/software hashes
and per-frame predictions. Frame IDs must be consecutive JPEG sources.
No GT-mask path is passed to the inference runner.

Provide a separate, **immutable, precomputed automatic initialization mask**
JSONL. Example (paths relative to prompt manifest, or absolute):

~~~json
{"video_id":"seq2","frame_index":3,"mask_path":"prompts/seq2/000003.png","image_sha256":"<64-lowercase-hex>","mask_sha256":"<64-lowercase-hex>","prompt_origin":"automatic_mask"}
~~~

One mask per eligible video. All absent-prompt videos are explicitly
recorded `no_initialization`, never silently discarded. The prompt PNG
must be 0/255 at original frame resolution and **must not be copied from
ground truth**. The input checker authenticates source-image and frozen-mask SHA256 and
refuses other prompt modes. An image-only inference manifest has no GT path
against which to compare a supplied prompt; therefore detector-origin
metadata must be independently audited to exclude disguised oracle prompts.

This protocol takes a frozen automatically predicted initialization MASK,
not oracle GT masks and not repeated detector calls. It does not regenerate
the initial mask from the box. To study FGPI prompt changes, export and
freeze those prompts separately, then match each prompt-condition run.

## Run native / identity / learned fusion

First pin the *same* SAM2.1 Hiera-B+ checkpoint in all arms and the same
original Lean-SAM2 source gitlink. The script uses
`scripts.run_author_baseline.build_author_predictor` and refuses modified
upstream Lean source.

~~~bash
# The SHA below is for the SAM2.1 Hiera-B+ backbone, NOT Lean bypass weights.
MODEL_SHA=$(shasum -a 256 /path/to/sam2.1_hiera_base_plus.pt | cut -d ' ' -f1)

python -m scripts.run_lean_memory_fusion_ablation \
  --arm native --manifest /path/to/images_only.jsonl \
  --prompt-manifest /path/to/frozen_automatic_masks.jsonl \
  --checkpoint /path/to/sam2.1_hiera_base_plus.pt \
  --checkpoint-sha256 "$MODEL_SHA" --device mps \
  --output outputs/ablation/native

python -m scripts.run_lean_memory_fusion_ablation \
  --arm identity --manifest /path/to/images_only.jsonl \
  --prompt-manifest /path/to/frozen_automatic_masks.jsonl \
  --checkpoint /path/to/sam2.1_hiera_base_plus.pt \
  --checkpoint-sha256 "$MODEL_SHA" --device mps \
  --output outputs/ablation/identity
~~~

**B2 requires a real training checkpoint.** In addition to the same args
above, set `--arm fusion --fusion-checkpoint /path/to/trained.pt`,
`--fusion-sha256 <exact-file-hash>`, and
`--train-manifest-sha256 <source-verified-train-manifest-hash>`.
A fake or zero-initialized checkpoint is NOT a valid learned Dice result.
A TRAIN-only memory-capture producer and **same-frame oracle-memory
distillation** trainer are now implemented (see
[training and evaluation guide](LEAN_MEMORY_TRAINING.md)).
This is NOT end-to-end backpropagation through future video frames, and there
are still no trained weights or new measured video Dice in the repository.

The inference receipts live in `<arm-output>/receipt.json`, and post-prompt
prediction masks in `<arm-output>/predictions/<sequence>/*.png`.
Each output folder must not pre-exist.

## Independently score Dice after inference

~~~bash
python -m scripts.score_lean_memory_ablations \
  --manifest /path/to/labeled_videos.jsonl \
  --run native=outputs/ablation/native \
  --run identity=outputs/ablation/identity \
  --run fusion=outputs/ablation/fusion \
  --baseline native \
  --output outputs/ablation/dice_report
~~~

Omit the `fusion=` line until a trained checkpoint and its run exist.

Outputs:

- `report.md`: arm-by-arm **with / without** Dice and empty FP table.
- `results.json`: patient/video-macro Dice, frame-weighted Dice, paired
  per-sequence delta, paired sequence-bootstrap descriptive 95% interval.
- `per_sequence.csv`: every eligible sequence and arm.
- `per_frame.jsonl`: per-frame Dice/IoU for GT-present; separate
  false-alarm status for GT-empty frames.

When using Apple Silicon, use `--device mps`; on a supported NVIDIA runner,
replace it with `--device cuda`. Run native twice on the same machine before
treating cross-process bitwise-identity differences as adapter regressions.

The scorer fails closed on mismatched prompt receipts, split/model hashes,
runtime/software/device and config digests, missing videos or frames, stale
prediction PNG hashes, out-of-order frames, wrong mask sizes and
any native-vs-identity **logit or binary-mask** discrepancy. GT file bytes are
read ONLY by the independent scorer; evaluation-manifest SHA256 is recorded.
Resident post-prune spatial token and insurance-entry mismatches are
reported at **each frame** and at the final state. It refuses
partial intersections or reusing an existing score output directory.

## Scientific and engineering gates

1. Run `pytest -q tests/test_lean_memory_feature_fusion.py
   tests/test_lean_memory_ablation.py` and full `pytest -q tests`.
2. Use a DEV-only smoke first, verify identical B0 and B1 logits,
   memory positions, object pointers, TAMP selections and mask hashes.
   Mask hashes alone do NOT establish every internal invariant.
3. Confirm **trained B2** has nonzero memory-write telemetry and that its
   retained tokens and memory-attention bytes match B0/B1. Enriched
   values cost temporary compute; log peak VRAM and latency.
4. Predeclare group-disjoint TRAIN/DEV/TEST and choose hyperparameters on DEV.
   Previously inspected PolypGen is exploratory; it is colonoscopy, not
   a clinical adenoid dataset.
5. Compare the SAME post-prompt source frames. Report video-macro Dice,
   frame-weighted Dice, full video distribution, GT-empty alarm rate and
   time. Use independent patient groups for inferential uncertainty.
6. Only describe B3/B4 as tested after their live Lean hooks have passed
   retention/position/TCIM and source-verified parity tests.

**CI caveat:** Existing hosted GitHub Actions workflows have failed with
zero executed job steps. Local CPU checks and GPU video smoke must be
reported separately; do not claim the GitHub CI suite passed.
