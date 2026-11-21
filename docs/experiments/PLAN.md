# Experimental methodology and preflight

**Current priority:** multiscale enrichment of native Lean-SAM2
mask-conditioned memory, with its original TAMP/TCIM/TARR preserved.
For commands, strict frozen-input controls, model checkpoints and
independent scoring, use the
[active memory-fusion protocol](LEAN_MEMORY_FUSION_PREFLIGHT.md).

**Status boundary:** the active memory-fusion runner consumes frozen
automatic initial-mask prompts; it does not yet execute the full
YOLO/FGPI-to-Lean pipeline. The original RMFA and TTMR designs remain
separate legacy research tracks, not interventions enabled by that runner.

## Study design

Use a frozen dataset manifest, pinned detector and SAM2/Lean-SAM2 checkpoints, fixed initial prompt policy, and matched per-sequence evaluation. Patient/source-level partitions should be disjoint across TRAIN, DEV and TEST; use DEV for hyperparameter selection and freeze before TEST.

## Active multiscale-memory experiment

| Arm | Method | Specific question |
|---|---|---|
| M0 | Frozen automatic prompt + native Lean-SAM2 | Video propagation baseline |
| M1 | M0 + zero-initialized identity memory fusion | Prove routing/bitwise prediction parity |
| M2 | M0 + trained full early-FPN memory fusion | Does memory enrichment help future-frame segmentation? |
| M3 | M0 + separately trained no-early-feature memory adapter | Are earlier feature levels actually responsible? |
| M4 | M0 + separately trained one-level controls | Is each early spatial scale useful? |
| M5 | M0 + separately trained no-additional-mask control | Does second mask conditioning help or reinforce errors? |

All M2–M5 arms must be trained on the **same sealed TRAIN capture
cohort**, use fixed image-only inference inputs, and be paired at the
video/source-group level. An unrolled temporal segmentation loss is
**not** implemented; the trainer presently uses same-frame
ground-truth-conditioned memory distillation as a surrogate target.

Before claiming any benefit, verify BF16-effective writes that precede
subsequent evaluated frames, TAMP anchor availability, native/identity
bitwise parity, and an improvement over M3 (not only M0). TAMP retained
indices are compared using a **complete matched event set** by default;
partial comparisons are explicitly exploratory.

## Historical proposed FGPI/RMFA/TTMR ablations (not one active runner)

| Arm | Method | Specific question |
|---|---|---|
| A0 | YOLO box only + native prompt encoder | Detection/prompt baseline |
| A1 | Same box + geometry-selected points | Do extra points alone help? |
| A2 | Same box + random points (fixed seed) | Does selection add value over chance? |
| A3 | Same box + FGPI points | Does early-feature token scoring improve initialization? |
| B0 | Selected prompt + native Lean-SAM2 | Spatial comparison baseline |
| B1 | B0 + trained RMFA, output-only | Does region-aware residual refinement help? |
| B2 | B1 + routed masks into future memory | Does feedback help or amplify errors? |
| C0 | B1 + native Lean memory | Temporal comparison baseline |
| C1 | B1 + TTMR, same retained-token budget | Does TTMR improve historical token choice? |

The current FGPI video runner is on MedSAM2 Hiera-T; A0/A3 must be rerun on Lean Hiera-B+ (using identical prompt policy) before interpreting their change as an integrated Lean contribution. A1/A2 require separate implementations.

## Primary outcomes

**FGPI:** initialization-frame Dice/IoU, detection coverage and invalid-initialization rate. **RMFA:** GT-present Dice/IoU, boundary F and GT-empty false alarms. **TTMR:** late-frame Dice, temporal IoU, propagation failures and cost (retained tokens, inference time and memory use).

Report both frame-weighted and equal-video means, including exclusions and the distribution of per-video effects. For uncertainty, resample verified independent source groups instead of correlated frames.

## Legacy-track preflight (historical notes)

1. Check Git revision, all submodule revisions, exact model config and checkpoint hashes.
2. Validate the contiguous video manifest and label-mask geometry.
3. Export YOLO detections once, keep their original-image coordinates and freeze detector outputs across all prompt ablations.
4. Run `python -m pytest -q tests`; prioritize `tests/test_fgpi_entrypoint.py`, `tests/test_fgpi_video.py`, `tests/test_fgpi_target_anchor.py`, and `tests/test_research_module_interfaces.py`.
5. Verify a single first frame: valid feature-map channels, detector coordinates, positive/negative point locations, SAM2 prompt API, initial mask-logit geometry.
6. Complete a short DEV sequence before attempting a 21-sequence run. Compare outputs to matched box-only initialization.
7. For learned RMFA, verify trained checkpoint provenance and zero-residual identity control. Do not run untrained adapter as learned inference.
8. Enable TTMR only after live memory-token/key/value and positional-convention checks; the existing offline selector alone does not modify video propagation.

The historical note about a GitHub Actions runner lock is no longer
current. Check the status of the **specific final commit's** CI workflows
before relying on them. CPU CI passing is not a substitute for CUDA
video-propagation smoke tests and paired DEV results.
