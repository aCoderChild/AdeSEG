# Experimental methodology and preflight

## Study design

Use a frozen dataset manifest, pinned detector and SAM2/Lean-SAM2 checkpoints, fixed initial prompt policy, and matched per-sequence evaluation. Patient/source-level partitions should be disjoint across TRAIN, DEV and TEST; use DEV for hyperparameter selection and freeze before TEST.

## Sequential ablations

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

## Preflight for Sunday experiments

1. Check Git revision, all submodule revisions, exact model config and checkpoint hashes.
2. Validate the contiguous video manifest and label-mask geometry.
3. Export YOLO detections once, keep their original-image coordinates and freeze detector outputs across all prompt ablations.
4. Run `python -m pytest -q tests`; prioritize `tests/test_fgpi_entrypoint.py`, `tests/test_fgpi_video.py`, `tests/test_fgpi_target_anchor.py`, and `tests/test_research_module_interfaces.py`.
5. Verify a single first frame: valid feature-map channels, detector coordinates, positive/negative point locations, SAM2 prompt API, initial mask-logit geometry.
6. Complete a short DEV sequence before attempting a 21-sequence run. Compare outputs to matched box-only initialization.
7. For learned RMFA, verify trained checkpoint provenance and zero-residual identity control. Do not run untrained adapter as learned inference.
8. Enable TTMR only after live memory-token/key/value and positional-convention checks; the existing offline selector alone does not modify video propagation.

The account's GitHub Actions runner lock prevents hosted CI execution until GitHub resolves it. Running the same tests locally is the correct preflight; green source checks cannot be assumed.
