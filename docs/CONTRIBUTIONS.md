# AdeSEG contribution boundaries

**Scientific status:** these are hypotheses and implemented research directions, NOT proven improvements.

## A — MedSAM2 high-resolution refinement

- **Pretrained component:** native MedSAM2 Hiera-T / 512 backbone and mask decoder.
- **Motivation:** test whether spatial detail lost through deep representations can be restored without damaging pretrained behavior.
- **Modification:** `adeseg/adapters/high_resolution_skip.py`, implemented in `modeling/medical_highres/`. Project 32-channel early skips and 256-channel deep features; fuse and refine; inject a bounded residual into the first native decoder skip.
- **Training:** supervised segmentation plus optional native-teacher KL and positive-region preservation. Teacher and GT are training-only; never inference inputs.
- **Ablations:** M0 native, M1 adapter only, M2 + KL, M3 + positive preservation; boundary supervision is a separate optional hypothesis.
- **Necessary evidence:** matched checkpoint/config, splits, initial prompts, resolution, parameters, training schedule, and seed; full-video paired performance with boundary and empty-frame errors.
- **Known limitation:** the historical adapter forward source is missing. Saved weight geometry does not prove forward parity or recovered historical performance.

## B — Lean-SAM2 multi-stage feature aggregation

- **Pretrained component:** Lean-SAM2 built on SAM2.1 Hiera-B+.
- **Motivation:** test whether learned foreground/background channel-value mixing improves object/background discrimination and reduces drift.
- **Modification:** `adeseg/adapters/multiscale_channel_attention.py`, `modeling/aggregation_blocks/`, and `modeling/structured_feature_aggregation.py`. Project early/deep features, spatial context, FG/BG Q/K/V correlation, residual feature fusion, foreground contrast and residual logits.
- **Ablations:** L0 native Lean, L1 projections/fusion, L2 region attention, L3 contrast, L4 complete refiner, L5 future-memory routing versus output-only. Implemented variants need verified feature plumbing, not only module flags.
- **Necessary evidence:** actual trained-adapter full-video results with paired native baseline and identical prompts and eligible frames.
- **Known limitation:** 0.742071 historical patient-macro Dice was a compatibility-corrected Lean author run WITHOUT the trained feature refiner.

## Evidence rules

1. Never call an oracle-GT-initialized run automatic.
2. Do not compare different backbones and claim the difference was caused by an adapter.
3. Separate empty-frame false alarm rate from GT-present Dice.
4. Distinguish frame-weighted means from patient/video-macro means.
5. Preserve the exact historical dependency and checkpoint identities.
6. Colonoscopy PolypGen is an engineering proxy, not evidence of adenoid clinical generalization.

See `docs/ARCHITECTURE.md` and `docs/CHECKPOINT_PROVENANCE.md`.

## FGPI fixed joint prompting without feature reliability gating

The research **FGPI** arm always combines a YOLO-provided bounding box with
one foreground and one background point derived from *frozen early FPN features*
using relative region-prototype similarity. The `box_only` arm is a separate,
matched ablation and **never** an automatic fallback. No feature contrast,
margin, uncertainty or learned reliability cutoff controls whether points are
added. Even an uninformative/uniform feature map produces deterministic points;
there is no empirical assertion that such prompts help.

Only malformed inputs or genuinely impossible spatial geometry cause a
fail-closed exception; these are input validity checks, **not confidence
gating**. The default detector confidence cutoff in
`scripts.infer_fgpi_video` is **zero**. An optional threshold controls
*which YOLO detections enter the study*, not whether FGPI feature prompts
are added to an accepted detection. Keep that optional threshold identical
between `box_only` and `fgpi` when evaluating.

This FGPI module **does not itself run YOLO detection**. Its input is a frozen
YOLO detection manifest with source-image hashes. End-to-end benefit requires
execution with real YOLO detections, a pinned MedSAM2 checkpoint, and
matched patient-independent test videos. No result has yet demonstrated gain.
