# SASVi and SAMRefiner on PolypGen

This is an exploratory engineering and performance study. The existing
PolypGen detector's training split is unknown and the evaluation frames have
already been inspected. Experimental artifacts are written to Drive under
`outputs/experiment/sasvi_samrefiner_20261114/`.

## SAMRefiner

- Author checkout: `external/SAMRefiner`, commit
  `4bb7f95738c7a2e805f189210fb09e02a2197557`; its bundled modified SAM and
  `sam_refiner` implementation are used. Original SAM ViT-B checkpoint,
  CPU float32, 2 PyTorch threads, FastGeodis 1.0.5, and author defaults
  including five iterations.
- The input is the saved `yolo_box` output of the independently audited
  `prompt_feature_refinement_v1/pilot_auto_attempt01` run. Evaluation uses its
  exact 80 selected frames across 21 sequences; reference is the input SAM2
  mask on the same frame. An empty initial mask remains empty, since it cannot
  provide a meaningful refinement prompt. No GT enters refinement; GT selects
  the retrospective evaluation cohort and scores predictions.
- Runner `scripts/run_samrefiner_polypgen.py`; independent pixel-level rescore
  `scripts/audit_samrefiner_polypgen.py`. Seq11 gate and cohort results are
  independently executed and match. Store source, asset, and mask hashes.

## SASVi adaptation

- Author checkout: `external/SASVi`, commit
  `ba87d22267da6fd05155829aee35da102c1a6567`. The released inference loop is
  used with SAM2.1 Hiera B+ on CPU float32. CUDA autocast is disabled, and
  output masks are saved at original size. Dataset-only model imports are
  replaced with inert stubs; those functions are not used by this adapter.
- The released loop expects a frame-wise segmentation overseer. As no matching
  pretrained polyp overseer is supplied, this adaptation supplies the saved
  unprompted CMSA-Net predictions (released column-wise preprocessing,
  foreground threshold 128) as one binary polyp class. The saved predictions
  are verified against the earlier CMSA receipt. They were generated with
  overlapping video clips that can use future images. SASVi also uses future
  overseer masks for confirmation; consequently this is retrospective and
  must not be described as causal online inference. It is not the published
  SASVi overseer or an unmodified author benchmark.
- The released loop requests a new class be present in three following frames,
  but a truncated final slice would accept fewer confirmations. The minimal
  correction is in `patches/sasvi_confirmation.patch` and
  `tests/test_sasvi_confirmation.py` exercises both an unconfirmed last-frame
  detection and a detection with three confirmations. Original and corrected
  runs are retained separately. A short gate differs only at the final frame,
  where the original re-prompted without a confirmation window.
- The upstream sampler silently swallowed every exception when requesting 20
  KMeans clusters. A mask with fewer than 20 pixels therefore lost its point
  prompt entirely. The patch limits the requested cluster count to the number
  of foreground pixels, leaves a truly empty mask without points, and lets
  unexpected errors surface. A small-mask regression test covers this case.
- The released video loader sorts by the filename substring starting at
  position five, while its inference loop sorts by the entire numeric name.
  With staged six-digit names, this makes `000010.jpg` precede `000001.jpg`
  and pairs the wrong images with overseer predictions. The patch restores
  full numeric sorting. The runner also validates the actual predictor frame
  list against the overseer list before propagation. A 21-frame test exercises
  ordering across two digit boundaries. **All earlier SASVi runs exceeding
  nine frames are invalid**, including the first cohort and seq11/seq20
  results. The eight-frame gate did not expose the bug and does not validate
  long-clip ordering. SAMRefiner is unaffected.
- The same patch converts JPEG tensors to float32 before transferring them to
  the model device: the released ordering attempts to move a float64 tensor to
  MPS, which fails. CPU conversion has the same order of numerical operations
  because both old and new transfers are to CPU. MPS results must be compared
  with CPU runs rather than presumed bit-identical.
- Before the sampler fix, one of two 21-frame MPS checks diverged substantially
  after a re-prompt; the cause was not established. Subsequent checks with
  traced point prompts were substantially more consistent. The final two
  independently executed, corrected 21-frame MPS gates had identical prompt
  events and masks. This short check is not proof of whole-cohort determinism.
- `scripts/run_sasvi_cmsa_polypgen.py` checks the author revision, exact patched
  file hash, earlier CMSA receipt, model and input hashes and writes predictions
  to Drive. `scripts/audit_sasvi_cmsa_polypgen.py` independently rescores
  those masks and CMSA masks against the current labels. Evaluation excludes
  frames through the first labeled-present frame solely to match earlier
  comparison conventions; SASVi itself starts on frame zero and receives no
  ground-truth prompt. Neither that label-derived evaluation cutoff nor any
  later annotation is used in inference.
- The corrected full cohort was first run using MPS FP32, but full seq20
  replays with identical recorded prompts produced Dice 0.000, 0.105 and 0.193.
  Two repeats with a fixed seed and deterministic algorithms also disagreed;
  the cause is unresolved. The MPS cohort is therefore provisional only.
  Two fresh full-sequence CPU runs reproduced all 52 seq20 masks exactly.
  The full CPU cohort runs in two disjoint partitions, guarded by those CPU
  checks. The cohort driver records each sequence's log, mask and image hashes,
  prompt events, independent audit, and progress on Drive.
