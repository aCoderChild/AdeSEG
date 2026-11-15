# Priority-sequence contrast module: research design

Status: proposed, not implemented or evaluated. Target seq11 overextension first; seq20 and cropped seq23 are stress tests. Keep the original one-shot detector, existing recent-window Lean checkpoint and preprocessing, and no redetection or new confidence gating. All experiment outputs go to the Google Drive experiment directory.

## Hypothesis and contribution

Use early image features for spatial discrimination and deeper features for semantic context to compute a continuous foreground–background contrast map around the current predicted mask. Fuse that information into a mask-logit correction through an existing trained-compatible decoder/refiner, or train a small adapter on separate videos. A source/foreground/background description is computed from automatic predictions, never target labels at inference. Preserve an unchanged Lean mask as the baseline and avoid claiming this is CMSA's trained CMA or a full CFBI implementation.

Compare the residual feature branch against the same-capacity mask-only refiner, unweighted early/deep features, RGB contrast and shuffled controls. Separate same-capacity training controls are required to establish whether any gain comes from contrast rather than extra capacity. Parameter-free SimAM is a saliency baseline, not foreground/background or channel-correlation equivalence.

## Data and evaluation

Freeze preprocessing, feature coordinates, scale alignment, pretrained checkpoint and prompt. Build train/validation splits by video; keep 11, 20 and 23 out of training and hyperparameter selection. Verify patient/source overlap as far as metadata permits. Train from saved automatic masks including overextension, missing targets and empty frames; use no ground-truth inputs at inference. Document label-derived mask perturbations only as training augmentation and avoid copying held-out examples. Report how often foreground/background regions are empty or contaminated.

First evaluate fixed saved masks, with precision, recall, Dice, boundary metrics, false-positive pixel area and presence/absence separated. Same frame IDs, crop and prompt for all arms. Include predictions with no polyp overlap: a branch that only changes boundaries cannot be credited with localizing a missed object. No new probability threshold or confidence gate; retain the underlying model's established output convention. Run at least one unchanged/refiner-noop integrity check and verify checkpoint/preprocessing, original image geometry and all masks. Score labels after all predictions are frozen.

Then, only for a frozen candidate with a meaningful benefit over controls and acceptable empty-frame behavior, run matched full-video propagation with identical automatic prompt, recent-window budget and sequence schedule; explicitly state whether corrected logits feed the memory encoder and compare otherwise identical feedback policies. Report per-sequence results, failure intervals and uncertainty. Keep the baseline and all results, logs, hashes and limitations.

## Risks and scope

An initially wrong or empty mask can make prototypes describe background; feature contrast may amplify illumination or reflections. A semantic mechanism that discovers a polyp absent from the current mask acts as new localization, even if fused into a segmenter; do not silently treat it as boundary refinement or add a framewise detector. The seq23 late raw-window recovery is retrospective and does not isolate the cause. Existing seq11 point and dense-prior pilots and memory gate studies were not convincing. No claimed improvement before proper controls.
