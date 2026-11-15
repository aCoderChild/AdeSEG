# Comparator implementation and provenance audit

Audit date: 2026-11-14. This is a code and existing-record review, not a new
model-quality experiment or a guarantee that the implementations have no other
bugs. Historical native SAM2, Lean-SAM2, SAMURAI and SAM2Long use a
ground-truth-mask initial prompt. The later automatic-prompt system and seq23
inset-excluded crop use a different protocol. The automatically prompted
SAMURAI experiment remains incomplete.

## Confirmed upstream code issue: CMSA preprocessing

The author dataset's train and evaluation transforms convert images to CHW
tensors before calling `Normalize_video`
([dataloader.py:238](../external/CMSA-Net/cmsa_project/lib/dataloader/dataloader.py#L238),
[dataloader.py:412](../external/CMSA-Net/cmsa_project/lib/dataloader/dataloader.py#L412)).
The published normalization indexes `img[:, :, i]`
([preprocess.py:168](../external/CMSA-Net/cmsa_project/lib/dataloader/preprocess.py#L168)),
which modifies three spatial columns rather than all pixels of each channel.
It is a real indexing error, not conventional channel normalization. The
published training and test paths share the error, but the code does not prove
how the released checkpoint was trained. The earlier inference-only correction
performed worse on the matched exploratory cohort; the user selected the
released behavior. Preserve both runs and identify the preprocessing explicitly.
See [CMSA_NET_POLYPGEN.md](CMSA_NET_POLYPGEN.md).

This is not grounds to silently change the pretrained checkpoint's input at
inference or claim the corrected result is an author baseline. The adapter
also uses label-file dimensions to restore output geometry; its model inputs
are images and the labels used in the transform do not become predicted-mask
inputs. Check that image and label geometry agree for any new dataset.

## Upstream SAM2Long propagation bounds issue

The author predictor constructs `processing_order` from the requested
`reverse` and `max_frame_num_to_track` values
([sam2_video_predictor.py:693](../external/SAM2Long/sam2/sam2_video_predictor.py#L693)),
but constructs the returned list using `range(start_frame_idx + 1, num_frames)`
([sam2_video_predictor.py:742](../external/SAM2Long/sam2/sam2_video_predictor.py#L742)).
These bounds are inconsistent for a partial or reverse request and can access
outputs outside the frames just processed. This is a source-code finding; this
audit did not execute partial/reverse requests. The historical project runner
requests complete forward propagation and verifies output coverage
([run_author_baseline.py:119](../scripts/run_author_baseline.py#L119)), so
this issue alone does not establish an error in the archived full-forward
scores. Do not use partial or reverse mode as if verified. Any repair should
be a separately documented patch and should receive a targeted author-model
test; the pinned checkout was not modified during this audit.

## Verified protocol differences, not implementation bugs

* SAM2Long gathers masks after processing the sequence, using the final
  `mem_pick_indexs` to select each earlier frame's pathway
  ([sam2_video_predictor.py:709](../external/SAM2Long/sam2/sam2_video_predictor.py#L709),
  [sam2_video_predictor.py:742](../external/SAM2Long/sam2/sam2_video_predictor.py#L742)).
  Its released full-trajectory output is noncausal; compare separately from a
  strictly online method and do not call the saved output real-time predictions.
  The project uses author CLI values `3 / 0.3 / 1` for pathways, IoU and
  uncertainty ([run_author_baseline.py:213](../scripts/run_author_baseline.py#L213),
  [vos_inference.py:480](../external/SAM2Long/tools/vos_inference.py#L480));
  the helper function defaults differ, so cite the CLI settings.
* The historical SAMURAI/SAM2Long runner loads a ground-truth mask at the first
  labeled frame ([run_author_baseline.py:218](../scripts/run_author_baseline.py#L218)).
  Such scores are a GT-prompt track, not automatic YOLO-prompt performance.
  For SAMURAI it rebases the video at that prompt while retaining source-frame
  identifiers ([run_author_baseline.py:506](../scripts/run_author_baseline.py#L506)).
  SAMURAI's published configuration includes its own memory-selection
  thresholds ([sam2.1_hiera_b+.yaml:117](../external/SAMURAI/sam2/sam2/configs/samurai/sam2.1_hiera_b%2B.yaml#L117));
  those are author-system behavior, distinct from adding a new project gate.
* CMSA is an unprompted clip model with a different checkpoint. The initial
  reference includes frame 1, so its frame 0 can use a future image. Frame 0
  is excluded from the historical matched post-prompt targets. Temporal
  attention uses causal clip positions for the later saved frames; nonetheless,
  the released test loop computes a complete initial clip before saving its
  outputs ([my_test.py:169](../external/CMSA-Net/cmsa_project/scripts/my_test.py#L169),
  [my_test.py:296](../external/CMSA-Net/cmsa_project/scripts/my_test.py#L296)).
  This adds batching latency and the saved clip benchmark is not an equivalent
  latency measurement for an online tracker.
* Patient independence, detector training exposure, and annotation
  completeness remain unresolved. Seq11/20/23 and other videos have already
  been inspected and are exploratory, not a locked external test.

## Reproducibility and checks

The pinned CMSA-Net, SAMURAI and SAM2Long tracked checkouts were clean at this
audit. Both historical SAMURAI and SAM2Long folders contain an archived
`runner_source.py` with SHA-256
`a1a7688ab7420ef45a20ab4660ede08dab2a85e778512d0411e6aac8ec04ff6f`,
matching their respective setup receipts. The current project runner has a
different hash (`6d45eb32fe7a52763509b27af638a3ffb22e887b1f9725083ca75f0c8ed72b56`);
its diff adds optional model-configuration and smoke-sequence arguments. The
archived source, not an unqualified current-runner claim, identifies the
historical implementation. The comparison used CPU float32 and has not been
shown to reproduce author CUDA results exactly.

`pytest -q tests/test_cmsa_polypgen.py tests/test_samurai_auto_prompt.py
tests/test_davis_author_comparison.py` passed 16 tests. These cover adapter,
geometry, and comparison behavior; they do not establish full runtime
equivalence. A previous exhaustive SAMURAI/SAM2Long saved-mask rescore was
interrupted and has no completion receipt. The checked subsets and other
spot-checks must not be described as an exhaustive audit.
